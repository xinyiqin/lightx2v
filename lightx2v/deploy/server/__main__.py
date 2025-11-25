import argparse
import asyncio
import base64
import json
import mimetypes
import os
import tempfile
import time
import traceback
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from lightx2v.deploy.common.audio_separator import AudioSeparator
from lightx2v.deploy.common.face_detector import FaceDetector
from lightx2v.deploy.common.pipeline import Pipeline
from lightx2v.deploy.common.podcasts import VolcEnginePodcastClient
from lightx2v.deploy.common.utils import check_params, data_name, load_inputs
from lightx2v.deploy.common.volcengine_tts import VolcEngineTTSClient
from lightx2v.deploy.data_manager import LocalDataManager, S3DataManager
from lightx2v.deploy.podcast_manager.local_podcast_manager import LocalPodcastManager
from lightx2v.deploy.podcast_manager.sql_podcast_manager import SQLPodcastManager
from lightx2v.deploy.queue_manager import LocalQueueManager, RabbitMQQueueManager
from lightx2v.deploy.server.auth import AuthManager
from lightx2v.deploy.server.metrics import MetricMonitor
from lightx2v.deploy.server.monitor import ServerMonitor, WorkerStatus
from lightx2v.deploy.server.redis_monitor import RedisServerMonitor
from lightx2v.deploy.task_manager import FinishedStatus, LocalTaskManager, PostgresSQLTaskManager, TaskStatus
from lightx2v.utils.service_utils import ProcessManager

# =========================
# Pydantic Models
# =========================


class TTSRequest(BaseModel):
    text: str
    voice_type: str
    context_texts: str = ""
    emotion: str = ""
    emotion_scale: int = 3
    speech_rate: int = 0
    pitch: int = 0
    loudness_rate: int = 0
    resource_id: str = "seed-tts-1.0"


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class PodcastRequest(BaseModel):
    text: str = ""
    input_url: str = ""
    prompt_text: str = ""
    nlp_texts: str = ""
    action: int = 0
    resource_id: str = "volc.service_type.10050"
    encoding: str = "mp3"
    input_id: str = "podcast"
    speaker_info: str = '{"random_order":false}'
    use_head_music: bool = False
    use_tail_music: bool = False
    only_nlp_text: bool = False
    return_audio_url: bool = False
    skip_round_audio_save: bool = False


# =========================
# FastAPI Related Code
# =========================

model_pipelines = None
task_manager = None
data_manager = None
queue_manager = None
server_monitor = None
auth_manager = None
metrics_monitor = MetricMonitor()
volcengine_tts_client = None
volcengine_podcast_client = None
face_detector = None
audio_separator = None
podcast_manager = None  # 播客管理器（数据库存储）

# 存储正在生成的播客会话的临时目录路径
# key: session_id, value: temp_dir Path
active_podcast_sessions = {}

# 播客历史数据缓存（内存缓存，减少重复读取）
# key: user_id, value: {data: sessions_list, timestamp: cache_time}
podcast_history_cache = {}
PODCAST_HISTORY_CACHE_TTL = 30  # 缓存30秒
PODCAST_HISTORY_MAX_RESULTS = 100  # 最多返回100条历史记录


@asynccontextmanager
async def lifespan(app: FastAPI):
    await task_manager.init()
    await task_manager.mark_server_restart()
    await data_manager.init()
    await queue_manager.init()
    await server_monitor.init()
    # 初始化播客管理器（如果使用数据库）
    if podcast_manager:
        await podcast_manager.init()
    asyncio.create_task(server_monitor.loop())
    yield
    await server_monitor.close()
    await queue_manager.close()
    await data_manager.close()
    await task_manager.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail} for {request.url}")
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})


static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 添加assets目录的静态文件服务
assets_dir = os.path.join(os.path.dirname(__file__), "static", "assets")
app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
security = HTTPBearer()


async def verify_user_access(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = auth_manager.verify_jwt_token(token)
    user_id = payload.get("user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user")
    user = await task_manager.query_user(user_id)
    # logger.info(f"Verfiy user access: {payload}")
    if user is None or user["user_id"] != user_id:
        raise HTTPException(status_code=401, detail="Invalid user")
    return user


async def verify_user_access_from_query(request: Request):
    """从查询参数中验证用户访问权限"""
    # 首先尝试从 Authorization 头部获取 token
    auth_header = request.headers.get("Authorization")
    token = None

    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]  # 移除 "Bearer " 前缀
    else:
        # 如果没有 Authorization 头部，尝试从查询参数获取
        token = request.query_params.get("token")

    payload = auth_manager.verify_jwt_token(token)
    user_id = payload.get("user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user")
    user = await task_manager.query_user(user_id)
    if user is None or user["user_id"] != user_id:
        raise HTTPException(status_code=401, detail="Invalid user")
    return user


async def verify_worker_access(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if not auth_manager.verify_worker_token(token):
        raise HTTPException(status_code=403, detail="Invalid worker token")
    return True


def error_response(e, code):
    return JSONResponse({"message": f"error: {e}!"}, status_code=code)


def format_user_response(user):
    return {
        "user_id": user.get("user_id"),
        "id": user.get("id"),
        "source": user.get("source"),
        "username": user.get("username") or "",
        "email": user.get("email") or "",
        "homepage": user.get("homepage") or "",
        "avatar_url": user.get("avatar_url") or "",
    }


def guess_file_type(name, default_type):
    content_type, _ = mimetypes.guess_type(name)
    if content_type is None:
        content_type = default_type
    return content_type


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/sitemap.xml", response_class=HTMLResponse)
async def sitemap():
    with open(os.path.join(os.path.dirname(__file__), "frontend", "dist", "sitemap.xml"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/auth/login/github")
async def github_auth(request: Request):
    client_id = auth_manager.github_client_id
    redirect_uri = f"{request.base_url}"
    auth_url = f"https://github.com/login/oauth/authorize?client_id={client_id}&redirect_uri={redirect_uri}"
    return {"auth_url": auth_url}


@app.get("/auth/callback/github")
async def github_callback(request: Request):
    try:
        code = request.query_params.get("code")
        if not code:
            return error_response("Missing authorization code", 400)
        user_info = await auth_manager.auth_github(code)
        user_id = await task_manager.create_user(user_info)
        user_info["user_id"] = user_id
        user_response = format_user_response(user_info)
        access_token, refresh_token = auth_manager.create_tokens(user_response)
        logger.info(f"GitHub callback: user_info: {user_response}, access token issued")
        return {"access_token": access_token, "refresh_token": refresh_token, "user_info": user_response}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/auth/login/google")
async def google_auth(request: Request):
    client_id = auth_manager.google_client_id
    redirect_uri = auth_manager.google_redirect_uri
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?client_id={client_id}&redirect_uri={redirect_uri}&response_type=code&scope=openid%20email%20profile&access_type=offline"
    logger.info(f"Google auth: auth_url: {auth_url}")
    return {"auth_url": auth_url}


@app.get("/auth/callback/google")
async def google_callback(request: Request):
    try:
        code = request.query_params.get("code")
        if not code:
            return error_response("Missing authorization code", 400)
        user_info = await auth_manager.auth_google(code)
        user_id = await task_manager.create_user(user_info)
        user_info["user_id"] = user_id
        user_response = format_user_response(user_info)
        access_token, refresh_token = auth_manager.create_tokens(user_response)
        logger.info(f"Google callback: user_info: {user_response}, access token issued")
        return {"access_token": access_token, "refresh_token": refresh_token, "user_info": user_response}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/auth/login/sms")
async def sms_auth(request: Request):
    try:
        phone_number = request.query_params.get("phone_number")
        if not phone_number:
            return error_response("Missing phone number", 400)
        ok = await auth_manager.send_sms(phone_number)
        if not ok:
            return error_response("SMS send failed", 400)
        return {"msg": "SMS send successfully"}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/auth/callback/sms")
async def sms_callback(request: Request):
    try:
        phone_number = request.query_params.get("phone_number")
        verify_code = request.query_params.get("verify_code")
        if not phone_number or not verify_code:
            return error_response("Missing phone number or verify code", 400)
        user_info = await auth_manager.check_sms(phone_number, verify_code)
        if not user_info:
            return error_response("SMS verify failed", 400)

        user_id = await task_manager.create_user(user_info)
        user_info["user_id"] = user_id
        user_response = format_user_response(user_info)
        access_token, refresh_token = auth_manager.create_tokens(user_response)
        logger.info(f"SMS callback: user_info: {user_response}, access token issued")
        return {"access_token": access_token, "refresh_token": refresh_token, "user_info": user_response}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/auth/refresh")
async def refresh_access_token(request: RefreshTokenRequest):
    try:
        payload = auth_manager.verify_refresh_token(request.refresh_token)
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        user = await task_manager.query_user(user_id)
        if user is None or user.get("user_id") != user_id:
            raise HTTPException(status_code=401, detail="Invalid user")
        user_info = format_user_response(user)
        access_token, refresh_token = auth_manager.create_tokens(user_info)
        return {"access_token": access_token, "refresh_token": refresh_token, "user_info": user_info}
    except HTTPException as exc:
        raise exc
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


async def prepare_subtasks(task_id):
    # schedule next subtasks and pend, put to message queue
    subtasks = await task_manager.next_subtasks(task_id)
    for sub in subtasks:
        logger.info(f"Prepare ready subtask: ({task_id}, {sub['worker_name']})")
        r = await queue_manager.put_subtask(sub)
        assert r, "put subtask to queue error"
        await server_monitor.pending_subtasks_add(sub["queue"], sub["task_id"])


def format_task(task):
    task["status"] = task["status"].name
    task["model_cls"] = model_pipelines.outer_model_name(task["model_cls"])


@app.get("/api/v1/model/list")
async def api_v1_model_list(user=Depends(verify_user_access)):
    try:
        return {"models": model_pipelines.get_model_lists()}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/task/submit")
async def api_v1_task_submit(request: Request, user=Depends(verify_user_access)):
    task_id = None
    try:
        msg = await server_monitor.check_user_busy(user["user_id"], active_new_task=True)
        if msg is not True:
            return error_response(msg, 400)
        params = await request.json()
        keys = [params.pop("task"), params.pop("model_cls"), params.pop("stage")]
        keys[1] = model_pipelines.inner_model_name(keys[1])
        assert len(params["prompt"]) > 0, "valid prompt is required"

        # get worker infos, model input names
        workers = model_pipelines.get_workers(keys)
        inputs = model_pipelines.get_inputs(keys)
        outputs = model_pipelines.get_outputs(keys)
        types = model_pipelines.get_types(keys)
        check_params(params, inputs, outputs, types)

        # check if task can be published to queues
        queues = [v["queue"] for v in workers.values()]
        wait_time = await server_monitor.check_queue_busy(keys, queues)
        if wait_time is None:
            return error_response(f"Queue busy, please try again later", 500)

        # process multimodal inputs data
        inputs_data = await load_inputs(params, inputs, types)

        # init task (we need task_id before preprocessing to save processed files)
        task_id = await task_manager.create_task(keys, workers, params, inputs, outputs, user["user_id"])
        logger.info(f"Submit task: {task_id} {params}")

        # save multimodal inputs data
        for inp, data in inputs_data.items():
            await data_manager.save_bytes(data, data_name(inp, task_id))

        await prepare_subtasks(task_id)
        return {"task_id": task_id, "workers": workers, "params": params, "wait_time": wait_time}

    except Exception as e:
        traceback.print_exc()
        if task_id:
            await task_manager.finish_subtasks(task_id, TaskStatus.FAILED, fail_msg=f"submit failed: {e}")
        return error_response(str(e), 500)


@app.get("/api/v1/task/query")
async def api_v1_task_query(request: Request, user=Depends(verify_user_access)):
    try:
        if "task_ids" in request.query_params:
            task_ids = request.query_params["task_ids"].split(",")
            tasks = []
            for task_id in task_ids:
                task_id = task_id.strip()
                if task_id:
                    task, subtasks = await task_manager.query_task(task_id, user["user_id"], only_task=False)
                    if task is not None:
                        task["subtasks"] = await server_monitor.format_subtask(subtasks)
                        format_task(task)
                        tasks.append(task)
            return {"tasks": tasks}

        # 单个任务查询
        task_id = request.query_params["task_id"]
        task, subtasks = await task_manager.query_task(task_id, user["user_id"], only_task=False)
        if task is None:
            return error_response(f"Task {task_id} not found", 404)
        task["subtasks"] = await server_monitor.format_subtask(subtasks)
        format_task(task)
        return task
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/list")
async def api_v1_task_list(request: Request, user=Depends(verify_user_access)):
    try:
        user_id = user["user_id"]
        page = int(request.query_params.get("page", 1))
        page_size = int(request.query_params.get("page_size", 10))
        assert page > 0 and page_size > 0, "page and page_size must be greater than 0"
        status_filter = request.query_params.get("status", None)

        query_params = {"user_id": user_id}
        if status_filter and status_filter != "ALL":
            query_params["status"] = TaskStatus[status_filter.upper()]

        total_tasks = await task_manager.list_tasks(count=True, **query_params)
        total_pages = (total_tasks + page_size - 1) // page_size
        page_info = {"page": page, "page_size": page_size, "total": total_tasks, "total_pages": total_pages}
        if page > total_pages:
            return {"tasks": [], "pagination": page_info}

        query_params["offset"] = (page - 1) * page_size
        query_params["limit"] = page_size

        tasks = await task_manager.list_tasks(**query_params)
        for task in tasks:
            format_task(task)

        return {"tasks": tasks, "pagination": page_info}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/result_url")
async def api_v1_task_result_url(request: Request, user=Depends(verify_user_access)):
    try:
        name = request.query_params["name"]
        task_id = request.query_params["task_id"]
        task = await task_manager.query_task(task_id, user_id=user["user_id"])
        assert task is not None, f"Task {task_id} not found"
        assert task["status"] == TaskStatus.SUCCEED, f"Task {task_id} not succeed"
        assert name in task["outputs"], f"Output {name} not found in task {task_id}"
        assert name not in task["params"], f"Output {name} is a stream"

        url = await data_manager.presign_url(task["outputs"][name])
        if url is None:
            url = f"./assets/task/result?task_id={task_id}&name={name}"
        return {"url": url}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/input_url")
async def api_v1_task_input_url(request: Request, user=Depends(verify_user_access)):
    try:
        name = request.query_params["name"]
        task_id = request.query_params["task_id"]
        filename = request.query_params.get("filename", None)

        task = await task_manager.query_task(task_id, user_id=user["user_id"])
        assert task is not None, f"Task {task_id} not found"
        assert name in task["inputs"], f"Input {name} not found in task {task_id}"
        assert name not in task["params"], f"Input {name} is a stream"

        # eg, multi person audio directory input
        if filename is not None:
            extra_inputs = task["params"]["extra_inputs"][name]
            name = f"{name}/{filename}"
            assert name in task["inputs"], f"Extra input {name} not found in task {task_id}"
            assert name in extra_inputs, f"Filename {filename} not found in extra inputs"

        url = await data_manager.presign_url(task["inputs"][name])
        if url is None:
            url = f"./assets/task/input?task_id={task_id}&name={name}"
            if filename is not None:
                url += f"&filename={filename}"
        return {"url": url}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/assets/task/result")
async def assets_task_result(request: Request, user=Depends(verify_user_access_from_query)):
    try:
        name = request.query_params["name"]
        task_id = request.query_params["task_id"]
        task = await task_manager.query_task(task_id, user_id=user["user_id"])
        assert task is not None, f"Task {task_id} not found"
        assert task["status"] == TaskStatus.SUCCEED, f"Task {task_id} not succeed"
        assert name in task["outputs"], f"Output {name} not found in task {task_id}"
        assert name not in task["params"], f"Output {name} is a stream"
        data = await data_manager.load_bytes(task["outputs"][name])

        #  set correct Content-Type
        content_type = guess_file_type(name, "application/octet-stream")
        headers = {"Content-Disposition": f'attachment; filename="{name}"'}
        headers["Cache-Control"] = "public, max-age=3600"
        return Response(content=data, media_type=content_type, headers=headers)

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/assets/task/input")
async def assets_task_input(request: Request, user=Depends(verify_user_access_from_query)):
    try:
        name = request.query_params["name"]
        task_id = request.query_params["task_id"]
        filename = request.query_params.get("filename", None)

        task = await task_manager.query_task(task_id, user_id=user["user_id"])
        assert task is not None, f"Task {task_id} not found"
        assert name in task["inputs"], f"Input {name} not found in task {task_id}"
        assert name not in task["params"], f"Input {name} is a stream"

        # eg, multi person audio directory input
        if filename is not None:
            extra_inputs = task["params"]["extra_inputs"][name]
            name = f"{name}/{filename}"
            assert name in task["inputs"], f"Extra input {name} not found in task {task_id}"
            assert name in extra_inputs, f"Filename {filename} not found in extra inputs"
        data = await data_manager.load_bytes(task["inputs"][name])

        #  set correct Content-Type
        content_type = guess_file_type(name, "application/octet-stream")
        headers = {"Content-Disposition": f'attachment; filename="{name}"'}
        headers["Cache-Control"] = "public, max-age=3600"
        return Response(content=data, media_type=content_type, headers=headers)

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/cancel")
async def api_v1_task_cancel(request: Request, user=Depends(verify_user_access)):
    try:
        task_id = request.query_params["task_id"]
        ret = await task_manager.cancel_task(task_id, user_id=user["user_id"])
        logger.warning(f"Task {task_id} cancelled: {ret}")
        if ret is True:
            return {"msg": "Task cancelled successfully"}
        else:
            return error_response({"error": f"Task {task_id} cancel failed: {ret}"}, 400)
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/resume")
async def api_v1_task_resume(request: Request, user=Depends(verify_user_access)):
    try:
        task_id = request.query_params["task_id"]

        task = await task_manager.query_task(task_id, user_id=user["user_id"])
        keys = [task["task_type"], task["model_cls"], task["stage"]]
        if not model_pipelines.check_item_by_keys(keys):
            return error_response(f"Model {keys} is not supported now, please submit a new task", 400)

        ret = await task_manager.resume_task(task_id, user_id=user["user_id"], all_subtask=False)
        if ret is True:
            await prepare_subtasks(task_id)
            return {"msg": "ok"}
        else:
            return error_response(f"Task {task_id} resume failed: {ret}", 400)
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.delete("/api/v1/task/delete")
async def api_v1_task_delete(request: Request, user=Depends(verify_user_access)):
    try:
        task_id = request.query_params["task_id"]

        task = await task_manager.query_task(task_id, user["user_id"], only_task=True)
        if not task:
            return error_response("Task not found", 404)

        if task["status"] not in FinishedStatus:
            return error_response("Only finished tasks can be deleted", 400)

        # delete task record
        success = await task_manager.delete_task(task_id, user["user_id"])
        if success:
            logger.info(f"Task {task_id} deleted by user {user['user_id']}")
            return JSONResponse({"message": "Task deleted successfully"})
        else:
            return error_response("Failed to delete task", 400)
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/worker/fetch")
async def api_v1_worker_fetch(request: Request, valid=Depends(verify_worker_access)):
    try:
        params = await request.json()
        logger.info(f"Worker fetching: {params}")
        keys = params.pop("worker_keys")
        identity = params.pop("worker_identity")
        max_batch = params.get("max_batch", 1)
        timeout = params.get("timeout", 5)

        # check client disconnected
        async def check_client(request, fetch_task, identity, queue):
            while True:
                msg = await request.receive()
                if msg["type"] == "http.disconnect":
                    logger.warning(f"Worker {identity} {queue} disconnected, req: {request.client}, {msg}")
                    fetch_task.cancel()
                    await server_monitor.worker_update(queue, identity, WorkerStatus.DISCONNECT)
                    return
                await asyncio.sleep(1)

        # get worker info
        worker = model_pipelines.get_worker(keys)
        await server_monitor.worker_update(worker["queue"], identity, WorkerStatus.FETCHING)

        fetch_task = asyncio.create_task(queue_manager.get_subtasks(worker["queue"], max_batch, timeout))
        check_task = asyncio.create_task(check_client(request, fetch_task, identity, worker["queue"]))
        try:
            subtasks = await asyncio.wait_for(fetch_task, timeout=timeout)
        except asyncio.TimeoutError:
            subtasks = []
            fetch_task.cancel()
        check_task.cancel()

        subtasks = [] if subtasks is None else subtasks
        for sub in subtasks:
            await server_monitor.pending_subtasks_sub(sub["queue"], sub["task_id"])
        valid_subtasks = await task_manager.run_subtasks(subtasks, identity)
        valids = [sub["task_id"] for sub in valid_subtasks]

        if len(valid_subtasks) > 0:
            await server_monitor.worker_update(worker["queue"], identity, WorkerStatus.FETCHED)
            logger.info(f"Worker {identity} {keys} {request.client} fetched {valids}")
        else:
            await server_monitor.worker_update(worker["queue"], identity, WorkerStatus.DISCONNECT)
        return {"subtasks": valid_subtasks}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/worker/report")
async def api_v1_worker_report(request: Request, valid=Depends(verify_worker_access)):
    try:
        params = await request.json()
        logger.info(f"{params}")
        task_id = params.pop("task_id")
        worker_name = params.pop("worker_name")
        status = TaskStatus[params.pop("status")]
        identity = params.pop("worker_identity")
        queue = params.pop("queue")
        fail_msg = params.pop("fail_msg", None)
        await server_monitor.worker_update(queue, identity, WorkerStatus.REPORT)

        ret = await task_manager.finish_subtasks(task_id, status, worker_identity=identity, worker_name=worker_name, fail_msg=fail_msg, should_running=True)

        # not all subtasks finished, prepare new ready subtasks
        if ret not in [TaskStatus.SUCCEED, TaskStatus.FAILED]:
            await prepare_subtasks(task_id)

        # all subtasks succeed, delete temp data
        elif ret == TaskStatus.SUCCEED:
            logger.info(f"Task {task_id} succeed")
            task = await task_manager.query_task(task_id)
            keys = [task["task_type"], task["model_cls"], task["stage"]]
            temps = model_pipelines.get_temps(keys)
            for temp in temps:
                type = model_pipelines.get_type(temp)
                name = data_name(temp, task_id)
                await data_manager.get_delete_func(type)(name)

        elif ret == TaskStatus.FAILED:
            logger.warning(f"Task {task_id} failed")

        return {"msg": "ok"}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/worker/ping/subtask")
async def api_v1_worker_ping_subtask(request: Request, valid=Depends(verify_worker_access)):
    try:
        params = await request.json()
        logger.info(f"{params}")
        task_id = params.pop("task_id")
        worker_name = params.pop("worker_name")
        identity = params.pop("worker_identity")
        queue = params.pop("queue")

        task = await task_manager.query_task(task_id)
        if task is None or task["status"] != TaskStatus.RUNNING:
            return {"msg": "delete"}

        assert await task_manager.ping_subtask(task_id, worker_name, identity)
        await server_monitor.worker_update(queue, identity, WorkerStatus.PING)
        return {"msg": "ok"}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/metrics")
async def api_v1_monitor_metrics():
    try:
        return Response(content=metrics_monitor.get_metrics(), media_type="text/plain")
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/template/asset_url/{template_type}/{filename}")
async def api_v1_template_asset_url(template_type: str, filename: str):
    """get template asset URL - no authentication required"""
    try:
        url = await data_manager.presign_template_url(template_type, filename)
        if url is None:
            url = f"./assets/template/{template_type}/{filename}"
        headers = {"Cache-Control": "public, max-age=3600"}
        return Response(content=json.dumps({"url": url}), media_type="application/json", headers=headers)
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


# Template API endpoints
@app.get("/assets/template/{template_type}/{filename}")
async def assets_template(template_type: str, filename: str):
    """get template file - no authentication required"""
    try:
        if not await data_manager.template_file_exists(template_type, filename):
            return error_response(f"template file {template_type} {filename} not found", 404)
        data = await data_manager.load_template_file(template_type, filename)

        # set media type according to file type
        if template_type == "images":
            if filename.lower().endswith(".png"):
                media_type = "image/png"
            elif filename.lower().endswith((".jpg", ".jpeg")):
                media_type = "image/jpeg"
            else:
                media_type = "application/octet-stream"
        elif template_type == "audios":
            if filename.lower().endswith(".mp3"):
                media_type = "audio/mpeg"
            elif filename.lower().endswith(".wav"):
                media_type = "audio/wav"
            else:
                media_type = "application/octet-stream"
        elif template_type == "videos":
            if filename.lower().endswith(".mp4"):
                media_type = "video/mp4"
            elif filename.lower().endswith(".webm"):
                media_type = "video/webm"
            elif filename.lower().endswith(".avi"):
                media_type = "video/x-msvideo"
            else:
                media_type = "video/mp4"  # default to mp4
        else:
            media_type = "application/octet-stream"

        headers = {"Cache-Control": "public, max-age=3600"}
        return Response(content=data, media_type=media_type, headers=headers)
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/template/list")
async def api_v1_template_list(request: Request):
    """get template file list (support pagination) - no authentication required"""
    try:
        # check page params
        page = int(request.query_params.get("page", 1))
        page_size = int(request.query_params.get("page_size", 12))
        if page < 1 or page_size < 1:
            return error_response("page and page_size must be greater than 0", 400)
        # limit page size
        page_size = min(page_size, 100)

        all_images = await data_manager.list_template_files("images")
        all_audios = await data_manager.list_template_files("audios")
        all_videos = await data_manager.list_template_files("videos")
        all_images = [] if all_images is None else all_images
        all_audios = [] if all_audios is None else all_audios
        all_videos = [] if all_videos is None else all_videos

        # 创建图片文件名（不含扩展名）到图片信息的映射
        all_images_sorted = sorted(all_images)
        image_map = {}  # 文件名（不含扩展名） -> {"filename": 完整文件名, "url": URL}
        for img_name in all_images_sorted:
            img_name_without_ext = img_name.rsplit('.', 1)[0] if '.' in img_name else img_name
            url = await data_manager.presign_template_url("images", img_name)
            if url is None:
                url = f"./assets/template/images/{img_name}"
            image_map[img_name_without_ext] = {"filename": img_name, "url": url}
        
        # 创建音频文件名（不含扩展名）到音频信息的映射
        all_audios_sorted = sorted(all_audios)
        audio_map = {}  # 文件名（不含扩展名） -> {"filename": 完整文件名, "url": URL}
        for audio_name in all_audios_sorted:
            audio_name_without_ext = audio_name.rsplit('.', 1)[0] if '.' in audio_name else audio_name
            url = await data_manager.presign_template_url("audios", audio_name)
            if url is None:
                url = f"./assets/template/audios/{audio_name}"
            audio_map[audio_name_without_ext] = {"filename": audio_name, "url": url}
        
        # 合并音频和图片模板，基于文件名前缀匹配
        # 获取所有唯一的基础文件名（不含扩展名）
        all_base_names = set(list(image_map.keys()) + list(audio_map.keys()))
        all_base_names_sorted = sorted(all_base_names)
        
        # 构建合并后的模板列表
        merged_templates = []
        for base_name in all_base_names_sorted:
            template_item = {
                "id": base_name,  # 使用基础文件名作为ID
                "image": image_map.get(base_name),
                "audio": audio_map.get(base_name)
            }
            merged_templates.append(template_item)
        
        # 分页处理
        total = len(merged_templates)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        
        paginated_templates = []
        if page <= total_pages:
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_templates = merged_templates[start_idx:end_idx]
        
        # 为了保持向后兼容，仍然返回images和audios字段（但可能为空）
        # 同时添加新的merged字段
        return {
            "templates": {
                "images": [],  # 保持向后兼容，但设为空
                "audios": [],  # 保持向后兼容，但设为空
                "videos": [],  # 保持向后兼容
                "merged": paginated_templates  # 新的合并列表
            },
            "pagination": {"page": page, "page_size": page_size, "total": total, "total_pages": total_pages},
        }
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/template/tasks")
async def api_v1_template_tasks(request: Request):
    """get template task list (support pagination) - no authentication required"""
    try:
        # check page params
        page = int(request.query_params.get("page", 1))
        page_size = int(request.query_params.get("page_size", 12))
        category = request.query_params.get("category", None)
        search = request.query_params.get("search", None)
        if page < 1 or page_size < 1:
            return error_response("page and page_size must be greater than 0", 400)
        # limit page size
        page_size = min(page_size, 100)

        all_templates = []
        all_categories = set()
        template_files = await data_manager.list_template_files("tasks")
        template_files = [] if template_files is None else template_files

        for template_file in template_files:
            try:
                bytes_data = await data_manager.load_template_file("tasks", template_file)
                template_data = json.loads(bytes_data)
                template_data["task"]["model_cls"] = model_pipelines.outer_model_name(template_data["task"]["model_cls"])
                all_categories.update(template_data["task"]["tags"])
                if category and category not in template_data["task"]["tags"]:
                    continue
                if search and search not in template_data["task"]["params"]["prompt"] + template_data["task"]["params"]["negative_prompt"] + template_data["task"]["model_cls"] + template_data["task"][
                    "stage"
                ] + template_data["task"]["task_type"] + ",".join(template_data["task"]["tags"]):
                    continue
                all_templates.append(template_data["task"])
            except Exception as e:
                logger.warning(f"Failed to load template file {template_file}: {e}")

        # page info
        total_templates = len(all_templates)
        total_pages = (total_templates + page_size - 1) // page_size
        paginated_templates = []

        if page <= total_pages:
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_templates = all_templates[start_idx:end_idx]

        return {"templates": paginated_templates, "pagination": {"page": page, "page_size": page_size, "total": total_templates, "total_pages": total_pages}, "categories": list(all_categories)}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/template/{template_id}")
async def api_v1_template_get(template_id: str, user=None):
    try:
        template_files = await data_manager.list_template_files("tasks")
        template_files = [] if template_files is None else template_files

        for template_file in template_files:
            try:
                bytes_data = await data_manager.load_template_file("tasks", template_file)
                template_data = json.loads(bytes_data)
                if template_data["task"]["task_id"] == template_id:
                    return template_data["task"]
            except Exception as e:
                logger.warning(f"Failed to load template file {template_file}: {e}")
                continue
        return error_response("Template not found", 404)

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/share/create")
async def api_v1_share_create(request: Request, user=Depends(verify_user_access)):
    try:
        params = await request.json()
        task_id = params["task_id"]
        valid_days = params.get("valid_days", 7)
        auth_type = params.get("auth_type", "public")
        auth_value = params.get("auth_value", "")
        share_type = params.get("share_type", "task")
        assert auth_type == "public", "Only public share is supported now"

        if share_type == "template":
            template = await api_v1_template_get(task_id, user)
            assert isinstance(template, dict) and template["task_id"] == task_id, f"Template {task_id} not found"
        else:
            task = await task_manager.query_task(task_id, user["user_id"], only_task=True)
            assert task, f"Task {task_id} not found"

        if auth_type == "user_id":
            assert auth_value != "", "Target user is required for auth_type = user_id"
            target_user = await task_manager.query_user(auth_value)
            assert target_user and target_user["user_id"] == auth_value, f"Target user {auth_value} not found"

        share_id = await task_manager.create_share(task_id, user["user_id"], share_type, valid_days, auth_type, auth_value)
        return {"share_id": share_id, "share_url": f"/share/{share_id}"}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/share/{share_id}")
async def api_v1_share_get(share_id: str):
    try:
        share_data = await task_manager.query_share(share_id)
        assert share_data, f"Share {share_id} not found or expired or deleted"
        task_id = share_data["task_id"]
        share_type = share_data["share_type"]
        assert share_data["auth_type"] == "public", "Only public share is supported now"

        if share_type == "template":
            task = await api_v1_template_get(task_id, None)
            assert isinstance(task, dict) and task["task_id"] == task_id, f"Template {task_id} not found"
        else:
            task = await task_manager.query_task(task_id, only_task=True)
            assert task, f"Task {task_id} not found"

        user_info = await task_manager.query_user(share_data["user_id"])
        username = user_info.get("username", "用户") if user_info else "用户"

        share_info = {
            "task_id": task_id,
            "share_type": share_type,
            "user_id": share_data["user_id"],
            "username": username,
            "task_type": task["task_type"],
            "model_cls": task["model_cls"],
            "stage": task["stage"],
            "prompt": task["params"].get("prompt", ""),
            "negative_prompt": task["params"].get("negative_prompt", ""),
            "inputs": task["inputs"],
            "outputs": task["outputs"],
            "create_t": task["create_t"],
            "valid_days": share_data["valid_days"],
            "valid_t": share_data["valid_t"],
            "auth_type": share_data["auth_type"],
            "auth_value": share_data["auth_value"],
            "output_video_url": None,
            "input_urls": {},
        }

        for input_name, input_filename in task["inputs"].items():
            if share_type == "template":
                template_type = "images" if "image" in input_name else "audios"
                input_url = await data_manager.presign_template_url(template_type, input_filename)
            else:
                input_url = await data_manager.presign_url(input_filename)
            share_info["input_urls"][input_name] = input_url

        for output_name, output_filename in task["outputs"].items():
            if share_type == "template":
                assert "video" in output_name, "Only video output is supported for template share"
                output_url = await data_manager.presign_template_url("videos", output_filename)
            else:
                output_url = await data_manager.presign_url(output_filename)
            share_info["output_video_url"] = output_url

        return share_info

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/voices/list")
async def api_v1_voices_list():
    try:
        if volcengine_tts_client is None:
            return error_response("Volcengine TTS client not loaded", 500)
        voices = volcengine_tts_client.get_voice_list()
        if voices is None:
            return error_response("No voice list found", 404)
        return voices
    except Exception as e:
        traceback.print_exc()
        return error_response("Failed to get voice list", 500)


@app.post("/api/v1/tts/generate")
async def api_v1_tts_generate(request: TTSRequest):
    """Generate TTS audio from text"""
    try:
        # Validate parameters
        if not request.text.strip():
            return JSONResponse({"error": "Text cannot be empty"}, status_code=400)

        if not request.voice_type:
            return JSONResponse({"error": "Voice type is required"}, status_code=400)

        # Generate unique output filename
        output_filename = f"tts_output_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        # Generate TTS
        success = await volcengine_tts_client.tts_request(
            text=request.text,
            voice_type=request.voice_type,
            context_texts=request.context_texts,
            emotion=request.emotion,
            emotion_scale=request.emotion_scale,
            speech_rate=request.speech_rate,
            loudness_rate=request.loudness_rate,
            pitch=request.pitch,
            output=output_path,
            resource_id=request.resource_id,
        )

        if success and os.path.exists(output_path):
            # Return the audio file
            return FileResponse(output_path, media_type="audio/mpeg", filename=output_filename)
        else:
            return JSONResponse({"error": "TTS generation failed"}, status_code=500)

    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        return JSONResponse({"error": f"TTS generation failed: {str(e)}"}, status_code=500)


@app.websocket("/api/v1/podcast/generate")
async def api_v1_podcast_generate_ws(websocket: WebSocket):
    """WebSocket接口：实时生成播客"""
    await websocket.accept()
    should_stop = False

    # 从查询参数或 headers 获取用户信息
    # 优化：只验证 JWT token，不查询数据库（避免握手超时）
    # 如果需要用户信息，可以在后续异步查询
    user_id = None
    try:
        # 尝试从查询参数获取 token
        token = websocket.query_params.get("token")
        if not token:
            # 尝试从 headers 获取
            auth_header = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]

        if token:
            # 只验证 JWT token，不查询数据库（避免握手超时）
            # 如果需要验证用户存在，可以在后续异步查询
            try:
                payload = auth_manager.verify_jwt_token(token)
                user_id = payload.get("user_id", None)
                if user_id:
                    logger.info(f"WebSocket authenticated user (from token): {user_id}")
                    # 可选：异步验证用户是否存在（不阻塞握手）
                    # 如果验证失败，后续使用 "anonymous"
                    try:
                        user = await asyncio.wait_for(task_manager.query_user(user_id), timeout=2.0)
                        if not user or user.get("user_id") != user_id:
                            logger.warning(f"WebSocket user verification failed: user not found")
                            user_id = None
                    except asyncio.TimeoutError:
                        logger.warning(f"WebSocket user verification timeout, using user_id from token: {user_id}")
                        # 超时情况下，仍然使用 token 中的 user_id
                    except Exception as e:
                        logger.warning(f"WebSocket user verification error: {e}, using user_id from token: {user_id}")
                        # 验证失败时，仍然使用 token 中的 user_id
            except Exception as e:
                logger.warning(f"Failed to verify JWT token: {e}")
                user_id = None
    except Exception as e:
        logger.warning(f"Failed to authenticate WebSocket user: {e}")
        user_id = None

    async def safe_send_json(payload):
        try:
            if websocket.application_state != WebSocketState.CONNECTED:
                return
            await websocket.send_json(payload)
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.info(f"WebSocket send skipped: {e}")

    try:
        # 接收输入
        try:
            data = await websocket.receive_text()
        except Exception as receive_error:
            logger.error(f"Failed to receive WebSocket message: {receive_error}", exc_info=True)
            await safe_send_json({"type": "error", "error": "无法接收消息，请重试"})
            return

        try:
            request_data = json.loads(data)
        except json.JSONDecodeError as json_error:
            logger.error(f"Failed to parse WebSocket message as JSON: {json_error}, data: {data[:200]}")
            await safe_send_json({"type": "error", "error": "消息格式错误，请重试"})
            return

        # 检查是否是停止请求
        if request_data.get("type") == "stop":
            logger.info("Received stop signal from client")
            await safe_send_json({"type": "stopped"})
            return

        input_text = request_data.get("input", "")

        if not input_text:
            await safe_send_json({"error": "输入不能为空"})
            return

        # 判断是URL还是文本
        is_url = input_text.startswith(("http://", "https://"))

        # 使用 data_manager 保存到 podcast 文件夹（按用户组织）
        from pathlib import Path

        timestamp = int(time.time())
        session_id = f"session_{timestamp}"

        # 确定用户ID（如果未认证，使用 "anonymous"）
        if not user_id:
            user_id = "anonymous"

        # 保存用户输入到 data_manager（按用户文件夹组织）
        user_input_path = f"podcast/{user_id}/{session_id}/user_input.txt"
        await data_manager.save_bytes(input_text.encode("utf-8"), user_input_path)
        logger.info(f"Saved user input to data_manager: {user_input_path}")

        # 创建临时目录用于播客生成（podcast_request 需要本地目录）
        temp_session_dir = Path(tempfile.gettempdir()) / f"podcast_temp_{timestamp}"
        temp_session_dir.mkdir(exist_ok=True)
        output_dir = str(temp_session_dir)

        # 将临时目录路径存储到全局字典中，供前端实时访问
        active_podcast_sessions[session_id] = temp_session_dir
        logger.info(f"Stored active session: {session_id} -> {temp_session_dir}")

        params = {
            "text": "" if is_url else input_text,
            "input_url": input_text if is_url else "",
            "action": 0,
            "output_dir": output_dir,
            "use_head_music": False,
            "use_tail_music": False,
            "skip_round_audio_save": False,
        }

        logger.info(f"WebSocket generating podcast with params: {params}")

        merged_audio_file = temp_session_dir / "merged_audio.mp3"
        # HLS playlist (audio-only with mp3 segments)
        hls_playlist_file = temp_session_dir / "playlist.m3u8"
        hls_target_duration = 10  # safe default seconds
        hls_seg_index = 0  # HLS 切片序号

        # 记录字幕时间戳
        subtitle_timestamps = []
        current_audio_duration = 0.0
        merged_audio = None  # 用于存储合并的音频对象

        # 内存缓存：存储所有需要保存到 data_manager 的数据
        memory_cache = {
            "merged_audio_bytes": None,  # 合并后的音频文件
            "hls_playlist_content": "",  # HLS 播放列表内容
            "audio_segments": {},  # 音频分片 {filename: bytes}
            "round_audios": {},  # 轮次音频 {filename: bytes}
            "podcast_texts_content": None,  # 字幕文本内容
            "subtitle_timestamps_json": None,  # 字幕时间戳 JSON
            "metadata_json": None,  # 元数据 JSON（包含 user_id、user_input、rounds）
        }

        # Create initial HLS playlist so前端第一次请求不会404
        hls_header = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:{}\n#EXT-X-MEDIA-SEQUENCE:0\n#EXT-X-PLAYLIST-TYPE:EVENT\n".format(hls_target_duration)
        try:
            with open(hls_playlist_file, "w", encoding="utf-8") as pf:
                pf.write(hls_header)
            # 初始化内存缓存中的 HLS playlist
            memory_cache["hls_playlist_content"] = hls_header
        except Exception as e:
            logger.warning(f"Failed to create initial HLS playlist: {e}")

        # 使用回调函数实时推送音频
        async def on_round_complete(round_data):
            """轮次完成回调 - 实时追加到完整音频，并通过 WebSocket 发送 PCM/WAV 块"""
            nonlocal current_audio_duration, merged_audio, hls_target_duration, hls_seg_index, should_stop

            # 检查是否收到停止信号
            if should_stop:
                logger.info("Stop signal detected in on_round_complete callback")
                return

            async def safe_send_bytes(data: bytes):
                """安全发送二进制数据"""
                try:
                    if websocket.application_state != WebSocketState.CONNECTED:
                        return
                    await websocket.send_bytes(data)
                except (WebSocketDisconnect, RuntimeError) as e:
                    logger.info(f"WebSocket send_bytes skipped: {e}")

            try:
                import struct

                from pydub import AudioSegment

                round_filename = temp_session_dir / f"{round_data['speaker']}_{round_data['round']}.mp3"
                if round_filename.exists():
                    try:
                        new_segment = AudioSegment.from_mp3(str(round_filename))
                        round_duration = len(new_segment) / 1000.0

                        # 转换为 PCM 16-bit LE，单声道，采样率 24000 Hz（兼容 iOS Safari）
                        pcm_segment = new_segment.set_frame_rate(24000).set_channels(1).set_sample_width(2)

                        # 获取 PCM 原始数据
                        pcm_data = pcm_segment.raw_data

                        # 创建 WAV header（44 bytes）
                        sample_rate = 24000
                        num_channels = 1
                        bits_per_sample = 16
                        data_size = len(pcm_data)
                        file_size = 36 + data_size

                        wav_header = struct.pack(
                            "<4sI4s4sIHHIIHH4sI",
                            b"RIFF",  # ChunkID
                            file_size,  # ChunkSize
                            b"WAVE",  # Format
                            b"fmt ",  # Subchunk1ID
                            16,  # Subchunk1Size (PCM)
                            1,  # AudioFormat (PCM)
                            num_channels,  # NumChannels
                            sample_rate,  # SampleRate
                            sample_rate * num_channels * bits_per_sample // 8,  # ByteRate
                            num_channels * bits_per_sample // 8,  # BlockAlign
                            bits_per_sample,  # BitsPerSample
                            b"data",  # Subchunk2ID
                            data_size,  # Subchunk2Size
                        )

                        # 组合 WAV header + PCM data
                        wav_chunk = wav_header + pcm_data

                        # 通过 WebSocket 发送 WAV chunk（二进制）
                        await safe_send_bytes(wav_chunk)
                        logger.info(f"Sent WAV chunk: {len(wav_chunk)} bytes, duration: {round_duration:.2f}s")

                        if merged_audio is None:
                            merged_audio = new_segment
                        else:
                            merged_audio = merged_audio + new_segment

                        # 保存合并后的音频到临时文件（用于前端实时访问）
                        merged_audio.export(str(merged_audio_file), format="mp3")

                        # 等待文件写入完成
                        time.sleep(0.3)

                        # 读取合并后的音频
                        with open(merged_audio_file, "rb") as f:
                            merged_audio_bytes = f.read()
                        merged_file_size = len(merged_audio_bytes)

                        # 保存到内存缓存（最后统一保存）
                        memory_cache["merged_audio_bytes"] = merged_audio_bytes

                        # 不需要实时保存到 data_manager，前端直接从临时目录读取

                        # 记录字幕时间戳
                        subtitle_timestamps.append(
                            {"start": current_audio_duration, "end": current_audio_duration + round_duration, "text": round_data.get("text", ""), "speaker": round_data.get("speaker")}
                        )
                        current_audio_duration += round_duration

                        logger.info(f"Merged audio updated: {merged_file_size} bytes, duration: {current_audio_duration:.2f}s")

                        # 更新/写入 HLS playlist（如单段 >20s，进行 6s 切分）
                        try:
                            # Ensure header exists
                            if not hls_playlist_file.exists():
                                with open(hls_playlist_file, "w", encoding="utf-8") as pf:
                                    pf.write("#EXTM3U\n")
                                    pf.write("#EXT-X-VERSION:3\n")
                                    pf.write(f"#EXT-X-TARGETDURATION:{hls_target_duration}\n")
                                    pf.write("#EXT-X-MEDIA-SEQUENCE:0\n")
                                    pf.write("#EXT-X-PLAYLIST-TYPE:EVENT\n")

                            def append_entry(duration_sec: float, file_path: Path, is_segment=False):
                                nonlocal hls_target_duration, memory_cache
                                d = max(0.1, float(duration_sec))
                                # 更新目标分片时长
                                ceil_d = int(d + 0.999)
                                if ceil_d > hls_target_duration:
                                    hls_target_duration = ceil_d
                                # 直接追加条目到临时文件和内存缓存（使用新的路径结构，只传递文件名）
                                seg_filename = file_path.name
                                entry_line = f"#EXTINF:{d:.3f},\n/api/v1/podcast/audio?session_id={session_id}&filename={seg_filename}\n"
                                # 写入临时文件（用于前端实时访问）
                                with open(hls_playlist_file, "a", encoding="utf-8") as pf:
                                    pf.write(f"#EXTINF:{d:.3f},\n")
                                    pf.write(f"/api/v1/podcast/audio?session_id={session_id}&filename={seg_filename}\n")
                                # 追加到内存缓存
                                memory_cache["hls_playlist_content"] += entry_line

                            if round_duration > 20.0:
                                # 将该轮音频切为 ~6s 小片段
                                slice_ms = 6000
                                total_ms = len(new_segment)
                                start = 0
                                while start < total_ms:
                                    end = min(total_ms, start + slice_ms)
                                    chunk = new_segment[start:end]
                                    chunk_name = f"seg_{hls_seg_index}.mp3"
                                    hls_seg_index += 1
                                    chunk_path = temp_session_dir / chunk_name
                                    chunk.export(str(chunk_path), format="mp3")

                                    # 读取分片并保存到内存缓存（最后统一保存）
                                    with open(chunk_path, "rb") as f:
                                        chunk_bytes = f.read()
                                    memory_cache["audio_segments"][chunk_name] = chunk_bytes

                                    append_entry(len(chunk) / 1000.0, chunk_path, is_segment=True)
                                    start = end
                            else:
                                # 直接作为一个 HLS 分片条目
                                # 读取轮次音频并保存到内存缓存（最后统一保存）
                                with open(round_filename, "rb") as f:
                                    round_bytes = f.read()
                                memory_cache["round_audios"][round_filename.name] = round_bytes
                                append_entry(round_duration, round_filename, is_segment=False)

                            # HLS playlist 内容已经在 append_entry 中更新到内存缓存，这里不需要再次读取
                            # 但为了确保一致性，从临时文件读取最新内容
                            with open(hls_playlist_file, "r", encoding="utf-8") as pf:
                                hls_content = pf.read()
                            memory_cache["hls_playlist_content"] = hls_content

                            # 不需要实时保存到 data_manager，前端直接从临时目录读取
                        except Exception as e:
                            logger.warning(f"Failed updating HLS playlist: {e}")

                        # 推送更新通知（使用新的路径结构，只传递文件名）
                        await safe_send_json(
                            {
                                "type": "audio_update",
                                "data": {
                                    "url": f"/api/v1/podcast/audio?session_id={session_id}&filename=merged_audio.mp3",
                                    "size": merged_file_size,
                                    "duration": current_audio_duration,
                                    "round": round_data.get("round"),
                                    "text": round_data.get("text", ""),
                                    "speaker": round_data.get("speaker"),
                                },
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error processing audio segment: {e}")
            except Exception as e:
                logger.error(f"Error sending round data: {e}")

        params["on_round_complete"] = on_round_complete

        # 创建一个任务来处理停止信号
        async def listen_for_stop():
            nonlocal should_stop
            try:
                while True:
                    if should_stop:
                        break
                    try:
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                        request = json.loads(data)
                        if request.get("type") == "stop":
                            logger.info("Stop signal received during generation")
                            should_stop = True
                            break
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break
            except Exception as e:
                logger.info(f"Stop listener ended: {e}")

        stop_listener_task = asyncio.create_task(listen_for_stop())

        try:
            podcast_texts, podcast_audio = await volcengine_podcast_client.podcast_request(**params)
        except RuntimeError as e:
            # 捕获播客生成错误，发送到前端
            error_msg = str(e)
            logger.error(f"Podcast generation failed: {error_msg}", exc_info=True)
            # 尝试提取更友好的错误信息
            if "invalid url" in error_msg.lower() or "could not extract content" in error_msg.lower():
                user_friendly_error = "无法从提供的链接提取内容，请检查链接是否有效或尝试使用其他链接"
            else:
                user_friendly_error = f"播客生成失败: {error_msg}"

            # 发送错误消息到前端
            try:
                if websocket.application_state == WebSocketState.CONNECTED:
                    await safe_send_json({"type": "error", "error": user_friendly_error})
            except Exception as send_error:
                logger.warning(f"Failed to send error message to client: {send_error}")

            # 清理临时目录
            if "session_id" in locals() and session_id in active_podcast_sessions:
                temp_dir = active_podcast_sessions[session_id]
                del active_podcast_sessions[session_id]
                try:
                    import shutil

                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp directory: {cleanup_error}")
            return
        finally:
            stop_listener_task.cancel()
            try:
                await stop_listener_task
            except asyncio.CancelledError:
                pass

        if should_stop:
            logger.info("Generation stopped by user")
            await safe_send_json({"type": "stopped"})
            return

        # 发送字幕数据
        subtitles = []
        if podcast_texts:
            for item in podcast_texts:
                subtitles.append({"text": item.get("text", ""), "speaker": item.get("speaker", "Unknown")})

        # 保存字幕文本到内存缓存（podcast_texts_*.json）
        if podcast_texts:
            try:
                # 查找临时目录中的 podcast_texts_*.json 文件
                podcast_texts_files = list(temp_session_dir.glob("podcast_texts_*.json"))
                if podcast_texts_files:
                    podcast_texts_file = podcast_texts_files[0]  # 取第一个
                    with open(podcast_texts_file, "r", encoding="utf-8") as f:
                        podcast_texts_content = f.read()
                    memory_cache["podcast_texts_content"] = podcast_texts_content
            except Exception as e:
                logger.warning(f"Failed to read podcast texts: {e}")

        # 保存字幕时间戳到内存缓存
        try:
            timestamp_json = json.dumps(subtitle_timestamps, ensure_ascii=False, indent=2)
            memory_cache["subtitle_timestamps_json"] = timestamp_json
        except Exception as e:
            logger.error(f"Error preparing subtitle timestamps: {e}")

        # 创建统一的元数据文件（包含 user_id、user_input、所有轮次信息）
        # 注意：不保存 CDN URL，因为 CDN URL 可能有过期时间，应该在需要时动态生成
        metadata = {
            "session_id": session_id,
            "user_id": user_id,  # 可能为 None（未认证用户）
            "user_input": input_text,
            "created_at": timestamp,
            "rounds": [],  # 将填充所有轮次信息
            # 不保存 audio_url、hls_url 等，因为这些 URL 可能有过期时间
            # 在需要时通过 data_manager.presign_url() 动态生成 CDN URL
        }

        # 从 subtitle_timestamps 和 podcast_texts 构建 rounds 数据
        if subtitle_timestamps and podcast_texts:
            # 合并时间戳和文本信息
            for i, timestamp_item in enumerate(subtitle_timestamps):
                round_info = {
                    "round": i,  # 轮次序号
                    "text": timestamp_item.get("text", ""),
                    "speaker": timestamp_item.get("speaker", ""),
                    "start": timestamp_item.get("start", 0.0),
                    "end": timestamp_item.get("end", 0.0),
                }
                # 如果 podcast_texts 中有对应项，使用其文本（可能更准确）
                if i < len(podcast_texts):
                    round_info["text"] = podcast_texts[i].get("text", round_info["text"])
                    round_info["speaker"] = podcast_texts[i].get("speaker", round_info["speaker"])
                metadata["rounds"].append(round_info)

        # 保存元数据到内存缓存
        try:
            metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
            memory_cache["metadata_json"] = metadata_json
        except Exception as e:
            logger.error(f"Error preparing metadata: {e}")

        # Append ENDLIST to HLS playlist（临时文件和内存缓存）
        try:
            if hls_playlist_file.exists():
                with open(hls_playlist_file, "a", encoding="utf-8") as pf:
                    pf.write("\n#EXT-X-ENDLIST\n")
                # 更新内存缓存
                memory_cache["hls_playlist_content"] += "\n#EXT-X-ENDLIST\n"
        except Exception as e:
            logger.warning(f"Failed to finalize HLS playlist: {e}")

        # 一次性保存所有数据到 data_manager（只保存音频文件，不按用户区分，类似任务系统）
        logger.info(f"开始批量保存播客数据到 data_manager (session: {session_id})")
        # 使用类似任务系统的命名方式：{session_id}-merged_audio.mp3（不按用户区分）
        merged_audio_path = f"{session_id}-merged_audio.mp3"

        save_success = False
        try:
            # 只保存 merged_audio.mp3 到 S3（不保存metadata.json到S3）
            if memory_cache["merged_audio_bytes"]:
                # 添加重试机制（最多重试3次）
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        await data_manager.save_bytes(memory_cache["merged_audio_bytes"], merged_audio_path)
                        # 验证文件是否真的保存成功
                        if await data_manager.file_exists(merged_audio_path):
                            logger.info(f"Saved merged audio to data_manager: {merged_audio_path}")
                            save_success = True
                            break
                        else:
                            logger.warning(f"File saved but not found, retrying... (attempt {retry_count + 1}/{max_retries})")
                            retry_count += 1
                            if retry_count < max_retries:
                                await asyncio.sleep(1)  # 等待1秒后重试
                    except Exception as e:
                        logger.warning(f"Error saving merged audio (attempt {retry_count + 1}/{max_retries}): {e}")
                        retry_count += 1
                        if retry_count < max_retries:
                            await asyncio.sleep(1)  # 等待1秒后重试
                        else:
                            raise  # 最后一次重试失败，抛出异常

                if not save_success:
                    raise Exception(f"Failed to save merged audio after {max_retries} attempts")

            # 保存元数据到数据库/本地文件（不保存到S3）
            if memory_cache.get("metadata_json"):
                import json as json_module

                metadata = json_module.loads(memory_cache["metadata_json"])

                # 在元数据中添加outputs字段（类似任务系统）
                metadata["outputs"] = {
                    "merged_audio": merged_audio_path  # 存储音频文件路径/ID
                }

                # 保存到数据库/本地文件
                if podcast_manager and user_id:
                    try:
                        await podcast_manager.insert_podcast(
                            session_id=session_id,
                            user_id=user_id,
                            user_input=input_text,
                            audio_path=merged_audio_path,
                            metadata_path="",  # 不再保存metadata.json到S3
                            rounds=metadata.get("rounds", []),
                            subtitles=subtitles,
                            extra_info={"timestamp": timestamp, "outputs": metadata["outputs"]},
                        )
                        logger.info(f"Saved podcast metadata to database: {session_id}")
                    except Exception as e:
                        logger.warning(f"Failed to save podcast to database: {e}")

            logger.info(f"批量保存完成 (session: {session_id})")
        except Exception as e:
            logger.error(f"Error saving data to data_manager: {e}", exc_info=True)
            # 如果保存失败，不要从 active_podcast_sessions 中移除，让前端可以从内存中获取
            if not save_success and session_id in active_podcast_sessions:
                logger.warning(f"Audio save failed, keeping session in active_podcast_sessions for memory access")

        # 尝试获取 CDN URL（如果支持）- 使用新的路径结构（类似任务系统）
        # 如果保存失败，尝试从内存中提供音频
        if save_success:
            audio_url = await data_manager.presign_url(merged_audio_path)
            if not audio_url:
                # 使用类似任务系统的URL格式
                audio_url = f"/api/v1/podcast/audio?session_id={session_id}&filename=merged_audio.mp3"
        else:
            # 保存失败，使用内存中的音频（通过 active_podcast_sessions）
            logger.warning(f"Using memory audio URL due to save failure")
            audio_url = f"/api/v1/podcast/audio?session_id={session_id}&filename=merged_audio.mp3"

        # timestamps 信息已包含在 metadata.json 中，不需要单独的文件
        timestamps_url = None

        # 发送完成消息（优先使用 CDN URL）
        await safe_send_json({"type": "complete", "data": {"audio_url": audio_url, "subtitles": subtitles, "session_id": session_id, "user_id": user_id}})

        # 只有在保存成功时才从全局字典中移除（生成完成后，前端应该从 data_manager 读取）
        # 如果保存失败，保留在 active_podcast_sessions 中，让前端可以从内存中获取
        if save_success and session_id in active_podcast_sessions:
            del active_podcast_sessions[session_id]
            logger.info(f"Removed session from active_podcast_sessions: {session_id}")
            # 清除该用户的历史缓存，确保新生成的数据能立即显示
            if user_id in podcast_history_cache:
                del podcast_history_cache[user_id]
                logger.debug(f"Cleared podcast history cache for user {user_id}")
        elif not save_success:
            logger.warning(f"Keeping session in active_podcast_sessions due to save failure: {session_id}")

        # 清理临时目录
        try:
            import shutil

            shutil.rmtree(temp_session_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        # 清理临时目录（如果存在）
        if "session_id" in locals() and session_id in active_podcast_sessions:
            temp_dir = active_podcast_sessions[session_id]
            del active_podcast_sessions[session_id]
            try:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory on disconnect: {e}")
    except asyncio.TimeoutError as e:
        logger.error(f"WebSocket timeout error: {e}")
        # 清理临时目录（如果存在）
        if "session_id" in locals() and session_id in active_podcast_sessions:
            temp_dir = active_podcast_sessions[session_id]
            del active_podcast_sessions[session_id]
            try:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp directory on timeout: {cleanup_error}")
        # 尝试发送错误消息（如果连接仍然有效）
        try:
            if websocket.application_state == WebSocketState.CONNECTED:
                await safe_send_json({"error": "Request timeout"})
        except Exception:
            pass  # 忽略发送错误
    except Exception as e:
        # 修复日志格式化错误：使用 f-string 或 repr() 避免格式化冲突
        try:
            error_str = repr(e) if isinstance(e, (dict, str)) and "{" in str(e) else str(e)
            logger.error(f"Error in WebSocket: {error_str}", exc_info=True)
        except Exception as log_error:
            # 如果日志记录也失败，使用最基本的错误信息
            logger.error(f"Error in WebSocket (logging failed: {log_error}): {type(e).__name__}: {e}", exc_info=True)
        # 清理临时目录（如果存在）
        if "session_id" in locals() and session_id in active_podcast_sessions:
            temp_dir = active_podcast_sessions[session_id]
            del active_podcast_sessions[session_id]
            try:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as cleanup_error:
                logger.warning("Failed to cleanup temp directory on error: %s", cleanup_error)
        # 尝试发送错误消息（如果连接仍然有效）
        try:
            if websocket.application_state == WebSocketState.CONNECTED:
                # 提取更友好的错误信息
                error_msg = str(e)
                if "invalid url" in error_msg.lower() or "could not extract content" in error_msg.lower():
                    user_friendly_error = "无法从提供的链接提取内容，请检查链接是否有效或尝试使用其他链接"
                elif "RuntimeError" in error_msg:
                    # 尝试从 RuntimeError 中提取实际错误信息
                    if "Server error:" in error_msg:
                        try:
                            # 使用全局导入的 json 模块，不要重新导入
                            error_json = json.loads(error_msg.split("Server error: ")[1])
                            if "error" in error_json:
                                user_friendly_error = error_json["error"]
                            else:
                                user_friendly_error = error_msg
                        except Exception:
                            user_friendly_error = error_msg
                    else:
                        user_friendly_error = error_msg
                else:
                    user_friendly_error = error_msg

                await safe_send_json({"type": "error", "error": user_friendly_error})
        except Exception:
            pass  # 忽略发送错误


@app.post("/api/v1/podcast/generate")
async def api_v1_podcast_generate(request: PodcastRequest, user=Depends(verify_user_access)):
    """HTTP接口：生成播客"""
    try:
        if not volcengine_podcast_client:
            return JSONResponse({"error": "Podcast client not initialized"}, status_code=500)

        # 判断是URL还是文本
        is_url = request.input_url.startswith(("http://", "https://")) if request.input_url else False

        if not request.text and not request.input_url:
            return JSONResponse({"error": "Text or input_url is required"}, status_code=400)

        # 创建输出目录
        output_dir = os.path.join(tempfile.gettempdir(), f"podcast_{uuid.uuid4().hex}")
        os.makedirs(output_dir, exist_ok=True)

        params = {
            "text": "" if is_url else request.text,
            "input_url": request.input_url if is_url else "",
            "prompt_text": request.prompt_text,
            "nlp_texts": request.nlp_texts,
            "action": request.action,
            "resource_id": request.resource_id,
            "encoding": request.encoding,
            "input_id": request.input_id,
            "speaker_info": request.speaker_info,
            "use_head_music": request.use_head_music,
            "use_tail_music": request.use_tail_music,
            "only_nlp_text": request.only_nlp_text,
            "return_audio_url": request.return_audio_url,
            "skip_round_audio_save": request.skip_round_audio_save,
            "output_dir": output_dir,
        }

        logger.info(f"Generating podcast with params: {params}")

        podcast_texts, podcast_audio = await volcengine_podcast_client.podcast_request(**params)

        if not podcast_audio:
            return JSONResponse({"error": "未生成音频数据"}, status_code=500)

        # 保存音频文件
        timestamp = time.time()
        audio_filename = f"podcast_{timestamp}.{request.encoding}"
        audio_path = os.path.join(output_dir, audio_filename)

        with open(audio_path, "wb") as f:
            f.write(bytes(podcast_audio))

        logger.info(f"Audio saved to: {audio_path}")

        # 构建响应
        subtitles = []
        if podcast_texts:
            for item in podcast_texts:
                subtitles.append({"text": item.get("text", ""), "speaker": item.get("speaker", "Unknown")})

        return {"audio_url": f"/api/v1/podcast/audio?session_id={os.path.basename(output_dir)}&filename={audio_filename}", "subtitles": subtitles, "message": "播客生成成功"}

    except Exception as e:
        logger.error(f"Error generating podcast: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/v1/podcast/audio")
@app.head("/api/v1/podcast/audio")
async def api_v1_podcast_audio(request: Request, user=Depends(verify_user_access)):
    """返回播客音频文件（优先从临时目录读取，生成完成后从 data_manager 读取，使用新路径格式：{session_id}-merged_audio.mp3）"""
    try:
        user_id = user.get("user_id")
        if not user_id:
            return JSONResponse({"error": "User authentication required"}, status_code=401)

        session_id = request.query_params.get("session_id")
        filename = request.query_params.get("filename")

        if not session_id or not filename:
            return JSONResponse({"error": "session_id and filename are required"}, status_code=400)

        # 先移除 filename 中的查询参数（前端可能在 URL 中添加 ?t=timestamp 来避免缓存）
        if "?" in filename:
            filename = filename.split("?")[0]

        # 优先从数据库/本地文件读取元数据，获取outputs中的音频路径
        file_path = None
        if podcast_manager:
            try:
                podcast_data = await podcast_manager.query_podcast(session_id, user_id)
                if podcast_data:
                    # 优先从extra_info中的outputs获取
                    outputs = podcast_data.get("extra_info", {}).get("outputs", {})
                    if outputs and "merged_audio" in outputs:
                        file_path = outputs["merged_audio"]
                    elif podcast_data.get("audio_path"):
                        # 兼容旧数据
                        file_path = podcast_data.get("audio_path")
            except Exception as e:
                logger.warning(f"Failed to read podcast data for {session_id}: {e}")

        # 如果数据库中没有，尝试新路径格式：{session_id}-merged_audio.mp3
        if not file_path:
            if filename == "merged_audio.mp3":
                file_path = f"{session_id}-merged_audio.mp3"
            else:
                file_path = f"{session_id}-{filename}"

        # 兼容旧路径格式（如果新路径不存在）
        if not await data_manager.file_exists(file_path):
            old_file_path = f"podcast/{user_id}/{session_id}/{filename}"
            if await data_manager.file_exists(old_file_path):
                file_path = old_file_path

        # 实际文件名就是 filename（已经是文件名）
        actual_filename = filename

        # 检查 Range 请求头
        range_header = request.headers.get("Range")
        start_byte = None
        end_byte = None

        if range_header:
            # 解析 Range 头，格式：bytes=start-end 或 bytes=start-
            import re

            match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if match:
                start_byte = int(match.group(1))
                end_byte = int(match.group(2)) if match.group(2) else None

        # 优先从临时目录读取（如果会话正在生成中）
        logger.debug(f"Checking active sessions: {list(active_podcast_sessions.keys())}, looking for: {session_id}")
        file_size = 0
        file_bytes = None

        if session_id in active_podcast_sessions:
            temp_dir = active_podcast_sessions[session_id]
            temp_file = temp_dir / actual_filename

            logger.debug(f"Session {session_id} is active, checking temp file: {temp_file}")
            if temp_file.exists():
                logger.info(f"Serving audio file from temp directory: {temp_file}")
                file_size = temp_file.stat().st_size

                # HEAD 请求不需要读取文件内容
                if request.method != "HEAD":
                    if range_header:
                        # Range 请求：只读取指定范围
                        with open(temp_file, "rb") as f:
                            f.seek(start_byte)
                            if end_byte is not None:
                                file_bytes = f.read(end_byte - start_byte + 1)
                            else:
                                file_bytes = f.read()
                    else:
                        file_bytes = temp_file.read_bytes()
            else:
                # 临时文件不存在，尝试从 data_manager 读取
                logger.warning(f"Temp file not found: {temp_file}, trying data_manager: {file_path}")
                if not await data_manager.file_exists(file_path):
                    logger.error(f"Audio file not found in temp or data_manager: {file_path}")
                    return JSONResponse({"error": f"Audio file not found: {filename}"}, status_code=404)
                logger.debug(f"Serving audio file from data_manager: {file_path}")

                # 对于 data_manager，需要读取文件才能获取大小
                # 但如果是 HEAD 请求且没有 Range，可以尝试只获取元数据
                if request.method == "HEAD" and not range_header:
                    # HEAD 请求：尝试获取文件大小（如果 data_manager 支持）
                    # 否则需要读取文件
                    file_bytes_temp = await data_manager.load_bytes(file_path)
                    file_size = len(file_bytes_temp)
                else:
                    file_bytes = await data_manager.load_bytes(file_path)
                    file_size = len(file_bytes)
                    if range_header:
                        # Range 请求：只返回指定范围
                        if end_byte is not None:
                            file_bytes = file_bytes[start_byte : end_byte + 1]
                        else:
                            file_bytes = file_bytes[start_byte:]
        else:
            # 会话已完成，从 data_manager 读取
            logger.debug(f"Session {session_id} not in active sessions, reading from data_manager: {file_path}")
            if not await data_manager.file_exists(file_path):
                logger.error(f"Audio file not found in data_manager: {file_path}")
                return JSONResponse({"error": f"Audio file not found: {filename}"}, status_code=404)
            logger.info(f"Serving audio file from data_manager: {file_path}")

            # HEAD 请求：只获取文件大小
            if request.method == "HEAD" and not range_header:
                file_bytes_temp = await data_manager.load_bytes(file_path)
                file_size = len(file_bytes_temp)
            else:
                file_bytes = await data_manager.load_bytes(file_path)
                file_size = len(file_bytes)
                if range_header:
                    # Range 请求：只返回指定范围
                    if end_byte is not None:
                        file_bytes = file_bytes[start_byte : end_byte + 1]
                    else:
                        file_bytes = file_bytes[start_byte:]

        # set content-type by extension
        from pathlib import Path

        ext = Path(actual_filename).suffix.lower()
        media_type = "application/octet-stream"
        if ext == ".mp3":
            media_type = "audio/mpeg"
        elif ext == ".wav":
            media_type = "audio/wav"
        elif ext == ".m3u8":
            media_type = "application/vnd.apple.mpegurl"
        elif ext == ".json":
            media_type = "application/json"

        # For HLS playlists that are frequently updated, read a snapshot to avoid
        # Content-Length mismatches when the file is appended during response.
        if ext == ".m3u8":
            # HEAD 请求：只返回头部
            if request.method == "HEAD":
                headers = {
                    "Cache-Control": "no-store, max-age=0",
                    "Pragma": "no-cache",
                    "Content-Type": media_type,
                    "Content-Length": str(file_size),
                }
                return Response(status_code=200, headers=headers)

            # GET 请求：返回内容（需要读取文件）
            if file_bytes is None:
                # 如果 HEAD 请求时没有读取文件，现在需要读取
                if session_id in active_podcast_sessions:
                    temp_dir = active_podcast_sessions[session_id]
                    temp_file = temp_dir / actual_filename
                    if temp_file.exists():
                        file_bytes = temp_file.read_bytes()
                    else:
                        file_bytes = await data_manager.load_bytes(file_path)
                else:
                    file_bytes = await data_manager.load_bytes(file_path)

            try:
                content = file_bytes.decode("utf-8")
            except Exception as e:
                logger.error(f"Error reading m3u8 file: {e}")
                return JSONResponse({"error": "Failed to read playlist"}, status_code=500)
            headers = {
                "Cache-Control": "no-store, max-age=0",
                "Pragma": "no-cache",
            }
            return Response(content=content, media_type=media_type, headers=headers)

        # 处理 HEAD 请求（只返回头部信息，不返回内容）
        if request.method == "HEAD":
            headers = {
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Content-Type": media_type,
            }
            if range_header:
                # HEAD 请求 + Range：返回 206 状态码和 Content-Range
                actual_start = start_byte
                actual_end = end_byte if end_byte is not None else file_size - 1
                content_length = actual_end - actual_start + 1 if end_byte is not None else file_size - actual_start
                headers["Content-Range"] = f"bytes {actual_start}-{actual_start + content_length - 1}/{file_size}"
                headers["Content-Length"] = str(content_length)
                return Response(status_code=206, headers=headers)
            else:
                return Response(status_code=200, headers=headers)

        # GET 请求：确保文件内容已读取
        if file_bytes is None:
            # 如果之前是 HEAD 请求，现在需要读取文件内容
            if session_id in active_podcast_sessions:
                temp_dir = active_podcast_sessions[session_id]
                temp_file = temp_dir / actual_filename
                if temp_file.exists():
                    if range_header:
                        with open(temp_file, "rb") as f:
                            f.seek(start_byte)
                            if end_byte is not None:
                                file_bytes = f.read(end_byte - start_byte + 1)
                            else:
                                file_bytes = f.read()
                    else:
                        file_bytes = temp_file.read_bytes()
                else:
                    file_bytes = await data_manager.load_bytes(file_path)
                    if range_header:
                        if end_byte is not None:
                            file_bytes = file_bytes[start_byte : end_byte + 1]
                        else:
                            file_bytes = file_bytes[start_byte:]
            else:
                file_bytes = await data_manager.load_bytes(file_path)
                if range_header:
                    if end_byte is not None:
                        file_bytes = file_bytes[start_byte : end_byte + 1]
                    else:
                        file_bytes = file_bytes[start_byte:]

        # 处理 Range 请求（GET 方法）
        if range_header:
            # 计算实际返回的范围
            actual_start = start_byte
            actual_end = end_byte if end_byte is not None else file_size - 1
            content_length = len(file_bytes)

            headers = {
                "Content-Range": f"bytes {actual_start}-{actual_start + content_length - 1}/{file_size}",
                "Content-Length": str(content_length),
                "Accept-Ranges": "bytes",
                "Content-Type": media_type,
                "Content-Disposition": f'attachment; filename="{Path(file_path).name}"',
            }
            return Response(
                content=file_bytes,
                media_type=media_type,
                status_code=206,  # Partial Content
                headers=headers,
            )

        # 返回完整音频文件（GET 方法）
        headers = {"Content-Length": str(len(file_bytes)), "Accept-Ranges": "bytes", "Content-Type": media_type, "Content-Disposition": f'attachment; filename="{Path(file_path).name}"'}
        return Response(content=file_bytes, media_type=media_type, headers=headers)

    except Exception as e:
        logger.error(f"Error serving audio: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/v1/podcast/history")
async def api_v1_podcast_history(request: Request, user=Depends(verify_user_access)):
    """获取播客历史记录列表（只返回当前用户的metadata，不返回音频URL）"""
    try:
        user_id = user.get("user_id")
        if not user_id:
            return {"sessions": []}

        # 如果使用数据库，优先使用数据库查询（快速）
        if podcast_manager:
            try:
                page = int(request.query_params.get("page", 1))
                page_size = int(request.query_params.get("page_size", 10))
                status = request.query_params.get("status", None)  # has_audio, no_audio

                result = await podcast_manager.list_podcasts(user_id=user_id, page=page, page_size=page_size, status=status)
                logger.debug(f"Query podcast history from database: {len(result['sessions'])} sessions")
                return result
            except Exception as e:
                logger.warning(f"Database query failed, falling back to file system: {e}")
                # 如果数据库查询失败，回退到文件系统查询

        # 文件系统模式：从S3查找音频文件（兼容旧数据）
        # 注意：新数据已不再保存metadata.json到S3，只保存音频文件
        # 这里主要用于兼容旧数据，新数据应该通过数据库查询

        # 检查缓存（文件系统模式）
        current_time = time.time()
        if user_id in podcast_history_cache:
            cache_entry = podcast_history_cache[user_id]
            if current_time - cache_entry["timestamp"] < PODCAST_HISTORY_CACHE_TTL:
                logger.debug(f"Returning cached podcast history for user {user_id}")
                return {"sessions": cache_entry["data"]}

        # 从S3查找所有音频文件（新格式：{session_id}-merged_audio.mp3）
        # 列出所有以"-merged_audio.mp3"结尾的文件
        all_files = await data_manager.list_files("")  # 列出根目录所有文件
        session_ids = set()

        for item in all_files:
            # 新格式：{session_id}-merged_audio.mp3
            if item.endswith("-merged_audio.mp3"):
                session_id = item.replace("-merged_audio.mp3", "")
                if session_id.startswith("session_"):
                    session_ids.add(session_id)
            # 兼容旧格式：podcast/{user_id}/{session_id}/merged_audio.mp3
            elif item.startswith(f"podcast/{user_id}/") and "/merged_audio.mp3" in item:
                parts = item.split("/")
                if len(parts) >= 3 and parts[2].startswith("session_"):
                    session_ids.add(parts[2])

        if not session_ids:
            result = {"sessions": []}
            podcast_history_cache[user_id] = {"data": [], "timestamp": current_time}
            return result

        # 按时间倒序排序，只处理最新的 N 条记录
        sorted_session_ids = sorted(session_ids, reverse=True)[:PODCAST_HISTORY_MAX_RESULTS]

        # 并行处理所有 session
        async def process_session(session_id):
            # 检查新格式音频文件是否存在
            new_audio_path = f"{session_id}-merged_audio.mp3"
            has_audio = await data_manager.file_exists(new_audio_path)

            # 如果新格式不存在，检查旧格式（兼容）
            if not has_audio:
                old_audio_path = f"podcast/{user_id}/{session_id}/merged_audio.mp3"
                has_audio = await data_manager.file_exists(old_audio_path)

            # 尝试从数据库/本地文件读取元数据
            user_input_preview = ""
            created_at = None

            if podcast_manager:
                try:
                    podcast_data = await podcast_manager.query_podcast(session_id, user_id)
                    if podcast_data:
                        user_input_preview = (podcast_data.get("user_input", "") or "")[:100]
                        created_at = podcast_data.get("created_at")
                        # 处理datetime对象
                        if hasattr(created_at, "isoformat"):
                            created_at = created_at.isoformat()
                        elif isinstance(created_at, (int, float)):
                            from datetime import datetime as dt

                            created_at = dt.fromtimestamp(created_at).isoformat()
                except Exception as e:
                    logger.debug(f"Failed to read podcast data for {session_id}: {e}")

            # 如果数据库中没有，尝试从S3读取旧metadata（兼容）
            if not user_input_preview:
                old_metadata_path = f"podcast/{user_id}/{session_id}/metadata.json"
                if await data_manager.file_exists(old_metadata_path):
                    try:
                        metadata_bytes = await data_manager.load_bytes(old_metadata_path)
                        if metadata_bytes:
                            import json as json_module

                            metadata = json_module.loads(metadata_bytes.decode("utf-8"))
                            user_input_preview = (metadata.get("user_input", "") or "")[:100]
                            created_at = metadata.get("created_at")
                    except Exception as e:
                        logger.debug(f"Failed to read old metadata for {session_id}: {e}")

            return {
                "session_id": session_id,
                "user_id": user_id,
                "user_input": user_input_preview or f"会话 {session_id}",
                "has_audio": has_audio,
                "created_at": created_at,
            }

        # 并行处理所有 session（增加并发度到 50，提高性能）
        batch_size = 50
        sessions = []

        for i in range(0, len(sorted_session_ids), batch_size):
            batch = sorted_session_ids[i : i + batch_size]
            batch_results = await asyncio.gather(*[process_session(session_id) for session_id in batch], return_exceptions=True)

            for result in batch_results:
                if result and not isinstance(result, Exception):
                    sessions.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Error processing session: {result}")

        # 按 created_at 排序（如果存在），确保最新的在前面
        if sessions and any(s.get("created_at") for s in sessions):
            sessions.sort(key=lambda x: x.get("created_at") or "", reverse=True)

        result = {"sessions": sessions}

        # 更新缓存
        podcast_history_cache[user_id] = {"data": sessions, "timestamp": current_time}

        return result
    except Exception as e:
        logger.error(f"Error getting podcast history: {e}")
        traceback.print_exc()
        return {"sessions": []}


@app.get("/api/v1/podcast/session/{session_id}")
async def api_v1_podcast_session_detail(session_id: str, user=Depends(verify_user_access)):
    """获取播客会话的详细信息（字幕、时间戳等，从数据库/本地文件读取）"""
    try:
        user_id = user.get("user_id")
        if not user_id:
            return JSONResponse({"error": "User authentication required"}, status_code=401)

        import json as json_module

        # 优先从数据库/本地文件读取元数据
        metadata = None
        rounds = []
        outputs = {}

        if podcast_manager:
            try:
                podcast_data = await podcast_manager.query_podcast(session_id, user_id)
                if podcast_data:
                    # 从数据库读取
                    rounds = podcast_data.get("rounds", [])
                    if isinstance(rounds, str):
                        rounds = json_module.loads(rounds)
                    outputs = podcast_data.get("extra_info", {}).get("outputs", {})
                    if not outputs and podcast_data.get("audio_path"):
                        # 兼容旧数据：使用audio_path
                        outputs = {"merged_audio": podcast_data.get("audio_path")}

                    metadata = {
                        "session_id": session_id,
                        "user_id": user_id,
                        "user_input": podcast_data.get("user_input", ""),
                        "created_at": podcast_data.get("created_at"),
                        "rounds": rounds,
                        "outputs": outputs,
                    }
            except Exception as e:
                logger.warning(f"Failed to read metadata from database for {session_id}: {e}")

        # 如果数据库中没有，尝试从S3读取（兼容旧数据）
        if not metadata:
            metadata_path = f"podcast/{user_id}/{session_id}/metadata.json"
            if await data_manager.file_exists(metadata_path):
                try:
                    metadata_bytes = await data_manager.load_bytes(metadata_path)
                    if metadata_bytes is not None:
                        metadata = json_module.loads(metadata_bytes.decode("utf-8"))
                        rounds = metadata.get("rounds", [])
                        outputs = metadata.get("outputs", {})
                except Exception as e:
                    logger.warning(f"Failed to read metadata from S3 for {session_id}: {e}")

        if not metadata:
            return JSONResponse({"error": "Podcast session not found"}, status_code=404)

        # 从 rounds 构建 subtitles 和 timestamps
        subtitles = []
        timestamps = []
        if rounds:
            for round_info in rounds:
                subtitles.append({"text": round_info.get("text", ""), "speaker": round_info.get("speaker", "")})
                timestamps.append({"start": round_info.get("start", 0.0), "end": round_info.get("end", 0.0), "text": round_info.get("text", ""), "speaker": round_info.get("speaker", "")})

        return {
            "session_id": session_id,
            "user_id": user_id,
            "rounds": rounds,
            "subtitles": subtitles,
            "timestamps": timestamps,
            "outputs": outputs,  # 返回outputs，前端可以通过这个获取音频URL
            "metadata": metadata,
        }
    except Exception as e:
        logger.error(f"Error getting podcast session detail: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/v1/podcast/session/{session_id}/audio_url")
async def api_v1_podcast_session_audio_url(session_id: str, user=Depends(verify_user_access)):
    """获取播客会话的音频 CDN URL（通过元数据中的outputs获取）"""
    try:
        user_id = user.get("user_id")
        if not user_id:
            return JSONResponse({"error": "User authentication required"}, status_code=401)

        # 从数据库/本地文件读取元数据，获取outputs
        audio_path = None
        if podcast_manager:
            try:
                podcast_data = await podcast_manager.query_podcast(session_id, user_id)
                if podcast_data:
                    # 优先从extra_info中的outputs获取
                    outputs = podcast_data.get("extra_info", {}).get("outputs", {})
                    if outputs and "merged_audio" in outputs:
                        audio_path = outputs["merged_audio"]
                    elif podcast_data.get("audio_path"):
                        # 兼容旧数据
                        audio_path = podcast_data.get("audio_path")
            except Exception as e:
                logger.warning(f"Failed to read podcast data for {session_id}: {e}")

        # 如果数据库中没有，尝试从S3读取（兼容旧数据）
        if not audio_path:
            old_audio_path = f"podcast/{user_id}/{session_id}/merged_audio.mp3"
            if await data_manager.file_exists(old_audio_path):
                audio_path = old_audio_path

        # 如果还是没有，使用新的路径格式
        if not audio_path:
            audio_path = f"{session_id}-merged_audio.mp3"

        # 检查文件是否存在
        if not await data_manager.file_exists(audio_path):
            return JSONResponse({"error": "Audio file not found"}, status_code=404)

        # 尝试获取 CDN URL
        audio_cdn_url = await data_manager.presign_url(audio_path)
        if audio_cdn_url:
            return {"audio_url": audio_cdn_url}
        else:
            # 回退到 API URL
            audio_url = f"/api/v1/podcast/audio?session_id={session_id}&filename=merged_audio.mp3"
            return {"audio_url": audio_url}
    except Exception as e:
        logger.error(f"Error getting podcast session audio URL: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


class FaceDetectRequest(BaseModel):
    image: str  # Base64 encoded image


class AudioSeparateRequest(BaseModel):
    audio: str  # Base64 encoded audio
    num_speakers: int = None  # Optional: number of speakers to separate


@app.post("/api/v1/face/detect")
async def api_v1_face_detect(request: FaceDetectRequest, user=Depends(verify_user_access)):
    """Detect faces in image (only detection, no cropping - cropping is done on frontend)
    Supports both base64 encoded images and URLs (blob URLs, http URLs, etc.)
    """
    try:
        if not face_detector:
            return error_response("Face detector not initialized", 500)

        # 验证输入
        if not request.image or not request.image.strip():
            logger.error("Face detection request: image is empty")
            return error_response("Image input is empty", 400)

        image_bytes = None

        # Check if input is a URL (blob:, http:, https:, or data: URL)
        if request.image.startswith("data:image"):
            # Data URL format: "data:image/png;base64,..."
            try:
                header, encoded = request.image.split(",", 1)
                image_bytes = base64.b64decode(encoded)
            except ValueError as e:
                logger.error(f"Failed to parse data URL: {e}, image length: {len(request.image)}")
                return error_response(f"Invalid data URL format: {str(e)}", 400)
            except Exception as e:
                logger.error(f"Failed to decode base64 from data URL: {e}")
                return error_response(f"Invalid base64 data in data URL: {str(e)}", 400)
        elif request.image.startswith(("http://", "https://")):
            # HTTP/HTTPS URL format: fetch the image from URL
            try:
                from lightx2v.deploy.common.utils import fetch_resource

                timeout = int(os.getenv("REQUEST_TIMEOUT", "10"))
                image_bytes = await fetch_resource(request.image, timeout=timeout)
                logger.info(f"Fetched image from URL for face detection: {request.image[:100]}... (size: {len(image_bytes)} bytes)")
            except Exception as e:
                logger.error(f"Failed to fetch image from URL: {e}, URL: {request.image[:200] if request.image else 'None'}")
                return error_response(f"Failed to fetch image from URL: {str(e)}", 400)
        else:
            # Assume it's base64 encoded string (without data URL prefix)
            try:
                image_bytes = base64.b64decode(request.image)
            except Exception as e:
                logger.error(f"Failed to decode base64 image: {e}, image length: {len(request.image) if request.image else 0}")
                return error_response(f"Invalid image format: {str(e)}", 400)

        if image_bytes is None:
            return error_response("Failed to get image bytes", 400)

        # Validate image format before passing to face detector
        # This will catch invalid image data early
        try:
            from lightx2v.deploy.common.utils import format_image_data

            image_bytes = await asyncio.to_thread(format_image_data, image_bytes)
        except Exception as e:
            logger.error(f"Invalid image data: {e}")
            return error_response(f"Invalid image format: {str(e)}", 400)

        # Detect faces only (no cropping)
        result = face_detector.detect_faces(image_bytes, return_image=False)

        faces_data = []
        for i, face in enumerate(result["faces"]):
            faces_data.append(
                {
                    "index": i,
                    "bbox": face["bbox"],  # [x1, y1, x2, y2] - absolute pixel coordinates in original image
                    "confidence": face["confidence"],
                    "class_id": face["class_id"],
                    "class_name": face["class_name"],
                    # Note: face_image is not included - frontend will crop it based on bbox
                }
            )

        return {
            "faces": faces_data,
            "total": len(faces_data),
        }

    except Exception as e:
        logger.error(f"Face detection error: {traceback.format_exc()}")
        return error_response(f"Face detection failed: {str(e)}", 500)


@app.post("/api/v1/audio/separate")
async def api_v1_audio_separate(request: AudioSeparateRequest, user=Depends(verify_user_access)):
    """Separate different speakers in audio"""
    try:
        if not audio_separator:
            return error_response("Audio separator not initialized", 500)

        # Decode base64 audio
        try:
            if request.audio.startswith("data:"):
                # Remove data URL prefix (e.g., "data:audio/mpeg;base64," or "data:application/octet-stream;base64,")
                header, encoded = request.audio.split(",", 1)
                logger.debug(f"Audio data URL header: {header}, encoded length: {len(encoded)}")
            else:
                encoded = request.audio
                logger.debug(f"Audio base64 string length: {len(encoded)}")
            
            # Clean the base64 string: remove whitespace and ensure proper padding
            encoded = encoded.strip().replace('\n', '').replace('\r', '').replace(' ', '')
            original_length = len(encoded)
            
            # Add padding if needed (base64 strings must be multiples of 4)
            missing_padding = len(encoded) % 4
            if missing_padding:
                logger.warning(f"Base64 string length ({original_length}) is not a multiple of 4, adding {4 - missing_padding} padding characters")
                encoded += '=' * (4 - missing_padding)
            
            # Validate base64 string before decoding
            if len(encoded) % 4 != 0:
                raise ValueError(f"Base64 string length ({len(encoded)}) is still not a multiple of 4 after padding")
            
            audio_bytes = base64.b64decode(encoded, validate=True)
            logger.debug(f"Successfully decoded base64 audio, size: {len(audio_bytes)} bytes")
        except Exception as e:
            logger.error(f"Failed to decode base64 audio: {str(e)}")
            logger.error(f"Audio string length: {len(request.audio)}, first 100 chars: {request.audio[:100]}")
            raise ValueError(f"Invalid base64 audio data: {str(e)}")

        # Separate speakers
        result = audio_separator.separate_speakers(audio_bytes, num_speakers=request.num_speakers)

        # Convert audio tensors to base64 strings (without saving to file)
        speakers_data = []
        for speaker in result["speakers"]:
            # Convert audio tensor directly to base64
            audio_base64 = audio_separator.speaker_audio_to_base64(speaker["audio"], speaker["sample_rate"], format="wav")

            speakers_data.append(
                {
                    "speaker_id": speaker["speaker_id"],
                    "audio": audio_base64,  # Base64 encoded audio
                    "segments": speaker["segments"],
                    "sample_rate": speaker["sample_rate"],
                }
            )

        return {
            "speakers": speakers_data,
            "total": len(speakers_data),
            "method": result.get("method", "pyannote"),
        }

    except Exception as e:
        logger.error(f"Audio separation error: {traceback.format_exc()}")
        return error_response(f"Audio separation failed: {str(e)}", 500)


# 所有未知路由 fallback 到 index.html (必须在所有API路由之后)
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def vue_fallback(full_path: str):
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)


# =========================
# Main Entry
# =========================

if __name__ == "__main__":
    ProcessManager.register_signal_handler()
    parser = argparse.ArgumentParser()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(cur_dir, "../../.."))
    dft_pipeline_json = os.path.join(base_dir, "configs/model_pipeline.json")
    dft_task_url = os.path.join(base_dir, "local_task")
    dft_data_url = os.path.join(base_dir, "local_data")
    dft_queue_url = os.path.join(base_dir, "local_queue")
    dft_volcengine_tts_list_json = os.path.join(base_dir, "configs/volcengine_voices_list.json")

    parser.add_argument("--pipeline_json", type=str, default=dft_pipeline_json)
    parser.add_argument("--task_url", type=str, default=dft_task_url)
    parser.add_argument("--data_url", type=str, default=dft_data_url)
    parser.add_argument("--queue_url", type=str, default=dft_queue_url)
    parser.add_argument("--redis_url", type=str, default="")
    parser.add_argument("--template_dir", type=str, default="")
    parser.add_argument("--volcengine_tts_list_json", type=str, default=dft_volcengine_tts_list_json)
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--audio_separator_model_path", type=str, default="")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    model_pipelines = Pipeline(args.pipeline_json)
    volcengine_tts_client = VolcEngineTTSClient(args.volcengine_tts_list_json)
    volcengine_podcast_client = VolcEnginePodcastClient()
    face_detector = FaceDetector()
    audio_separator = AudioSeparator(model_path=args.audio_separator_model_path)
    auth_manager = AuthManager()
    if args.task_url.startswith("/"):
        task_manager = LocalTaskManager(args.task_url, metrics_monitor)
        podcast_manager = LocalPodcastManager(args.task_url)
    elif args.task_url.startswith("postgresql://"):
        task_manager = PostgresSQLTaskManager(args.task_url, metrics_monitor)
        podcast_manager = SQLPodcastManager(args.task_url)
    else:
        raise NotImplementedError
    if args.data_url.startswith("/"):
        data_manager = LocalDataManager(args.data_url, args.template_dir)
    elif args.data_url.startswith("{"):
        data_manager = S3DataManager(args.data_url, args.template_dir)
    else:
        raise NotImplementedError
    if args.queue_url.startswith("/"):
        queue_manager = LocalQueueManager(args.queue_url)
    elif args.queue_url.startswith("amqp://"):
        queue_manager = RabbitMQQueueManager(args.queue_url)
    else:
        raise NotImplementedError
    if args.redis_url:
        server_monitor = RedisServerMonitor(model_pipelines, task_manager, queue_manager, args.redis_url)
    else:
        server_monitor = ServerMonitor(model_pipelines, task_manager, queue_manager)

    uvicorn.run(app, host=args.ip, port=args.port, reload=False, workers=1)
