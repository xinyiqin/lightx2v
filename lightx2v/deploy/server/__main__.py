import argparse
import asyncio
import base64
import copy
import json
import mimetypes
import os
import re
import tempfile
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any

import aiofiles
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from starlette.background import BackgroundTask

from lightx2v.deploy.common.audio_separator import AudioSeparator
from lightx2v.deploy.common.face_detector import FaceDetector
from lightx2v.deploy.common.pipeline import Pipeline
from lightx2v.deploy.common.podcasts import VolcEnginePodcastClient
from lightx2v.deploy.common.sensetime_voice_clone import SenseTimeTTSClient
from lightx2v.deploy.common.utils import check_params, data_name, fetch_resource, format_audio_data, format_image_data, load_inputs, media_to_audio
from lightx2v.deploy.common.volcengine_asr import VolcEngineASRClient
from lightx2v.deploy.common.volcengine_tts import VolcEngineTTSClient
from lightx2v.deploy.data_manager import BaseDataManager, LocalDataManager, S3DataManager
from lightx2v.deploy.queue_manager import LocalQueueManager, RabbitMQQueueManager
from lightx2v.deploy.server.auth import AuthManager
from lightx2v.deploy.server.metrics import MetricMonitor
from lightx2v.deploy.server.monitor import ServerMonitor, WorkerStatus
from lightx2v.deploy.server.redis_monitor import RedisServerMonitor
from lightx2v.deploy.task_manager import BaseTaskManager, FinishedStatus, LocalTaskManager, PostgresSQLTaskManager, TaskStatus
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


class VoiceCloneSaveRequest(BaseModel):
    speaker_id: str
    name: str = ""  # 音色名称


class VoiceCloneTTSRequest(BaseModel):
    text: str
    speaker_id: str
    style: str = "正常"
    speed: float = 1.0
    volume: float = 0
    pitch: float = 0
    language: str = "ZH_CN"


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
volcengine_asr_client = None
sensetime_voice_clone_client = None
volcengine_podcast_client = None
face_detector = None
audio_separator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await task_manager.init()
    await task_manager.mark_server_restart()
    await data_manager.init()
    await queue_manager.init()
    await server_monitor.init()
    asyncio.create_task(server_monitor.loop())
    yield
    await server_monitor.close()
    await queue_manager.close()
    await data_manager.close()
    await task_manager.close()


app = FastAPI(lifespan=lifespan)

# 确保 mimetypes 模块正确配置 JavaScript 和其他文件类型
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/manifest+json", ".webmanifest")

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
        # Get token from request for API calls
        token = None
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix

        # Pass data_manager, task_manager, user_id, and token to load_inputs
        inputs_data = await load_inputs(params, inputs, types, data_manager=data_manager, task_manager=task_manager, user_id=user["user_id"], token=token)

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


@app.get("/api/v1/workflow/{workflow_id}/input")
async def api_v1_workflow_input(request: Request, user=Depends(verify_user_access)):
    try:
        workflow_id = request.path_params["workflow_id"]
        filename = request.query_params.get("filename")
        if not filename:
            return error_response("filename is required", 400)

        workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
        if not workflow:
            return error_response("Workflow not found", 404)

        try:
            if not filename.startswith("workflows/"):
                filename = f"workflows/{workflow_id}/{filename}"
            data = await data_manager.load_bytes(filename)
        except Exception as e:
            logger.error(f"Failed to load workflow input file {filename}: {e}")
            return error_response("File not found", 404)

        ext = os.path.splitext(filename)[1].lower()
        mime_type = WORKFLOW_FILE_MIME.get(ext, "application/octet-stream")
        headers = {"Cache-Control": "public, max-age=3600"}
        return Response(content=data, media_type=mime_type, headers=headers)
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
        if name in task["params"]:
            return error_response(f"Input {name} is a stream", 400)

        # eg, multi person audio directory input
        if filename is not None:
            extra_inputs = task["params"]["extra_inputs"][name]
            name = f"{name}/{filename}"
            assert name in task["inputs"], f"Extra input {name} not found in task {task_id}"
            assert name in extra_inputs, f"Filename {filename} not found in extra inputs"

        # multi-images, rename input_image to input_image/input_image_1
        all_extra_inputs = task["params"].get("extra_inputs", {})
        if name in all_extra_inputs:
            name = all_extra_inputs[name][0]

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

        # Standard case: name exists in inputs
        assert name in task["inputs"], f"Input {name} not found in task {task_id}"
        if name in task["params"]:
            return error_response(f"Input {name} is a stream", 400)

        # eg, multi person audio directory input, multi image input
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
            img_name_without_ext = img_name.rsplit(".", 1)[0] if "." in img_name else img_name
            url = await data_manager.presign_template_url("images", img_name)
            if url is None:
                url = f"./assets/template/images/{img_name}"
            image_map[img_name_without_ext] = {"filename": img_name, "url": url}

        # 创建音频文件名（不含扩展名）到音频信息的映射
        all_audios_sorted = sorted(all_audios)
        audio_map = {}  # 文件名（不含扩展名） -> {"filename": 完整文件名, "url": URL}
        for audio_name in all_audios_sorted:
            audio_name_without_ext = audio_name.rsplit(".", 1)[0] if "." in audio_name else audio_name
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
                "audio": audio_map.get(base_name),
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
                "merged": paginated_templates,  # 新的合并列表
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

        # 判断是否是图片输出任务（i2i 或 t2i）
        is_image_task = task["task_type"] in ["i2i", "t2i"]

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
            "output_image_url": None,
            "input_urls": {},
        }

        for input_name, input_filename in task["inputs"].items():
            if share_type == "template":
                template_type = "images" if "image" in input_name else "audios"
                input_url = await data_manager.presign_template_url(template_type, input_filename)
            else:
                input_url = await data_manager.presign_url(input_filename)
            share_info["input_urls"][input_name] = input_url

        # 根据任务类型处理输出URL
        for output_name, output_filename in task["outputs"].items():
            if share_type == "template":
                if is_image_task:
                    # 图片输出任务：使用 images 类型
                    assert "image" in output_name, "Only image output is supported for image task template share"
                    output_url = await data_manager.presign_template_url("images", output_filename)
                    share_info["output_image_url"] = output_url
                else:
                    # 视频输出任务：使用 videos 类型
                    assert "video" in output_name, "Only video output is supported for video task template share"
                    output_url = await data_manager.presign_template_url("videos", output_filename)
                    share_info["output_video_url"] = output_url
            else:
                # 任务分享：根据任务类型设置对应的输出URL
                output_url = await data_manager.presign_url(output_filename)
                if is_image_task and "image" in output_name:
                    share_info["output_image_url"] = output_url
                elif not is_image_task and "video" in output_name:
                    share_info["output_video_url"] = output_url

        return share_info

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/voices/list")
async def api_v1_voices_list(request: Request):
    try:
        version = request.query_params.get("version", "all")
        if volcengine_tts_client is None:
            return error_response("Volcengine TTS client not loaded", 500)
        voices = volcengine_tts_client.get_voice_list()
        if voices is None:
            return error_response("No voice list found", 404)
        if version != "all":
            voices = copy.deepcopy(voices)
            voices["voices"] = [v for v in voices["voices"] if v["version"] == version]
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
            return FileResponse(output_path, media_type="audio/mpeg", filename=output_filename, background=BackgroundTask(lambda: os.unlink(output_path) if os.path.exists(output_path) else None))
        else:
            return JSONResponse({"error": "TTS generation failed"}, status_code=500)

    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        return JSONResponse({"error": f"TTS generation failed: {str(e)}"}, status_code=500)


@app.post("/api/v1/voice/clone")
async def api_v1_voice_clone(request: Request, user=Depends(verify_user_access)):
    try:
        if volcengine_asr_client is None:
            return JSONResponse({"error": "ASR client not initialized"}, status_code=500)
        if sensetime_voice_clone_client is None:
            return JSONResponse({"error": "Voice clone client not initialized"}, status_code=500)

        form = await request.form()
        file = form.get("file")
        if not file:
            return JSONResponse({"error": "No file uploaded"}, status_code=400)
        raw_data = await file.read()

        user_text = form.get("text", "").strip() if form.get("text") else ""
        use_user_text = bool(user_text)

        cur_duration, min_duration, step_duration = 10.0, 5.0, 2.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            while True:
                audio_data = await asyncio.to_thread(format_audio_data, raw_data, max_duration=cur_duration)
                audio_path = os.path.join(tmp_dir, "formatted_audio.wav")
                async with aiofiles.open(audio_path, "wb") as fout:
                    await fout.write(audio_data)

                if use_user_text:
                    asr_text = user_text
                    logger.info(f"Using user input text (skipping ASR): {asr_text}")
                else:
                    if volcengine_asr_client is None:
                        return JSONResponse({"error": "ASR client not initialized"}, status_code=500)
                    asr_success, asr_result = await volcengine_asr_client.recognize_request(file_path=audio_path)
                    if not asr_success:
                        return JSONResponse({"error": f"ASR failed: {asr_result}"}, status_code=500)
                    asr_text = asr_result.get("result", {}).get("text", "")
                    if not asr_text:
                        return JSONResponse({"error": "Failed to extract text from audio"}, status_code=500)
                    logger.info(f"ASR recognized text: {asr_text}")

                clone_success, clone_result = await sensetime_voice_clone_client.upload_audio_clone(
                    audio_path=audio_path,
                    audio_text=asr_text,
                )
                if not clone_success:
                    err_msg = str(clone_result).lower()
                    cur_duration -= step_duration
                    if "text length" in err_msg and "too long" in err_msg and cur_duration >= min_duration:
                        logger.warning(f"Voice clone failed: {err_msg}, reducing duration to {cur_duration}s")
                        continue
                    return JSONResponse({"error": f"Voice clone failed: {clone_result}"}, status_code=500)
                logger.info(f"Voice clone successful with duration: {cur_duration}s, speaker_id: {clone_result}")
                return JSONResponse({"speaker_id": clone_result, "text": asr_text, "message": "Voice clone successful. Please save the voice to add it to your collection."}, status_code=200)
    except Exception:
        traceback.print_exc()
        return JSONResponse({"error": "Voice clone failed"}, status_code=500)


@app.post("/api/v1/voice/clone/tts")
async def api_v1_voice_clone_tts(request: VoiceCloneTTSRequest):
    try:
        if not request.text.strip():
            return JSONResponse({"error": "Text cannot be empty"}, status_code=400)
        if not request.speaker_id:
            return JSONResponse({"error": "Speaker ID is required"}, status_code=400)
        if sensetime_voice_clone_client is None:
            return JSONResponse({"error": "Voice clone client not initialized"}, status_code=500)
        output_filename = f"voice_clone_tts_{uuid.uuid4().hex}.wav"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        success = await sensetime_voice_clone_client.tts_request(
            text=request.text,
            speaker=request.speaker_id,
            style=request.style,
            speed=request.speed,
            volume=request.volume,
            pitch=request.pitch,
            language=request.language,
            output=output_path,
            sample_rate=24000,
            audio_format="pcm",
            stream_output=True,
            output_subtitles=False,
        )
        if success and os.path.exists(output_path):
            return FileResponse(output_path, media_type="audio/wav", filename=output_filename, background=BackgroundTask(lambda: os.unlink(output_path) if os.path.exists(output_path) else None))
        else:
            return JSONResponse({"error": "TTS generation failed"}, status_code=500)
    except Exception:
        traceback.print_exc()
        return JSONResponse({"error": "TTS generation failed"}, status_code=500)


@app.post("/api/v1/voice/clone/save")
async def api_v1_voice_clone_save(request: VoiceCloneSaveRequest, user=Depends(verify_user_access)):
    try:
        if not request.speaker_id:
            return JSONResponse({"error": "Speaker ID is required"}, status_code=400)
        if not request.name.strip():
            return JSONResponse({"error": "Name is required"}, status_code=400)
        ret = await task_manager.create_voice_clone(user["user_id"], request.speaker_id, request.name)
        if not ret:
            return JSONResponse({"error": "Failed to create voice clone"}, status_code=500)
        return {"message": "Voice clone saved successfully", "speaker_id": request.speaker_id, "name": request.name}
    except Exception:
        traceback.print_exc()
        return JSONResponse({"error": "Failed to save voice clone"}, status_code=500)


@app.delete("/api/v1/voice/clone/{speaker_id}")
async def api_v1_voice_clone_delete(speaker_id: str, user=Depends(verify_user_access)):
    try:
        if not speaker_id:
            return JSONResponse({"error": "Speaker ID is required"}, status_code=400)
        if sensetime_voice_clone_client is None:
            return JSONResponse({"error": "Voice clone client not initialized"}, status_code=500)

        delete_success = await sensetime_voice_clone_client.delete_speaker(speaker_id)
        if not delete_success:
            return JSONResponse({"error": "Failed to delete voice clone from server"}, status_code=500)

        ret = await task_manager.delete_voice_clone(user["user_id"], speaker_id)
        if not ret:
            return JSONResponse({"error": "Failed to delete voice clone"}, status_code=500)
        return {"message": "Voice clone deleted successfully"}

    except Exception:
        traceback.print_exc()
        return JSONResponse({"error": "Failed to delete voice clone"}, status_code=500)


@app.get("/api/v1/voice/clone/list")
async def api_v1_voice_clone_list(user=Depends(verify_user_access)):
    try:
        voice_clones = await task_manager.list_voice_clones(user["user_id"])
        if voice_clones is None:
            return JSONResponse({"error": "Failed to get voice clone list"}, status_code=500)
        return {"voice_clones": voice_clones}
    except Exception:
        traceback.print_exc()
        return JSONResponse({"error": "Failed to get voice clone list"}, status_code=500)


@app.websocket("/api/v1/podcast/generate")
async def api_v1_podcast_generate_ws(websocket: WebSocket):
    await websocket.accept()

    def ws_get_user_id():
        token = websocket.query_params.get("token")
        if not token:
            token = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
            if token and token.startswith("Bearer "):
                token = token[7:]
        payload = auth_manager.verify_jwt_token(token)
        user_id = payload["user_id"]
        return user_id

    async def safe_send_json(payload):
        try:
            await websocket.send_json(payload)
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.warning(f"WebSocket send skipped: {e}")

    try:
        user_id = ws_get_user_id()
        data = await websocket.receive_text()
        request_data = json.loads(data)

        # stop request
        if request_data.get("type") == "stop":
            logger.info("Received stop signal from client")
            await safe_send_json({"type": "stopped"})
            return

        # user input prompt
        input_text = request_data.get("input", "")
        is_url = input_text.startswith(("http://", "https://"))
        if not input_text:
            await safe_send_json({"error": "输入不能为空"})
            return

        session_id = "session_" + str(uuid.uuid4())
        params = {
            "session_id": session_id,
            "data_manager": data_manager,
            "text": "" if is_url else input_text,
            "input_url": input_text if is_url else "",
            "action": 0,
            "use_head_music": False,
            "use_tail_music": False,
            "skip_round_audio_save": False,
        }
        logger.info(f"WebSocket generating podcast with params: {params}")

        # 使用回调函数实时推送音频
        async def on_round_complete(round_info):
            await safe_send_json({"type": "audio_update", "data": round_info})

        params["on_round_complete"] = on_round_complete

        # 创建一个任务来处理停止信号
        async def listen_for_stop(podcast_task):
            while True:
                try:
                    if podcast_task.done():
                        return
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                    request = json.loads(data)
                    if request.get("type") == "stop":
                        logger.warning("Stop signal received during podcast generation")
                        podcast_task.cancel()
                        return
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.warning(f"Stop listener ended: {e}")
                    return

        podcast_task = asyncio.create_task(volcengine_podcast_client.podcast_request(**params))
        stop_listener_task = asyncio.create_task(listen_for_stop(podcast_task))
        podcast_info = None

        try:
            podcast_info = await podcast_task
        except asyncio.CancelledError:
            logger.warning("Podcast generation cancelled by user")
            await safe_send_json({"type": "stopped"})
            return
        finally:
            stop_listener_task.cancel()
        if podcast_info is None:
            await safe_send_json({"error": "播客生成失败，请稍后重试"})
            return

        audio_path = podcast_info["audio_name"]
        rounds = podcast_info["subtitles"]
        await task_manager.create_podcast(session_id, user_id, input_text, audio_path, rounds)
        audio_url = await data_manager.presign_podcast_output_url(audio_path)
        if not audio_url:
            audio_url = f"/api/v1/podcast/audio?session_id={session_id}&filename={audio_path}"
        logger.info(f"completed podcast generation (session: {session_id})")

        await safe_send_json(
            {
                "type": "complete",
                "data": {
                    "audio_url": audio_url,
                    "subtitles": podcast_info["subtitles"],
                    "session_id": session_id,
                    "user_id": user_id,
                },
            }
        )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

    except Exception:
        logger.error(f"Error in websocket: {traceback.format_exc()}")
        await safe_send_json({"error": "websocket internal error, please try again later!"})


@app.get("/api/v1/podcast/audio")
async def api_v1_podcast_audio(request: Request, user=Depends(verify_user_access_from_query)):
    try:
        user_id = user["user_id"]
        session_id = request.query_params.get("session_id")
        filename = request.query_params.get("filename")
        if not session_id or not filename:
            return JSONResponse({"error": "session_id and filename are required"}, status_code=400)

        ext = os.path.splitext(filename)[1].lower()
        assert ext == ".mp3", f"Unsupported file extension: {ext}"

        # 解析 Range 头，格式：bytes=start-end 或 bytes=start-
        range_header = request.headers.get("Range")
        start_byte, end_byte = None, None
        if range_header:
            match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if match:
                start_byte = int(match.group(1))
                end_byte = int(match.group(2)) + 1 if match.group(2) else None

        podcast_data = await task_manager.query_podcast(session_id, user_id)
        if podcast_data:
            # generate is finished and save info to database
            func = data_manager.load_podcast_output_file
            filename = podcast_data["audio_path"]
            func_args = (filename,)
        else:
            func = data_manager.load_podcast_temp_session_file
            func_args = (session_id, filename)

        logger.debug(f"Serving audio file from {func.__name__} with args: {func_args}, start_byte: {start_byte}, end_byte: {end_byte}")
        file_bytes = await func(*func_args)
        file_size = len(file_bytes)
        file_bytes = file_bytes[start_byte:end_byte]

        content_length = len(file_bytes)
        media_type = "audio/mpeg"
        status_code = 200
        headers = {"Content-Length": str(content_length), "Accept-Ranges": "bytes", "Content-Type": media_type, "Content-Disposition": f'attachment; filename="{filename}"'}

        if start_byte is not None and start_byte > 0:
            status_code = 206  # Partial Content
            headers["Content-Range"] = f"bytes {start_byte}-{start_byte + content_length - 1}/{file_size}"
        return Response(content=file_bytes, media_type=media_type, status_code=status_code, headers=headers)

    except Exception as e:
        logger.error(f"Error serving audio: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/v1/podcast/history")
async def api_v1_podcast_history(request: Request, user=Depends(verify_user_access)):
    try:
        user_id = user["user_id"]
        page = int(request.query_params.get("page", 1))
        page_size = int(request.query_params.get("page_size", 10))
        assert page > 0 and page_size > 0, "page and page_size must be greater than 0"
        status = request.query_params.get("status", None)  # has_audio, no_audio

        query_params = {"user_id": user_id}
        if status == "has_audio":
            query_params["has_audio"] = True
        elif status == "no_audio":
            query_params["has_audio"] = False

        total_tasks = await task_manager.list_podcasts(count=True, **query_params)
        total_pages = (total_tasks + page_size - 1) // page_size
        page_info = {"page": page, "page_size": page_size, "total": total_tasks, "total_pages": total_pages}
        if page > total_pages:
            return {"sessions": [], "pagination": page_info}

        query_params["offset"] = (page - 1) * page_size
        query_params["limit"] = page_size
        sessions = await task_manager.list_podcasts(**query_params)
        return {"sessions": sessions, "pagination": page_info}

    except Exception as e:
        logger.error(f"Error getting podcast history: {e}")
        traceback.print_exc()
        return {"sessions": []}


@app.get("/api/v1/podcast/session/{session_id}/audio_url")
async def api_v1_podcast_session_audio_url(session_id: str, user=Depends(verify_user_access)):
    try:
        user_id = user["user_id"]
        podcast_data = await task_manager.query_podcast(session_id, user_id)
        if not podcast_data:
            return JSONResponse({"error": "Podcast session not found"}, status_code=404)

        audio_path = podcast_data["audio_path"]
        audio_url = await data_manager.presign_podcast_output_url(audio_path)
        if not audio_url:
            audio_url = f"/api/v1/podcast/audio?session_id={session_id}&filename={audio_path}"
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


class AudioExtractRequest(BaseModel):
    video: str  # Base64 encoded video
    output_format: str = "wav"  # Output audio format: wav, mp3
    sample_rate: int = 44100  # Output sample rate
    channels: int = 2  # Output channels: 1=mono, 2=stereo


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
        try:
            # Check if input is a URL (blob:, http:, https:, or data: URL)
            if request.image.startswith(("http://", "https://")):
                timeout = int(os.getenv("REQUEST_TIMEOUT", "10"))
                image_bytes = await fetch_resource(request.image, timeout=timeout)
                logger.debug(f"Fetched image from URL for face detection: {request.image[:100]}... (size: {len(image_bytes)} bytes)")
            else:
                encoded = request.image
                # Data URL format: "data:image/png;base64,..."
                if encoded.startswith("data:image"):
                    _, encoded = encoded.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                logger.debug(f"Decoded base64 image: {request.image[:100]}... (size: {len(image_bytes)} bytes)")

            # Validate image format before passing to face detector
            image_bytes = await asyncio.to_thread(format_image_data, image_bytes)

        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}, image length: {len(request.image) if request.image else 0}")
            return error_response(f"Invalid image format: {str(e)}", 400)

        # Detect faces only (no cropping)
        result = await asyncio.to_thread(face_detector.detect_faces, image_bytes, return_image=False)
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
        return {"faces": faces_data, "total": len(faces_data)}

    except Exception as e:
        logger.error(f"Face detection error: {traceback.format_exc()}")
        return error_response(f"Face detection failed: {str(e)}", 500)


@app.post("/api/v1/audio/separate")
async def api_v1_audio_separate(request: AudioSeparateRequest, user=Depends(verify_user_access)):
    """Separate different speakers in audio"""
    try:
        if not audio_separator:
            return error_response("Audio separator not initialized", 500)
        audio_bytes = None
        try:
            encoded = request.audio
            if encoded.startswith("data:"):
                # Remove data URL prefix (e.g., "data:audio/mpeg;base64," or "data:application/octet-stream;base64,")
                _, encoded = encoded.split(",", 1)
            audio_bytes = await asyncio.to_thread(base64.b64decode, encoded, validate=True)
            logger.debug(f"Successfully decoded base64 audio, size: {len(audio_bytes)} bytes")

        except Exception as e:
            logger.error(f"Failed to decode base64 audio {request.audio[:100]}..., error: {str(e)}")
            return error_response(f"Invalid base64 audio data", 400)

        # Separate speakers
        result = await asyncio.to_thread(audio_separator.separate_speakers, audio_bytes, num_speakers=request.num_speakers)

        # Convert audio tensors to base64 strings (without saving to file)
        speakers_data = []
        for speaker in result["speakers"]:
            # Convert audio tensor directly to base64
            audio_base64 = await asyncio.to_thread(audio_separator.speaker_audio_to_base64, speaker["audio"], speaker["sample_rate"], format="wav")
            speakers_data.append(
                {
                    "speaker_id": speaker["speaker_id"],
                    "audio": audio_base64,  # Base64 encoded audio
                    "segments": speaker["segments"],
                    "sample_rate": speaker["sample_rate"],
                }
            )
        return {"speakers": speakers_data, "total": len(speakers_data), "method": result.get("method", "pyannote")}

    except Exception as e:
        logger.error(f"Audio separation error: {traceback.format_exc()}")
        return error_response(f"Audio separation failed: {str(e)}", 500)


@app.post("/api/v1/audio/extract")
async def api_v1_audio_extract(request: AudioExtractRequest, user=Depends(verify_user_access)):
    """Extract audio from video file"""
    try:
        # Validate request parameters
        output_formats = ["wav", "mp3"]
        sample_rates = [8000, 16000, 24000, 32000, 44100, 48000]
        channels = [1, 2]
        if request.output_format not in output_formats or request.sample_rate not in sample_rates or request.channels not in channels:
            return error_response(
                f"Unsupported request parameters: output_format={request.output_format}, sample_rate={request.sample_rate}, channels={request.channels}. Supported formats: {output_formats}, sample rates: {sample_rates}, channels: {channels}",
                400,
            )

        video_bytes = None
        try:
            encoded = request.video
            if encoded.startswith("data:"):
                # Remove data URL prefix (e.g., "data:video/mp4;base64,")
                _, encoded = encoded.split(",", 1)
            video_bytes = await asyncio.to_thread(base64.b64decode, encoded, validate=True)
            logger.debug(f"Successfully decoded base64 video, size: {len(video_bytes)} bytes")

        except Exception as e:
            logger.error(f"Failed to decode base64 video {request.video[:100]}..., error: {str(e)}")
            return error_response(f"Invalid base64 video data", 400)

        # Extract audio from video
        audio_bytes = await asyncio.to_thread(media_to_audio, video_bytes, None, request.sample_rate, request.channels, request.output_format)

        # Convert audio bytes to base64
        audio_base64 = await asyncio.to_thread(base64.b64encode, audio_bytes)
        audio_base64 = audio_base64.decode("utf-8")

        # Determine MIME type based on output format
        mime_type = "audio/wav" if request.output_format == "wav" else "audio/mpeg"
        audio_data_url = f"data:{mime_type};base64,{audio_base64}"

        logger.info(f"Successfully extracted audio from video: {len(audio_bytes)} bytes, format={request.output_format}")
        return {
            "audio": audio_data_url,  # Data URL format for easy use in frontend
            "format": request.output_format,
            "sample_rate": request.sample_rate,
            "channels": request.channels,
            "size": len(audio_bytes),
        }

    except Exception as e:
        logger.error(f"Audio extraction error: {traceback.format_exc()}")
        error_msg = str(e).lower()
        code = 500
        if "does not contain an audio track" in error_msg or "no audio track" in error_msg:
            code = 400
        return error_response(f"Audio extraction failed: {error_msg}", code)


# =========================
# Static file mount (must be after all dynamic routes)
# =========================
# 添加assets目录的静态文件服务（用于前端静态资源）
assets_dir = os.path.join(os.path.dirname(__file__), "static", "assets")
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# 添加 Canvas 子应用的 assets 目录的静态文件服务
canvas_assets_dir = os.path.join(static_dir, "canvas", "assets")
if os.path.exists(canvas_assets_dir):
    app.mount("/canvas/assets", StaticFiles(directory=canvas_assets_dir), name="canvas-assets")


# Canvas 子应用路由处理（必须在 vue_fallback 之前）
@app.get("/canvas/{full_path:path}")
async def canvas_fallback(full_path: str):
    # 跳过 assets 路径，由上面的 mount 处理
    if full_path.startswith("assets/"):
        return HTMLResponse("<h1>Asset not found</h1>", status_code=404)

    # 从 static 目录读取 Canvas 文件
    canvas_dir = os.path.join(static_dir, "canvas")

    # 处理 /canvas/index.html 或 /canvas/ 路径
    if not full_path or full_path == "index.html":
        canvas_index_path = os.path.join(canvas_dir, "index.html")
        if os.path.exists(canvas_index_path):
            return FileResponse(canvas_index_path, media_type="text/html")

    # 对于其他 canvas 路径下的文件，尝试直接返回文件
    canvas_file_path = os.path.join(canvas_dir, full_path)
    if os.path.exists(canvas_file_path) and os.path.isfile(canvas_file_path):
        # 根据文件扩展名设置正确的 MIME 类型
        ext = os.path.splitext(full_path)[1].lower()
        media_type_map = {
            ".html": "text/html",
            ".js": "application/javascript",
            ".mjs": "application/javascript",
            ".json": "application/json",
            ".css": "text/css",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".webp": "image/webp",
            ".webmanifest": "application/manifest+json",
            ".wasm": "application/wasm",
            ".woff": "font/woff",
            ".woff2": "font/woff2",
            ".ttf": "font/ttf",
            ".eot": "application/vnd.ms-fontobject",
            ".otf": "font/otf",
        }
        media_type = media_type_map.get(ext, None)
        if media_type:
            return FileResponse(canvas_file_path, media_type=media_type)
        else:
            # 使用 mimetypes 模块推断 MIME 类型
            guessed_type, _ = mimetypes.guess_type(canvas_file_path)
            return FileResponse(canvas_file_path, media_type=guessed_type or "application/octet-stream")

    # 如果文件不存在，返回 canvas/index.html
    canvas_index_path = os.path.join(canvas_dir, "index.html")
    if os.path.exists(canvas_index_path):
        return FileResponse(canvas_index_path, media_type="text/html")
    return HTMLResponse("<h1>Canvas app not found</h1>", status_code=404)


# 所有未知路由 fallback 到 index.html (必须在所有API路由之后)
# =========================
# Workflow API Endpoints
# =========================

# Shared MIME types for workflow file responses (input & output file GET)
WORKFLOW_FILE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".txt": "text/plain",
    ".json": "application/json",
}


@app.post("/api/v1/workflow/create")
async def api_v1_workflow_create(request: Request, user=Depends(verify_user_access)):
    """Create a new workflow."""
    try:
        params = await request.json()
        name = params.get("name", "Untitled Workflow")
        description = params.get("description", "")
        nodes = params.get("nodes", [])
        connections = params.get("connections", [])
        visibility = params.get("visibility", "private")
        workflow_id = params.get("workflow_id")  # Optional: allow frontend to specify workflow_id
        tags = params.get("tags", [])
        node_output_history = params.get("node_output_history") or {}
        author_id = params.get("author_id") or user.get("user_id")
        author_name = params.get("author_name") or user.get("username") or ""
        if not isinstance(tags, list):
            tags = []
        else:
            tags = [str(tag) for tag in tags if isinstance(tag, (str, int, float))]
        if not isinstance(node_output_history, dict):
            node_output_history = {}
        node_output_history = _truncate_node_output_history(node_output_history)
        try:
            workflow_id = await task_manager.create_workflow(
                user_id=user["user_id"],
                name=name,
                description=description,
                nodes=nodes,
                connections=connections,
                visibility=visibility,
                workflow_id=workflow_id,  # Pass optional workflow_id
                tags=tags,
                node_output_history=node_output_history,
                author_id=author_id,
                author_name=author_name,
            )

            logger.info(f"Workflow {workflow_id} created by user {user['user_id']}")
            return {"workflow_id": workflow_id, "message": "Workflow created successfully"}
        except ValueError as ve:
            if "already exists" in str(ve):
                logger.warning(f"Workflow {workflow_id} already exists, returning 409")
                return error_response(f"Workflow {workflow_id} already exists. Use PUT /api/v1/workflow/{workflow_id} to update.", 409)
            raise
    except Exception as e:
        error_msg = f"Failed to create workflow: {str(e)}"
        logger.error(error_msg)
        traceback.print_exc()
        return error_response(error_msg, 500)


async def _format_user_name(user_id):
    try:
        if not user_id:
            return ""
        user = await task_manager.query_user(user_id)
        return user.get("username", "")
    except Exception:
        logger.error(f"Failed to format user name for user {user_id}: {traceback.format_exc()}")
        return ""


async def _list_workflows_paginated(request: Request, user: dict, public: bool):
    """Shared logic for listing workflows: user's list (public=False) or public workflows (public=True)."""
    page = int(request.query_params.get("page", 1))
    page_size = int(request.query_params.get("page_size", 10))
    search = request.query_params.get("search", None)
    if page < 1 or page_size < 1:
        return error_response("page and page_size must be greater than 0", 400)

    query_params = {}
    if public:
        query_params["visibility"] = "public"
    else:
        query_params["user_id"] = user["user_id"]
    if search:
        query_params["search"] = search

    total_workflows = await task_manager.list_workflows(count=True, **query_params)
    total_pages = (total_workflows + page_size - 1) // page_size
    page_info = {"page": page, "page_size": page_size, "total": total_workflows, "total_pages": total_pages}
    if page > total_pages:
        return {"workflows": [], "pagination": page_info}

    query_params["offset"] = (page - 1) * page_size
    query_params["limit"] = page_size
    workflows = await task_manager.list_workflows(**query_params)

    rets = []
    for wf in workflows:
        thumsup = await task_manager.get_workflow_thumsup(wf["workflow_id"], user["user_id"])
        wf_user_id = wf.get("user_id", "")
        wf_user_name = await _format_user_name(wf_user_id)
        stored_author_id = wf.get("author_id") or wf_user_id
        stored_author_name = wf.get("author_name") or (await _format_user_name(stored_author_id) if stored_author_id else "")
        tags_value = wf.get("tags", [])
        if not isinstance(tags_value, list):
            tags_value = []
        else:
            tags_value = [str(tag) for tag in tags_value if isinstance(tag, (str, int, float))]
        item = {
            "workflow_id": wf["workflow_id"],
            "name": wf["name"],
            "description": wf.get("description", ""),
            "create_t": wf["create_t"],
            "update_t": wf["update_t"],
            "visibility": wf.get("visibility", "private"),
            "thumsup_count": thumsup.get("count", 0),
            "thumsup_liked": thumsup.get("liked", False),
            "user_id": wf_user_id,
            "author_name": stored_author_name or wf_user_name,
            "author_id": stored_author_id,
            "tags": tags_value,
            "nodes": wf.get("nodes", []),
            "connections": wf.get("connections", []),
        }
        rets.append(item)
    return {"workflows": rets, "pagination": page_info}


@app.get("/api/v1/workflow/list")
async def api_v1_workflow_list(request: Request, user=Depends(verify_user_access)):
    """List workflows with pagination and search."""
    try:
        result = await _list_workflows_paginated(request, user, public=False)
        return result
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/workflow/public")
async def api_v1_workflow_public(request: Request, user=Depends(verify_user_access)):
    """List public workflows for community view."""
    try:
        result = await _list_workflows_paginated(request, user, public=True)
        return result
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/workflow/{workflow_id}")
async def api_v1_workflow_get(request: Request, user=Depends(verify_user_access)):
    """Get a workflow by ID."""
    try:
        workflow_id = request.path_params["workflow_id"]
        workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
        if not workflow:
            return error_response(f"Workflow {workflow_id} not found", 404)
        thumsup = await task_manager.get_workflow_thumsup(workflow_id, user["user_id"])
        workflow["thumsup_count"] = thumsup.get("count", 0)
        workflow["thumsup_liked"] = thumsup.get("liked", False)
        tags_value = workflow.get("tags", [])
        if not isinstance(tags_value, list):
            tags_value = []
        else:
            tags_value = [str(tag) for tag in tags_value if isinstance(tag, (str, int, float))]
        workflow["tags"] = tags_value
        node_history = workflow.get("node_output_history", {})
        if not isinstance(node_history, dict):
            node_history = {}
        workflow["node_output_history"] = node_history
        if "author_id" in workflow and workflow["author_id"] is not None:
            workflow["author_id"] = str(workflow["author_id"])
        if "author_name" in workflow and workflow["author_name"] is not None:
            workflow["author_name"] = str(workflow["author_name"])
        # Don't expose legacy fields to frontend
        workflow.pop("last_run_t", None)
        if workflow.get("tag") != "delete":
            workflow.pop("tag", None)
        return workflow
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


async def _update_workflow(workflow_id, params, user):
    updates = {}

    if "name" in params:
        updates["name"] = params["name"]
    if "description" in params:
        updates["description"] = params["description"]
    if "nodes" in params:
        updates["nodes"] = params["nodes"]
    if "connections" in params:
        updates["connections"] = params["connections"]
    if "chat_history" in params:
        pass  # chat_history 已拆到独立存储，不再写入 workflow
    if "extra_info" in params:
        updates["extra_info"] = params["extra_info"]
    if "tags" in params:
        incoming_tags = params.get("tags") or []
        if not isinstance(incoming_tags, list):
            incoming_tags = []
        else:
            incoming_tags = [str(tag) for tag in incoming_tags if isinstance(tag, (str, int, float))]
        updates["tags"] = incoming_tags
    if "node_output_history" in params:
        node_history = params.get("node_output_history") or {}
        if not isinstance(node_history, dict):
            node_history = {}
        updates["node_output_history"] = _truncate_node_output_history(node_history)
    if "author_id" in params:
        updates["author_id"] = params["author_id"]
    if "author_name" in params:
        updates["author_name"] = params["author_name"]
    if "visibility" in params:
        if params["visibility"] not in ["private", "public"]:
            return error_response("visibility must be 'private' or 'public'", 400)
        updates["visibility"] = params["visibility"]

    # Check if workflow exists first (user must own it to update)
    existing_workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
    if not existing_workflow:
        logger.warning(f"Workflow {workflow_id} not found for user {user['user_id']}")
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    success = await task_manager.update_workflow(workflow_id, updates, user["user_id"])
    return success


@app.put("/api/v1/workflow/{workflow_id}")
async def api_v1_workflow_update(request: Request, user=Depends(verify_user_access)):
    """Update a workflow."""
    try:
        workflow_id = request.path_params["workflow_id"]
        params = await request.json()
        success = await _update_workflow(workflow_id, params, user)
        if success:
            logger.info(f"Workflow {workflow_id} updated by user {user['user_id']}")
            return {"message": "Workflow updated successfully"}
        else:
            logger.error(f"Failed to update workflow {workflow_id} for user {user['user_id']}")
            return error_response("Failed to update workflow", 400)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.put("/api/v1/workflow/{workflow_id}/visibility")
async def api_v1_workflow_visibility(request: Request, user=Depends(verify_user_access)):
    """Update workflow visibility (private/public)."""
    try:
        params = await request.json()
        workflow_id = request.path_params["workflow_id"]
        visibility = params.get("visibility", "")
        if visibility not in ["private", "public"]:
            return error_response("visibility must be 'private' or 'public'", 400)

        existing_workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
        if not existing_workflow:
            logger.warning(f"Workflow {workflow_id} not found for user {user['user_id']}")
            return error_response(f"Workflow {workflow_id} not found", 404)

        success = await task_manager.update_workflow(workflow_id, {"visibility": visibility}, user["user_id"])
        if success:
            logger.info(f"Workflow {workflow_id} visibility updated by user {user['user_id']}")
            return {"message": "Workflow visibility updated successfully", "visibility": visibility}
        else:
            logger.error(f"Failed to update workflow visibility {workflow_id} for user {user['user_id']}")
            return error_response("Failed to update workflow visibility", 400)
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/workflow/{workflow_id}/thumsup")
async def api_v1_workflow_thumsup(request: Request, user=Depends(verify_user_access)):
    """Toggle workflow thumsup."""
    try:
        workflow_id = request.path_params["workflow_id"]
        workflow = await task_manager.query_workflow(workflow_id, user_id=None)
        if not workflow:
            return error_response("Workflow not found", 404)
        if workflow.get("visibility", "private") != "public" and workflow.get("user_id") != user["user_id"]:
            return error_response("Workflow is private", 403)
        result = await task_manager.toggle_workflow_thumsup(workflow_id, user["user_id"])
        return {
            "workflow_id": workflow_id,
            "thumsup_count": result.get("count", 0),
            "thumsup_liked": result.get("liked", False),
        }
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/workflow/{workflow_id}/thumsup")
async def api_v1_workflow_thumsup_get(request: Request, user=Depends(verify_user_access)):
    """Get workflow thumsup status."""
    try:
        workflow_id = request.path_params["workflow_id"]
        workflow = await task_manager.query_workflow(workflow_id, user_id=None)
        if not workflow:
            return error_response("Workflow not found", 404)
        if workflow.get("visibility", "private") != "public" and workflow.get("user_id") != user["user_id"]:
            return error_response("Workflow is private", 403)
        result = await task_manager.get_workflow_thumsup(workflow_id, user["user_id"])
        return {
            "workflow_id": workflow_id,
            "thumsup_count": result.get("count", 0),
            "thumsup_liked": result.get("liked", False),
        }
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/workflow/{workflow_id}/copy")
async def api_v1_workflow_copy(request: Request, user=Depends(verify_user_access)):
    """Copy a workflow (for preset workflows). Creates a new workflow with the same content but new workflow_id and current user_id."""
    try:
        params = await request.json()
        new_workflow_id = params.get("workflow_id")
        workflow_id = request.path_params["workflow_id"]
        original_workflow = await task_manager.query_workflow(workflow_id, user_id=None)
        if not original_workflow:
            logger.warning(f"Workflow {workflow_id} not found for copying")
            return error_response(f"Workflow {workflow_id} not found", 404)

        original_author_id = original_workflow.get("author_id") or original_workflow.get("user_id")
        original_author_name = original_workflow.get("author_name")
        if not original_author_name and original_author_id:
            original_author_name = await _format_user_name(original_author_id)
        original_tags = original_workflow.get("tags", [])
        if not isinstance(original_tags, list):
            original_tags = []
        else:
            original_tags = [str(tag) for tag in original_tags if isinstance(tag, (str, int, float))]
        original_node_history = original_workflow.get("node_output_history", {})
        if not isinstance(original_node_history, dict):
            original_node_history = {}
        original_node_history = _truncate_node_output_history(original_node_history)

        copied_workflow_id = await task_manager.create_workflow(
            user_id=user["user_id"],
            name=original_workflow.get("name", "Untitled Workflow") + " (Copy)",
            description=original_workflow.get("description", ""),
            nodes=original_workflow.get("nodes", []),
            connections=original_workflow.get("connections", []),
            visibility=original_workflow.get("visibility", "private"),
            workflow_id=new_workflow_id,  # Use provided workflow_id or generate new one
            tags=original_tags,
            node_output_history=original_node_history,
            author_id=original_author_id,
            author_name=original_author_name,
        )

        updates = {}
        if "extra_info" in original_workflow:
            updates["extra_info"] = original_workflow["extra_info"]

        if updates:
            await task_manager.update_workflow(copied_workflow_id, updates, user["user_id"])

        if hasattr(task_manager, "save_workflow_chat_history") and hasattr(task_manager, "get_workflow_chat_history"):
            chat_data = await task_manager.get_workflow_chat_history(workflow_id, user_id=None)
            messages = chat_data.get("messages", []) if chat_data else []
            if not messages:
                legacy_chat = original_workflow.get("chat_history") or []
                messages = [_to_persisted_message(m) for m in legacy_chat if isinstance(m, dict)]
            if messages:
                messages = messages[-MAX_CHAT_HISTORY:]
                await task_manager.save_workflow_chat_history(copied_workflow_id, messages, user["user_id"])

        logger.info(f"Workflow {workflow_id} copied to {copied_workflow_id} by user {user['user_id']}")
        return {"workflow_id": copied_workflow_id, "message": "Workflow copied successfully"}
    except Exception as e:
        error_msg = f"Failed to copy workflow: {str(e)}"
        logger.error(error_msg)
        traceback.print_exc()
        return error_response(error_msg, 500)


# =========================
# Workflow Data Management Functions
# =========================


MAX_NODE_HISTORY = 20
MAX_CHAT_HISTORY = 50


def _truncate_chat_history(history: list) -> list:
    """Keep only the most recent MAX_CHAT_HISTORY messages."""
    if not isinstance(history, list):
        return []
    return history[-MAX_CHAT_HISTORY:]


def _truncate_node_output_history(history: dict) -> dict:
    """Keep only the most recent MAX_NODE_HISTORY entries per node."""
    if not isinstance(history, dict):
        return {}
    out: dict = {}
    for node_id, entries in history.items():
        if not isinstance(entries, list):
            continue
        out[node_id] = entries[:MAX_NODE_HISTORY]
    return out


def ensure_node_output_history(workflow: dict) -> dict:
    history = workflow.get("node_output_history")
    if not isinstance(history, dict):
        history = {}
        workflow["node_output_history"] = history
    return history


def append_history_entry(workflow: dict, node_id: str, entry: dict) -> list[dict]:
    history = ensure_node_output_history(workflow)
    entries = history.get(node_id)
    if not isinstance(entries, list):
        entries = []
    entries.insert(0, entry)
    trimmed: list[dict] = []
    if len(entries) > MAX_NODE_HISTORY:
        trimmed = entries[MAX_NODE_HISTORY:]
        entries = entries[:MAX_NODE_HISTORY]
    history[node_id] = entries
    return trimmed


def iter_file_history_entries(workflow: dict):
    history = workflow.get("node_output_history", {})
    if not isinstance(history, dict):
        return
    for node_id, entries in history.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get("kind") == "file":
                yield node_id, entry


def get_entry_storage_paths(entry: dict) -> list[str]:
    metadata = entry.get("metadata") or {}
    storage_path = metadata.get("storage_path")
    if isinstance(storage_path, list):
        return [path for path in storage_path if isinstance(path, str)]
    if isinstance(storage_path, str):
        return [storage_path]
    return []


def find_file_path_in_outputs(workflow: dict, file_id: str) -> str | None:
    """Find storage path for file_id. Returns path from metadata.storage_path if present."""
    for _, entry in iter_file_history_entries(workflow):
        value = entry.get("value") if isinstance(entry.get("value"), dict) else {}
        if value.get("fileId") == file_id:
            paths = get_entry_storage_paths(entry)
            if paths:
                return paths[0]
    return None


# Common extensions for workflow output files (audio, image, video)
_WORKFLOW_FILE_EXTENSIONS = (".mp3", ".wav", ".ogg", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".bin")


async def _try_conventional_file_path(
    workflow_id: str, file_id: str, dm: BaseDataManager, entry_ext: str | None = None
) -> str | None:
    """
    Fallback: when node_output_history has fileId but no metadata.storage_path
    (e.g. entries from frontend autosave), try conventional path
    workflows/{workflow_id}_{file_id}{ext}. Uses entry_ext if provided, else tries common extensions.
    """
    base_path = f"workflows/{workflow_id}_{file_id}"
    exts_to_try = [entry_ext] if entry_ext and entry_ext.startswith(".") else []
    exts_to_try.extend(_WORKFLOW_FILE_EXTENSIONS)
    for ext in exts_to_try:
        if not ext:
            continue
        path = base_path + (ext if ext.startswith(".") else f".{ext}")
        try:
            if hasattr(dm, "file_exists"):
                if await dm.file_exists(path):
                    return path
            else:
                await dm.load_bytes(path)
                return path
        except Exception:
            continue
    return None


def get_entry_ext(workflow: dict, file_id: str) -> str | None:
    """Get ext from the first matching file history entry (value.ext)."""
    for _, entry in iter_file_history_entries(workflow):
        value = entry.get("value") if isinstance(entry.get("value"), dict) else {}
        if value.get("fileId") == file_id:
            ext = value.get("ext")
            if isinstance(ext, str) and ext:
                return ext if ext.startswith(".") else f".{ext}"
    return None


def count_file_references(workflow: dict, file_id: str) -> int:
    occurrences = 0
    for _, entry in iter_file_history_entries(workflow):
        value = entry.get("value") if isinstance(entry.get("value"), dict) else {}
        if value.get("fileId") == file_id:
            occurrences += 1
    return occurrences


async def cleanup_history_files(entries: list[dict], workflow_id: str, workflow: dict, data_manager: BaseDataManager) -> None:
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("kind") != "file":
            continue
        storage_paths = get_entry_storage_paths(entry)
        if not storage_paths:
            continue
        value = entry.get("value") if isinstance(entry.get("value"), dict) else {}
        file_id = value.get("fileId")
        for storage_path in storage_paths:
            if not isinstance(storage_path, str):
                continue
            if not storage_path.startswith(f"workflows/{workflow_id}_"):
                continue
            if file_id:
                ref_count = count_file_references(workflow, file_id)
                if ref_count > 0:
                    continue
            try:
                await data_manager.delete_bytes(storage_path)
                logger.info(f"Deleted history file: {storage_path}")
            except Exception as e:
                logger.error(f"Failed to delete history file {storage_path}: {e}")


async def cleanup_node_data(workflow_id: str, node_id: str, workflow: dict, data_manager: BaseDataManager, task_manager: BaseTaskManager) -> None:
    history = ensure_node_output_history(workflow)
    node_entries = history.pop(node_id, [])
    if not node_entries:
        return

    await cleanup_history_files(node_entries, workflow_id, workflow, data_manager)
    await task_manager.update_workflow(workflow_id, {"node_output_history": history}, user_id=None)


async def cleanup_workflow_files(workflow_id: str, workflow: dict, data_manager: BaseDataManager) -> None:
    files_to_cleanup = set()
    for _, entry in iter_file_history_entries(workflow):
        for path in get_entry_storage_paths(entry):
            if isinstance(path, str):
                files_to_cleanup.add(path)

    for file_path in files_to_cleanup:
        try:
            await data_manager.delete_bytes(file_path)
            logger.info(f"Deleted workflow file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")


def _parse_output_data_raw(output_data_raw: Any, file_info: dict | None) -> tuple[bytes | str | dict, dict | None, str]:
    """Parse request body output_data_raw into (output_data, file_info, file_ext)."""
    ext_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/ogg": ".ogg",
        "text/plain": ".txt",
        "application/json": ".json",
    }
    file_ext = ".bin"
    if isinstance(output_data_raw, str):
        if output_data_raw.startswith("data:"):
            header, encoded = output_data_raw.split(",", 1)
            output_data = base64.b64decode(encoded)
            mime_type = header.split(":")[1].split(";")[0] if ":" in header else "application/octet-stream"
            file_ext = ext_map.get(mime_type, ".bin")
            if not file_info:
                file_info = {"file_id": str(uuid.uuid4())}
            return output_data, file_info, file_ext
        if output_data_raw.startswith("./assets/task/result") or output_data_raw.startswith("http://") or output_data_raw.startswith("https://"):
            return output_data_raw, None, file_ext
        return output_data_raw, file_info, file_ext
    if isinstance(output_data_raw, dict):
        return output_data_raw, file_info, file_ext
    raise ValueError("Invalid output_data format")


async def _append_port_output_to_workflow(
    workflow_id: str,
    node_id: str,
    port_id: str,
    output_data: bytes | str | dict,
    file_info: dict | None,
    workflow: dict,
    data_manager: BaseDataManager,
    file_ext: str = ".bin",
) -> dict[str, Any]:
    """Append one port's output as history entry to workflow (in memory). Does NOT update DB.
    Returns the created entry for response building."""
    timestamp_ms = int(time.time() * 1000)
    entry_id = str(uuid.uuid4())
    metadata: dict[str, Any] = {"created_at": timestamp_ms, "port_id": port_id}
    entry: dict[str, Any]

    def looks_like_file_string(value: str) -> bool:
        prefixes = ('http://', 'https://', './', '/assets/', '/api/v1/workflow/', 'local://', 'blob:')
        return any(value.startswith(prefix) for prefix in prefixes)

    if isinstance(output_data, bytes):
        file_id = file_info["file_id"] if file_info and file_info.get("file_id") else str(uuid.uuid4())
        storage_path = f"workflows/{workflow_id}_{file_id}{file_ext}"
        await data_manager.save_bytes(output_data, storage_path)
        saved_file_url = await data_manager.presign_url(storage_path)
        metadata["storage_path"] = storage_path
        metadata["file_ext"] = file_ext
        metadata["size"] = len(output_data)
        if file_info and "mime_type" in file_info:
            metadata["mime_type"] = file_info["mime_type"]
        entry = {
            "id": entry_id,
            "timestamp": timestamp_ms,
            "kind": "file",
            "value": {
                "fileId": file_id,
                "url": saved_file_url,
                "ext": file_ext,
            },
            "metadata": metadata,
        }
    elif isinstance(output_data, str):
        if looks_like_file_string(output_data):
            entry = {
                "id": entry_id,
                "timestamp": timestamp_ms,
                "kind": "file",
                "value": {
                    "fileId": file_info["file_id"] if file_info and file_info.get("file_id") else None,
                    "url": output_data,
                    "ext": file_ext,
                },
                "metadata": metadata,
            }
        else:
            entry = {
                "id": entry_id,
                "timestamp": timestamp_ms,
                "kind": "text",
                "value": {"text": output_data},
                "metadata": metadata,
            }
    elif isinstance(output_data, dict):
        # Unwrap if client sent { "json": { "__type": "lightx2v_result", ... } } so we persist kind: lightx2v_result
        data = output_data
        if data.get("__type") is None and "json" in data and isinstance(data.get("json"), dict):
            data = data["json"]
        if data.get("type") == "reference" and data.get("file_id"):
            file_id_val = data.get("file_id")
            ext_val = data.get("ext") or ".bin"
            entry = {
                "id": entry_id,
                "timestamp": timestamp_ms,
                "kind": "file",
                "value": {
                    "fileId": file_id_val,
                    "url": data.get("file_url") or data.get("url"),
                    "ext": ext_val,
                },
                "metadata": metadata,
            }
        elif data.get("__type") == "lightx2v_result":
            entry = {
                "id": entry_id,
                "timestamp": timestamp_ms,
                "kind": "lightx2v_result",
                "value": {
                    "taskId": data.get("task_id") or data.get("taskId"),
                    "outputName": data.get("output_name") or data.get("outputName") or "output",
                    "isCloud": bool(data.get("is_cloud")),
                },
                "metadata": metadata,
            }
        else:
            entry = {
                "id": entry_id,
                "timestamp": timestamp_ms,
                "kind": "json",
                "value": {"json": output_data},
                "metadata": metadata,
            }
    else:
        entry = {
            "id": entry_id,
            "timestamp": timestamp_ms,
            "kind": "text",
            "value": {"text": str(output_data)},
            "metadata": metadata,
        }

    trimmed = append_history_entry(workflow, node_id, entry)
    if trimmed:
        await cleanup_history_files(trimmed, workflow_id, workflow, data_manager)
    return entry


async def save_node_output_data_with_rollback(
    workflow_id: str,
    node_id: str,
    port_id: str,
    output_data: bytes | str | dict,
    file_info: dict | None,
    workflow: dict,
    data_manager: BaseDataManager,
    task_manager: BaseTaskManager,
    file_ext: str = ".bin",
) -> None:
    await _append_port_output_to_workflow(workflow_id, node_id, port_id, output_data, file_info, workflow, data_manager, file_ext)
    success = await task_manager.update_workflow(workflow_id, {"node_output_history": workflow["node_output_history"]}, user_id=None)
    if not success:
        raise Exception("Failed to update workflow in database")


@app.delete("/api/v1/workflow/{workflow_id}")
async def api_v1_workflow_delete(request: Request, user=Depends(verify_user_access)):
    try:
        workflow_id = request.path_params["workflow_id"]
        workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
        success = await task_manager.delete_workflow(workflow_id, user["user_id"])
        if not success:
            return error_response("Failed to delete workflow", 400)

        if workflow and workflow.get("node_output_history"):
            await cleanup_workflow_files(workflow_id, workflow, data_manager)

        logger.info(f"Workflow {workflow_id} deleted by user {user['user_id']}")
        return {"message": "Workflow deleted successfully"}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.delete("/api/v1/workflow/{workflow_id}/node/{node_id}")
async def api_v1_workflow_delete_node(request: Request, user=Depends(verify_user_access)):
    try:
        workflow_id = request.path_params["workflow_id"]
        node_id = request.path_params["node_id"]
        workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
        if not workflow:
            return error_response("Workflow not found", 404)

        await cleanup_node_data(workflow_id, node_id, workflow, data_manager, task_manager)

        logger.info(f"Node {node_id} deleted from workflow {workflow_id}")
        return {"message": "Node deleted successfully"}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


def _build_single_output_response(entry: dict) -> dict:
    """Build { data, entry, url? } for one history entry."""
    kind = entry.get("kind", "")
    value = entry.get("value") if isinstance(entry.get("value"), dict) else {}
    resp: dict[str, Any] = {"entry": entry}
    if kind == "text":
        resp["data"] = {"data_type": "text", "text_value": value.get("text", "")}
    elif kind == "json":
        resp["data"] = {"data_type": "json", "json_value": value.get("json")}
    elif kind == "file":
        data_url = value.get("dataUrl")
        url = value.get("url")
        if data_url:
            resp["data"] = {"data_type": "text", "text_value": data_url}
        elif url:
            resp["data"] = {"data_type": "url", "url_value": url}
            resp["url"] = url
        else:
            storage_path = (entry.get("metadata") or {}).get("storage_path")
            resp["data"] = {"data_type": "file", "file_path": storage_path}
    elif kind == "lightx2v_result":
        resp["data"] = {"data_type": "url", "url_value": None}
    else:
        resp["data"] = {"data_type": "json", "json_value": value}
    return resp


@app.get("/api/v1/workflow/{workflow_id}/node/{node_id}/output")
async def api_v1_workflow_node_output(request: Request, user=Depends(verify_user_access)):
    """Read current output(s) for the node. Returns outputs keyed by port_id (or field from __json__)."""
    try:
        workflow_id = request.path_params["workflow_id"]
        node_id = request.path_params["node_id"]
        workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
        if not workflow:
            return error_response("Workflow not found", 404)

        history_map = workflow.get("node_output_history", {})
        if not isinstance(history_map, dict):
            return {"outputs": {}}

        entries = history_map.get(node_id, [])
        if not isinstance(entries, list) or len(entries) == 0:
            return {"outputs": {}}

        # Latest entry per port_id (by timestamp desc)
        latest_by_port: dict[str, dict] = {}
        for e in entries:
            if not isinstance(e, dict):
                continue
            meta = e.get("metadata") or {}
            port_id = meta.get("port_id")
            if port_id is None:
                port_id = "output"
            ts = e.get("timestamp") or 0
            if port_id not in latest_by_port or (latest_by_port[port_id].get("timestamp") or 0) < ts:
                latest_by_port[port_id] = e

        # Expand __json__ into one virtual port per key
        json_entry = latest_by_port.pop(JSON_PORT_ID, None)
        if json_entry and json_entry.get("kind") == "json":
            json_value = (json_entry.get("value") or {}).get("json") if isinstance(json_entry.get("value"), dict) else None
            if isinstance(json_value, dict):
                for key, field_val in json_value.items():
                    synthetic = {
                        "id": json_entry.get("id"),
                        "timestamp": json_entry.get("timestamp"),
                        "kind": "json",
                        "value": {"json": field_val},
                        "metadata": {**(json_entry.get("metadata") or {}), "port_id": key},
                    }
                    if isinstance(field_val, str):
                        latest_by_port[key] = {"data": {"data_type": "text", "text_value": field_val}, "entry": synthetic}
                    else:
                        latest_by_port[key] = {"data": {"data_type": "json", "json_value": field_val}, "entry": synthetic}
            else:
                latest_by_port[JSON_PORT_ID] = _build_single_output_response(json_entry)

        outputs: dict[str, Any] = {}
        for port_id, entry in latest_by_port.items():
            if "data" in entry and "entry" in entry:
                outputs[port_id] = entry
            else:
                outputs[port_id] = _build_single_output_response(entry)
        return {"outputs": outputs}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/workflow/{workflow_id}/node/{node_id}/output/history")
async def api_v1_workflow_node_output_history(request: Request, user=Depends(verify_user_access)):
    """Read history list for the node (all entries, any port)."""
    try:
        workflow_id = request.path_params["workflow_id"]
        node_id = request.path_params["node_id"]
        workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
        if not workflow:
            return error_response("Workflow not found", 404)

        history_map = workflow.get("node_output_history", {})
        if not isinstance(history_map, dict):
            return {"history": []}

        entries = history_map.get(node_id, [])
        if not isinstance(entries, list):
            return {"history": []}
        return {"history": [e for e in entries if isinstance(e, dict)]}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/workflow/{workflow_id}/node/{node_id}/output/reuse")
async def api_v1_workflow_node_output_reuse(request: Request, user=Depends(verify_user_access)):
    """Reuse a history entry by index. Body: { \"port_id\": \"output\", \"history_index\": 0 }."""
    try:
        workflow_id = request.path_params["workflow_id"]
        node_id = request.path_params["node_id"]
        body = await request.json() or {}
        port_id = body.get("port_id") or "output"
        history_index = body.get("history_index", 0)
        workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
        if not workflow:
            return error_response("Workflow not found", 404)
        history_map = workflow.get("node_output_history", {})
        entries = history_map.get(node_id, []) if isinstance(history_map, dict) else []
        if not isinstance(entries, list):
            return error_response("History not found", 404)
        matching = [e for e in entries if isinstance(e, dict) and ((e.get("metadata") or {}).get("port_id") or "output") == port_id]
        if history_index < 0 or history_index >= len(matching):
            return error_response("History index out of range", 404)
        entry = matching[history_index]
        resp = _build_single_output_response(entry)
        return resp
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/workflow/{workflow_id}/node/{node_id}/output/upload")
async def api_v1_workflow_node_output_upload(request: Request, file: UploadFile = File(...), port_id: str = Form("output"), user=Depends(verify_user_access)):
    """Upload file as node output. Optional form field port_id (default \"output\")."""
    try:
        workflow_id = request.path_params["workflow_id"]
        node_id = request.path_params["node_id"]
        workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
        if not workflow:
            return error_response("Workflow not found", 404)

        file_content = await file.read()
        original_filename = file.filename or "file"
        ext = os.path.splitext(original_filename)[1].lower() or ".bin"
        file_id = str(uuid.uuid4())
        file_info = {"file_id": file_id}

        saved_file_url = await save_node_output_data_with_rollback(workflow_id, node_id, port_id, file_content, file_info, workflow, data_manager, task_manager, file_ext=ext)

        logger.info(f"Uploaded file for node {node_id} (port {port_id}) in workflow {workflow_id}")
        history_map = workflow.get("node_output_history", {})
        entry = None
        if isinstance(history_map, dict):
            node_entries = history_map.get(node_id, [])
            if isinstance(node_entries, list):
                def matches_port(e: dict) -> bool:
                    meta = e.get("metadata") or {}
                    return (meta.get("port_id") or "output") == port_id
                matching = [e for e in node_entries if isinstance(e, dict) and matches_port(e)]
                if matching:
                    entry = matching[0]

        response_payload: dict[str, Any] = {
            "message": "File uploaded successfully",
            "file_id": file_id,
            "file_url": saved_file_url if saved_file_url else f"/api/v1/workflow/{workflow_id}/file/{file_id}",
        }
        if entry:
            response_payload["entry"] = entry
        return response_payload
    except Exception as e:
        error_msg = f"Failed to upload file for node {node_id} in workflow {workflow_id}: {str(e)}"
        logger.error(error_msg)
        traceback.print_exc()
        return error_response(error_msg, 500)


@app.get("/api/v1/workflow/{workflow_id}/file/{file_id}")
async def api_v1_workflow_file(request: Request, user=Depends(verify_user_access)):
    try:
        workflow_id = request.path_params["workflow_id"]
        file_id = request.path_params["file_id"]
        workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
        if not workflow:
            return error_response("Workflow not found", 404)

        file_path = find_file_path_in_outputs(workflow, file_id)
        if not file_path:
            # Fallback: entries may have fileId but no metadata.storage_path (e.g. from frontend autosave).
            # Use value.ext if present, else try conventional path workflows/{workflow_id}_{file_id}.{ext}.
            if count_file_references(workflow, file_id) > 0:
                entry_ext = get_entry_ext(workflow, file_id)
                file_path = await _try_conventional_file_path(workflow_id, file_id, data_manager, entry_ext)
        if not file_path:
            return error_response("File not found", 404)

        try:
            data = await data_manager.load_bytes(file_path)
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            return error_response("File not found", 404)

        ext = os.path.splitext(file_path)[1].lower()
        mime_type = WORKFLOW_FILE_MIME.get(ext, "application/octet-stream")
        headers = {"Cache-Control": "public, max-age=3600"}
        return Response(content=data, media_type=mime_type, headers=headers)
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


JSON_PORT_ID = "__json__"


@app.post("/api/v1/workflow/{workflow_id}/node/{node_id}/outputs/save")
async def api_v1_workflow_node_outputs_save(request: Request, user=Depends(verify_user_access)):
    """Save all outputs for one node in a single request.
    For multi-port object output (e.g. text-generation customOutputs), pass {"__json__": fullObject}
    to store as one kind:json history entry. Reading by port_id will extract the field from the json."""
    try:
        workflow_id = request.path_params["workflow_id"]
        node_id = request.path_params["node_id"]
        workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
        if not workflow:
            return error_response("Workflow not found", 404)

        params = await request.json()
        outputs = params.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            return error_response("outputs object with at least one port_id is required", 400)

        # Single JSON mode: save whole object as one entry
        if JSON_PORT_ID in outputs:
            output_data_raw = outputs[JSON_PORT_ID]
            if output_data_raw is not None and isinstance(output_data_raw, dict):
                await _append_port_output_to_workflow(
                    workflow_id, node_id, JSON_PORT_ID, output_data_raw, None, workflow, data_manager, ".bin"
                )
                success = await task_manager.update_workflow(
                    workflow_id, {"node_output_history": workflow["node_output_history"]}, user_id=None
                )
                if not success:
                    raise Exception("Failed to update workflow in database")
                node_entries = workflow.get("node_output_history", {}).get(node_id, [])
                entry = None
                if isinstance(node_entries, list):
                    for e in node_entries:
                        if isinstance(e, dict) and (e.get("metadata") or {}).get("port_id") == JSON_PORT_ID:
                            entry = e
                            break
                results: dict[str, dict[str, Any]] = {}
                if entry:
                    for key in output_data_raw.keys():
                        results[key] = {}
                logger.info(f"Saved JSON output for node {node_id} in workflow {workflow_id}")
                return {"message": "Node outputs saved successfully", "results": results}
            return error_response("__json__ value must be a non-empty object", 400)

        # Per-port mode
        for port_id, output_data_raw in outputs.items():
            if not isinstance(port_id, str) or output_data_raw is None:
                continue
            try:
                output_data, file_info, file_ext = _parse_output_data_raw(output_data_raw, None)
            except ValueError:
                return error_response(f"Invalid output_data for port {port_id}", 400)
            await _append_port_output_to_workflow(
                workflow_id, node_id, port_id, output_data, file_info, workflow, data_manager, file_ext
            )

        success = await task_manager.update_workflow(workflow_id, {"node_output_history": workflow["node_output_history"]}, user_id=None)
        if not success:
            raise Exception("Failed to update workflow in database")

        logger.info(f"Saved {len(outputs)} output(s) for node {node_id} in workflow {workflow_id}")
        results = {}
        node_entries = workflow.get("node_output_history", {}).get(node_id, [])
        if isinstance(node_entries, list):
            for e in node_entries:
                if isinstance(e, dict):
                    pid = (e.get("metadata") or {}).get("port_id", "")
                    if pid not in results and pid in outputs:
                        entry_value = e.get("value") if isinstance(e.get("value"), dict) else {}
                        r: dict[str, Any] = {}
                        if entry_value.get("fileId"):
                            r["file_id"] = entry_value["fileId"]
                        if entry_value.get("url"):
                            r["file_url"] = r["url"] = entry_value["url"]
                        if entry_value.get("ext"):
                            r["ext"] = entry_value["ext"]
                        results[pid] = r
                        if len(results) >= len(outputs):
                            break
        return {"message": "Node outputs saved successfully", "results": results}
    except Exception as e:
        error_msg = f"Failed to save node outputs for {node_id} in workflow {workflow_id}: {str(e)}"
        logger.error(error_msg)
        traceback.print_exc()
        return error_response(error_msg, 500)


def _to_persisted_message(msg: dict) -> dict:
    """Strip non-persisted fields from a chat message."""
    out = {
        "id": msg.get("id"),
        "role": msg.get("role"),
        "content": msg.get("content", ""),
        "timestamp": msg.get("timestamp", 0),
    }
    for k in ["image", "useSearch", "sources", "error"]:
        if k in msg and msg[k] is not None:
            out[k] = msg[k]
    return out


@app.get("/api/v1/workflow/{workflow_id}/chat")
async def api_v1_workflow_chat_get(request: Request, user=Depends(verify_user_access)):
    """Get chat history for a workflow. Returns messages from separate store or workflow doc (fallback)."""
    try:
        workflow_id = request.path_params["workflow_id"]
        workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
        if not workflow:
            return error_response("Workflow not found", 404)

        if hasattr(task_manager, "get_workflow_chat_history"):
            data = await task_manager.get_workflow_chat_history(workflow_id, user["user_id"])
            if data:
                return {"workflow_id": workflow_id, "messages": data.get("messages", []), "updated_at": data.get("updated_at")}
        return {"workflow_id": workflow_id, "messages": workflow.get("chat_history") or [], "updated_at": None}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/workflow/{workflow_id}/chat")
async def api_v1_workflow_chat_post(request: Request, user=Depends(verify_user_access)):
    """Append messages to chat history."""
    try:
        workflow_id = request.path_params["workflow_id"]
        workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
        if not workflow:
            return error_response("Workflow not found", 404)
        if not hasattr(task_manager, "get_workflow_chat_history") or not hasattr(task_manager, "save_workflow_chat_history"):
            return error_response("Chat history storage not available", 501)

        params = await request.json()
        to_append = params.get("messages", [])
        if not isinstance(to_append, list):
            to_append = []
        persisted = [_to_persisted_message(m) for m in to_append if isinstance(m, dict)]

        existing = await task_manager.get_workflow_chat_history(workflow_id, user["user_id"])
        if existing:
            messages = existing.get("messages", []) + persisted
        else:
            legacy = workflow.get("chat_history") or []
            messages = [_to_persisted_message(m) for m in legacy if isinstance(m, dict)] + persisted
        updated_at = await task_manager.save_workflow_chat_history(workflow_id, messages, user["user_id"])
        return {"workflow_id": workflow_id, "messages": messages, "updated_at": updated_at}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.put("/api/v1/workflow/{workflow_id}/chat")
async def api_v1_workflow_chat_put(request: Request, user=Depends(verify_user_access)):
    """Replace full chat history."""
    try:
        workflow_id = request.path_params["workflow_id"]
        workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
        if not workflow:
            return error_response("Workflow not found", 404)
        if not hasattr(task_manager, "save_workflow_chat_history"):
            return error_response("Chat history storage not available", 501)

        params = await request.json()
        messages = params.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        persisted = [_to_persisted_message(m) for m in messages if isinstance(m, dict)]
        updated_at = await task_manager.save_workflow_chat_history(workflow_id, persisted, user["user_id"])
        return {"workflow_id": workflow_id, "messages": persisted, "updated_at": updated_at}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/workflow/{workflow_id}/autosave")
async def api_v1_workflow_autosave(request: Request, user=Depends(verify_user_access)):
    try:
        workflow_id = request.path_params["workflow_id"]
        params = await request.json()
        success = await _update_workflow(workflow_id, params, user)
        if success:
            logger.info(f"Workflow {workflow_id} autosaved")
            workflow = await task_manager.query_workflow(workflow_id, user["user_id"])
            update_t = workflow["update_t"] if workflow else None
            return {"message": "Workflow autosaved", "workflow_id": workflow_id, "update_t": update_t}
        else:
            return error_response("Failed to autosave workflow", 400)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


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
    parser.add_argument("--face_detector_model_path", type=str, default=None)
    parser.add_argument("--face_detector_method", type=str, default="yolo")
    parser.add_argument("--audio_separator_model_path", type=str, default="")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    model_pipelines = Pipeline(args.pipeline_json)
    volcengine_tts_client = VolcEngineTTSClient(args.volcengine_tts_list_json)
    volcengine_asr_client = VolcEngineASRClient()
    sensetime_voice_clone_client = SenseTimeTTSClient()
    volcengine_podcast_client = VolcEnginePodcastClient()
    face_detector = FaceDetector(method=args.face_detector_method, model_path=args.face_detector_model_path)
    try:
        audio_separator = AudioSeparator(model_path=args.audio_separator_model_path)
    except Exception as e:
        logger.warning(f"Failed to initialize audio_separator, audio separation feature will be disabled: {e}")
        audio_separator = None
    auth_manager = AuthManager()
    if args.task_url.startswith("/"):
        task_manager = LocalTaskManager(args.task_url, metrics_monitor)
    elif args.task_url.startswith("postgresql://"):
        task_manager = PostgresSQLTaskManager(args.task_url, metrics_monitor)
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
