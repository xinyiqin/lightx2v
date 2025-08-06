import os
import asyncio
import uvicorn
import argparse
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger

from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.service_utils import ProcessManager
from lightx2v.deploy.common.pipeline import Pipeline
from lightx2v.deploy.common.utils import load_inputs, data_name
from lightx2v.deploy.task_manager import LocalTaskManager, PostgresSQLTaskManager
from lightx2v.deploy.data_manager import LocalDataManager, S3DataManager
from lightx2v.deploy.queue_manager import LocalQueueManager, RabbitMQQueueManager
from lightx2v.deploy.task_manager import TaskStatus
from lightx2v.deploy.server.monitor import ServerMonitor, WorkerStatus
from lightx2v.deploy.server.auth import AuthManager

# =========================
# FastAPI Related Code
# =========================

model_pipelines = None
task_manager = None
data_manager = None
queue_manager = None
server_monitor = None
auth_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await task_manager.init()
    await data_manager.init()
    await queue_manager.init()
    asyncio.create_task(server_monitor.init())
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
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
security = HTTPBearer()


async def verify_user_access(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = auth_manager.verify_jwt_token(token)
    user_id = payload.get('user_id', None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user")
    user = await task_manager.query_user(user_id)
    # logger.info(f"Verfiy user access: {payload}")
    if user is None or user['user_id'] != user_id:
        raise HTTPException(status_code=401, detail="Invalid user")
    return user


async def verify_worker_access(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if not auth_manager.verify_worker_token(token):
        raise HTTPException(status_code=403, detail="Invalid worker token")
    return True 


def error_response(e, code):
    return JSONResponse({"message": f"error: {e}!"}, status_code=code)


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
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
        user_info['user_id'] = user_id
        access_token = auth_manager.create_jwt_token(user_info)
        logger.info(f"GitHub callback: user_info: {user_info}, access_token: {access_token}")
        return {"access_token": access_token, "user_info": user_info}
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


@app.get("/api/v1/model/list")
async def api_v1_model_list(user = Depends(verify_user_access)):
    try:
        return {'models': model_pipelines.get_model_lists()}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/task/submit")
async def api_v1_task_submit(request: Request, user = Depends(verify_user_access)):
    task_id = None
    try:
        params = await request.json()
        keys = [params.pop('task'), params.pop('model_cls'), params.pop('stage')]
        assert len(params["prompt"]) > 0, "valid prompt is required"

        # get worker infos, model input names
        workers = model_pipelines.get_workers(keys)
        inputs = model_pipelines.get_inputs(keys)
        outputs = model_pipelines.get_outputs(keys)
        types = model_pipelines.get_types(keys)

        # check if task can be published to queues
        queues = [v['queue'] for v in workers.values()]
        wait_time = await server_monitor.check_queue_busy(keys, queues)
        if wait_time is None:
            return error_response(f"Queue busy, please try again later", 500)

        # process multimodal inputs data
        inputs_data = await load_inputs(params, inputs, types)

        # init task
        task_id = await task_manager.create_task(
            keys, workers, params, inputs, outputs, user['user_id']
        )
        logger.info(f"Submit task: {task_id} {params}")

        # save multimodal inputs data
        for inp, data in inputs_data.items():
            await data_manager.save_bytes(data, data_name(inp, task_id))

        await prepare_subtasks(task_id)
        return {'task_id': task_id, "workers": workers, "params": params, "wait_time": wait_time}

    except Exception as e:
        traceback.print_exc()
        if task_id:
            await task_manager.finish_subtasks(task_id, TaskStatus.FAILED)
        return error_response(str(e), 500)


@app.get("/api/v1/task/query")
async def api_v1_task_query(request: Request, user = Depends(verify_user_access)):
    try:
        task_id = request.query_params['task_id']
        task = await task_manager.query_task(task_id, user['user_id'])
        if task is None:
            return {'msg': 'task not found'}
        task['status'] = task['status'].name
        return task
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/list")
async def api_v1_task_list(request: Request, user = Depends(verify_user_access)):
    try:
        user_id = user['user_id']
        tasks = await task_manager.list_tasks(user_id=user_id)
        for task in tasks:
            task['status'] = task['status'].name
        return {'tasks': tasks}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/result")
async def api_v1_task_result(request: Request, user = Depends(verify_user_access)):
    try:
        name = request.query_params['name']
        task_id = request.query_params['task_id']
        task = await task_manager.query_task(task_id, user_id=user['user_id'])
        if task is None:
            return error_response(f"Task {task_id} not found", 404)
        if task['status'] != TaskStatus.SUCCEED:
            return error_response(f"Task {task_id} not succeed", 400)
        assert name in task['outputs'], f"Output {name} not found in task {task_id}"
        data = await data_manager.load_bytes(task['outputs'][name])
        return Response(content=data, media_type="application/octet-stream")
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/cancel")
async def api_v1_task_cancel(request: Request, user = Depends(verify_user_access)):
    try:
        task_id = request.query_params['task_id']
        ret = await task_manager.cancel_task(task_id, user_id=user['user_id'])
        return {'msg': 'ok' if ret else 'failed'}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/resume")
async def api_v1_task_resume(request: Request, user = Depends(verify_user_access)):
    try:
        task_id = request.query_params['task_id']
        ret = await task_manager.resume_task(task_id, user_id=user['user_id'], all_subtask=True)
        if ret:
            await prepare_subtasks(task_id)
            return {'msg': 'ok'}
        else:
            return error_response(f"Task {task_id} resume failed", 400)
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/worker/fetch")
async def api_v1_worker_fetch(request: Request, valid = Depends(verify_worker_access)):
    try:
        params = await request.json()
        logger.info(f"Worker fetching: {params}")
        keys = params.pop('worker_keys')
        identity = params.pop('worker_identity')
        max_batch = params.get('max_batch', 1)
        timeout = params.get('timeout', 5)

        # check client disconnected
        async def check_client(request, fetch_task, identity, queue):
            while True:
                ret = await request.is_disconnected()
                if ret:
                    await server_monitor.worker_update(queue, identity, WorkerStatus.DISCONNECT)
                    fetch_task.cancel()
                    return
                await asyncio.sleep(1)

        # get worker info
        worker = model_pipelines.get_worker(keys)
        await server_monitor.worker_update(worker['queue'], identity, WorkerStatus.FETCHING)
        fetch_task = asyncio.create_task(queue_manager.get_subtasks(worker['queue'], max_batch, timeout))
        check_task = asyncio.create_task(check_client(request, fetch_task, identity, worker['queue']))
        subtasks = await fetch_task
        disconnected = await request.is_disconnected()

        if not subtasks or disconnected:
            return {'subtasks': []}

        worker_names = [sub['worker_name'] for sub in subtasks]
        task_ids = [sub['task_id'] for sub in subtasks]
        valids = await task_manager.run_subtasks(task_ids, worker_names, identity)
        valid_subtasks = [sub for sub in subtasks if (sub['task_id'], sub['worker_name']) in valids]

        if len(valid_subtasks) > 0:
            logger.info(f"Worker {identity} {keys} {request.client} fetched {valids}")
            check_task.cancel()
            await server_monitor.worker_update(worker['queue'], identity, WorkerStatus.FETCHED)
        return {'subtasks': valid_subtasks}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/worker/report")
async def api_v1_worker_report(request: Request, valid = Depends(verify_worker_access)):
    try:
        params = await request.json()
        logger.info(f"{params}")
        task_id = params.pop('task_id')
        worker_name = params.pop('worker_name')
        status = TaskStatus[params.pop('status')]
        identity = params.pop('worker_identity')
        await server_monitor.worker_update(None, identity, WorkerStatus.REPORT)

        ret = await task_manager.finish_subtasks(
            task_id, status, worker_identity=identity, worker_name=worker_name
        )
        # not all subtasks finished, prepare new ready subtasks
        if ret not in [TaskStatus.SUCCEED, TaskStatus.FAILED]:
            await prepare_subtasks(task_id)

        # all subtasks succeed, delete temp data
        elif ret == TaskStatus.SUCCEED:
            logger.info(f"Task {task_id} succeed")
            task = await task_manager.query_task(task_id)
            keys = [task['task_type'], task['model_cls'], task['stage']]
            temps = model_pipelines.get_temps(keys)
            for temp in temps:
                type = model_pipelines.get_type(temp)
                name = data_name(temp, task_id)
                await data_manager.get_delete_func(type)(name)

        elif ret == TaskStatus.FAILED:
            logger.warning(f"Task {task_id} failed")

        return {'msg': 'ok'}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/metrics")
async def api_v1_monitor_metrics(request: Request, valid = Depends(verify_worker_access)):
    try:
        metrics = await server_monitor.cal_metrics()
        return {'metrics': metrics}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


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

    parser.add_argument("--pipeline_json", type=str, default=dft_pipeline_json)
    parser.add_argument("--task_url", type=str, default=dft_task_url)
    parser.add_argument("--data_url", type=str, default=dft_data_url)
    parser.add_argument("--queue_url", type=str, default=dft_queue_url)
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    with ProfilingContext("Init Server Cost"):
        model_pipelines = Pipeline(args.pipeline_json)
        auth_manager = AuthManager()
        if args.task_url.startswith('/'):
            task_manager = LocalTaskManager(args.task_url)
        elif args.task_url.startswith('postgresql://'):
            task_manager = PostgresSQLTaskManager(args.task_url)
        else:
            raise NotImplementedError
        if args.data_url.startswith('/'):
            data_manager = LocalDataManager(args.data_url)
        elif args.data_url.startswith('{'):
            data_manager = S3DataManager(args.data_url)
        else:
            raise NotImplementedError
        if args.queue_url.startswith('/'):
            queue_manager = LocalQueueManager(args.queue_url)
        elif args.queue_url.startswith('amqp://'):
            queue_manager = RabbitMQQueueManager(args.queue_url)
        else:
            raise NotImplementedError
        server_monitor = ServerMonitor(model_pipelines, task_manager, queue_manager)

    uvicorn.run(app, host=args.ip, port=args.port, reload=False, workers=1)
