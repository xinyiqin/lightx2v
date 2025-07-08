import os
import uvicorn
import argparse
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.service_utils import ProcessManager
from lightx2v.deploy.common.pipeline import Pipeline
from lightx2v.deploy.common.utils import get_inputs_data, data_name
from lightx2v.deploy.task_manager.local_task_manager import LocalTaskManager
from lightx2v.deploy.data_manager.local_data_manager import LocalDataManager
from lightx2v.deploy.queue_manager.local_queue_manager import LocalQueueManager
from lightx2v.deploy.task_manager import TaskStatus

# =========================
# FastAPI Related Code
# =========================

model_pipelines = None
task_manager = None
data_manager = None
queue_manager = None

app = FastAPI()


@app.post("/api/v1/task/submit")
async def api_v1_task_submit(request: Request):
    task_id = None
    try:
        params = await request.json()
        task_type = params.pop('task')
        model_cls = params.pop('model_cls')
        stage = params.pop('stage')

        keys = [task_type, model_cls, stage]
        workers = model_pipelines.get_workers(keys)

        raw_inputs = model_pipelines.get_raw_inputs(keys)
        inputs_data = await get_inputs_data(params, raw_inputs) 

        logger.info(f"{params}")
        task_id = await task_manager.create_task(keys, workers, params)

        for inp, data in inputs_data.items():
            await data_manager.save_bytes(data, data_name(inp, task_id))

        subtasks = await task_manager.next_subtasks(task_id)
        for sub in subtasks:
            sub['params'] = params
            await task_manager.pend_subtask(task_id, sub['worker_name'])
            r = await queue_manager.put_subtask(sub)
            assert r, "put subtask to queue error"

        return {'task_id': task_id, "workers": workers, "params": params}

    except Exception as e:
        traceback.print_exc()
        if task_id:
            subtasks = await task_manager.query_subtasks(task_id)
            for sub in subtasks:
                await task_manager.finish_subtask(task_id, sub['worker_name'], TaskStatus.FAILED) 
        return JSONResponse({"message": f"error: {e}!"}, status_code=500)


@app.get("/v1/local/video/generate/service_status")
async def get_service_status():
    return ApiServerServiceStatus.get_status_service()


@app.get("/v1/local/video/generate/get_all_tasks")
async def get_all_tasks():
    return ApiServerServiceStatus.get_all_tasks()


@app.post("/v1/local/video/generate/task_status")
async def get_task_status(message):
    return ApiServerServiceStatus.get_status_task_id(message.task_id)


@app.get("/v1/local/video/generate/stop_running_task")
async def stop_running_task():
    global thread
    if thread and thread.is_alive():
        try:
            _async_raise(thread.ident, SystemExit)
            thread.join()

            # Clean up the thread reference
            thread = None
            ApiServerServiceStatus.clean_stopped_task()
            gc.collect()
            torch.cuda.empty_cache()
            return {"stop_status": "success", "reason": "Task stopped successfully."}
        except Exception as e:
            return {"stop_status": "error", "reason": str(e)}
    else:
        return {"stop_status": "do_nothing", "reason": "No running task found."}


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

    # global model_pipelines
    # global task_manager
    # global data_manager
    # global queue_manager

    with ProfilingContext("Init Server Cost"):
        model_pipelines = Pipeline(args.pipeline_json)
        task_manager = LocalTaskManager(args.task_url)
        data_manager = LocalDataManager(args.data_url)
        queue_manager = LocalQueueManager(args.queue_url)

    uvicorn.run(app, host=args.ip, port=args.port, reload=False, workers=1)
