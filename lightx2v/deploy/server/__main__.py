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
from lightx2v.deploy.common.utils import load_inputs, data_name
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


def error_response(e, code):
    return JSONResponse({"message": f"error: {e}!"}, status_code=code)


async def prepare_subtasks(task_id):
    # pend subtasks that could run, put to message queue
    subtasks = await task_manager.next_subtasks(task_id)
    logger.info(f"got next subtasks: {subtasks}.")
    for sub in subtasks:
        await task_manager.pend_subtask(task_id, sub['worker_name'])
        r = await queue_manager.put_subtask(sub)
        assert r, "put subtask to queue error"


@app.post("/api/v1/task/submit")
async def api_v1_task_submit(request: Request):
    task_id = None
    try:
        params = await request.json()
        keys = [params.pop('task'), params.pop('model_cls'), params.pop('stage')]

        # get worker infos, model input names
        workers = model_pipelines.get_workers(keys)
        inputs = model_pipelines.get_inputs(keys)
        outputs = model_pipelines.get_outputs(keys)

        # process multimodal inputs data
        inputs_data = await load_inputs(params, inputs) 
        logger.info(f"{params}")

        # init task
        task_id = await task_manager.create_task(keys, workers, params, inputs, outputs)

        # save multimodal inputs data
        for inp, data in inputs_data.items():
            await data_manager.save_bytes(data, data_name(inp, task_id))

        await prepare_subtasks(task_id)
        return {'task_id': task_id, "workers": workers, "params": params}

    except Exception as e:
        traceback.print_exc()
        if task_id:
            subtasks = await task_manager.query_subtasks(task_id)
            for sub in subtasks:
                await task_manager.finish_subtask(task_id, sub['worker_name'], TaskStatus.FAILED) 
        return error_response(str(e), 500)


@app.get("/api/v1/task/query")
async def api_v1_task_query(request: Request):
    try:
        params = await request.json()
        task_id = params.pop('task_id')
        task = await task_manager.query_task(task_id, fmt=True)
        return {'task': task}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/worker/fetch")
async def api_v1_worker_fetch(request: Request):
    try:
        params = await request.json()
        logger.info(f"{params}")
        keys = params.pop('worker_keys')
        identity = params.pop('worker_identity')
        max_batch = params.get('max_batch', 1)
        timeout = params.get('timeout', 5)

        # get worker info
        worker = model_pipelines.get_worker(keys)
        subtasks = await queue_manager.get_subtasks(
            worker['queue'], max_batch, timeout
        )
        if not subtasks:
            # return error_response(f"no subtask for worker {keys}!", 404)
            return {'subtasks': []}

        for sub in subtasks:
            await task_manager.run_subtask(
                sub['task_id'], sub['worker_name'], identity
            )
        return {'subtasks': subtasks}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/worker/report")
async def api_v1_worker_report(request: Request):
    try:
        params = await request.json()
        logger.info(f"{params}")
        task_id = params.pop('task_id')
        worker_name = params.pop('worker_name')
        status = TaskStatus[params.pop('status')]
        identity = params.pop('worker_identity')

        # check if task identity == worker identity ?
        await task_manager.check_identity(task_id, worker_name, identity, status)
        # update task status
        await task_manager.finish_subtask(task_id, worker_name, status)
        # prepare new ready subtasks
        await prepare_subtasks(task_id)

        return {'msg': 'ok'}

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
        if args.task_url.startswith('/'):
            task_manager = LocalTaskManager(args.task_url)
        else:
            raise NotImplementedError
        if args.data_url.startswith('/'):
            data_manager = LocalDataManager(args.data_url)
        else:
            raise NotImplementedError
        if args.queue_url.startswith('/'):
            queue_manager = LocalQueueManager(args.queue_url)
        else:
            raise NotImplementedError

    uvicorn.run(app, host=args.ip, port=args.port, reload=False, workers=1)
