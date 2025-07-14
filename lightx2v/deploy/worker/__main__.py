import os
import time
import uuid
import json
import asyncio
import argparse
import requests
from loguru import logger

from lightx2v.utils.service_utils import ProcessManager
from lightx2v.deploy.data_manager.local_data_manager import LocalDataManager
from lightx2v.deploy.task_manager import TaskStatus

from lightx2v.deploy.worker.pipeline import PipelineRunner
from lightx2v.deploy.worker.text_encoder import TextEncoderRunner


RUNNER_MAP = {
    "pipeline": PipelineRunner,
    "text_encoder": TextEncoderRunner,
    "image_encoder": None,
    "vae_encoder": None,
    "vae_decoder": None,
    "dit": None,
}


def fetch_subtasks(server_url, worker_keys, worker_identity):
    url = server_url + "/api/v1/worker/fetch"
    params = {
        "worker_keys": worker_keys,
        "worker_identity": worker_identity,
        "max_batch": 1,
        "timeout": 5,
    }
    ret = requests.get(url, data=json.dumps(params))
    if ret.status_code == 200:
        subtasks = ret.json()['subtasks']
        print(f"fetch ok: {subtasks}")
        return subtasks
    else:
        print(f"fetch fail: [{ret.status_code}], error: {ret.text}")
        return None


def report_task(server_url, task_id, worker_name, status, worker_identity):
    url = server_url + "/api/v1/worker/report"
    params = {
        "task_id": task_id,
        "worker_name": worker_name,
        "status": status,
        "worker_identity": worker_identity,
    }
    ret = requests.get(url, data=json.dumps(params))
    if ret.status_code == 200:
        ret = ret.json()
        print(f"report ok: {ret}")
        return True 
    else:
        print(f"report fail: [{ret.status_code}], error: {ret.text}")
        return False


async def main(args):
    worker_keys = [args.task, args.model_cls, args.stage, args.worker]

    data_manager = None
    if args.data_url.startswith("/"):
        data_manager = LocalDataManager(args.data_url)
    else:
        raise NotImplementedError
    runner = RUNNER_MAP[args.worker](args)

    while True:
        subtasks = fetch_subtasks(args.server, worker_keys, args.identity)
        if subtasks is not None and len(subtasks) > 0:
            for sub in subtasks:
                ret = await runner.run(sub['inputs'], sub['outputs'], sub['params'], data_manager)
                status = TaskStatus.SUCCEED.name if ret is True else TaskStatus.FAILED.name
                report_task(args.server, sub['task_id'], sub['worker_name'], status, args.identity)
        else:
            await asyncio.sleep(5)


# =========================
# Main Entry
# =========================

if __name__ == "__main__":
    ProcessManager.register_signal_handler()
    parser = argparse.ArgumentParser()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(cur_dir, "../../.."))
    dft_data_url = os.path.join(base_dir, "local_data")

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model_cls", type=str, required=True)
    parser.add_argument("--stage", type=str, required=True)
    parser.add_argument("--worker", type=str, required=True)
    parser.add_argument("--identity", type=str, default='')

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)

    parser.add_argument("--server", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--data_url", type=str, default=dft_data_url)

    args = parser.parse_args()
    if args.identity == '':
        # TODO: spec worker instance identity by k8s env
        args.identity = str(uuid.uuid4())
    logger.info(f"args: {args}")

    asyncio.run(main(args))
