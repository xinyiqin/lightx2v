import os
import sys
import signal
import time
import uuid
import json
import asyncio
import argparse
import requests
from loguru import logger

from lightx2v.utils.service_utils import ProcessManager
from lightx2v.deploy.data_manager import LocalDataManager, S3DataManager
from lightx2v.deploy.task_manager import TaskStatus
from lightx2v.deploy.common.utils import try_catch
from lightx2v.deploy.worker.hub import PipelineWorker, TextEncoderWorker, ImageEncoderWorker, VaeEncoderWorker, VaeDecoderWorker, DiTWorker


RUNNER_MAP = {
    "pipeline": PipelineWorker,
    "text_encoder": TextEncoderWorker,
    "image_encoder": ImageEncoderWorker,
    "vae_encoder": VaeEncoderWorker,
    "vae_decoder": VaeDecoderWorker,
    "dit": DiTWorker,
}

# {task_id: {"server": xx, "worker_name": xx, "identity": xx}}
RUNNING_SUBTASKS = {}


@try_catch
def fetch_subtasks(server_url, worker_keys, worker_identity):
    url = server_url + "/api/v1/worker/fetch"
    params = {
        "worker_keys": worker_keys,
        "worker_identity": worker_identity,
        "max_batch": 1,
        "timeout": 60,
    }
    ret = requests.get(url, data=json.dumps(params))
    if ret.status_code == 200:
        subtasks = ret.json()['subtasks']
        logger.info(f"{worker_identity} fetch {worker_keys} ok: {subtasks}")
        return subtasks
    else:
        logger.warning(f"{worker_identity} fetch {worker_keys} fail: [{ret.status_code}], error: {ret.text}")
        return None

@try_catch
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
        RUNNING_SUBTASKS.pop(task_id)
        ret = ret.json()
        logger.info(f"{worker_identity} report {task_id} {worker_name} {status} ok")
        return True
    else:
        logger.warning(f"{worker_identity} report {task_id} {worker_name} {status} fail: [{ret.status_code}], error: {ret.text}")
        return False


async def main(args):
    worker_keys = [args.task, args.model_cls, args.stage, args.worker]

    data_manager = None
    if args.data_url.startswith("/"):
        data_manager = LocalDataManager(args.data_url)
    elif args.data_url.startswith("{"):
        data_manager = S3DataManager(args.data_url)
    else:
        raise NotImplementedError
    runner = RUNNER_MAP[args.worker](args)
    await data_manager.init()

    while True:
        subtasks = fetch_subtasks(args.server, worker_keys, args.identity)
        if subtasks is not None and len(subtasks) > 0:
            for sub in subtasks:
                RUNNING_SUBTASKS[sub['task_id']] = {
                    "server": args.server,
                    "worker_name": sub['worker_name'],
                    "identity": args.identity,
                }
                ret = await runner.run(sub['inputs'], sub['outputs'], sub['params'], data_manager)
                status = TaskStatus.SUCCEED.name if ret is True else TaskStatus.FAILED.name
                report_task(args.server, sub['task_id'], sub['worker_name'], status, args.identity)
        else:
            await asyncio.sleep(5)


def signal_handler(signum, frame):
    logger.info("\nReceived Ctrl+C, report all running subtasks")
    for task_id, s in RUNNING_SUBTASKS.items():
        report_task(s['server'], task_id, s['worker_name'], TaskStatus.FAILED.name, s['identity'])
    ProcessManager.kill_all_related_processes()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)  # 捕获Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # 捕获终止信号


# =========================
# Main Entry
# =========================

if __name__ == "__main__":
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
        args.identity = 'worker-' + str(uuid.uuid4())[:8]
    logger.info(f"args: {args}")

    asyncio.run(main(args))
