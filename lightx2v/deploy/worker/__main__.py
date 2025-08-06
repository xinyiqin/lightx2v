import os
import sys
import signal
import uuid
import json
import asyncio
import argparse
import aiohttp
import traceback
from loguru import logger

from lightx2v.deploy.data_manager import LocalDataManager, S3DataManager
from lightx2v.deploy.task_manager import TaskStatus
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
WORKER_SECRET_KEY = os.getenv("WORKER_SECRET_KEY", "worker-secret-key-change-in-production")
HEADERS = {
    "Authorization": f"Bearer {WORKER_SECRET_KEY}",
    "Content-Type": "application/json"
}

async def fetch_subtasks(server_url, worker_keys, worker_identity, max_batch, timeout):
    url = server_url + "/api/v1/worker/fetch"
    params = {
        "worker_keys": worker_keys,
        "worker_identity": worker_identity,
        "max_batch": max_batch,
        "timeout": timeout,
    }
    try:
        logger.info(f"{worker_identity} fetching {worker_keys} with timeout: {timeout}s ...")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=json.dumps(params), headers=HEADERS, timeout=timeout + 10) as ret:
                if ret.status == 200:
                    ret = await ret.json()
                    subtasks = ret['subtasks']
                    for sub in subtasks:
                        RUNNING_SUBTASKS[sub['task_id']] = {
                            "server": server_url,
                            "worker_name": sub['worker_name'],
                            "identity": worker_identity,
                        }
                    logger.info(f"{worker_identity} fetch {worker_keys} ok: {subtasks}")
                    return subtasks
                else:
                    error_text = await ret.text()
                    logger.warning(f"{worker_identity} fetch {worker_keys} fail: [{ret.status}], error: {error_text}")
                    return None
    except asyncio.CancelledError:
        logger.warning("Fetch subtasks cancelled, shutting down...")
        raise asyncio.CancelledError
    except:
        logger.warning(f"Fetch subtasks failed: {traceback.format_exc()}")

async def report_task(server_url, task_id, worker_name, status, worker_identity):
    url = server_url + "/api/v1/worker/report"
    params = {
        "task_id": task_id,
        "worker_name": worker_name,
        "status": status,
        "worker_identity": worker_identity,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=json.dumps(params), headers=HEADERS) as ret:
                if ret.status == 200:
                    RUNNING_SUBTASKS.pop(task_id)
                    ret = await ret.json()
                    logger.info(f"{worker_identity} report {task_id} {worker_name} {status} ok")
                    return True
                else:
                    error_text = await ret.text()
                    logger.warning(f"{worker_identity} report {task_id} {worker_name} {status} fail: [{ret.status}], error: {error_text}")
                    return False
    except asyncio.CancelledError:
        logger.warning("Report task cancelled, shutting down...")
        raise asyncio.CancelledError
    except:
        logger.warning(f"Report task failed: {traceback.format_exc()}")

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

    try:
        while True:
            subtasks = await fetch_subtasks(args.server, worker_keys, args.identity, args.max_batch, args.timeout)
            if subtasks is not None and len(subtasks) > 0:
                for sub in subtasks:
                    ret = await runner.run(sub['inputs'], sub['outputs'], sub['params'], data_manager)
                    status = TaskStatus.SUCCEED.name if ret is True else TaskStatus.FAILED.name
                    await report_task(args.server, sub['task_id'], sub['worker_name'], status, args.identity)
            else:
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.warning("Main loop cancelled, shutting down...")
    except:
        logger.error(f"Main loop failed: {traceback.format_exc()}")


async def shutdown(loop):
    logger.warning("Received kill signal")
    for t in asyncio.all_tasks():
        if t is not asyncio.current_task():
            logger.warning(f"Cancel async task {t} ...")
            t.cancel()

    # Report any remaining running subtasks as failed
    task_ids = list(RUNNING_SUBTASKS.keys())
    for task_id in task_ids:
        try:
            s = RUNNING_SUBTASKS[task_id]
            logger.warning(f"Report {task_id} {s['worker_name']} {TaskStatus.FAILED.name} ...")
            await report_task(s['server'], task_id, s['worker_name'], TaskStatus.FAILED.name, s['identity'])
        except:
            logger.warning(f"Report task {task_id} failed: {traceback.format_exc()}")

    # Force exit after a short delay to ensure cleanup
    def force_exit():
        logger.warning("Force exiting process...")
        sys.exit(0)
    loop.call_later(2, force_exit)


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
    parser.add_argument("--max_batch", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)

    parser.add_argument("--server", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--data_url", type=str, default=dft_data_url)

    args = parser.parse_args()
    if args.identity == '':
        # TODO: spec worker instance identity by k8s env
        args.identity = 'worker-' + str(uuid.uuid4())[:8]
    logger.info(f"args: {args}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for s in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(s, lambda: asyncio.create_task(shutdown(loop)))

    try:
        loop.create_task(main(args), name="main")
        loop.run_forever()
    finally:
        loop.close()
        logger.warning("Event loop closed")
