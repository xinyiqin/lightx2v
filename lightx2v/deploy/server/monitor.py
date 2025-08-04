import time
import asyncio
from enum import Enum
from loguru import logger
from lightx2v.deploy.task_manager import TaskStatus
from lightx2v.deploy.common.utils import class_try_catch_async

class WorkerStatus(Enum):
    FETCHING = 1
    FETCHED = 2
    DISCONNECT = 3
    REPORT = 4


class CostWindow:
    def __init__(self, window):
        self.window = window
        self.costs = []
        self.avg = None

    def append(self, cost):
        self.costs.append(cost)
        if len(self.costs) > self.window:
            self.costs.pop(0)
        self.avg = sum(self.costs) / len(self.costs)


class WorkerClient:
    def __init__(self, queue, identity, infer_timeout, offline_timeout, avg_window):
        self.queue = queue
        self.identity = identity
        self.status = None
        self.update_t = time.time()
        self.infer_cost = CostWindow(avg_window)
        self.offline_cost = CostWindow(avg_window)
        self.infer_timeout = infer_timeout
        self.offline_timeout = offline_timeout

    # FETCHING -> FETCHED -> REPORT -> FETCHING
    # FETCHING -> DISCONNECT -> FETCHING
    def update(self, status: WorkerStatus):
        pre_status = self.status
        pre_t = self.update_t
        self.status = status
        self.update_t = time.time()
        cur_cost = self.update_t - pre_t

        if status == WorkerStatus.FETCHING:
            if pre_status in [WorkerStatus.FETCHED, WorkerStatus.REPORT] and pre_t is not None:
                if cur_cost < self.offline_timeout:
                    self.offline_cost.append(max(cur_cost, 1))

        elif status == WorkerStatus.REPORT:
            if pre_status == WorkerStatus.FETCHED and pre_t is not None:
                if cur_cost < self.infer_timeout:
                    self.infer_cost.append(max(cur_cost, 1))

    def check(self):
        elapse = time.time() - self.update_t
        logger.warning(f"Worker {self.queue} {self.identity} elapse {elapse:.2f} s, status={self.status} infer_avg={self.infer_cost.avg} offline_avg={self.offline_cost.avg} infer_timeout={self.infer_timeout} offline_timeout={self.offline_timeout}")
        if self.status == WorkerStatus.FETCHED:
            # infer too long
            if self.infer_cost.avg is not None and elapse > self.infer_cost.avg * 5:
                return False
            if elapse > self.infer_timeout:
                return False
        elif self.status == WorkerStatus.DISCONNECT:
            # offline too long
            if self.offline_cost.avg is not None and elapse > self.offline_cost.avg * 5:
                return False
            if elapse > self.offline_timeout:
                return False
        return True


class ServerMonitor:
    def __init__(self, model_pipelines, task_manager, queue_manager, interval=1):
        self.model_pipelines = model_pipelines
        self.task_manager = task_manager
        self.queue_manager = queue_manager
        self.interval = interval
        self.stop = False
        self.worker_clients = {}
        self.identity_to_queue = {}

        self.all_queues = self.model_pipelines.get_queues()
        self.subtask_created_timeout = self.model_pipelines.get_subtask_created_timeout()
        self.subtask_pending_timeout = self.model_pipelines.get_subtask_pending_timeout()
        self.worker_avg_window = self.model_pipelines.get_worker_avg_window()
        self.worker_offline_timeout = self.model_pipelines.get_worker_offline_timeout()
        self.worker_min_capacity = self.model_pipelines.get_worker_min_capacity()
        self.task_timeout = self.model_pipelines.get_task_tolerance_timeout()
        self.schedule_ratio_high = self.model_pipelines.get_schedule_ratio_high()
        self.schedule_ratio_low = self.model_pipelines.get_schedule_ratio_low()

    async def init(self):
        while True:
            if self.stop:
                break
            await self.clean_workers()
            await self.clean_subtasks()
            await asyncio.sleep(self.interval)
        logger.info("ServerMonitor stopped")

    async def close(self):
        self.stop = True
        self.model_pipelines = None
        self.task_manager = None
        self.queue_manager = None
        self.worker_clients = None

    def init_worker(self, queue, identity):
        if queue not in self.worker_clients:
            self.worker_clients[queue] = {}
        if identity not in self.worker_clients[queue]:
            infer_timeout = self.model_pipelines.get_subtask_running_timeout(queue)
            self.worker_clients[queue][identity] = WorkerClient(
                queue, identity, infer_timeout, self.worker_offline_timeout, self.worker_avg_window
            )
            self.identity_to_queue[identity] = queue
        return self.worker_clients[queue][identity]

    @class_try_catch_async
    async def worker_update(self, queue, identity, status):
        if queue is None:
            queue = self.identity_to_queue[identity]
        worker = self.init_worker(queue, identity)
        worker.update(status)
        logger.info(f"Worker {identity} {queue} update [{status}]")

    async def clean_workers(self):
        qs = list(self.worker_clients.keys())
        for queue in qs:
            idens = list(self.worker_clients[queue].keys())
            for identity in idens:
                if not self.worker_clients[queue][identity].check():
                    logger.warning(f"Worker {queue} {identity} out of contact, remove it")
                    self.worker_clients[queue].pop(identity)
                    self.identity_to_queue.pop(identity)

    async def clean_subtasks(self):
        created_end_t = time.time() - self.subtask_created_timeout
        pending_end_t = time.time() - self.subtask_pending_timeout
        fails = set()

        created_tasks = await self.task_manager.list_tasks(
            status=TaskStatus.CREATED, subtasks=True, end_created_t=created_end_t
        )
        pending_tasks = await self.task_manager.list_tasks(
            status=TaskStatus.PENDING, subtasks=True, end_created_t=pending_end_t
        )

        def fmt_subtask(t):
            return f"({t['task_id']}, {t['worker_name']}, {t['queue']}, {t['worker_identity']})"

        for t in created_tasks + pending_tasks:
            if t['task_id'] in fails:
                continue
            elapse = time.time() - t['update_t']
            logger.warning(f"Subtask {fmt_subtask(t)} CREATED / PENDING timeout: {elapse:.2f} s")
            await self.task_manager.finish_subtasks(
                t['task_id'], TaskStatus.FAILED, worker_name=t['worker_name']
            )
            fails.add(t['task_id'])

        running_tasks = await self.task_manager.list_tasks(
            status=TaskStatus.RUNNING, subtasks=True
        )

        for t in running_tasks:
            if t['task_id'] in fails:
                continue
            elapse = time.time() - t['update_t']
            limit = self.model_pipelines.get_subtask_running_timeout(t['queue'])
            if elapse >= limit:
                logger.warning(f"Subtask {fmt_subtask(t)} RUNNING timeout: {elapse:.2f} s")
                await self.task_manager.finish_subtasks(
                    t['task_id'], TaskStatus.FAILED, worker_name=t['worker_name']
                )
                fails.add(t['task_id'])

    def get_avg_worker_infer_cost(self, queue):
        if queue not in self.worker_clients:
            self.worker_clients[queue] = {}
        infer_costs = []
        for _, client in self.worker_clients[queue].items():
            if client.infer_cost.avg is not None:
                infer_costs.append(client.infer_cost.avg)
        if len(infer_costs) <= 0:
            return None
        return sum(infer_costs) / len(infer_costs)

    # check if a task can be published to queues
    @class_try_catch_async
    async def check_queue_busy(self, keys, queues):
        wait_time = 0

        for queue in queues:
            avg_cost = self.get_avg_worker_infer_cost(queue)
            if avg_cost is None:
                avg_cost = self.model_pipelines.get_subtask_running_timeout(queue)
            worker_cnt = len(self.worker_clients[queue])
            subtask_pending = await self.queue_manager.pending_num(queue)
            capacity = self.task_timeout * max(worker_cnt, 1) // avg_cost
            capacity = max(self.worker_min_capacity, capacity)

            if subtask_pending >= capacity:
                ss = f"pending={subtask_pending}, capacity={capacity}"
                logger.warning(f"Queue {queue} busy, {ss}, task {keys} cannot be publised!")
                return None
            wait_time += avg_cost * subtask_pending / max(worker_cnt, 1)
        return wait_time

    @class_try_catch_async
    async def cal_metrics(self):
        data = {}
        target_high = self.task_timeout * self.schedule_ratio_high
        target_low = self.task_timeout * self.schedule_ratio_low

        for queue in self.all_queues:
            avg_cost = self.get_avg_worker_infer_cost(queue)
            worker_cnt = len(self.worker_clients[queue])
            subtask_pending = await self.queue_manager.pending_num(queue)

            data[queue] = {
                "avg_cost": avg_cost,
                "worker_cnt": worker_cnt,
                "subtask_pending": subtask_pending,
                "max_worker": 0,
                "min_worker": 0,
                "need_add_worker": 0,
                "need_del_worker": 0,
                "del_worker_identities": [],
            }

            if avg_cost is not None:
                data[queue]["min_worker"] = max(1, subtask_pending * avg_cost // target_high)
                data[queue]["max_worker"] = max(1, subtask_pending * avg_cost // target_low)
            else:
                avg_cost = self.model_pipelines.get_subtask_running_timeout(queue)
                data[queue]["avg_cost"] = avg_cost
                base_cnt = min(subtask_pending // self.worker_min_capacity, 1)
                data[queue]["min_worker"] = max(1, min(base_cnt, subtask_pending * avg_cost // target_high))
                data[queue]["max_worker"] = max(1, min(base_cnt, subtask_pending * avg_cost // target_low))

            if worker_cnt < data[queue]["min_worker"]:
                data[queue]["need_add_worker"] = data[queue]["min_worker"] - worker_cnt

            if subtask_pending == 0 and worker_cnt > data[queue]["max_worker"]:
                data[queue]["need_del_worker"] = worker_cnt - data[queue]["max_worker"]
                if data[queue]["need_del_worker"] > 0:
                    for identity, client in self.worker_clients[queue].items():
                        if client.status in [WorkerStatus.FETCHING, WorkerStatus.DISCONNECT]:
                            data[queue]["del_worker_identities"].append(identity)
                            if len(data[queue]["del_worker_identities"]) >= data[queue]["need_del_worker"]:
                                break
        return data