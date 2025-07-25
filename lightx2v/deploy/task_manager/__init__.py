import uuid
from enum import Enum
from lightx2v.deploy.common.utils import current_time, data_name


class TaskStatus(Enum):
    CREATED = 1
    PENDING = 2
    RUNNING = 3
    SUCCEED = 4
    FAILED = 5
    CANCEL = 6


class BaseTaskManager:
    def __init__(self):
        pass

    async def init(self):
        pass

    async def close(self):
        pass

    async def insert_task(self, task, subtasks):
        raise NotImplementedError

    async def list_tasks(self, **kwargs):
        raise NotImplementedError

    async def query_task(self, task_id):
        raise NotImplementedError

    async def next_subtasks(self, task_id):
        raise NotImplementedError

    async def run_subtasks(self, task_ids, worker_names, worker_identity):
        raise NotImplementedError

    async def finish_subtasks(self, task_id, status, worker_identity=None, worker_name=None):
        raise NotImplementedError

    async def cancel_task(self, task_id):
        raise NotImplementedError

    async def resume_task(self, task_id, all_subtask=False):
        raise NotImplementedError

    def fmt_dict(self, data):
        for k in ['status']:
            if k in data:
                data[k] = data[k].name

    def parse_dict(self, data):
        for k in ['status']:
            if k in data:
                data[k] = TaskStatus[data[k]]

    async def create_task(self, worker_keys, workers, params, inputs, outputs):
        task_type, model_cls, stage = worker_keys
        cur_t = current_time()
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "task_type": task_type,
            "model_cls": model_cls,
            "stage": stage,
            "params": params,
            "create_t": cur_t,
            "update_t": cur_t,
            "status": TaskStatus.CREATED, 
            "extra_info": "",
            "tag": "",
            "inputs": {x: data_name(x, task_id) for x in inputs},
            "outputs": {x: data_name(x, task_id) for x in outputs},
        }
        subtasks = []
        for worker_name, worker_item in workers.items():
            subtasks.append({
                "task_id": task_id,
                "worker_name": worker_name,
                "inputs": {x: data_name(x, task_id) for x in worker_item['inputs']},
                "outputs": {x: data_name(x, task_id) for x in worker_item['outputs']},
                "queue": worker_item['queue'],
                "previous": worker_item['previous'], 
                "status": TaskStatus.CREATED,
                "worker_identity": "",
                "result": "",
                "fail_time": 0,
                "extra_info": "",
                "create_t": cur_t,
                "update_t": cur_t,
            })
        assert await self.insert_task(task, subtasks), f"create task {task_id} failed"
        return task_id


# Import task manager implementations
from .local_task_manager import LocalTaskManager
from .sql_task_manager import PostgresSQLTaskManager

__all__ = ['BaseTaskManager', 'LocalTaskManager', 'PostgresSQLTaskManager']