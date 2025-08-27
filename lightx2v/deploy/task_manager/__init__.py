from re import T
import uuid
from enum import Enum
from loguru import logger
from lightx2v.deploy.common.utils import current_time, data_name


class TaskStatus(Enum):
    CREATED = 1
    PENDING = 2
    RUNNING = 3
    SUCCEED = 4
    FAILED = 5
    CANCEL = 6

ActiveStatus = [TaskStatus.CREATED, TaskStatus.PENDING, TaskStatus.RUNNING]
FinishedStatus = [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCEL]


class BaseTaskManager:
    def __init__(self):
        pass

    async def init(self):
        pass

    async def close(self):
        pass

    async def insert_user_if_not_exists(self, user_info):
        raise NotImplementedError

    async def query_user(self, user_id):
        raise NotImplementedError

    async def insert_task(self, task, subtasks):
        raise NotImplementedError

    async def list_tasks(self, **kwargs):
        raise NotImplementedError

    async def query_task(self,task_id, user_id=None):
        raise NotImplementedError

    async def next_subtasks(self, task_id):
        raise NotImplementedError

    async def run_subtasks(self, subtasks, worker_identity):
        raise NotImplementedError

    async def ping_subtask(self, task_id, worker_name, worker_identity):
        raise NotImplementedError

    async def finish_subtasks(self, task_id, status, worker_identity=None, worker_name=None):
        raise NotImplementedError

    async def cancel_task(self, task_id, user_id=None):
        raise NotImplementedError

    async def resume_task(self, task_id, all_subtask=False, user_id=None):
        raise NotImplementedError

    def fmt_dict(self, data):
        for k in ['status']:
            if k in data:
                data[k] = data[k].name

    def parse_dict(self, data):
        for k in ['status']:
            if k in data:
                data[k] = TaskStatus[data[k]]

    async def create_user(self, user_info):
        assert user_info['source'] == 'github', f"do not support {user_info['source']} user!"
        cur_t = current_time()
        user_id = f"{user_info['source']}_{user_info['id']}"
        data = {
            'user_id': user_id,
            'source': user_info['source'],
            'id': user_info['id'],
            'username': user_info['username'],
            'email': user_info['email'],
            'homepage': user_info['homepage'],
            'avatar_url': user_info['avatar_url'],
            'create_t': cur_t,
            'update_t': cur_t,
            'extra_info': '',
            'tag': '',
        }
        assert await self.insert_user_if_not_exists(data), f"create user {data} failed"
        return user_id

    async def create_task(self, worker_keys, workers, params, inputs, outputs, user_id):
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
            "extra_info": {"active_start_t": cur_t},
            "tag": "",
            "inputs": {x: data_name(x, task_id) for x in inputs},
            "outputs": {x: data_name(x, task_id) for x in outputs},
            "user_id": user_id,
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
                "extra_info": {"CREATED_start_t": cur_t},
                "create_t": cur_t,
                "update_t": cur_t,
                "ping_t": 0.0,
                "infer_cost": -1.0,
            })
        assert await self.insert_task(task, subtasks), f"create task {task_id} failed"
        return task_id

    def mark_task_start(self, task):
        t = current_time()
        if not isinstance(task['extra_info'], dict):
            task['extra_info'] = {}
        task['extra_info']['active_start_t'] = t
        return task['extra_info']

    def mark_task_end(self, task):
        start_t = task['extra_info']['active_start_t']
        end_t = current_time()
        task['extra_info']['active_end_t'] = end_t
        task['extra_info']['active_elapse'] = end_t - start_t
        return task['extra_info']

    def mark_subtask(self, subtask, old_status, new_status):
        t = current_time()
        if not isinstance(subtask['extra_info'], dict):
            subtask['extra_info'] = {}
        if old_status == new_status:
            logger.warning(f"Subtask {subtask} update same status: {old_status} vs {new_status}")
            return subtask['extra_info']

        if old_status in ActiveStatus:
            if 'start_t' not in subtask['extra_info']:
                logger.warning(f"Subtask {subtask} has no start time, status: {old_status}")
            else:
                elapse = t - subtask['extra_info']['start_t']
                elapse_key = f"{old_status.name}-{new_status.name}"
                subtask['extra_info'][elapse_key] = elapse
                subtask['extra_info']['elapse_key'] = elapse_key
                del subtask['extra_info']['start_t']

        if new_status in ActiveStatus:
            subtask['extra_info']['start_t'] = t
        return subtask['extra_info']


# Import task manager implementations
from .local_task_manager import LocalTaskManager
from .sql_task_manager import PostgresSQLTaskManager

__all__ = ['BaseTaskManager', 'LocalTaskManager', 'PostgresSQLTaskManager']