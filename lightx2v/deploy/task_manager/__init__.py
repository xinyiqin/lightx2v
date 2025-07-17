import uuid
from enum import Enum
from lightx2v.deploy.common.utils import current_time, data_name, time2str, str2time


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

    async def insert_task(self, task, subtasks):
        raise NotImplementedError

    async def list_tasks(self, **kwargs):
        raise NotImplementedError

    async def query_task(self, task_id, fmt=False):
        raise NotImplementedError

    async def query_subtasks(self, task_id, worker_name=None):
        raise NotImplementedError

    async def update_task(self, task_id, **kwargs):
        raise NotImplementedError

    async def update_subtask(self, task_id, worker_name, **kwargs):
        raise NotImplementedError

    def fmt_dict(self, data):
        for k in ['status']:
            if k in data:
                data[k] = data[k].name
        for k in ['create_t', 'update_t']:
            if k in data:
                data[k] = time2str(data[k])

    def parse_dict(self, data):
        for k in ['status']:
            if k in data:
                data[k] = TaskStatus[data[k]]
        for k in ['create_t', 'update_t']:
            if k in data:
                data[k] = str2time(data[k])

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
        await self.insert_task(task, subtasks)
        return task_id

    async def pend_subtask(self, task_id, worker_name):
        await self.update_subtask(task_id, worker_name, status=TaskStatus.PENDING)
        await self.update_task(task_id, status=TaskStatus.PENDING)

    async def run_subtask(self, task_id, worker_name, worker_identity):
        await self.update_subtask(
            task_id,
            worker_name,
            worker_identity=worker_identity,
            status=TaskStatus.RUNNING,
        )
        await self.update_task(task_id, status=TaskStatus.RUNNING)

    async def finish_subtask(self, task_id, worker_name, status):
        await self.update_subtask(task_id, worker_name, status=status)
        subtasks = await self.query_subtasks(task_id)
        running_subs = []
        has_failed_sub = False
        for sub in subtasks:
            if sub['status'] not in [TaskStatus.SUCCEED, TaskStatus.FAILED]:
                running_subs.append(sub)
            if sub['status'] == TaskStatus.FAILED:
                has_failed_sub = True
        # some subtask failed, we should fail all other subtasks
        if has_failed_sub:
            await self.update_task(task_id, status=TaskStatus.FAILED)
            for sub in running_subs:
                await self.update_subtask(task_id, sub['worker_name'], status=TaskStatus.FAILED)
            return TaskStatus.FAILED
        # all subtasks finished and all succeed
        elif len(running_subs) == 0:
            await self.update_task(task_id, status=TaskStatus.SUCCEED)
            return TaskStatus.SUCCEED
        return None

    async def next_subtasks(self, task_id):
        task = await self.query_task(task_id)
        if task['status'] not in [TaskStatus.CREATED, TaskStatus.RUNNING, TaskStatus.PENDING]:
            return []
        subtasks = await self.query_subtasks(task_id)
        succeeds = set()
        for sub in subtasks:
            if sub['status'] == TaskStatus.SUCCEED:
                succeeds.add(sub['worker_name'])
        nexts = []
        for sub in subtasks:
            if sub['status'] == TaskStatus.CREATED:
                dep_ok = True
                for prev in sub['previous']:
                    if prev not in succeeds:
                        dep_ok = False
                        break
                if dep_ok:
                    sub['params'] = task['params']
                    nexts.append(sub)
        return nexts

    async def cancel_task(self, task_id):
        await self.update_task(task_id, status=TaskStatus.CANCEL)

    async def revoke_task(self, task_id, all_subtask=False):
        task = await self.query_task(task_id)
        # the task is not finished
        if task['status'] not in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCEL]:
            return False
        # the task is no need to revoke
        if not all_subtask and task['status'] == TaskStatus.SUCCEED:
            return False
        subtasks = await self.query_subtasks(task_id)
        for sub in subtasks:
            if all_subtask or sub['status'] == TaskStatus.FAILED:
                await self.update_subtask(task_id, sub['worker_name'], status=TaskStatus.CREATED)
        await self.update_task(task_id, status=TaskStatus.CREATED)
        return True

    async def check_identity(self, task_id, worker_name, worker_identity, status):
        subtasks = await self.query_subtasks(task_id, worker_name)
        assert len(subtasks) >= 1, f"no worker task_id={task_id} name={worker_name}"
        pre = subtasks[0]['worker_identity']
        assert pre == worker_identity, f"identity not matched: {pre} vs {worker_identity}"
