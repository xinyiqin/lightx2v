import uuid
from enum import Enum
from lightx2v.deploy.utils import current_time 


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

    def insert_task(self, task, subtasks):
        raise NotImplementedError

    def list_tasks(self, **kwargs):
        raise NotImplementedError

    def query_task(self, task_id):
        raise NotImplementedError

    def query_subtasks(self, task_id, worker_name=None):
        raise NotImplementedError

    def update_task(self, task_id, **kwargs):
        raise NotImplementedError

    def update_subtask(self, task_id, worker_name, **kwargs):
        raise NotImplementedError

    def create_task(self, task_type, model_cls, stage, workers, params):
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
        }
        subtasks = []
        for worker_name, worker_item in workers.items():
            subtasks.append({
                "task_id": task_id,
                "worker_name": worker_name,
                "inputs": [task_id + '-' + x for x in worker_item['inputs']],
                "outputs": [task_id + '-' + x for x in worker_item['outputs']],
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
        self.insert_task(task, subtasks)

    def pend_subtask(self, task_id, worker_name):
        self.update_subtask(task_id, worker_name, status=TaskStatus.PENDING)
        self.update_task(task_id, status=TaskStatus.RUNNING)

    def run_subtask(self, task_id, worker_name, worker_identity):
        self.update_subtask(
            task_id,
            worker_name,
            worker_identity=worker_identity,
            status=TaskStatus.RUNNING,
        )

    def finish_subtask(self, task_id, worker_name, status):
        self.update_subtask(task_id, worker_name, status=status)
        subtasks = self.query_subtasks(task_id)
        all_finished = True
        all_succeed = True
        for sub in subtasks:
            if sub['task'] not in [TaskStatus.SUCCEED, TaskStatus.FAILED]:
                all_finished = False
                break
            if sub['task'] == TaskStatus.FAILED:
                all_succeed = False
        if all_finished and all_succeed:
            self.update_task(task_id, status=TaskStatus.SUCCEED)
            return TaskStatus.SUCCEED
        if not all_succeed:
            self.update_task(task_id, status=TaskStatus.FAILED)
            return TaskStatus.FAILED
        return None

    def next_subtasks(self, task_id):
        task = self.query_task(task_id)
        if task['status'] != TaskStatus.CREATED:
            return []
        subtasks = self.query_subtasks(task_id)
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
                    nexts.append(sub)
        return nexts

    def cancel_task(self, task_id):
        self.update_task(task_id, status=TaskStatus.CANCEL)

    def revoke_task(self, task_id, all_subtask=False):
        task = self.query_task(task_id)
        # the task is not finished
        if task['status'] not in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCEL]:
            return False
        # the task is no need to revoke
        if not all_subtask and task['status'] == TaskStatus.SUCCEED:
            return False
        subtasks = self.query_subtasks(task_id)
        for sub in subtasks:
            if all_subtask or sub['status'] == TaskStatus.FAILED:
                self.update_subtask(task_id, sub['worker_name'], status=TaskStatus.CREATED)
        self.update_task(task_id, status=TaskStatus.CREATED) 
        return True
