import os
import json
import asyncio
from lightx2v.deploy.task_manager import BaseTaskManager, TaskStatus
from lightx2v.deploy.common.utils import current_time, class_try_catch_async, time2str, str2time


class LocalTaskManager(BaseTaskManager):
    def __init__(self, local_dir):
       self.local_dir = local_dir
       if not os.path.exists(self.local_dir):
           os.makedirs(self.local_dir)

    def get_task_filename(self, task_id):
        return os.path.join(self.local_dir, f"task_{task_id}.json")

    def get_user_filename(self, user_id):
        return os.path.join(self.local_dir, f"user_{user_id}.json")

    def fmt_dict(self, data):
        super().fmt_dict(data)
        for k in ['create_t', 'update_t', 'ping_t']:
            if k in data:
                data[k] = time2str(data[k])

    def parse_dict(self, data):
        super().parse_dict(data)
        for k in ['create_t', 'update_t', 'ping_t']:
            if k in data:
                data[k] = str2time(data[k])

    def save(self, task, subtasks, with_fmt=True):
        info = {"task": task, "subtasks": subtasks}
        if with_fmt:
            self.fmt_dict(info['task'])
            [self.fmt_dict(x) for x in info['subtasks']]
        out_name = self.get_task_filename(task['task_id'])
        with open(out_name, 'w') as fout:
            fout.write(json.dumps(info, indent=4, ensure_ascii=False))

    def load(self, task_id, user_id=None):
        fpath = self.get_task_filename(task_id)
        info = json.load(open(fpath))
        task, subtasks = info['task'], info['subtasks']
        if user_id is not None and task['user_id'] != user_id:
            raise Exception(f"Task {task_id} is not belong to user {user_id}")
        self.parse_dict(task)
        for sub in subtasks:
            self.parse_dict(sub)
        return task, subtasks

    @class_try_catch_async
    async def insert_task(self, task, subtasks):
        self.save(task, subtasks)
        return True

    @class_try_catch_async
    async def list_tasks(self, **kwargs):
        tasks = []
        fs = [os.path.join(self.local_dir, f) for f in os.listdir(self.local_dir)]
        for f in os.listdir(self.local_dir):
            if not f.startswith('task_'):
                continue
            fpath = os.path.join(self.local_dir, f)
            info = json.load(open(fpath))
            if kwargs.get('subtasks', False):
                items = info['subtasks']
                assert 'user_id' not in kwargs, "user_id is not allowed when subtasks is True"
            else:
                items = [info['task']]
            for task in items:
                self.parse_dict(task)
                if 'user_id' in kwargs and task['user_id'] != kwargs['user_id']:
                    continue
                if 'status' in kwargs:
                    if isinstance(kwargs['status'], list) and task['status'] not in kwargs['status']:
                        continue
                    elif kwargs['status'] != task['status']:
                        continue
                if 'start_created_t' in kwargs and kwargs['start_created_t'] > task['create_t']:
                    continue
                if 'end_created_t' in kwargs and kwargs['end_created_t'] < task['create_t']:
                    continue
                if 'start_updated_t' in kwargs and kwargs['start_updated_t'] > task['update_t']:
                    continue
                if 'end_updated_t' in kwargs and kwargs['end_updated_t'] < task['update_t']:
                    continue
                if 'start_ping_t' in kwargs and kwargs['start_ping_t'] > task['ping_t']:
                    continue
                if 'end_ping_t' in kwargs and kwargs['end_ping_t'] < task['ping_t']:
                    continue
                tasks.append(task)
        if 'count' in kwargs:
            return len(tasks)
        tasks = sorted(tasks, key=lambda x: x['create_t'], reverse=True)
        if 'offset' in kwargs:
            tasks = tasks[kwargs['offset']:]
        if 'limit' in kwargs:
            tasks = tasks[:kwargs['limit']]
        return tasks

    @class_try_catch_async
    async def query_task(self, task_id, user_id=None):
        task, subtasks = self.load(task_id, user_id)
        return task

    @class_try_catch_async
    async def next_subtasks(self, task_id):
        task, subtasks = self.load(task_id)
        if task['status'] not in [TaskStatus.CREATED, TaskStatus.RUNNING, TaskStatus.PENDING]:
            return []
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
                    sub['status'] = TaskStatus.PENDING
                    sub['update_t'] = current_time()
                    nexts.append(sub)
        if len(nexts) > 0:
            task['status'] = TaskStatus.PENDING
            task['update_t'] = current_time()
            self.save(task, subtasks)
        return nexts

    @class_try_catch_async
    async def run_subtasks(self, cands, worker_identity):
        valids = []
        for cand in cands:
            task_id = cand['task_id']
            worker_name = cand['worker_name']
            task, subtasks = self.load(task_id)
            if task['status'] in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCEL]:
                continue
            for sub in subtasks:
                if sub['worker_name'] == worker_name:
                    sub['status'] = TaskStatus.RUNNING
                    sub['worker_identity'] = worker_identity
                    sub['update_t'] = current_time()
                    task['status'] = TaskStatus.RUNNING
                    task['update_t'] = current_time()
                    task['ping_t'] = current_time()
                    self.save(task, subtasks)
                    valids.append(cand)
                    break
        return valids

    @class_try_catch_async
    async def ping_subtask(self, task_id, worker_name, worker_identity):
        task, subtasks = self.load(task_id)
        for sub in subtasks:
            if sub['worker_name'] == worker_name:
                pre = sub['worker_identity']
                assert pre == worker_identity, f"worker identity not matched: {pre} vs {worker_identity}"
                sub['ping_t'] = current_time()
                self.save(task, subtasks)
                return True
        return False

    @class_try_catch_async
    async def finish_subtasks(self, task_id, status, worker_identity=None, worker_name=None):
        task, subtasks = self.load(task_id)
        subs = subtasks

        if worker_name:
            subs = [sub for sub in subtasks if sub['worker_name'] == worker_name]
        assert len(subs) >= 1, f"no worker task_id={task_id}, name={worker_name}"

        if worker_identity:
            pre = subs[0]['worker_identity']
            assert pre == worker_identity, f"worker identity not matched: {pre} vs {worker_identity}"

        assert status in [TaskStatus.SUCCEED, TaskStatus.FAILED], f"invalid finish status: {status}"
        for sub in subs:
            pre_t = sub['update_t']
            sub['status'] = status
            sub['update_t'] = current_time()
            if status == TaskStatus.SUCCEED:
                sub['infer_cost'] = sub['update_t'] - pre_t

        if task['status'] == TaskStatus.CANCEL:
            return TaskStatus.CANCEL

        running_subs = []
        failed_sub = False
        for sub in subtasks:
            if sub['status'] not in [TaskStatus.SUCCEED, TaskStatus.FAILED]:
                running_subs.append(sub)
            if sub['status'] == TaskStatus.FAILED:
                failed_sub = True

        # some subtask failed, we should fail all other subtasks
        if failed_sub:
            task['status'] = TaskStatus.FAILED
            task['update_t'] = current_time()
            for sub in running_subs:
                sub['status'] = TaskStatus.FAILED
                sub['update_t'] = current_time()
            self.save(task, subtasks)
            return TaskStatus.FAILED

        # all subtasks finished and all succeed
        elif len(running_subs) == 0:
            task['status'] = TaskStatus.SUCCEED
            task['update_t'] = current_time()
            self.save(task, subtasks)
            return TaskStatus.SUCCEED

        self.save(task, subtasks)
        return None

    @class_try_catch_async
    async def cancel_task(self, task_id, user_id=None):
        task, subtasks = self.load(task_id, user_id)
        if task['status'] not in [TaskStatus.CREATED, TaskStatus.PENDING, TaskStatus.RUNNING]:
            return False
        task['status'] = TaskStatus.CANCEL
        task['update_t'] = current_time()
        self.save(task, subtasks)
        return True

    @class_try_catch_async
    async def resume_task(self, task_id, all_subtask=False, user_id=None):
        task, subtasks = self.load(task_id, user_id)
        # the task is not finished
        if task['status'] not in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCEL]:
            return False
        # the task is no need to resume
        if not all_subtask and task['status'] == TaskStatus.SUCCEED:
            return False
        for sub in subtasks:
            if all_subtask or sub['status'] != TaskStatus.SUCCEED:
                sub['status'] = TaskStatus.CREATED
                sub['update_t'] = current_time()
        task['status'] = TaskStatus.CREATED
        task['update_t'] = current_time()
        self.save(task, subtasks)
        return True

    @class_try_catch_async
    async def insert_user_if_not_exists(self, user_info):
        fpath = self.get_user_filename(user_info['user_id'])
        if os.path.exists(fpath):
            return True
        self.fmt_dict(user_info)
        with open(fpath, 'w') as fout:
            fout.write(json.dumps(user_info, indent=4, ensure_ascii=False))
        return True

    @class_try_catch_async
    async def query_user(self, user_id):
        fpath = self.get_user_filename(user_id)
        if not os.path.exists(fpath):
            return None
        data = json.load(open(fpath))
        self.parse_dict(data)
        return data


async def test():
    from lightx2v.deploy.common.pipeline import Pipeline
    p = Pipeline("/data/nvme1/liuliang1/lightx2v/configs/model_pipeline.json")
    m = LocalTaskManager("/data/nvme1/liuliang1/lightx2v/local_task")
    await m.init()

    keys = ["t2v", "wan2.1", "multi_stage"]
    workers = p.get_workers(keys)
    inputs = p.get_inputs(keys)
    outputs = p.get_outputs(keys)
    params = {
        "prompt": "fake input prompts",
        "resolution": {
            "height": 233,
            "width": 456,
        },
    }

    user_info = {
        "source": "github",
        "id": "test-id-233",
        "username": "test-username-233",
        "email": "test-email-233@test.com",
        "homepage": "https://test.com",
        "avatar_url": "https://test.com/avatar.png",
    }
    user_id = await m.create_user(user_info)
    print(" - create_user:", user_id)

    user = await m.query_user(user_id)
    print(" - query_user:", user)

    task_id = await m.create_task(keys, workers, params, inputs, outputs, user_id)
    print(" - create_task:", task_id)

    tasks = await m.list_tasks()
    print(" - list_tasks:", tasks)

    task = await m.query_task(task_id)
    print(" - query_task:", task)

    subtasks = await m.next_subtasks(task_id)
    print(" - next_subtasks:", subtasks)

    await m.run_subtasks(subtasks, 'fake-worker')
    await m.finish_subtasks(task_id, TaskStatus.FAILED)
    await m.cancel_task(task_id)
    await m.resume_task(task_id)
    for sub in subtasks:
        await m.finish_subtasks(
            sub['task_id'], TaskStatus.SUCCEED, worker_name=sub['worker_name'], worker_identity='fake-worker'
        )

    subtasks = await m.next_subtasks(task_id)
    print(" - final next_subtasks:", subtasks)

    task = await m.query_task(task_id)
    print(" - final task:", task)


    await m.close()

if __name__ == "__main__":
    asyncio.run(test())
