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

    def get_filename(self, task_id):
        return os.path.join(self.local_dir, f"{task_id}.json")

    def fmt_dict(self, data):
        super().fmt_dict(data)
        for k in ['create_t', 'update_t']:
            if k in data:
                data[k] = time2str(data[k])

    def parse_dict(self, data):
        super().parse_dict(data)
        for k in ['create_t', 'update_t']:
            if k in data:
                data[k] = str2time(data[k])

    def save(self, task, subtasks, with_fmt=True):
        info = {"task": task, "subtasks": subtasks}
        if with_fmt:
            self.fmt_dict(info['task'])
            [self.fmt_dict(x) for x in info['subtasks']]
        out_name = self.get_filename(task['task_id'])
        with open(out_name, 'w') as fout:
            fout.write(json.dumps(info, indent=4, ensure_ascii=False))

    def load(self, task_id):
        fpath = self.get_filename(task_id)
        info = json.load(open(fpath))
        task, subtasks = info['task'], info['subtasks']
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
            fpath = os.path.join(self.local_dir, f)
            task = json.load(open(fpath))['task']
            self.parse_dict(task)

            if 'status' in kwargs:
                if isinstance(kwargs['status'], list) and task['status'] not in kwargs['status']:
                    continue
                elif kwargs['status'] != task['status']:
                    continue
            if 'start_created_t' in kwargs and kwargs['start_created_t'] > task['create_t']:
                continue
            if 'end_created_t' in kwargs and kwargs['end_created_t'] < task['create_t']:
                continue
            tasks.append(task)
        return tasks

    @class_try_catch_async
    async def query_task(self, task_id):
        return self.load(task_id)[0]

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
    async def run_subtasks(self, task_ids, worker_names, worker_identity):
        valids = []
        for task_id, worker_name in zip(task_ids, worker_names):
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
                    self.save(task, subtasks)
                    valids.append((task_id, worker_name))
                    break
        return valids

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
            sub['status'] = status
            sub['update_t'] = current_time()
 
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
    async def cancel_task(self, task_id):
        task, subtasks = self.load(task_id)
        if task['status'] not in [TaskStatus.CREATED, TaskStatus.PENDING, TaskStatus.RUNNING]:
            return False
        task['status'] = TaskStatus.CANCEL
        task['update_t'] = current_time()
        self.save(task, subtasks)
        return True

    @class_try_catch_async
    async def resume_task(self, task_id, all_subtask=False):
        task, subtasks = self.load(task_id)
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

    task_id = await m.create_task(keys, workers, params, inputs, outputs)
    print(" - create_task:", task_id)

    tasks = await m.list_tasks()
    print(" - list_tasks:", tasks)

    task = await m.query_task(task_id)
    print(" - query_task:", task)

    subtasks = await m.next_subtasks(task_id)
    print(" - next_subtasks:", subtasks)

    task_ids = [sub['task_id'] for sub in subtasks]
    worker_names = [sub['worker_name'] for sub in subtasks]
    await m.run_subtasks(task_ids, worker_names, 'fake-worker')
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
