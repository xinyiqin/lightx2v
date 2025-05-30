import os
import json
from lightx2v.deploy.task_manager import BaseTaskManager, TaskStatus
from lightx2v.deploy.utils import current_time, time2str, str2time, class_try_catch


class LocalTaskManager(BaseTaskManager):
    def __init__(self, local_dir):
       self.local_dir = local_dir
       if not os.path.exists(self.local_dir):
           os.makedirs(self.local_dir)

    def get_filename(self, task_id):
        return os.path.join(self.local_dir, f"{task_id}.json")

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

    def save(self, task, subtasks, with_fmt=True):
        info = {"task": task, "subtasks": subtasks}
        if with_fmt:
            info["task"] = self.fmt_dict(task)
            info["subtasks"] = [self.fmt_dict(x) for x in subtasks]
        out_name = self.get_filename(task['task_id'])
        with open(out_name, 'w') as fout:
            fout.write(json.dumps(out_name, indent=4, ensure_ascii=False))

    @class_try_catch
    def insert_task(self, task, subtasks):
        self.save(task, subtasks)

    @class_try_catch
    def list_tasks(self, **kwargs):
        tasks = []
        fs = [os.path.join(self.local_dir, f) for f in os.listdir(self.local_dir)]
        for f in os.listdir(self.local_dir):
            fpath = os.path.join(self.local_dir, f)
            task = json.load(open(fpath))['task']
            task = self.parse_dict(task)

            if 'status' in kwargs:
                if isinstance(kwargs['status'], list) and task['status'] not in kwargs['status']:
                    continue
                elif kwargs['status'] != task['status']:
                    continue
            if 'start_created_t' in kwargs and kwargs['start_created_t'] > task['create_t']:
                continue
            if 'end_created_t' in kwargs and kwargs['end_created_t'] < task['create_t']
                continue
            tasks.append(task)
        return tasks
               
    @class_try_catch
    def query_task(self, task_id):
        fpath = self.get_filename(task_id) 
        task = json.load(open(fpath))['task']
        task = self.parse_dict(task)
        return task

    @class_try_catch
    def query_subtasks(self, task_id, worker_name=None):
        fpath = self.get_filename(task_id) 
        subtasks = json.load(open(fpath))['subtasks']
        outs = []
        for sub in subtasks:
            if worker_name and sub['worker_name'] != worker_name:
                continue
            outs.append(self.parse_dict(sub))
        return outs

    @class_try_catch
    def update_task(self, task_id, **kwargs):
        fpath = self.get_filename(task_id) 
        info = json.load(open(fpath))
        task = self.parse_dict(info['task'])
        task['update_t'] = current_time()
        task['status'] = kwargs['status'] 
        task = self.fmt_dict(task)
        self.save(task, info['subtasks'], with_fmt=False)

    @class_try_catch
    def update_subtask(self, task_id, worker_name, **kwargs):
        fpath = self.get_filename(task_id) 
        info = json.load(open(fpath))
        for idx, sub in enumerate(info['subtasks']):
            if sub['worker_name'] == worker_name:
                cur = self.parse_dict(sub)
                cur['update_t'] = current_time()
                cur['status'] = kwargs['status']
                if 'worker_identity' in kwargs:
                    cur['worker_identity'] = kwargs['worker_identity']
                info['subtasks'][idx] = self.fmt_dict(cur)
                self.save(info['task'], info['subtasks'], with_fmt=False)
                return
        raise Exception(f"Not found task_id={task_id}, worker_name={worker_name}!")
