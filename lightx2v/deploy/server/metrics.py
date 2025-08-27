from loguru import logger
from lightx2v.deploy.task_manager import TaskStatus, ActiveStatus, FinishedStatus
from prometheus_client import Counter, Gauge, Summary


class MetricMonitor:
    def __init__(self):
        self.task_all = Counter(
            'task_all_total',
            'Total count of all tasks',
            ['task_type', 'model_cls', 'stage']
        )
        self.task_end = Counter(
            'task_end_total',
            'Total count of ended tasks',
            ['task_type', 'model_cls', 'stage', 'status']
        )
        self.task_active = Gauge(
            'task_active_size',
            'Current count of active tasks',
            ['task_type', 'model_cls', 'stage']
        )
        self.task_elapse = Summary(
            'task_elapse_seconds',
            'Elapse time of tasks',
            ['task_type', 'model_cls', 'stage', 'end_status']
        )
        self.subtask_all = Counter(
            'subtask_all_total',
            'Total count of all subtasks',
            ['queue']
        )
        self.subtask_end = Counter(
            'subtask_end_total',
            'Total count of ended subtasks',
            ['queue', 'status']
        )
        self.subtask_active = Gauge(
            'subtask_active_size',
            'Current count of active subtasks',
            ['queue', 'status']
        )
        self.subtask_elapse = Summary(
            'subtask_elapse_seconds',
            'Elapse time of subtasks',
            ['queue', 'elapse_key']
        )

    def record_task_start(self, task):
        self.task_all.labels(task['task_type'], task['model_cls'], task['stage']).inc()
        self.task_active.labels(task['task_type'], task['model_cls'], task['stage']).inc()

    def record_task_end(self, task, status, elapse):
        self.task_end.labels(task['task_type'], task['model_cls'], task['stage'], status.name).inc()
        self.task_active.labels(task['task_type'], task['model_cls'], task['stage']).dec()
        self.task_elapse.labels(task['task_type'], task['model_cls'], task['stage'], status.name).observe(elapse)

    def record_subtask(self, subtask, old_status, new_status, elapse_key, elapse):
        if old_status in ActiveStatus and new_status in FinishedStatus:
            self.subtask_end.labels(subtask['queue'], elapse_key).inc()
            self.subtask_active.labels(subtask['queue'], old_status.name).dec()
        if old_status not in ActiveStatus and new_status in ActiveStatus:
            self.subtask_active.labels(subtask['queue'], new_status.name).inc()
            if new_status == TaskStatus.CREATED:
                self.subtask_all.labels(subtask['queue']).inc()
        if elapse and elapse_key:
            self.subtask_elapse.labels(subtask['queue'], elapse_key).observe(elapse)