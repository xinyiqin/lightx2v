class BaseQueueManager:
    def __init__(self):
        pass

    async def put_subtask(self, subtask):
        raise NotImplementedError

    async def get_subtasks(self, queue, max_batch, timeout):
        raise NotImplementedError
