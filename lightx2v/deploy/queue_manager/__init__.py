class BaseQueueManager:
    def __init__(self):
        pass

    async def put_subtask(self, subtask):
        raise NotImplementedError

    async def get_subtask(self, queue, timeout):
        raise NotImplementedError
