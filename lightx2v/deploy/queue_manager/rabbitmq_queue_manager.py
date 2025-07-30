import json
import asyncio
import time
import aio_pika
import traceback
from loguru import logger

from lightx2v.deploy.queue_manager import BaseQueueManager
from lightx2v.deploy.common.utils import class_try_catch_async


class RabbitMQQueueManager(BaseQueueManager):
    def __init__(self, conn_url, max_retries=3):
        self.conn_url = conn_url
        self.max_retries = max_retries
        self.conn = None
        self.chan = None
        self.queues = set()

    async def init(self):
        await self.get_conn()

    async def close(self):
        await self.del_conn()

    async def get_conn(self):
        if self.chan and self.conn:
            return
        for i in range(self.max_retries):
            try:
                logger.info(f"Connect to RabbitMQ (attempt {i + 1}/{self.max_retries}..)")
                self.conn = await aio_pika.connect_robust(self.conn_url)
                self.chan = await self.conn.channel()
                self.queues = set()
                await self.chan.set_qos(prefetch_count=10)
                logger.info("Successfully connected to RabbitMQ")
                return
            except Exception as e:
                logger.warning(f"Failed to connect to RabbitMQ: {e}")
                if i < self.max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    raise

    async def declare_queue(self, queue):
        if queue not in self.queues:
            await self.get_conn()
            await self.chan.declare_queue(queue, durable=True)
            self.queues.add(queue)
        return await self.chan.get_queue(queue)

    @class_try_catch_async
    async def put_subtask(self, subtask):
        queue = subtask['queue']
        await self.declare_queue(queue)
        keys = ['queue', 'task_id', 'worker_name', 'inputs', 'outputs', 'params']
        msg = json.dumps({k: subtask[k] for k in keys}).encode('utf-8')
        message = aio_pika.Message(
            body=msg,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            content_type='application/json'
        )
        await self.chan.default_exchange.publish(message, routing_key=queue)
        logger.info(f"Published subtask {subtask} to queue {queue}")
        return True

    async def get_subtasks(self, queue, max_batch, timeout):
        try:
            q = await self.declare_queue(queue)
            subtasks = []
            t0 = time.time()

            while True:
                cur_timeout = max(timeout - (time.time() - t0), 0.1)
                message = await q.get(no_ack=False, timeout=cur_timeout, fail=False)
                if message:
                    await message.ack()
                    subtask = json.loads(message.body.decode('utf-8'))
                    subtasks.append(subtask)
                    if len(subtasks) >= max_batch:
                        return subtasks
                    else:
                        continue
                else:
                    if len(subtasks) > 0:
                        return subtasks
                    if time.time() - t0 > timeout:
                        return None
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.warning("rabbitmq get_subtasks cancelled")
            return None
        except:
            logger.warning(f"rabbitmq get_subtasks failed: {traceback.format_exc()}")
            return None

    async def del_conn(self):
        if self.chan:
            await self.chan.close()
        if self.conn:
            await self.conn.close()


async def test():
    conn_url = "amqp://mtc:Sensetime666@127.0.0.1:5672"
    q = RabbitMQQueueManager(conn_url)
    await q.init()
    subtask = {
        "task_id": "test-subtask-id",
        "queue": "test_queue",
        "worker_name": "test_worker",
        "inputs": {},
        "outputs": {},
        "params": {},
    }
    await q.put_subtask(subtask)
    await asyncio.sleep(5)
    for i in range(2):
        subtask = await q.get_subtasks("test_queue", 3, 5)
        print("get subtask:", subtask)
    await q.close()

if __name__ == "__main__":
    asyncio.run(test())