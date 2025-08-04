import json
import asyncio
import asyncpg
import traceback
from loguru import logger
from datetime import datetime
from lightx2v.deploy.task_manager import BaseTaskManager, TaskStatus
from lightx2v.deploy.common.utils import class_try_catch_async


class PostgresSQLTaskManager(BaseTaskManager):

    def __init__(self, db_url):
        self.db_url = db_url
        self.table_tasks = "tasks"
        self.table_subtasks = "subtasks"
        self.table_versions = "versions"
        self.pool = None

    async def init(self):
        await self.upgrade_db()

    async def close(self):
        if self.pool:
            await self.pool.close()

    def fmt_dict(self, data):
        super().fmt_dict(data)
        for k in ['create_t', 'update_t']:
            if k in data and isinstance(data[k], float):
                data[k] = datetime.fromtimestamp(data[k])
        for k in ['params', 'extra_info', 'inputs', 'outputs', 'previous']:
            if k in data:
                data[k] = json.dumps(data[k], ensure_ascii=False)

    def parse_dict(self, data):
        super().parse_dict(data)
        for k in ['params', 'extra_info', 'inputs', 'outputs', 'previous']:
            if k in data:
                data[k] = json.loads(data[k])
        for k in ['create_t', 'update_t']:
            if k in data:
                data[k] = data[k].timestamp()

    async def get_conn(self):
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.db_url)
        return await self.pool.acquire()

    async def release_conn(self, conn):
        await self.pool.release(conn)

    async def query_version(self):
        conn = await self.get_conn()
        try:
            row = await conn.fetchrow(f"SELECT version FROM {self.table_versions} ORDER BY create_t DESC LIMIT 1")
            row = dict(row)
            return row['version'] if row else 0
        except:
            logger.error(f"query_version error: {traceback.format_exc()}")
            return 0
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def upgrade_db(self):
        versions = [
            (1, "Init tables", self.upgrade_v1),
            # (2, "Add new fields or indexes", self.upgrade_v2),
        ]
        logger.info(f"upgrade_db: {self.db_url}")
        cur_ver = await self.query_version()
        for ver, description, func in versions:
            if cur_ver < ver:
                logger.info(f"Upgrade to version {ver}: {description}")
                if not await func(ver, description):
                    logger.error(f"Upgrade to version {ver}: {description} func failed")
                    return False
                cur_ver = ver
        logger.info(f"upgrade_db: {self.db_url} done")
        return True

    async def upgrade_v1(self, version, description):
        conn = await self.get_conn()
        try:
            async with conn.transaction(isolation='read_uncommitted'):
                # create tasks table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_tasks} (
                        task_id VARCHAR(128) PRIMARY KEY,
                        task_type VARCHAR(64),
                        model_cls VARCHAR(64),
                        stage VARCHAR(64),
                        params JSONB,
                        create_t TIMESTAMPTZ,
                        update_t TIMESTAMPTZ,
                        status VARCHAR(64),
                        extra_info JSONB,
                        tag VARCHAR(64),
                        inputs JSONB,
                        outputs JSONB
                    )
                """)
                # create subtasks table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_subtasks} (
                        task_id VARCHAR(128),
                        worker_name VARCHAR(128),
                        inputs JSONB,
                        outputs JSONB,
                        queue VARCHAR(128),
                        previous JSONB,
                        status VARCHAR(64),
                        worker_identity VARCHAR(128),
                        result VARCHAR(128),
                        fail_time INTEGER,
                        extra_info JSONB,
                        create_t TIMESTAMPTZ,
                        update_t TIMESTAMPTZ,
                        PRIMARY KEY (task_id, worker_name),
                        FOREIGN KEY (task_id) REFERENCES {self.table_tasks}(task_id) ON DELETE CASCADE
                    )
                """)
                # create versions table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_versions} (
                        version INTEGER PRIMARY KEY,
                        description VARCHAR(255),
                        create_t TIMESTAMPTZ NOT NULL
                    )
                """)
                # create indexes
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_tasks}_status ON {self.table_tasks}(status)")
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_tasks}_create_t ON {self.table_tasks}(create_t)")
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_tasks}_tag ON {self.table_tasks}(tag)")
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_subtasks}_task_id ON {self.table_subtasks}(task_id)")
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_subtasks}_worker_name ON {self.table_subtasks}(worker_name)")
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_subtasks}_status ON {self.table_subtasks}(status)")

                # update version
                await conn.execute(
                    f"INSERT INTO {self.table_versions} (version, description, create_t) VALUES ($1, $2, $3)",
                    version, description, datetime.now()
                )
                return True
        except:
            logger.error(f"upgrade_v1 error: {traceback.format_exc()}")
            return False
        finally:
            await self.release_conn(conn)

    async def load(self, conn, task_id):
        row = await conn.fetchrow(f"SELECT * FROM {self.table_tasks} WHERE task_id = $1", task_id)
        task = dict(row)
        assert task, f"query_task: task not found: {task_id}"
        self.parse_dict(task)
        rows = await conn.fetch(f"SELECT * FROM {self.table_subtasks} WHERE task_id = $1", task_id)
        subtasks = []
        for row in rows:
            sub = dict(row)
            self.parse_dict(sub)
            subtasks.append(sub)
        return task, subtasks

    async def update_task(self, conn, task_id, **kwargs):
        query = f"UPDATE {self.table_tasks} SET "
        conds = ["update_t = $1"]
        params = [datetime.now()]
        param_idx = 1
        if 'status' in kwargs:
            param_idx += 1
            conds.append(f"status = ${param_idx}")
            params.append(kwargs['status'].name)
        query += " ,".join(conds)
        query += f" WHERE task_id = ${param_idx + 1}"
        params.append(task_id)
        await conn.execute(query, *params)

    async def update_subtask(self, conn, task_id, worker_name, **kwargs):
        query = f"UPDATE {self.table_subtasks} SET "
        conds = ["update_t = $1"]
        params = [datetime.now()]
        param_idx = 1
        if 'status' in kwargs:
            param_idx += 1
            conds.append(f"status = ${param_idx}")
            params.append(kwargs['status'].name)
        if 'worker_identity' in kwargs:
            param_idx += 1
            conds.append(f"worker_identity = ${param_idx}")
            params.append(kwargs['worker_identity'])
        query += " ,".join(conds)
        query += f" WHERE task_id = ${param_idx + 1} AND worker_name = ${param_idx + 2}"
        params.extend([task_id, worker_name])
        await conn.execute(query, *params)

    @class_try_catch_async
    async def insert_task(self, task, subtasks):
        conn = await self.get_conn()
        try:
            async with conn.transaction(isolation='read_uncommitted'):
                self.fmt_dict(task)
                await conn.execute(f"""
                    INSERT INTO {self.table_tasks} 
                    (task_id, task_type, model_cls, stage, params, create_t,
                        update_t, status, extra_info, tag, inputs, outputs)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """,
                    task['task_id'], task['task_type'], task['model_cls'],
                    task['stage'], task['params'], task['create_t'],
                    task['update_t'], task['status'], task['extra_info'],
                    task['tag'], task['inputs'], task['outputs']
                )
                for sub in subtasks:
                    self.fmt_dict(sub)
                    await conn.execute(f"""
                        INSERT INTO {self.table_subtasks}
                        (task_id, worker_name, inputs, outputs, queue, previous, status,
                            worker_identity, result, fail_time, extra_info, create_t, update_t)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                        """,
                        sub['task_id'], sub['worker_name'], sub['inputs'], sub['outputs'],
                        sub['queue'], sub['previous'], sub['status'], sub['worker_identity'],
                        sub['result'], sub['fail_time'],sub['extra_info'], sub['create_t'],
                        sub['update_t']
                    )
                return True
        except:
            logger.error(f"insert_task error: {traceback.format_exc()}")
            return False
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def list_tasks(self, **kwargs):
        conn = await self.get_conn()
        try:
            query = f"SELECT * FROM {self.table_tasks}"
            params = []
            conds = []
            param_idx = 0
            if kwargs.get('subtasks', False):
                query = f"SELECT * FROM {self.table_subtasks}"

            if 'status' in kwargs:
                param_idx += 1
                if isinstance(kwargs['status'], list):
                    next_idx = param_idx + len(kwargs['status'])
                    placeholders = ','.join([f'${i}' for i in range(param_idx, next_idx)])
                    conds.append(f"status IN ({placeholders})")
                    params.extend([x.name for x in kwargs['status']])
                    param_idx = next_idx
                else:
                    conds.append(f"status = ${param_idx}")
                    params.append(kwargs['status'].name)

            if 'start_created_t' in kwargs:
                param_idx += 1
                conds.append(f"create_t >= ${param_idx}")
                params.append(datetime.fromtimestamp(kwargs['start_created_t']))

            if 'end_created_t' in kwargs:
                param_idx += 1
                conds.append(f"create_t <= ${param_idx}")
                params.append(datetime.fromtimestamp(kwargs['end_created_t']))

            if conds:
                query += " WHERE " + " AND ".join(conds)
            query += " ORDER BY create_t ASC"

            rows = await conn.fetch(query, *params)
            tasks = []
            for row in rows:
                task = dict(row)
                self.parse_dict(task)
                tasks.append(task)
            return tasks
        except:
            logger.error(f"list_tasks error: {traceback.format_exc()}")
            return []
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def query_task(self, task_id):
        conn = await self.get_conn()
        try:
            task, _ = await self.load(conn, task_id)
            return task
        except:
            logger.error(f"query_task error: {traceback.format_exc()}")
            return None
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def next_subtasks(self, task_id):
        conn = await self.get_conn()
        try:
            async with conn.transaction(isolation='read_uncommitted'):
                task, subtasks = await self.load(conn, task_id)
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
                            await self.update_subtask(conn, task_id, sub['worker_name'], status=TaskStatus.PENDING)
                            nexts.append(sub)
                if len(nexts) > 0:
                    await self.update_task(conn, task_id, status=TaskStatus.PENDING)
                return nexts
        except:
            logger.error(f"next_subtasks error: {traceback.format_exc()}")
            return None
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def run_subtasks(self, task_ids, worker_names, worker_identity):
        conn = await self.get_conn()
        try:
            async with conn.transaction(isolation='read_uncommitted'):
                valids = []
                for task_id, worker_name in zip(task_ids, worker_names):
                    task, subtasks = await self.load(conn, task_id)
                    if task['status'] in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCEL]:
                        continue
                    for sub in subtasks:
                        if sub['worker_name'] == worker_name:
                            await self.update_subtask(conn, task_id, worker_name, status=TaskStatus.RUNNING, worker_identity=worker_identity)
                            await self.update_task(conn, task_id, status=TaskStatus.RUNNING)
                            valids.append((task_id, worker_name))
                            break
                return valids
        except:
            logger.error(f"run_subtasks error: {traceback.format_exc()}")
            return []
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def finish_subtasks(self, task_id, status, worker_identity=None, worker_name=None):
        conn = await self.get_conn()
        try:
            async with conn.transaction(isolation='read_uncommitted'):
                task, subtasks = await self.load(conn, task_id)
                subs = subtasks

                if worker_name:
                    subs = [sub for sub in subtasks if sub['worker_name'] == worker_name]
                assert len(subs) >= 1, f"no worker task_id={task_id}, name={worker_name}"

                if worker_identity:
                    pre = subs[0]['worker_identity']
                    assert pre == worker_identity, f"worker identity not matched: {pre} vs {worker_identity}"

                assert status in [TaskStatus.SUCCEED, TaskStatus.FAILED], f"invalid finish status: {status}"
                for sub in subs:
                    await self.update_subtask(conn, task_id, sub['worker_name'], status=status)
                    sub['status'] = status

                running_subs = []
                failed_sub = False
                for sub in subtasks:
                    if sub['status'] not in [TaskStatus.SUCCEED, TaskStatus.FAILED]:
                        running_subs.append(sub)
                    if sub['status'] == TaskStatus.FAILED:
                        failed_sub = True

                # some subtask failed, we should fail all other subtasks
                if failed_sub:
                    await self.update_task(conn, task_id, status=TaskStatus.FAILED)
                    for sub in running_subs:
                        await self.update_subtask(conn, task_id, sub['worker_name'], status=TaskStatus.FAILED)
                    return TaskStatus.FAILED

                # all subtasks finished and all succeed
                elif len(running_subs) == 0:
                    await self.update_task(conn, task_id, status=TaskStatus.SUCCEED)
                    return TaskStatus.SUCCEED
                return None
        except:
            logger.error(f"finish_subtasks error: {traceback.format_exc()}")
            return None
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def cancel_task(self, task_id):
        conn = await self.get_conn()
        try:
            async with conn.transaction(isolation='read_uncommitted'):
                task, subtasks = await self.load(conn, task_id)
                if task['status'] not in [TaskStatus.CREATED, TaskStatus.PENDING, TaskStatus.RUNNING]:
                    return False
                await self.update_task(conn, task_id, status=TaskStatus.CANCEL)
                return True
        except:
            logger.error(f"cancel_task error: {traceback.format_exc()}")
            return False
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def resume_task(self, task_id, all_subtask=False):
        conn = await self.get_conn()
        try:
            async with conn.transaction(isolation='read_uncommitted'):
                task, subtasks = await self.load(conn, task_id)
                # the task is not finished
                if task['status'] not in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCEL]:
                    return False
                # the task is no need to resume
                if not all_subtask and task['status'] == TaskStatus.SUCCEED:
                    return False
                for sub in subtasks:
                    if all_subtask or sub['status'] != TaskStatus.SUCCEED:
                        await self.update_subtask(conn, task_id, sub['worker_name'], status=TaskStatus.CREATED)
                await self.update_task(conn, task_id, status=TaskStatus.CREATED)
                return True
        except:
            logger.error(f"resume_task error: {traceback.format_exc()}")
            return False
        finally:
            await self.release_conn(conn)


async def test():
    from lightx2v.deploy.common.pipeline import Pipeline
    p = Pipeline("/data/nvme1/liuliang1/lightx2v/configs/model_pipeline.json")
    m = PostgresSQLTaskManager("postgresql://mtc:Sensetime666@127.0.0.1:5432/lightx2v_test")
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