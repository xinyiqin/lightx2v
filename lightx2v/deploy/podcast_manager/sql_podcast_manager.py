"""
播客管理器 - 使用数据库存储播客元数据，实现快速查询
参考任务管理器的设计，提供类似的高性能查询能力
"""

import json
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg
from loguru import logger

from lightx2v.deploy.common.utils import class_try_catch_async


class SQLPodcastManager:
    """播客管理器 - 使用PostgreSQL存储播客元数据"""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.table_podcasts = "podcasts"
        self.pool = None

    async def init(self):
        """初始化数据库连接和表结构"""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.db_url)
        await self.create_tables()

    async def close(self):
        """关闭数据库连接池"""
        if self.pool:
            await self.pool.close()

    async def get_conn(self):
        """获取数据库连接"""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.db_url)
        return await self.pool.acquire()

    async def release_conn(self, conn):
        """释放数据库连接"""
        await self.pool.release(conn)

    @class_try_catch_async
    async def create_tables(self):
        """创建播客表结构"""
        conn = await self.get_conn()
        try:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_podcasts} (
                    session_id VARCHAR(128) PRIMARY KEY,
                    user_id VARCHAR(256) NOT NULL,
                    user_input TEXT,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    has_audio BOOLEAN DEFAULT FALSE,
                    audio_path TEXT,
                    metadata_path TEXT,
                    rounds JSONB,
                    subtitles JSONB,
                    extra_info JSONB,
                    tag VARCHAR(64) DEFAULT ''
                )
            """)
            # 创建索引（PostgreSQL语法）
            try:
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_podcasts_user_id ON {self.table_podcasts}(user_id)")
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_podcasts_created_at ON {self.table_podcasts}(created_at)")
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_podcasts_user_created ON {self.table_podcasts}(user_id, created_at)")
            except Exception:
                pass  # 索引可能已存在
            logger.info(f"Podcast table {self.table_podcasts} created or already exists")
        except Exception as e:
            logger.error(f"Error creating podcast table: {e}")
            raise
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def insert_podcast(
        self, session_id: str, user_id: str, user_input: str, audio_path: str, metadata_path: str, rounds: List[Dict] = None, subtitles: List[Dict] = None, extra_info: Dict = None
    ):
        """插入播客记录"""
        conn = await self.get_conn()
        try:
            now = datetime.now()
            # 确保extra_info包含outputs字段
            if extra_info is None:
                extra_info = {}
            if "outputs" not in extra_info and audio_path:
                extra_info["outputs"] = {"merged_audio": audio_path}

            await conn.execute(
                f"""
                INSERT INTO {self.table_podcasts}
                (session_id, user_id, user_input, created_at, updated_at, has_audio,
                 audio_path, metadata_path, rounds, subtitles, extra_info, tag)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (session_id) DO UPDATE SET
                    updated_at = $5,
                    has_audio = $6,
                    audio_path = $7,
                    metadata_path = $8,
                    rounds = $9,
                    subtitles = $10,
                    extra_info = $11
                """,
                session_id,
                user_id,
                user_input,
                now,
                now,
                True,  # has_audio
                audio_path,
                metadata_path or "",  # metadata_path可能为空（不再保存到S3）
                json.dumps(rounds or [], ensure_ascii=False),
                json.dumps(subtitles or [], ensure_ascii=False),
                json.dumps(extra_info, ensure_ascii=False),
                "",
            )
            return True
        except Exception as e:
            logger.error(f"Error inserting podcast: {e}")
            return False
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def update_podcast(self, session_id: str, **kwargs):
        """更新播客记录"""
        conn = await self.get_conn()
        try:
            updates = []
            params = []
            param_idx = 1

            if "has_audio" in kwargs:
                updates.append(f"has_audio = ${param_idx}")
                params.append(kwargs["has_audio"])
                param_idx += 1

            if "rounds" in kwargs:
                updates.append(f"rounds = ${param_idx}")
                params.append(json.dumps(kwargs["rounds"], ensure_ascii=False))
                param_idx += 1

            if "subtitles" in kwargs:
                updates.append(f"subtitles = ${param_idx}")
                params.append(json.dumps(kwargs["subtitles"], ensure_ascii=False))
                param_idx += 1

            if "extra_info" in kwargs:
                updates.append(f"extra_info = ${param_idx}")
                params.append(json.dumps(kwargs["extra_info"], ensure_ascii=False))
                param_idx += 1

            if updates:
                updates.append(f"updated_at = ${param_idx}")
                params.append(datetime.now())
                param_idx += 1

                params.append(session_id)
                query = f"UPDATE {self.table_podcasts} SET {', '.join(updates)} WHERE session_id = ${param_idx}"
                await conn.execute(query, *params)
            return True
        except Exception as e:
            logger.error(f"Error updating podcast: {e}")
            return False
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def query_podcast(self, session_id: str, user_id: Optional[str] = None) -> Optional[Dict]:
        """查询单个播客记录"""
        conn = await self.get_conn()
        try:
            if user_id:
                row = await conn.fetchrow(f"SELECT * FROM {self.table_podcasts} WHERE session_id = $1 AND user_id = $2 AND tag != 'delete'", session_id, user_id)
            else:
                row = await conn.fetchrow(f"SELECT * FROM {self.table_podcasts} WHERE session_id = $1 AND tag != 'delete'", session_id)

            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Error querying podcast: {e}")
            return None
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def list_podcasts(self, user_id: str, page: int = 1, page_size: int = 10, status: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """列表查询播客记录（支持分页和过滤）"""
        conn = await self.get_conn()
        try:
            # 构建查询条件
            conds = ["user_id = $1", "tag != 'delete'"]
            params = [user_id]
            param_idx = 2

            if status == "has_audio":
                conds.append("has_audio = TRUE")
            elif status == "no_audio":
                conds.append("has_audio = FALSE")

            if "start_created_at" in kwargs:
                conds.append(f"created_at >= ${param_idx}")
                params.append(kwargs["start_created_at"])
                param_idx += 1

            if "end_created_at" in kwargs:
                conds.append(f"created_at <= ${param_idx}")
                params.append(kwargs["end_created_at"])
                param_idx += 1

            where_clause = " WHERE " + " AND ".join(conds) if conds else ""

            # 查询总数
            count_query = f"SELECT COUNT(*) FROM {self.table_podcasts}{where_clause}"
            total = await conn.fetchval(count_query, *params)

            # 查询数据（包含rounds和subtitles数据）
            offset = (page - 1) * page_size
            query = f"""
                SELECT session_id, user_id, user_input, created_at, updated_at, has_audio, rounds, subtitles
                FROM {self.table_podcasts}
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """
            params.extend([page_size, offset])

            rows = await conn.fetch(query, *params)

            sessions = []
            for row in rows:
                # 处理created_at时间格式
                created_at = row["created_at"]
                if isinstance(created_at, datetime):
                    created_at = created_at.isoformat()
                elif created_at is None:
                    created_at = None

                # 解析rounds和subtitles数据
                rounds = []
                subtitles = []
                timestamps = []

                if row.get("rounds"):
                    try:
                        if isinstance(row["rounds"], str):
                            rounds = json.loads(row["rounds"])
                        else:
                            rounds = row["rounds"]

                        # 从rounds构建subtitles和timestamps
                        for round_info in rounds:
                            subtitles.append({"text": round_info.get("text", ""), "speaker": round_info.get("speaker", "")})
                            timestamps.append({"start": round_info.get("start", 0.0), "end": round_info.get("end", 0.0), "text": round_info.get("text", ""), "speaker": round_info.get("speaker", "")})
                    except Exception as e:
                        logger.warning(f"Error parsing rounds for {row['session_id']}: {e}")

                sessions.append(
                    {
                        "session_id": row["session_id"],
                        "user_id": row["user_id"],
                        "user_input": (row["user_input"] or "")[:100],  # 只返回前100个字符
                        "created_at": created_at,
                        "has_audio": row["has_audio"],
                        "rounds": rounds,  # 包含完整轮次信息
                        "subtitles": subtitles,  # 字幕列表
                        "timestamps": timestamps,  # 时间戳列表
                    }
                )

            total_pages = (total + page_size - 1) // page_size

            return {"sessions": sessions, "pagination": {"page": page, "page_size": page_size, "total": total, "total_pages": total_pages}}
        except Exception as e:
            logger.error(f"Error listing podcasts: {e}")
            traceback.print_exc()
            return {"sessions": [], "pagination": {"page": page, "page_size": page_size, "total": 0, "total_pages": 0}}
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def delete_podcast(self, session_id: str, user_id: str) -> bool:
        """软删除播客记录"""
        conn = await self.get_conn()
        try:
            await conn.execute(f"UPDATE {self.table_podcasts} SET tag = 'delete', updated_at = $1 WHERE session_id = $2 AND user_id = $3", datetime.now(), session_id, user_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting podcast: {e}")
            return False
        finally:
            await self.release_conn(conn)
