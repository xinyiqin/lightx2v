"""
本地播客管理器 - 使用本地文件系统存储播客元数据
参考 LocalTaskManager 的设计，提供类似的功能
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

from loguru import logger

from lightx2v.deploy.common.utils import class_try_catch_async, current_time, str2time, time2str


class LocalPodcastManager:
    """本地播客管理器 - 使用JSON文件存储播客元数据"""
    
    def __init__(self, local_dir: str):
        self.local_dir = local_dir
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)
    
    def get_podcast_filename(self, session_id: str):
        """获取播客文件路径"""
        return os.path.join(self.local_dir, f"podcast_{session_id}.json")
    
    def fmt_dict(self, data: Dict):
        """格式化字典（时间戳转字符串）"""
        for k in ["created_at", "updated_at"]:
            if k in data and isinstance(data[k], (int, float)):
                data[k] = time2str(data[k])
            elif k in data and isinstance(data[k], datetime):
                data[k] = data[k].isoformat()
    
    def parse_dict(self, data: Dict):
        """解析字典（字符串转时间戳）"""
        for k in ["created_at", "updated_at"]:
            if k in data and isinstance(data[k], str):
                try:
                    data[k] = str2time(data[k])
                except:
                    # 如果是ISO格式，尝试解析
                    try:
                        data[k] = datetime.fromisoformat(data[k].replace('Z', '+00:00')).timestamp()
                    except:
                        pass
    
    def save(self, podcast_data: Dict, with_fmt: bool = True):
        """保存播客数据到文件"""
        if with_fmt:
            self.fmt_dict(podcast_data)
        out_name = self.get_podcast_filename(podcast_data["session_id"])
        with open(out_name, "w", encoding="utf-8") as fout:
            fout.write(json.dumps(podcast_data, indent=4, ensure_ascii=False))
    
    def load(self, session_id: str, user_id: Optional[str] = None) -> Optional[Dict]:
        """从文件加载播客数据"""
        fpath = self.get_podcast_filename(session_id)
        if not os.path.exists(fpath):
            return None
        
        data = json.load(open(fpath, encoding="utf-8"))
        if user_id is not None and data.get("user_id") != user_id:
            raise Exception(f"Podcast {session_id} is not belong to user {user_id}")
        if data.get("tag") == "delete":
            raise Exception(f"Podcast {session_id} is deleted")
        
        self.parse_dict(data)
        return data
    
    async def init(self):
        """初始化（本地文件系统无需特殊初始化）"""
        pass
    
    async def close(self):
        """关闭（本地文件系统无需关闭）"""
        pass
    
    @class_try_catch_async
    async def insert_podcast(self, session_id: str, user_id: str, user_input: str,
                            audio_path: str, metadata_path: str, rounds: List[Dict] = None,
                            subtitles: List[Dict] = None, extra_info: Dict = None):
        """插入播客记录"""
        now = current_time()
        # 确保extra_info包含outputs字段
        if extra_info is None:
            extra_info = {}
        if "outputs" not in extra_info and audio_path:
            extra_info["outputs"] = {"merged_audio": audio_path}
        
        podcast_data = {
            "session_id": session_id,
            "user_id": user_id,
            "user_input": user_input,
            "created_at": now,
            "updated_at": now,
            "has_audio": True,
            "audio_path": audio_path,
            "metadata_path": metadata_path or "",  # metadata_path可能为空（不再保存到S3）
            "rounds": rounds or [],
            "subtitles": subtitles or [],
            "extra_info": extra_info,
            "tag": ""
        }
        self.save(podcast_data)
        return True
    
    @class_try_catch_async
    async def update_podcast(self, session_id: str, **kwargs):
        """更新播客记录"""
        data = self.load(session_id)
        if not data:
            return False
        
        # 更新字段
        if "has_audio" in kwargs:
            data["has_audio"] = kwargs["has_audio"]
        if "rounds" in kwargs:
            data["rounds"] = kwargs["rounds"]
        if "subtitles" in kwargs:
            data["subtitles"] = kwargs["subtitles"]
        if "extra_info" in kwargs:
            data["extra_info"] = kwargs["extra_info"]
        
        data["updated_at"] = current_time()
        self.save(data)
        return True
    
    @class_try_catch_async
    async def query_podcast(self, session_id: str, user_id: Optional[str] = None) -> Optional[Dict]:
        """查询单个播客记录"""
        try:
            return self.load(session_id, user_id)
        except Exception as e:
            logger.warning(f"Error querying podcast {session_id}: {e}")
            return None
    
    @class_try_catch_async
    async def list_podcasts(self, user_id: str, page: int = 1, page_size: int = 10,
                           status: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """列表查询播客记录（支持分页和过滤）"""
        podcasts = []
        
        # 遍历所有播客文件
        if not os.path.exists(self.local_dir):
            return {"sessions": [], "pagination": {"page": page, "page_size": page_size, "total": 0, "total_pages": 0}}
        
        for f in os.listdir(self.local_dir):
            if not f.startswith("podcast_") or not f.endswith(".json"):
                continue
            
            fpath = os.path.join(self.local_dir, f)
            try:
                data = json.load(open(fpath, encoding="utf-8"))
                self.parse_dict(data)
                
                # 过滤条件
                if user_id and data.get("user_id") != user_id:
                    continue
                
                if data.get("tag") == "delete":
                    continue
                
                if status == "has_audio" and not data.get("has_audio", False):
                    continue
                elif status == "no_audio" and data.get("has_audio", False):
                    continue
                
                if "start_created_at" in kwargs:
                    created_at = data.get("created_at", 0)
                    if isinstance(created_at, str):
                        created_at = str2time(created_at)
                    if created_at < kwargs["start_created_at"]:
                        continue
                
                if "end_created_at" in kwargs:
                    created_at = data.get("created_at", 0)
                    if isinstance(created_at, str):
                        created_at = str2time(created_at)
                    if created_at > kwargs["end_created_at"]:
                        continue
                
                podcasts.append(data)
            except Exception as e:
                logger.warning(f"Error reading podcast file {f}: {e}")
                continue
        
        # 排序（按创建时间倒序）
        podcasts.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        
        # 计算总数
        total = len(podcasts)
        
        # 分页
        offset = (page - 1) * page_size
        paginated_podcasts = podcasts[offset:offset + page_size]
        
        # 格式化返回数据（包含rounds和subtitles数据）
        sessions = []
        for podcast in paginated_podcasts:
            created_at = podcast.get("created_at", 0)
            if isinstance(created_at, (int, float)):
                created_at = datetime.fromtimestamp(created_at).isoformat()
            elif isinstance(created_at, str):
                try:
                    # 尝试解析时间戳字符串
                    created_at = datetime.fromtimestamp(str2time(created_at)).isoformat()
                except:
                    pass  # 保持原样
            
            # 解析rounds和subtitles数据
            rounds = podcast.get("rounds", [])
            subtitles = []
            timestamps = []
            
            if rounds:
                # 从rounds构建subtitles和timestamps
                for round_info in rounds:
                    subtitles.append({
                        "text": round_info.get("text", ""),
                        "speaker": round_info.get("speaker", "")
                    })
                    timestamps.append({
                        "start": round_info.get("start", 0.0),
                        "end": round_info.get("end", 0.0),
                        "text": round_info.get("text", ""),
                        "speaker": round_info.get("speaker", "")
                    })
            
            sessions.append({
                "session_id": podcast.get("session_id"),
                "user_id": podcast.get("user_id"),
                "user_input": (podcast.get("user_input") or "")[:100],  # 只返回前100个字符
                "created_at": created_at,
                "has_audio": podcast.get("has_audio", False),
                "rounds": rounds,  # 包含完整轮次信息
                "subtitles": subtitles,  # 字幕列表
                "timestamps": timestamps  # 时间戳列表
            })
        
        total_pages = (total + page_size - 1) // page_size
        
        return {
            "sessions": sessions,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages
            }
        }
    
    @class_try_catch_async
    async def delete_podcast(self, session_id: str, user_id: str) -> bool:
        """软删除播客记录"""
        data = self.load(session_id, user_id)
        if not data:
            return False
        
        data["tag"] = "delete"
        data["updated_at"] = current_time()
        self.save(data)
        return True

