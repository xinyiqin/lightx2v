import asyncio
import time
from functools import wraps

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.envs import *


class _ProfilingContext:
    def __init__(self, name):
        self.name = name
        if dist.is_initialized():
            self.rank_info = f"Rank {dist.get_rank()}"
        else:
            self.rank_info = "Single GPU"

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_time
        logger.info(f"[Profile] {self.rank_info} - {self.name} cost {elapsed:.6f} seconds")
        return False

    async def __aenter__(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_time
        logger.info(f"[Profile] {self.rank_info} - {self.name} cost {elapsed:.6f} seconds")
        return False

    def __call__(self, func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with self:
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return sync_wrapper


class _NullContext:
    # Context manager without decision branch logic overhead
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __call__(self, func):
        return func


class _ProfilingContextL1(_ProfilingContext):
    """Level 1 profiling context with Level1_Log prefix."""

    def __init__(self, name):
        super().__init__(f"Level1_Log {name}")


class _ProfilingContextL2(_ProfilingContext):
    """Level 2 profiling context with Level2_Log prefix."""

    def __init__(self, name):
        super().__init__(f"Level2_Log {name}")


"""
PROFILING_DEBUG_LEVEL=0: [Default] disable all profiling
PROFILING_DEBUG_LEVEL=1: enable ProfilingContext4DebugL1
PROFILING_DEBUG_LEVEL=2: enable ProfilingContext4DebugL1 and ProfilingContext4DebugL2
"""
ProfilingContext4DebugL1 = _ProfilingContextL1 if CHECK_PROFILING_DEBUG_LEVEL(1) else _NullContext  # if user >= 1, enable profiling
ProfilingContext4DebugL2 = _ProfilingContextL2 if CHECK_PROFILING_DEBUG_LEVEL(2) else _NullContext  # if user >= 2, enable profiling
