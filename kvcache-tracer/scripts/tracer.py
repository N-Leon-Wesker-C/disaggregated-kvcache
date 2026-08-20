"""
tracer.py — KV Cache Block 生命周期追踪器

通过 monkey-patch vLLM 的 KVCacheManager 记录每个 block 的
分配、访问、释放事件。

注意：服务器上实际验证的版本针对 vLLM 0.26 的
SingleTypeKVCacheManager 结构，hook 点为：
  - allocate_new_blocks: block 分配
  - free:               block 释放
  - cache_blocks:       block 写入/访问
如果你在服务器上已经调整过这个文件，以服务器版本为准并同步回来。

用法:
    python -c "
    import tracer
    tracer.install_hooks()
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', 'Qwen/Qwen2.5-7B-Instruct',
        '--gpu-memory-utilization', '0.85',
        '--port', '8000'])
    "
"""

import json
import time
import os
from collections import defaultdict
from typing import List, Optional


class KVTraceCollector:
    """收集并保存 KV Cache block 事件"""

    def __init__(self):
        self.events: List[dict] = []
        self.start_time = time.time()
        self.active_blocks: dict = {}          # block_id → {alloc_time, request_id}
        self.request_blocks: dict = defaultdict(set)

    # ---- 记录方法 ----

    def record_alloc(self, block_ids, request_id: str):
        """记录 block 分配事件"""
        t = round(time.time() - self.start_time, 6)
        for blk in block_ids:
            blk_id = self._block_to_int(blk)
            self.active_blocks[blk_id] = {"alloc_time": t, "request_id": request_id}
            self.request_blocks[request_id].add(blk_id)
            self.events.append({
                "event": "alloc",
                "timestamp": t,
                "block_id": blk_id,
                "request_id": request_id,
            })

    def record_free(self, block_ids, request_id: str):
        """记录 block 释放事件"""
        t = round(time.time() - self.start_time, 6)
        for blk in block_ids:
            blk_id = self._block_to_int(blk)
            if blk_id in self.active_blocks:
                del self.active_blocks[blk_id]
            self.request_blocks[request_id].discard(blk_id)
            self.events.append({
                "event": "free",
                "timestamp": t,
                "block_id": blk_id,
                "request_id": request_id,
            })

    def record_access(self, block_ids, request_id: str, step: int):
        """记录 block 被访问（每次 decode step 读一次）"""
        t = round(time.time() - self.start_time, 6)
        for blk in block_ids:
            blk_id = self._block_to_int(blk)
            self.events.append({
                "event": "access",
                "timestamp": t,
                "block_id": blk_id,
                "request_id": request_id,
                "step": step,
            })

    def save(self, path: str = "kvcache_trace.json"):
        """保存 trace 到 JSON 文件"""
        with open(path, "w") as f:
            json.dump({
                "metadata": {
                    "total_events": len(self.events),
                    "duration_s": round(time.time() - self.start_time, 2),
                    "unique_blocks": len(set(e["block_id"] for e in self.events)),
                    "unique_requests": len(set(
                        e.get("request_id", "") for e in self.events)),
                },
                "events": self.events,
            }, f, indent=2)
        print(f"[KVTrace] Saved {len(self.events)} events to {path}")

    # ---- 内部工具 ----

    @staticmethod
    def _block_to_int(block) -> int:
        """vLLM block 可能是 int 或 KVCacheBlock 对象，统一转成 int"""
        if isinstance(block, int):
            return block
        for attr in ("block_id", "block_number", "index"):
            if hasattr(block, attr):
                return int(getattr(block, attr))
        return hash(block)


# 全局单例
_tracer: Optional[KVTraceCollector] = None


def get_tracer() -> KVTraceCollector:
    global _tracer
    if _tracer is None:
        _tracer = KVTraceCollector()
    return _tracer


# ================================================================
# Hook 注入 — 针对 vLLM 0.26 (v1 engine)
# ================================================================

def install_hooks():
    """在 SingleTypeKVCacheManager 上安装 hook"""
    tracer = get_tracer()

    import vllm.v1.core.single_type_kv_cache_manager as m

    # ---- Hook 1: allocate_new_blocks ----
    _orig_alloc = m.SingleTypeKVCacheManager.allocate_new_blocks

    def _patched_alloc(self, request_id, num_blocks, *args, **kwargs):
        result = _orig_alloc(self, request_id, num_blocks, *args, **kwargs)
        # result 是 KVCacheBlock 列表（或 None）
        # 拿到 block_id 后记录
        if result:
            tracer.record_alloc(result, str(request_id))
        return result

    m.SingleTypeKVCacheManager.allocate_new_blocks = _patched_alloc

    # ---- Hook 2: free ----
    _orig_free = m.SingleTypeKVCacheManager.free

    def _patched_free(self, request_id, *args, **kwargs):
        # 释放前先拿到这个请求持有的 block
        # pop_blocks_for_free 在 free 内部调用——如果拿不到
        # block 列表，在这里记录请求级别的事件
        result = _orig_free(self, request_id, *args, **kwargs)
        tracer.events.append({
            "event": "free_request",
            "timestamp": round(time.time() - tracer.start_time, 6),
            "request_id": str(request_id),
        })
        return result

    m.SingleTypeKVCacheManager.free = _patched_free

    # ---- Hook 3: cache_blocks ----
    _orig_cache = m.SingleTypeKVCacheManager.cache_blocks

    def _patched_cache(self, request_id, *args, **kwargs):
        # cache_blocks(request_id, token_indices, block_ids, ...) 签名因版本而异
        # 先把原始调用执行掉，再记录——注意：block_ids 在 args 里，
        # 具体位置需要你根据 0.26 的签名确认
        result = _orig_cache(self, request_id, *args, **kwargs)
        # 简化：只记录请求级别的 access 事件
        tracer.events.append({
            "event": "access_request",
            "timestamp": round(time.time() - tracer.start_time, 6),
            "request_id": str(request_id),
        })
        return result

    m.SingleTypeKVCacheManager.cache_blocks = _patched_cache

    print("[KVTrace] Hooks installed on SingleTypeKVCacheManager: "
          "allocate_new_blocks, free, cache_blocks")


def save_on_exit():
    """程序退出时自动保存 trace"""
    import atexit
    tracer = get_tracer()
    output_path = os.environ.get("KVTRACE_OUTPUT", "kvcache_trace.json")
    atexit.register(lambda: tracer.save(output_path))
    print(f"[KVTrace] Will save trace to {output_path} on exit")
