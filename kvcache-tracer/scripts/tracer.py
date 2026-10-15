"""
tracer.py — KV Cache Block 生命周期追踪器

通过 monkey-patch vLLM 的 SingleTypeKVCacheManager 记录每个 block 的
分配、写入、释放事件。

============================================================
实测环境：vLLM 0.25.1（本文件按 0.25.1 的实际签名校对过）
============================================================

四个 hook 点：

  SingleTypeKVCacheManager.allocate_new_blocks(self, request_id, num_tokens, ...)
      → 返回本次新分配的 KVCacheBlock 列表          [alloc]

  SingleTypeKVCacheManager.cache_blocks(self, request, num_tokens, ...)
      → 参数是 Request 对象，不是 request_id！
      → 每当一个 block 被写满并写入 prefix cache 时触发  [access]

  SingleTypeKVCacheManager.pop_blocks_for_free(self, request_id)
      → 返回该请求持有的全部 block（尚未归还 block pool）
      → 这是 block 释放的唯一收口点：free() 内部调用它，
        scheduler 抢占（preemption）也直接调用它        [free]

  KVCacheManager.allocate_slots(self, request, ...)
      → 注意是另一个类。scheduler 每个 step 对每个 running request
        调一次（sched/scheduler.py:535），是「每步读取」的唯一可见信号
      → 只做内存计数，计数在 block 释放时写进 free 事件  [reads]

------------------------------------------------------------
两个必须知道的运行时事实（否则 hook 一定不生效）
------------------------------------------------------------
1. monkey-patch 只存在于「安装它的那个解释器」里。

2. vLLM v1 的 EngineCore 是 **spawn** 出来的独立进程，不是 fork ——
   而 KVCacheManager 只在那个进程里。frontend 进程里做的 patch
   对它是不可见的。实测（vLLM 0.25.1 / 单卡 4090）：

       frontend 刚 import vllm 时问 get_mp_context()  → 'fork'
       vLLM 真正创建 EngineCore 时再问                 → 'spawn'

   因为 vllm/utils/system_utils.py:_maybe_force_spawn() 会在
   cuda_is_initialized() 为真时强制改写成 spawn，而 frontend 到那个
   时刻已经初始化过 CUDA 了。**所以要验证 start method，必须插在
   vLLM 真正创建 EngineCore 的那一刻去测，早测会得到假阳性。**

   正确做法是用 vLLM 自带的 plugin 机制，它会加载进所有进程
   （frontend / EngineCore / worker），见 kvcache-tracer/plugin/。

3. 顺带一提：EngineCore 进程退出走 os._exit()，不执行 atexit ——
   事件必须「边产生边落盘」，不能等最后统一 save。本实现用 JSONL
   增量写盘，文件名带 pid：kvcache_trace.<pid>.jsonl

用法:
    # 一次性安装（装完就不用再管启动方式了）
    cd kvcache-tracer/plugin && pip install -e .
    # 正常启动，hook 自动生效
    vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
    # 发请求，然后
    python scripts/analyze.py kvcache_trace.*.jsonl
"""

import json
import time
import os
from collections import defaultdict
from typing import List, Optional

#: KVTRACE_DEBUG=1 时每个 hook 触发都会往 stdout 打一行（带 pid），
#: 用来确认 hook 到底有没有被调用、在哪个进程里被调用。
DEBUG = os.environ.get("KVTRACE_DEBUG") == "1"


def _dbg(msg: str):
    if DEBUG:
        print(f"[KVTrace][pid={os.getpid()}] {msg}", flush=True)


class KVTraceCollector:
    """收集并保存 KV Cache block 事件。

    事件同时进内存（self.events）和 JSONL 文件（增量、flush）。
    落盘是主通道 —— fork 出的 EngineCore 进程内存中的事件会随
    os._exit() 一起消失，只有已经写进文件的部分能留下来。
    """

    #: 攒够多少条事件刷一次盘。
    #: 默认 1 = 每条事件立即落盘。事件的产生速率本来就低（一个 decode step
    #: 最多产出一条 access，alloc/free 更稀疏），一次 write+flush 的开销可以
    #: 忽略；而攒批会让数据滞留在缓冲区里 —— 服务还在跑的时候你去看文件是空的，
    #: 子进程被 kill 时缓冲区里的数据直接丢。用 KVTRACE_FLUSH_EVERY 调大可以
    #: 换吞吐，但要知道代价。
    FLUSH_EVERY = int(os.environ.get("KVTRACE_FLUSH_EVERY", "1"))

    def __init__(self, path: Optional[str] = None):
        self.events: List[dict] = []

        # 时间原点必须跨进程统一。spawn 出的 EngineCore 是全新解释器，
        # 如果各自在 __init__ 里取 time.time()，父子进程的时间戳就差了
        # 一个「加载时间」，analyze.py 按 timestamp 归并多进程事件时会错位。
        # 做法：第一个注册的进程把 T0 写进环境变量，spawn 继承环境变量，
        # 于是所有进程共用同一个原点。
        t0 = os.environ.get("KVTRACE_T0")
        if t0 is None:
            t0 = repr(time.time())
            os.environ["KVTRACE_T0"] = t0
        self.start_time = float(t0)
        self.active_blocks: dict = {}          # block_id → {alloc_time, request_id}
        self.request_blocks: dict = defaultdict(set)

        #: (block_id, request_id) → 该请求读该 block 的次数。
        #:
        #: 键必须是二元组，不能只用 block_id。一个 block 会被多个请求持有
        #: （prefix cache 命中时共享、释放后又被下一个请求复用），而计数器
        #: 在释放时被取走清零。只按 block_id 键控的话，谁先释放谁就把所有
        #: 持有者的读数一并取走，后面的持有者取到 0 —— 实测 8 并发时
        #: shared-prefix 类别里出现 128 条 reads=0，而先释放的那个拿到
        #: 被夸大的 129→329（并发越高越离谱）。
        #:
        #: 只在内存里累加，不落盘 —— 每步每 block 写一条事件会把文件撑爆
        #: （长请求 129 blocks × 64 steps ≈ 8000 条）。真正的计数在 block
        #: 释放时一次性写进 free 事件，I/O 是每个 block 一条。
        self.reads: dict = defaultdict(int)

        # 输出前缀。真正的文件名是 "<prefix>.<pid>.jsonl"，pid 在首次
        # 刷盘时才取 —— 因为 fork 出的子进程 pid 与父进程不同，
        # 必须让子进程把事件写进自己的文件。
        self.path = path or os.environ.get("KVTRACE_OUTPUT", "kvcache_trace")
        self._fh = None
        self._fh_path: Optional[str] = None
        self._buf: List[dict] = []

    # ---- 落盘 ----

    def _emit(self, event: dict) -> None:
        """记录一条事件：进内存 + 进缓冲区，缓冲区满则刷盘。"""
        self.events.append(event)
        self._buf.append(event)
        if len(self._buf) >= self.FLUSH_EVERY:
            self.flush()

    def _open(self):
        if self._fh is None:
            self._fh_path = f"{self.path}.{os.getpid()}.jsonl"
            self._fh = open(self._fh_path, "a")

    def flush(self) -> None:
        """把缓冲区写进 JSONL 文件。"""
        if not self._buf:
            return
        self._open()
        assert self._fh is not None
        for ev in self._buf:
            self._fh.write(json.dumps(ev, separators=(",", ":")) + "\n")
        self._buf.clear()
        self._fh.flush()

    # ---- 记录方法 ----

    def record_alloc(self, blocks, request_id: str, source: str = "new"):
        """记录 block 获取事件（出生）

        source="new"   由 allocate_new_blocks 新分配
        source="cache" 由 add_local_computed_blocks 从 prefix cache 命中复用

        两者必须都记，否则统计会失衡：命中缓存的 block 是「凭空」出现在
        请求里的，没有 alloc 事件却有 free 事件。实测 4 个请求共享 3000
        token 前缀时，free 比 alloc 多了 562 条 —— 正好是 3 个请求
        × ~188 个缓存 block。
        """
        t = round(time.time() - self.start_time, 6)
        for blk in blocks:
            blk_id = self._block_to_int(blk)
            self.active_blocks[blk_id] = {"alloc_time": t, "request_id": request_id}
            self.request_blocks[request_id].add(blk_id)
            self._emit({
                "event": "alloc",
                "timestamp": t,
                "block_id": blk_id,
                "request_id": request_id,
                "source": source,
                "pid": os.getpid(),
            })

    def record_free(self, blocks, request_id: str):
        """记录 block 释放事件（死亡）

        顺带把该 block 累计的读取次数写进事件 —— 这是 hot/cold skew 的唯一
        数据来源，见 record_step_reads()。
        """
        t = round(time.time() - self.start_time, 6)
        for blk in blocks:
            blk_id = self._block_to_int(blk)
            self.active_blocks.pop(blk_id, None)
            self.request_blocks[request_id].discard(blk_id)
            self._emit({
                "event": "free",
                "timestamp": t,
                "block_id": blk_id,
                "request_id": request_id,
                "reads": self.reads.pop((blk_id, request_id), 0),
                "pid": os.getpid(),
            })

    def record_step_reads(self, blocks, request_id: str):
        """累加「该请求读该 block」的次数（每个 decode step 调一次，**不落盘**）。

        为什么需要它：cache_blocks 只在 block 写满时触发一次，所以
        access 事件的计数恒为 1，回答不了「哪些 block 是热的」。真正
        「每步读一次」发生在 attention kernel 里，KVCacheManager 这层
        看不到 —— 唯一可见的等价信号是 allocate_slots()：
        scheduler 每个 step 对每个 running request 调一次。
        这里只做内存计数，等 block 释放时由 record_free() 一次性写出。

        语义是「这个请求持有期间读了这个 block 多少次」。加总所有请求对
        同一个 block_id 的读数，得到的是这个 block 服务过的总次数 ——
        analyze.py 的 compute_access_counts 就是这么算热度的。
        """
        for blk in blocks:
            self.reads[(self._block_to_int(blk), request_id)] += 1

    def record_access(self, blocks, request_id: str, step: int = -1):
        """记录 block 被写入（写满一个 block 并进入 prefix cache）

        注意语义：这个事件来自 cache_blocks()，一个 block 填满时触发一次。
        它**不是**「每个 decode step 读一次」—— 那种每步读取发生在 attention
        kernel 里，KVCacheManager 这一层看不到（见文末说明）。
        """
        t = round(time.time() - self.start_time, 6)
        for blk in blocks:
            blk_id = self._block_to_int(blk)
            self._emit({
                "event": "access",
                "timestamp": t,
                "block_id": blk_id,
                "request_id": request_id,
                "step": step,
                "pid": os.getpid(),
            })

    def save(self, path: Optional[str] = None):
        """把内存中的事件汇总成一个 JSON 文件（便于人眼查看）。

        注意：这是「尽力而为」的汇总。fork 出的 EngineCore 进程里的
        事件到不了这里，真正的数据在 JSONL 文件里。
        """
        self.flush()
        out = path or (os.environ.get("KVTRACE_OUTPUT", "kvcache_trace") + ".json")
        with open(out, "w") as f:
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
        print(f"[KVTrace] Saved {len(self.events)} events to {out}")
        if self._fh_path:
            print(f"[KVTrace] Incremental trace: {self._fh_path}")

    # ---- 内部工具 ----

    @staticmethod
    def _block_to_int(block) -> int:
        """vLLM block 可能是 int 或 KVCacheBlock 对象，统一转成 int。

        vLLM 0.25.1 的 KVCacheBlock 是 dataclass，字段名就是 block_id。
        """
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
# Hook 注入
# ================================================================
#
# 包装函数一律写成 (self, *args, **kwargs) 透传的形式，不硬编码参数名。
# 原因：vLLM 各版本这三个方法的签名一直在变（0.25 的 cache_blocks 第一个
# 参数是 Request 对象，旧版本是 request_id），透传可以避免版本差异导致的
# TypeError；真正的参数在包装体内按位置/属性取。

def install_hooks():
    """在 SingleTypeKVCacheManager 上安装 hook"""
    tracer = get_tracer()

    import vllm.v1.core.single_type_kv_cache_manager as m

    M = m.SingleTypeKVCacheManager

    # ---- Hook 1: allocate_new_blocks → alloc ----
    _orig_alloc = M.allocate_new_blocks

    def _patched_alloc(self, *args, **kwargs):
        blocks = _orig_alloc(self, *args, **kwargs)
        request_id = args[0] if args else kwargs.get("request_id")
        _dbg(f"HOOK allocate_new_blocks request={request_id} → {len(blocks)} blocks")
        if blocks:                      # 可能返回 []（无需新 block）
            tracer.record_alloc(blocks, str(request_id))
        return blocks

    M.allocate_new_blocks = _patched_alloc

    # ---- Hook 1.5: add_local_computed_blocks → alloc(source="cache") ----
    # prefix cache 命中的 block 不走 allocate_new_blocks，是直接挂到请求上的。
    # 不记这里，统计就只剩「新分配」的一半，free 数会远大于 alloc 数。
    _orig_add_cached = M.add_local_computed_blocks

    def _patched_add_cached(self, *args, **kwargs):
        result = _orig_add_cached(self, *args, **kwargs)
        request_id = args[0] if args else kwargs.get("request_id")
        blocks = (args[1] if len(args) > 1
                  else kwargs.get("new_computed_blocks", []))
        if blocks:
            tracer.record_alloc(blocks, str(request_id), source="cache")
        return result

    M.add_local_computed_blocks = _patched_add_cached

    # ---- Hook 2: pop_blocks_for_free → free ----
    # 这是 block 释放的唯一收口点：
    #   free() 内部调它（single_type_kv_cache_manager.py:410）
    #   scheduler 抢占时直接调它（sched/scheduler.py:2141）
    _orig_pop = M.pop_blocks_for_free

    def _patched_pop(self, *args, **kwargs):
        blocks = _orig_pop(self, *args, **kwargs)
        request_id = args[0] if args else kwargs.get("request_id")
        _dbg(f"HOOK pop_blocks_for_free request={request_id} → {len(blocks)} blocks")
        if blocks:
            tracer.record_free(blocks, str(request_id))
        return blocks

    M.pop_blocks_for_free = _patched_pop

    # ---- Hook 3: cache_blocks → access ----
    # 签名: cache_blocks(self, request: Request, num_tokens, retention_interval=None)
    # 第一个参数是 Request 对象，request_id 要取 request.request_id
    _orig_cache = M.cache_blocks

    def _patched_cache(self, *args, **kwargs):
        request = args[0] if args else kwargs.get("request")
        request_id = getattr(request, "request_id", None) or str(request)

        # 调用前记录「已缓存到第几个 block」，调用后对比，
        # 差值就是本次新写满的 block —— 即真正发生了写入的 block。
        before = self.num_cached_block.get(request_id, 0)
        result = _orig_cache(self, *args, **kwargs)
        after = self.num_cached_block.get(request_id, 0)
        _dbg(f"HOOK cache_blocks request={request_id} cached {before}→{after}")

        if after > before:
            req_blocks = self.req_to_blocks.get(request_id, [])
            tracer.record_access(req_blocks[before:after], request_id)
        return result

    M.cache_blocks = _patched_cache

    # ---- Hook 4: KVCacheManager.allocate_slots → 每步读取计数 ----
    # scheduler 每个 step 对每个 running request 调一次 allocate_slots
    # （sched/scheduler.py:535），这是「这个请求这一步被调度了」的信号，
    # 等价于「它持有的 block 这一步被读了一遍」。
    # 注意：这是另一个类（KVCacheManager），不是 SingleTypeKVCacheManager。
    import vllm.v1.core.kv_cache_manager as kcm

    _orig_alloc_slots = kcm.KVCacheManager.allocate_slots

    def _patched_alloc_slots(self, *args, **kwargs):
        result = _orig_alloc_slots(self, *args, **kwargs)
        request = args[0] if args else kwargs.get("request")
        # 返回 None 表示分配失败（显存不够），这一步没跑起来，不算读取
        if result is not None and request is not None:
            request_id = getattr(request, "request_id", str(request))
            blocks = []
            for mgr in self.coordinator.single_type_managers:
                blocks.extend(mgr.req_to_blocks.get(request_id, []))
            # 转成 str 再交给 tracer：reads 的键是 (block_id, request_id)，
            # 必须和 record_free 里那个 str(request_id) 逐字符相同才能对上。
            # 上面的 req_to_blocks 查找仍用原对象，dict 的键类型不能变。
            tracer.record_step_reads(blocks, str(request_id))
        return result

    kcm.KVCacheManager.allocate_slots = _patched_alloc_slots

    print("[KVTrace] Hooks installed: "
          "SingleTypeKVCacheManager.{allocate_new_blocks→alloc(new), "
          "add_local_computed_blocks→alloc(cache), "
          "pop_blocks_for_free→free, cache_blocks→access}, "
          "KVCacheManager.allocate_slots→read counter")


def save_on_exit():
    """正常退出时刷盘 + 落一份汇总 JSON。

    警告：只对「正常退出」的进程有效。fork 出的 EngineCore 子进程退出时走
    os._exit()，不执行 atexit —— 它的数据靠 flush() 增量写盘保住，不靠这里。
    """
    import atexit
    tracer = get_tracer()

    def _on_exit():
        tracer.flush()
        print(f"[KVTrace] flushed to {tracer._fh_path or '(nothing recorded)'}")

    atexit.register(_on_exit)
    print(f"[KVTrace] Will flush JSONL incrementally to "
          f"{tracer.path}.<pid>.jsonl")
