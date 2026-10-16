"""
vllm_steptrace — 给 EngineCore 的每个 iteration 打点

============================================================
为什么要单独做这个，而不是用 --enable-logging-iteration-details
============================================================

vLLM 自带的那个日志（vllm/v1/engine/core.py:452-471）只覆盖**模型执行段**，
**按构造排除了 schedule 阶段**。而 PCIe 迁移流量最可能的落点恰恰是
schedule / CPU 那一侧：D2H 拷贝占主机 DRAM 带宽、IOMMU 与 DMA 映射吃 CPU。

拿它当主信号，会测出「几乎没影响」，而用户可见的 ITL 明明在动 ——
然后得出「PCIe 无所谓」的结论。这是整场实验里最容易踩、也最不容易发现的坑。

所以主信号换成 `EngineCoreProc._process_engine_step`（core.py:1300）。
它是**唯一漏斗**：

    run_busy_loop()
      ├─ _process_input_queue()
      └─ _process_engine_step()        ← 包这里
           ├─ self.step_fn()           （schedule + forward）
           ├─ 输出入队
           ├─ post_step()
           └─ if not model_executed and scheduler.has_requests():
                  time.sleep(0.001)    ← WAITING_FOR_REMOTE_KVS，整毫秒

只包 `step` 会漏掉输出处理和那个 1ms sleep。

============================================================
两个跨进程的坑（这个仓库已经被咬过一次，commit d3207c4）
============================================================

1. `time.perf_counter()` 的**原点逐进程不同**，不能拿来和外部进程（负载发生器）
   的时间线拼接。所以每行同时记 `time.time_ns()`（CLOCK_REALTIME，可跨进程序）
   和 `perf_counter_ns()` 的**差值**（时长本身仍然用单调钟测才准）。
2. 别在热循环里逐行落盘。攒批，每 `KVSTEP_FLUSH_EVERY` 行刷一次。

============================================================
用法
============================================================

    KVSTEP_OUT=/abs/path/traces/step_preload \
    vllm serve ... --enable-logging-iteration-details

不设 `KVSTEP_OUT` 时 `register()` 直接返回，零开销。
输出：`<KVSTEP_OUT>.<pid>.jsonl`，每行一个 iteration。
"""

import atexit
import json
import os

#: 攒多少行刷一次盘。热循环里逐行写会污染测量本身。
DEFAULT_FLUSH_EVERY = 1000

#: 和行数**或**时间（秒）两个条件里先到的那个触发刷盘。
#:
#: 只按行数刷会踩一个坑：实测一个 block 只有几百个 iteration，永远够不到
#: 1000 行，于是整场测量期间文件一直是 0 字节 —— 看起来像 hook 没装上。
#: 按时间刷才能边跑边读。
DEFAULT_FLUSH_INTERVAL_S = 0.5

_buffer: list[str] = []
_fh = None
_flush_every = DEFAULT_FLUSH_EVERY
_flush_interval = DEFAULT_FLUSH_INTERVAL_S
_last_flush = 0.0
_path: str | None = None


def _flush(force: bool = False):
    global _buffer, _last_flush
    import time
    now = time.monotonic()
    if _fh is None or not _buffer:
        return
    if not force and (now - _last_flush) < _flush_interval:
        return
    _fh.write("".join(_buffer))
    _fh.flush()
    _buffer = []
    _last_flush = now


def _ensure_open():
    """首次写时才建文件。

    入口点在 frontend / worker 进程里也会执行，那里没有 EngineCoreProc，
    但 `install()` 仍会包到一个空的类属性上。延迟建文件可以避免留下
    一堆 0 字节的 `*.jsonl`，让人误以为 hook 生效了。
    """
    global _fh
    if _fh is None and _path is not None:
        _fh = open(_path, "w", buffering=1 << 20)


def install(out_prefix: str, flush_every: int = DEFAULT_FLUSH_EVERY) -> bool:
    """包住 EngineCoreProc._process_engine_step。装上了返回 True。"""
    global _flush_every, _path
    _flush_every = flush_every
    _path = f"{out_prefix}.{os.getpid()}.jsonl"

    try:
        from vllm.v1.engine import core as core_mod
    except ImportError:
        return False

    cls = getattr(core_mod, "EngineCoreProc", None)
    if cls is None or not hasattr(cls, "_process_engine_step"):
        return False

    import time

    atexit.register(lambda: _flush(force=True))

    orig = cls._process_engine_step
    if getattr(orig, "_kvtrace_step_wrapped", False):
        return True

    def patched(self, *args, **kwargs):
        # 实时钟用于跨进程拼接；单调钟的差值用于时长（不受 NTP 调整影响）
        t0_rt = time.time_ns()
        t0_pc = time.perf_counter_ns()
        ok = orig(self, *args, **kwargs)
        t1_pc = time.perf_counter_ns()

        # iter_index 是 EngineCore 自己的计数器，`--enable-logging-iteration-details`
        # 打的 `Iteration(N)` 用的是同一个 —— 靠它把「完整步」和日志里的
        # 「仅执行段」对齐，两者之差就是 schedule + 输出 + post_step 那段。
        # 注意 orig() 返回时 index 已经自增，所以记的是**刚完成的那一步**的序号。
        idx = getattr(self, "_iteration_index", None)
        _buffer.append(json.dumps({
            "t": t0_rt,                      # 步开始（CLOCK_REALTIME）
            "dur_ns": t1_pc - t0_pc,         # 本步耗时
            "iter": None if idx is None else int(idx) - 1,
            "model_executed": bool(ok),
            "pid": os.getpid(),
        }) + "\n")
        _ensure_open()
        if len(_buffer) >= _flush_every:
            _flush(force=True)
        else:
            _flush()
        return ok

    patched._kvtrace_step_wrapped = True
    cls._process_engine_step = patched
    print(f"[KVStep][pid={os.getpid()}] 已包住 EngineCoreProc._process_engine_step "
          f"→ {_path}", flush=True)
    return True


def register():
    """vLLM ``vllm.general_plugins`` 入口点。

    vLLM 会在 frontend / EngineCore / worker 每个进程里各调一次。只有
    EngineCore 进程里存在 EngineCoreProc，别处 hasattr 一挡就退出。
    """
    out = os.environ.get("KVSTEP_OUT")
    if not out:
        return
    every = int(os.environ.get("KVSTEP_FLUSH_EVERY", DEFAULT_FLUSH_EVERY))
    install(out, every)
