#!/usr/bin/env python3
"""
gsm.py — libgpuswap.so 的 ctypes 绑定（真库，不是模拟器）

这一层只做三件事：**装库、转类型、把返回码变成异常**。任何"策略"或
"记账"逻辑都不许写在这里 —— 写进来就等于又造了一个双实现，
而双实现漂移正是本项目已经栽过两次的坑（README 里的对比图必须来自
这一个真库，不是 Python 侧的影子实现）。

用法：
    from gsm import GpuSwap

    with GpuSwap(hot_bytes=512 << 20, cold_bytes=4 << 30) as m:
        m.set_policy("adaptive")
        m.reserve_cold(256)          # 见下：不预热的话测出来的异步是假的
        m.allocate(block_id=1, size=2 << 20)
        m.evict(1)                   # 立即返回
        m.sync()
"""

import ctypes
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)                       # gpu-swap/
_DEFAULT_LIB = os.path.join(_REPO, "build", "libgpuswap.so")

# 结果码 ↔ MigrateResult，数值必须和 include/gpu_swap_c.h 一致。
_OK = 0
_STRATEGIES = ("lru", "lfu", "gdsf", "adaptive")


class GpuSwapError(RuntimeError):
    """带结果码的库错误。msg 里同时带上 gsm_last_error 的原文。"""

    def __init__(self, code, op, detail=""):
        self.code = code
        self.op = op
        self.detail = detail
        super().__init__(f"{op} 失败 (code={code}): {detail}")


class GpuSwap:
    """GpuSwapManager 的 Python 句柄。

    ⚠️ CUDA 上下文是 per-thread 的：**一个实例只能在创建它的那个线程里用**。
    ⚠️ 所有访问器都不推进状态机（不收割在途迁移），所以
       `evict(); pending_migrations` 这个组合是确定性的，跟拷贝快不快无关。
    """

    def __init__(self, hot_bytes, cold_bytes, lib_path=None):
        self._lib = _load(lib_path)
        _declare(self._lib)
        self._h = self._lib.gsm_create(ctypes.c_size_t(hot_bytes),
                                       ctypes.c_size_t(cold_bytes))
        if not self._h:
            raise GpuSwapError(-1, "gsm_create",
                               "返回了空句柄（CUDA 初始化失败？）")
        self.hot_bytes = hot_bytes
        self.cold_bytes = cold_bytes

    # ---- 生涯 ----

    def close(self):
        if getattr(self, "_h", None):
            self._lib.gsm_destroy(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ---- 内部：调用 + 错误 ----

    def _call(self, name, *args):
        fn = getattr(self._lib, name)
        code = fn(self._h, *args)
        if code != _OK:
            raise GpuSwapError(code, name, self.last_error())
        return code

    def _try(self, name, *args):
        """返回结果码而不抛异常 —— 给"这一步本来就该失败"的用例用。"""
        return getattr(self._lib, name)(self._h, *args)

    # ---- 分配 / 释放 / 访问 ----

    def allocate(self, block_id, size):
        if self._try("gsm_allocate", ctypes.c_int64(block_id),
                     ctypes.c_size_t(size)) != _OK:
            return False
        return True

    def free(self, block_id):
        return self._try("gsm_free", ctypes.c_int64(block_id))

    def touch(self, block_id):
        return self._call("gsm_touch", ctypes.c_int64(block_id))

    # ---- 迁移 ----

    def evict(self, block_id):
        return self._try("gsm_evict", ctypes.c_int64(block_id))

    def prefetch(self, block_id):
        return self._try("gsm_prefetch", ctypes.c_int64(block_id))

    def sync(self):
        return self._call("gsm_sync")

    def wait_block(self, block_id):
        return self._call("gsm_wait_block", ctypes.c_int64(block_id))

    def settle(self):
        return self._call("gsm_settle")

    def reserve_cold(self, n_slots):
        """预热冷池。

        **必须在任何 kernel 之前调**：冷池是惰性 cudaMallocHost 的，而那是
        驱动文档点名的隐式同步点。第一次换出才去分配的话，那次换出会等设备
        空闲 —— 异步性恰好在最需要它的地方失效（实测撞上过：一个 250ms 自旋
        kernel 把紧随其后的一次 2MB 换出整个挡住）。
        """
        return self._call("gsm_reserve_cold", ctypes.c_size_t(n_slots))

    # ---- 时钟与策略 ----

    def set_clock(self, now_ms):
        self._lib.gsm_set_clock(self._h, ctypes.c_int64(int(now_ms)))

    def clock(self):
        return self._lib.gsm_clock(self._h)

    def set_policy(self, name, young_age_ms=-1, hot_threshold=-1,
                   protect_max_rate=-1.0):
        if name not in _STRATEGIES:
            raise ValueError(f"未知策略 {name!r}，可选 {_STRATEGIES}")
        self._call("gsm_set_policy", name.encode(),
                   ctypes.c_int64(young_age_ms),
                   ctypes.c_int64(hot_threshold),
                   ctypes.c_double(protect_max_rate))

    def policy_report(self):
        """策略自身的统计。非 Adaptive 的策略没有保护率，返回 None
        （库那边会明确报 GSM_ERR_BAD_ARG，不是静默给一堆 0）。"""
        r = _PolicyReport()
        if self._try("gsm_policy_report", ctypes.byref(r)) != _OK:
            return None
        return {k: getattr(r, k) for k, _ in _PolicyReport._fields_}

    # ---- 统计（全部不推进状态机）----

    @property
    def hot_used(self):        return self._lib.gsm_hot_used(self._h)
    @property
    def hot_valid(self):       return self._lib.gsm_hot_valid(self._h)
    @property
    def cold_used(self):       return self._lib.gsm_cold_used(self._h)
    @property
    def hot_capacity(self):    return self._lib.gsm_hot_capacity(self._h)
    @property
    def cold_capacity(self):   return self._lib.gsm_cold_capacity(self._h)
    @property
    def hot_peak(self):        return self._lib.gsm_hot_peak(self._h)
    @property
    def slot_bytes(self):      return self._lib.gsm_slot_bytes(self._h)
    @property
    def num_blocks(self):      return self._lib.gsm_num_blocks(self._h)
    @property
    def num_blocks_in_gpu(self): return self._lib.gsm_num_blocks_in_gpu(self._h)
    @property
    def pending_migrations(self): return self._lib.gsm_pending_migrations(self._h)
    @property
    def eviction_count(self):  return self._lib.gsm_eviction_count(self._h)
    @property
    def eviction_fingerprint(self):
        return self._lib.gsm_eviction_fingerprint(self._h)

    def is_migrating(self, block_id):
        return bool(self._lib.gsm_is_migrating(self._h, ctypes.c_int64(block_id)))

    def is_pinned(self, block_id):
        return bool(self._lib.gsm_is_pinned(self._h, ctypes.c_int64(block_id)))

    def block_info(self, block_id):
        info = _BlockInfo()
        if self._lib.gsm_block_info(self._h, ctypes.c_int64(block_id),
                                    ctypes.byref(info)) != _OK:
            return None
        return {k: getattr(info, k) for k, _ in _BlockInfo._fields_}

    def last_error(self):
        if not getattr(self, "_h", None):
            return ""
        s = self._lib.gsm_last_error(self._h)
        return s.decode("utf-8", "replace") if s else ""


# ================================================================
# ctypes 胶水
# ================================================================

class _PolicyReport(ctypes.Structure):
    _fields_ = [
        ("decisions", ctypes.c_int64), ("victims", ctypes.c_int64),
        ("clamped", ctypes.c_int64), ("relaxed_hot", ctypes.c_int64),
        ("relaxed_young", ctypes.c_int64),
        ("young_block_hits", ctypes.c_int64), ("hot_block_hits", ctypes.c_int64),
        ("resident_seen", ctypes.c_int64),
        ("protect_min", ctypes.c_double), ("protect_mean", ctypes.c_double),
        ("protect_max", ctypes.c_double),
        ("protect_first_quartile", ctypes.c_double),
        ("protect_last_quartile", ctypes.c_double),
        ("cutoff_last", ctypes.c_int64), ("cutoff_clamped_last", ctypes.c_int64),
        ("inflation", ctypes.c_double),
        ("inflation_rebases", ctypes.c_int64),
        ("is_adaptive", ctypes.c_int32),
    ]


class _BlockInfo(ctypes.Structure):
    _fields_ = [
        ("in_gpu", ctypes.c_int32), ("valid", ctypes.c_int32),
        ("migrating", ctypes.c_int32), ("pinned", ctypes.c_int32),
        ("has_cold", ctypes.c_int32), ("_pad", ctypes.c_int32),
        ("size", ctypes.c_size_t), ("gpu_addr", ctypes.c_uint64),
        ("last_access", ctypes.c_int64), ("access_count", ctypes.c_int64),
        ("alloc_time", ctypes.c_int64),
    ]


def _check_fresh(path):
    """拒绝加载一个比源码旧的 .so。

    这个东西是被真实事故逼出来的：一次 25 分钟的真实对比跑完，才发现
    `libgpuswap.so` 比 `eviction_policy.h` 早 9 分钟 —— 整轮量的都是旧内核，
    而**结果看起来完全正常**（数字合理、趋势合理），只有和模拟器对账才露馅。
    静默用旧代码是最贵的一类错误，所以在入口处直接炸掉。
    """
    src_dir = os.path.join(_REPO, "src")
    inc_dir = os.path.join(_REPO, "include")
    lib_mtime = os.path.getmtime(path)
    newer = []
    for d in (src_dir, inc_dir):
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if not name.endswith((".cu", ".h", ".cpp")):
                continue
            f = os.path.join(d, name)
            if os.path.getmtime(f) > lib_mtime:
                newer.append(os.path.relpath(f, _REPO))
    if newer:
        raise GpuSwapError(
            -1, "stale-lib",
            f"{os.path.relpath(path, _REPO)} 比这些源码旧：{', '.join(sorted(newer))}。"
            f"先跑 `make` 重建，否则你量的是旧内核。")


def _load(lib_path=None):
    path = lib_path or os.environ.get("GSM_LIB") or _DEFAULT_LIB
    if not os.path.exists(path):
        raise GpuSwapError(
            -1, "dlopen",
            f"找不到 {path}。先在 gpu-swap/ 下跑 `make`。"
            f"（也可以用 GSM_LIB 环境变量指到别处）")
    if not os.environ.get("GSM_ALLOW_STALE"):
        _check_fresh(path)
    try:
        return ctypes.CDLL(path)
    except OSError as e:
        raise GpuSwapError(-1, "dlopen", f"{path}: {e}") from e


def _declare(lib):
    """显式声明每个入口的签名。

    不声明的话 ctypes 默认按 int 处理返回值，而 size_t / int64_t 在
    64 位下会**被截断**——症状是统计量偶尔变成负数或小得离谱，
    而且只在数据量大了以后才出现。这里逐条写死，不留默认。
    """
    c = ctypes
    lib.gsm_create.restype = c.c_void_p
    lib.gsm_create.argtypes = [c.c_size_t, c.c_size_t]
    lib.gsm_destroy.restype = None
    lib.gsm_destroy.argtypes = [c.c_void_p]

    _i64, _sz, _dbl = c.c_int64, c.c_size_t, c.c_double
    for name, argtypes in [
        ("gsm_allocate", [_i64, _sz]), ("gsm_free", [_i64]),
        ("gsm_touch", [_i64]), ("gsm_evict", [_i64]), ("gsm_prefetch", [_i64]),
        ("gsm_sync", []), ("gsm_wait_block", [_i64]), ("gsm_settle", []),
        ("gsm_reserve_cold", [_sz]),
        ("gsm_pin", [_i64, c.c_void_p]), ("gsm_unpin", [_i64]),
        ("gsm_set_policy", [c.c_char_p, _i64, _i64, _dbl]),
        ("gsm_policy_report", [c.POINTER(_PolicyReport)]),
        ("gsm_block_info", [_i64, c.POINTER(_BlockInfo)]),
        ("gsm_fill_pattern", [_i64, c.c_uint64]),
        ("gsm_verify_pattern", [_i64, c.c_uint64, c.POINTER(c.c_int64)]),
    ]:
        fn = getattr(lib, name)
        fn.restype = c.c_int
        fn.argtypes = [c.c_void_p] + argtypes

    lib.gsm_set_clock.restype = None
    lib.gsm_set_clock.argtypes = [c.c_void_p, _i64]
    lib.gsm_clock.restype = _i64
    lib.gsm_clock.argtypes = [c.c_void_p]

    for name in ["gsm_hot_used", "gsm_hot_valid", "gsm_cold_used",
                 "gsm_hot_capacity", "gsm_cold_capacity", "gsm_hot_peak",
                 "gsm_slot_bytes"]:
        fn = getattr(lib, name)
        fn.restype = _sz
        fn.argtypes = [c.c_void_p]

    for name in ["gsm_num_blocks", "gsm_num_blocks_in_gpu",
                 "gsm_pending_migrations"]:
        fn = getattr(lib, name)
        fn.restype = c.c_int
        fn.argtypes = [c.c_void_p]

    for name in ["gsm_is_migrating", "gsm_is_pinned"]:
        fn = getattr(lib, name)
        fn.restype = c.c_int
        fn.argtypes = [c.c_void_p, _i64]

    lib.gsm_eviction_count.restype = _i64
    lib.gsm_eviction_count.argtypes = [c.c_void_p]
    lib.gsm_eviction_fingerprint.restype = c.c_uint32
    lib.gsm_eviction_fingerprint.argtypes = [c.c_void_p]
    lib.gsm_gpu_addr.restype = c.c_uint64
    lib.gsm_gpu_addr.argtypes = [c.c_void_p, _i64]

    lib.gsm_last_error.restype = c.c_char_p
    lib.gsm_last_error.argtypes = [c.c_void_p]
    lib.gsm_result_str.restype = c.c_char_p
    lib.gsm_result_str.argtypes = [c.c_int]
