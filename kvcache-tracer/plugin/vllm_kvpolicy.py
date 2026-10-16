"""
vllm_kvpolicy — 给 vLLM 的 CPU offloading 层补两个逐出策略：LFU / Adaptive

============================================================
为什么只补策略、不补迁移层
============================================================

`vllm/v1/kv_offload/cpu/` 已经是一个完整的异步迁移层：独立 CUDA stream、
event 串链、`enable_timing` 的传输计时、引用计数、块池管理，全都是现成的。
配置入口是 `--kv-offloading-size <GiB>`。

唯一缺的是**逐出决策**。策略基类在
`kv_offload/cpu/policies/base.py` 的 `CachePolicy`
（`get/insert/remove/touch/evict(n, protected)/clear`），注册表在
`kv_offload/cpu/manager.py:30` 的 `_CACHE_POLICIES` —— 一个普通模块级 dict，
里面只有 `"lru"` 和 `"arc"`。

普通 dict 意味着**可以直接补键，不用 fork vLLM**。本文件补两个。

============================================================
一个容易忽略的事实：vLLM 已经有"按频次准入"，缺的是"按频次逐出"
============================================================

`manager.py` 的 `store_threshold`（走 extra_config 配）用 `self.counts` 这个
OrderedDict 统计每个 key 在 `lookup()` 里出现过几次，不到阈值的块根本不会被换出
到 CPU。**准入这一侧已经有频次概念了**，而且 `counts` 自己做 LRU 裁剪
（`max_tracker_size`）。

但 `_CACHE_POLICIES` 里的 LRU / ARC 在**逐出**这一侧都不看频次：一个刚被命中
一次的新块，可以把一个被命中过一千次的老块挤掉。本文件补的就是这一侧。

============================================================
用法
============================================================

    vllm serve ... --kv-offloading-size 0.25 \\
        --kv-transfer-config '{"kv_connector_extra_config":{"eviction_policy":"lfu"}}'

把 `"lfu"` 换成 `"adaptive"` 即可换策略。

（`--kv-offloading-size` 只往 extra_config 里塞 `cpu_bytes_to_use`，而且是
 `update` 不是覆盖 —— 见 `config/vllm.py:805` —— 所以 eviction_policy 会保留。）

打开观测（周期快照写 JSONL，用来算命中率/换出量/换回量）：

    KVTRACE_POLICY_OUT=/tmp/policy.jsonl vllm serve ...
"""

import heapq
import json
import math
import os
import sys
import threading
import time
import zlib
from collections import OrderedDict, deque
from collections.abc import Iterable
from typing import override

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy

# ================================================================
# 自适应衰减常数 tau 的在线估计
# ================================================================
#
# tau 是 adaptive 唯一的超参，也是它唯一会**静默失败**的地方：
#
#   tau → ∞   衰减项恒为 1，退化成纯 LFU
#   tau → 0   老块的 score 迅速归零，退化成 LRU
#
# 两种退化都不报错，报表上看不出异常 —— 离线 replay 就踩过这一版，报告里只
# 表现为"adaptive 和 LRU 居然一样"（见 scripts/replay.py:auto_tau）。
# 所以这里**不拍一个常数**，而是按 replay 用的同一定义在线估计：
# "同一个块相邻两次被使用的时间间隔"的中位数 —— 这才是"热"的时间尺度。

_TAU_SEED_S = 1.0          # 样本不足时的兜底值
_TAU_MIN_SAMPLES = 32      # 少于这么多样本就不信中位数
_TAU_WINDOW = 4096         # 滑动窗口内的间隔样本数
_TAU_RECOMPUTE_EVERY = 256  # 中位数是 O(k log k)，攒一批再算
# 低于这个间隔的不算"复用"。必须过滤：一次 load 会连着触发 touch 和
# mark_evictable，两者相隔几微秒，喂进估计器会把 tau 直接压到 ~0，
# adaptive 就**静默退化成 LRU** —— 正是上面说的那种失败。
_TAU_MIN_GAP_S = 1e-3


class _TauEstimator:
    """块被复用的间隔中位数，在线滑动估计。"""

    def __init__(self) -> None:
        self.gaps: deque[float] = deque(maxlen=_TAU_WINDOW)
        self._since = 0
        self.n = 0
        self.value = _TAU_SEED_S

    def observe(self, gap_s: float) -> None:
        if gap_s < _TAU_MIN_GAP_S:
            return
        self.gaps.append(gap_s)
        self.n += 1
        self._since += 1
        if self.n >= _TAU_MIN_SAMPLES and self._since >= _TAU_RECOMPUTE_EVERY:
            self._since = 0
            self.value = sorted(self.gaps)[len(self.gaps) // 2]

    @property
    def ready(self) -> bool:
        return self.n >= _TAU_MIN_SAMPLES


# ================================================================
# 频次类策略的公共骨架
# ================================================================

class _FreqCachePolicy(CachePolicy):
    """按"使用频次"逐出的公共部分，两个子类只差 score 怎么算。

    **`freq` / `last` 故意不放进 evictable 结构里。** 离线 replay 踩过这个坑：
    频次塞在池子记录里，块一被钉住（从池里 pop 出来）频次就丢了，释放回池时又从
    0 开始 —— LFU 会静悄悄地退化成"随便挑一个"，指标上还看不出异常
    （scripts/replay.py 里那段注释）。这里 `BlockStatus.ref_cnt > 0` 就是"钉住"：
    这些块不在 evictable 里，但频次必须留着。

    `freq` / `last` 只按 key 增长，要自己设上界。vLLM 的 `counts` 用的是
    `max_tracker_size` 的 LRU 裁剪，这里照抄同一个思路。

    "一次使用"的口径**和 `LRUCachePolicy` 逐字对齐**：insert（store 发出）、
    mark_evictable（store/load 完成）、mark_non_evictable（load 开始，把块钉住）。
    所以 `freq` 是 LRU 内部"最近使用"顺序的同一个事件流的计数版本，两者可比。
    一次 load 会触发 mark_non_evictable + mark_evictable 两次，所以每个复用周期
    通常计 2 次 —— 单调且保序，不影响 LFU 的比较结果，但别把它当成"被命中了几
    次"直接引用。

    `touch` 也算一次使用，但它传的是**整个请求**的 key 列表而不只是命中的那段，
    所以那里必须自己判在不在池子里（见 `touch` 的注释）—— 不判的话会给从没进过
    池子的块记频次，把 `_seen` 挤满，反而把真在池子里的块的频次挤掉。
    """

    def __init__(self, cache_capacity: int, max_tracker_size: int = 64_000):
        self.cache_capacity = cache_capacity
        self.max_tracker_size = max_tracker_size
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        # 只有 ref_cnt == 0 的块在里面 —— 和 LRUCachePolicy 的口径一致。
        self.evictable: dict[OffloadKey, None] = {}
        self.freq: dict[OffloadKey, float] = {}
        self.last: dict[OffloadKey, float] = {}
        # `last` 的插入顺序，用来做上界裁剪（OrderedDict 天然是 LRU 序）。
        self._seen: OrderedDict[OffloadKey, None] = OrderedDict()
        # 块"闲着没人要"的起点（见 `_use`）。不是 `last` 的别名。
        self._idle_since: dict[OffloadKey, float] = {}
        self.n_uses = 0
        self.n_evicted = 0

    # --- 内部 ---

    def _use(self, key: OffloadKey) -> None:
        """记一次使用。**只有真正的复用间隔才喂给 `_on_reuse`。**

        这里有个必须绕开的坑：**"相邻两次使用的间隔"不是复用间隔**。产生使用事件
        的有 insert / mark_evictable / touch，其中前两个之间只隔着一次 store 传输
        （几十毫秒），而 `touch` 在每次请求 lookup 后都会对整张 key 列表跑一遍 ——
        并发请求共用同一个热前缀时，同一个块会在几十毫秒内被连着 touch 好几次。

        早先就是按"相邻两次使用的间隔"算的，中位数被这类**短间隔**占满：实测
        tau = 63 ms（n = 2769），而真正的复用尺度是秒级。

        后果是静默的：`exp(-age/tau)` 在 tau = 63 ms 时，任何歇了半秒以上的块
        指数项都下溢成 0，逐出退化成"挑最旧的"，**adaptive 跑出来和 LRU 一模一样**
        —— 实测四个策略的计数正好两两相等（adaptive≡lru、arc≡lfu，逐位相同），
        这不是巧合，是 adaptive 已经变成 LRU 了。

        正确口径是**空闲时长**：块被放回池子（`mark_evictable`，ref_cnt 归零）到
        它下次被需要之间的时间。这才是"它闲了多久才又被要"，也就是 tau 该量的
        东西 —— 逐出决策面对的正是"这个块现在是闲的，它还要闲多久"。
        实现上就是在每个"释放"点打一个起点，下一次"使用"时结算。
        """
        now = time.monotonic()
        self.n_uses += 1
        self._settle_idle(key)
        self.last[key] = now
        self.freq[key] = self.freq.get(key, 0.0) + 1.0
        if key in self._seen:
            self._seen.move_to_end(key)
        else:
            self._seen[key] = None
            while len(self._seen) > self.max_tracker_size:
                old, _ = self._seen.popitem(last=False)
                self.freq.pop(old, None)
                self.last.pop(old, None)

    def _settle_idle(self, key: OffloadKey) -> None:
        """块从"闲着"变成"又被要了"，结算这段空闲时长。只为喂 tau 估计器。

        和 `_use` 分开是必须的：空闲时长要在**块"又被要了"的那一刻**结算才准
        （`touch` 或 `mark_non_evictable`，看哪个先到），但频次的口径必须和
        `LRUCachePolicy` 逐字一致。`mark_non_evictable` 那条路径上顺手 +1 会让一次
        load 从计 1 次变成计 2 次，`_FreqCachePolicy` 就不再是离线 replay 里那个
        被差分测试验过等价的 LFU 了（LFU/adaptive 各 400 余次逐出、0 分歧）。
        一次只改一件事。
        """
        idle = self._idle_since.pop(key, None)
        if idle is not None:
            self._on_reuse(time.monotonic() - idle)

    def _on_reuse(self, gap_s: float) -> None:
        """子类可用来喂 tau 估计器。"""

    def _score(self, key: OffloadKey, now: float) -> float:
        raise NotImplementedError

    # --- CachePolicy 接口 ---

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        if block.ref_cnt == 0:
            self.evictable[key] = None
        self._use(key)

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        self.evictable.pop(key, None)
        # 被换走之后就没有"复用间隔"可谈了：它再回来是一次 miss，不是命中。
        # 留着起点会把之后的重新存入算成一个巨大的间隔，把 tau 往上拽。
        self._idle_since.pop(key, None)

    @override
    def touch(self, keys: Iterable[OffloadKey]) -> None:
        """命中路径的"最近用过"标记。

        调用者是 offloading connector 的 scheduler：`_lookup` 之后每次请求都调
        一次（`kv_connector/v1/offloading/scheduler.py:691`），store 路径也调
        （同文件 `:935`）。**传进来的 `keys` 是整个请求的 offload key 列表，不是
        只有命中的那一段** —— 所以必须自己判在不在池子里。

        `if key in self.evictable` 这道守卫和 `LRUCachePolicy.touch` 一致（它写的是
        `if key in self.evictable_blocks`），不是可有可无：少了它，请求里那些**从没
        进过池子**的块也会被记进 `freq` / `last` / `_seen`。而 `_seen` 是有上界的
        LRU（`max_tracker_size`），被这些块灌满之后就会把**真正在池子里**的块的
        频次挤掉 —— 频次归零，LFU 悄悄退化成"随便挑一个"。这正是
        `scripts/replay.py` 里记过的那个坑，只是换了个入口。

        按 reversed 走：`keys` 是前缀顺序，排在前面的块是更靠前的公共前缀、复用
        价值更高，所以要让它显得"最近"。LRU 用 move_to_end，这里用 last 的时间戳
        —— 但同一批里 time.monotonic() 的分辨率会把它们打平，于是显式加一个随 i
        递增的微小偏置。
        """
        now = time.monotonic()
        for i, key in enumerate(reversed(list(keys))):
            if key not in self.evictable:
                continue
            self._use(key)
            self.last[key] = now + i * 1e-9

    @override
    def clear(self) -> None:
        self.blocks.clear()
        self.evictable.clear()
        self.freq.clear()
        self.last.clear()
        self._seen.clear()
        self._idle_since.clear()

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []
        now = time.monotonic()
        # 原子性契约（base.py）：凑不齐 n 个就返回 None 且**不改状态**。
        candidates = [key for key in self.evictable if key not in protected]
        if len(candidates) < n:
            return None
        # O(m log n) 而不是全排序：池子大、逐出少时差别很明显。
        picked = heapq.nsmallest(
            n, candidates,
            key=lambda k: (self._score(k, now), self.last.get(k, 0.0), k))
        out: list[tuple[OffloadKey, BlockStatus]] = []
        for key in picked:
            block = self.blocks[key]
            assert block.ref_cnt == 0
            del self.evictable[key]
            del self.blocks[key]
            self._idle_since.pop(key, None)
            out.append((key, block))
        self.n_evicted += len(out)
        return out

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        if key in self.blocks:
            self.evictable[key] = None
            self._use(key)
            # ref_cnt 归零 = 块回到池子里待着，空闲计时从这一刻开始。
            self._idle_since[key] = time.monotonic()

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        """`prepare_load` 把块钉住（ref_cnt 0 -> 1）时调用 —— 这是"它又被要了"的
        那一刻，空闲时长在这里结算最准（等到 `complete_load` 会把换回传输的时间
        也算进空闲里）。只结算空闲，不记频次，理由见 `_settle_idle`。

        实测这个 pin 点通常是**后到的**：`touch` 在 lookup 之后就跑过了
        （`kv_connector/v1/offloading/scheduler.py:691`），空闲时长多半在那一刻
        就已经结算掉了。留这一手是为了覆盖 touch 漏掉的块 —— 配了 sliding window
        的组只 touch 尾部一段（同文件 `:456-461`），前面那些不会被 touch 到。
        """
        if key in self.blocks:
            self._settle_idle(key)

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        self.evictable.pop(key, None)

    # --- 供观测 ---

    @property
    def tau(self) -> float | None:
        return None


class LFUCachePolicy(_FreqCachePolicy):
    """最少使用次数者先走；次数相同则最久没用的先走。

    和离线 replay 的 `Evictor("lfu")` 同构：`min((freq, last_access_t))`。
    """

    @override
    def _score(self, key: OffloadKey, now: float) -> float:
        return self.freq.get(key, 0.0)


class AdaptiveCachePolicy(_FreqCachePolicy):
    """频次 × 时间衰减；score 最小的先走。

        score = freq × exp(-(now - last) / tau)

    tau 由 `_TauEstimator` 在线估计。这个形式把 LRU 和 LFU 当成两个极端包含进来
    （tau → ∞ 退化成 LFU，tau → 0 退化成 LRU），所以"adaptive 比两者都好/都差"
    是有信息量的，而不是在比一个随手拍出来的启发式。
    """

    def __init__(self, cache_capacity: int, max_tracker_size: int = 64_000):
        super().__init__(cache_capacity, max_tracker_size)
        self.tau_est = _TauEstimator()

    @override
    def _on_reuse(self, gap_s: float) -> None:
        self.tau_est.observe(gap_s)

    @override
    def _score(self, key: OffloadKey, now: float) -> float:
        age = now - self.last.get(key, now)
        return self.freq.get(key, 0.0) * math.exp(-age / self.tau_est.value)

    @property
    @override
    def tau(self) -> float | None:
        return self.tau_est.value


# ================================================================
# 注册
# ================================================================

POLICIES: dict[str, type[CachePolicy]] = {
    "lfu": LFUCachePolicy,
    "adaptive": AdaptiveCachePolicy,
}


def install_policies() -> None:
    """把两个策略补进 vLLM 的 `_CACHE_POLICIES`，并盯住最终选中的是哪一个。"""
    from vllm.v1.kv_offload.cpu import manager as mgr

    for name, cls in POLICIES.items():
        if name in mgr._CACHE_POLICIES:
            print(f"[KVPolicy][pid={os.getpid()}] 跳过 {name!r}：vLLM 已自带")
            continue
        mgr._CACHE_POLICIES[name] = cls
        print(f"[KVPolicy][pid={os.getpid()}] 已注册策略 {name!r} -> {cls.__name__}")

    _instrument_policy_choice()


def _policy_probe(policy) -> dict:
    """两次快照之间顺带记下的策略内部状态。

    加 `tau` 是因为它会**静悄悄地退化**：adaptive 的分数是
    `freq × exp(-age/tau)`，`tau → 0` 时指数项除最近一块外全下溢成 0，逐出就变成
    "挑最旧的" —— 也就是 LRU。这时它跑出来和 LRU 一模一样，而客观指标上没有任何
    异常。不看 tau，就分不清"adaptive 和 LRU 打平"和"adaptive 根本就是 LRU"。
    """
    out = {}
    est = getattr(policy, "tau_est", None)
    if est is not None:
        out["tau_s"] = round(est.value, 4)
        out["tau_n"] = est.n
    for name in ("n_uses", "n_evicted"):
        v = getattr(policy, name, None)
        if isinstance(v, int):
            out[name] = v
    ev = getattr(policy, "evictable", None)
    if ev is not None:
        out["pool_evictable"] = len(ev)
    return out


def _instrument_policy_choice() -> None:
    """把"这一轮到底用了哪个策略"打出来，并**在这里**装观测器。

    两件事都挂在 manager 的构造函数上，理由不同：

    1. **策略名字**。只在自定义策略的 `__init__` 里打日志是不够的：选 lru/arc
       时走的是 vLLM 自带的类，那行**永远不会出现**。于是"没看到"和"没生效"在
       脚本里长得一模一样 —— 实测就是这样，闸门脚本把 lru 误判成"策略没生效"
       直接中止。挂在这里，四种策略就都有同一行证据。

    2. **观测器**。`register()` 在**每个** vLLM 进程里跑，包括 frontend；而
       frontend 从来不建 manager。早先在 `register()` 里直接开文件，结果
       frontend 和 EngineCore **两个进程往同一个 JSONL 里交错写**，frontend 那
       份是恒为 0 的计数器 —— 分析脚本读到哪一行全看运气，实测读到的全是 0，
       看起来就像"四个策略完全一样"。挂在构造函数上，写文件的人必然是有
       manager 的那个进程。

    顺带把 `store_threshold` 带出来：它 >= 2 时 vLLM 自己就在做按频次**准入**
    （见文件头），那是和逐出策略相互作用的一个开关，不该是默认值里看不见的。
    """
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

    out = os.environ.get("KVTRACE_POLICY_OUT")
    orig_init = CPUOffloadingManager.__init__
    state = {"observer": None}

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        actual = type(self._policy).__name__
        print(f"[KVPolicy][pid={os.getpid()}] 实际逐出策略 = "
              f"{actual}"
              f"（容量 {self._num_blocks} 块，store_threshold="
              f"{self.store_threshold}）")
        if out and state["observer"] is None:
            # 用 manager 里**实际**的策略类名，不用环境变量：环境变量说的是
            # "想要哪个"，这里记的是"真的是哪个"。两者不一致时，计数文件自己
            # 就说明了问题，不必去翻日志。
            obs = _install_observer(out, actual)
            obs.probe = (lambda p: lambda: _policy_probe(p))(self._policy)
            state["observer"] = obs

    CPUOffloadingManager.__init__ = __init__


# ================================================================
# 观测：把命中/搬运/逐出的计数落盘
# ================================================================
#
# 只在 KVTRACE_POLICY_OUT 被设置时才装：scheduler 的 `_maximal_prefix_lookup`
# 是**逐块**循环调 lookup() 的，给热路径加一层 Python 调用不算免费。
#
# 落盘形式是**周期快照**而不是逐事件：策略对比要的是命中率/换出量/换回量这些
# 累计量，逐块写盘会把磁盘 IO 变成测量的一部分。文件里既能取最后一次当总量，
# 也能两行相减当速率。
#
# 时钟用 time.time_ns()：perf_counter 的原点逐进程不同，这个仓库已经被咬过一次
# （commit d3207c4）。

_SNAPSHOT_EVERY_N = 2000
_SNAPSHOT_EVERY_S = 2.0


class _PolicyObserver:
    def __init__(self, path: str, policy_name: str):
        self.path = path
        self.policy_name = policy_name
        self.fh = open(path, "a", buffering=1)  # noqa: SIM115
        self.counters = {
            "lookup_hit": 0, "lookup_pending": 0, "lookup_miss": 0,
            "store_blocks": 0, "store_evicted": 0, "store_dropped": 0,
            "load_blocks": 0,
        }
        # 逐出序列的指纹（顺序敏感）。**计数相等不等于决策相同** —— 实测四个
        # 策略正好两两相等（arc≡lfu、lru≡adaptive，各 3 次重复逐位一致），那可能
        # 是真的一样，也可能只是这些计数太粗、把不同的决策序列压成了同一个数。
        # 指纹能分开这两种情况：一样就是真的一样，不一样就说明计数在骗人。
        #
        # 用 crc32 而不是内置 hash()：后者对 str 加了逐进程随机盐
        # （PYTHONHASHSEED），跨进程根本不可比。
        self.evict_fp = 0
        self._fp_n = 0
        self._since_n = 0
        self._since_t = time.monotonic()
        # 每轮快照额外附带的字段，由装观测器的人给（拿得到策略对象才拿得到）。
        self.probe = None
        # 定时刷盘。只在 bump() 里检查阈值是不够的：一轮 workload 结束后就没有
        # 新事件了，最后最多 2000 次计数会永远留在内存里，外面读到的最后一行是
        # **陈旧**的 —— 而"跑完读最后一行"正是分析脚本的取数方式。
        self._stop = False
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()

    def _tick(self) -> None:
        while not self._stop:
            time.sleep(_SNAPSHOT_EVERY_S)
            try:
                self.flush()
            except Exception:  # noqa: BLE001
                return

    def fingerprint(self, keys) -> None:
        """把一批被逐出的 key 折进指纹。"""
        for k in keys:
            self.evict_fp = zlib.crc32(repr(k).encode(), self.evict_fp)
            self._fp_n += 1

    def bump(self, key: str, n: int = 1) -> None:
        self.counters[key] += n
        self._since_n += n
        if (self._since_n >= _SNAPSHOT_EVERY_N
                or time.monotonic() - self._since_t >= _SNAPSHOT_EVERY_S):
            self.flush()

    def flush(self) -> None:
        self._since_n = 0
        self._since_t = time.monotonic()
        rec = {"t": time.time_ns(), "pid": os.getpid(),
               "policy": self.policy_name, "evict_fp": self.evict_fp,
               **self.counters}
        if self.probe is not None:
            try:
                rec.update(self.probe())
            except Exception:  # noqa: BLE001
                self.probe = None  # 探针坏了不该拖垮服务
        self.fh.write(json.dumps(rec) + "\n")


def _install_observer(path: str, policy_name: str) -> "_PolicyObserver":
    from vllm.v1.kv_offload.base import LookupResult
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

    obs = _PolicyObserver(path, policy_name)

    orig_lookup = CPUOffloadingManager.lookup
    orig_prepare_store = CPUOffloadingManager.prepare_store
    orig_prepare_load = CPUOffloadingManager.prepare_load

    def lookup(self, key, req_context):
        r = orig_lookup(self, key, req_context)
        if r is LookupResult.HIT:
            obs.bump("lookup_hit")
        elif r is LookupResult.HIT_PENDING:
            obs.bump("lookup_pending")
        else:
            obs.bump("lookup_miss")
        return r

    def prepare_store(self, keys, req_context):
        out = orig_prepare_store(self, keys, req_context)
        if out is None:
            # 凑不齐可逐出的块 —— vLLM 会放弃这一批 store。
            obs.bump("store_dropped")
            return out
        obs.bump("store_blocks", len(out.keys_to_store))
        obs.bump("store_evicted", len(out.evicted_keys))
        obs.fingerprint(out.evicted_keys)
        return out

    def prepare_load(self, keys, req_context):
        spec = orig_prepare_load(self, keys, req_context)
        obs.bump("load_blocks", len(keys))
        return spec

    CPUOffloadingManager.lookup = lookup
    CPUOffloadingManager.prepare_store = prepare_store
    CPUOffloadingManager.prepare_load = prepare_load
    print(f"[KVPolicy][pid={os.getpid()}] 观测已开启 -> {path}")
    return obs


def register() -> None:
    """vLLM ``vllm.general_plugins`` 入口点调用的函数，每个进程一次。

    **不在这里开观测文件。** 这个函数在每个 vLLM 进程里都跑一次，包括根本不建
    manager 的 frontend；早先在这里开文件，结果是 frontend 和 EngineCore 两个
    进程往同一个 JSONL 里交错写，frontend 那份计数器恒为 0，分析脚本取到哪一行
    全看运气 —— 实测取到的全是 0，四个策略看起来一模一样。观测器改由
    `_instrument_policy_choice` 挂在 manager 的构造函数上：谁有 manager，谁才
    写文件。
    """
    try:
        install_policies()
    except Exception as e:  # noqa: BLE001
        # 版本对不上、或 frontend 进程里 import 失败，都不该让整个 serve 起不来。
        print(f"[KVPolicy][pid={os.getpid()}] 注册失败：{e!r}", file=sys.stderr)
