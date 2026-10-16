#!/usr/bin/env python3
"""
test_e2e.py — gpu-swap 端到端测试（真库 / 模拟两条路）

模拟 LLM 推理过程中 KV Cache block 的分配/访问/释放模式，
驱动 gpu-swap 并统计 miss 率、换出次数、峰值利用率。

用法:
    python scripts/test_e2e.py --real --strategy adaptive    # 走 ctypes 真库
    python scripts/test_e2e.py --mock --strategy adaptive    # 纯 Python 模拟

两条路跑**同一个** workload 生成器、**同一套**策略常数（从
include/policy_thresholds.h 现场解析，不是抄一遍），所以趋势可比。
绝对值允许不同：模拟器按字节线性计价，漏掉每个描述符约 0.44 µs 的
固定开销（见下 DEFAULT_PCIE_LATENCY_US 的注释）。

⚠️ 别把模拟器当权威。它不碰 CUDA，测不到真实迁移的任何一种失效
（同步点、带宽争抢、假锁）。权威结果一律来自 `--real`。
"""

import argparse
import ctypes
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gsm import GpuSwap, GpuSwapError       # noqa: E402  (真库绑定)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THRESHOLDS_H = os.path.join(_REPO, "include", "policy_thresholds.h")


# ================================================================
# 策略常数：从生成的头文件里读，**不要**在这里抄一遍
# ================================================================

def load_thresholds(path=_THRESHOLDS_H):
    """解析 policy_thresholds.h 里的 GSM_* 常数。

    这个文件是 scripts/derive_thresholds.py 从项目一的 trace 现算出来的。
    抄一份到 Python 里就等于埋了一个"两边的 50 不是同一个 50"的坑 ——
    这个项目已经因为"同一个名字、两个值"栽过（analyze.py 用 >50、
    compare_runs.py 用 >100）。
    """
    out = {}
    if not os.path.exists(path):
        raise SystemExit(
            f"找不到 {path}。跑 `make thresholds` 从项目一的 trace 生成。")
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(
                r"static constexpr\s+(int64_t|double)\s+(GSM_\w+)\s*=\s*"
                r"([0-9.]+)\s*;", line.strip())
            if m:
                kind, name, val = m.groups()
                out[name] = int(val) if kind == "int64_t" else float(val)
    for req in ("GSM_HOT_ACCESS_THRESHOLD", "GSM_MAX_PROTECTION_RATE",
                "GSM_YOUNG_AGE_MS", "GSM_RECOMPUTE_US_PER_BLOCK",
                "GSM_SWAP_IN_US_PER_BLOCK"):
        if req not in out:
            raise SystemExit(f"{path} 里没有 {req} —— 生成脚本改过格式？")
    return out


TH = load_thresholds()


# ================================================================
# 策略（Python 侧镜像；权威实现在 include/eviction_policy.h）
# ================================================================

class LRUPolicy:
    """最近最少使用——选 last_access 最小的。（C++ 侧有 (last_access,
    block_id) 的字典序 tie-break 保证可复现；这里靠 min() 的扫描序，
    效果等价：只在 last_access 相等时才由 block_id 决定。）"""

    def select_victim(self, gpu_blocks):
        if not gpu_blocks:
            return None
        return min(gpu_blocks, key=lambda b: (b.last_access, b.block_id)).block_id


class LFUPolicy:
    """最不频繁使用——选 access_count 最小的"""

    def select_victim(self, gpu_blocks):
        if not gpu_blocks:
            return None
        return min(gpu_blocks,
                   key=lambda b: (b.access_count, b.last_access, b.block_id)).block_id


class GDSFPolicy:
    """代价加权的频率密度：score = (base - origin) + cost_per_hit*hits/size_kb

    GDSF（Cherkasova 的 GD-Size-Frequency）。`base` 是块进入热池时的通胀
    水位 L，此后冻结；每次逐出 L ← 被逐出者的分数（GreedyDual 原文，**赋值**）。

    ⚠️ 通胀 L 是单调递增的，无界增长后会压过频率项，把排序退化成
    "逐出最老的"（≈ FIFO/LRU）。实测（60 请求 / 254 槽）：L 从 64 涨到 214，
    miss 从 34.7% 涨到 74.0%，而同期 LRU 是 90.7% —— GDSF 悄悄变成了 LRU。
    所以加一条**不变量**：`L - origin ≤ max(频率项 over 驻留块)`，
    超了就 O(1) 平移到 0（`origin = L`，不改任何两块之间的相对次序）。
    修完 74.0% → 60.0%。重基准次数计入报告。C++ 侧同规则，见 eviction_policy.h。
    """
    COST_PER_HIT_US = (TH["GSM_RECOMPUTE_US_PER_BLOCK"]
                       - TH["GSM_SWAP_IN_US_PER_BLOCK"])     # 31x 的代价比

    def __init__(self):
        self.inflation = 0.0     # 绝对值
        self.origin = 0.0        # 基准；生效的通胀 = inflation - origin
        self.rebases = 0
        self.base = {}           # block_id → 进入热池时的通胀水位（绝对）

    def on_alloc(self, block_id):
        self.base[block_id] = self.inflation

    def on_free(self, block_id):
        self.base.pop(block_id, None)

    def _freq(self, b):
        return self.COST_PER_HIT_US * b.access_count / max(b.size, 1) * 1024.0

    def _score(self, b):
        return (self.base.get(b.block_id, 0.0) - self.origin) + self._freq(b)

    def select_victim(self, gpu_blocks):
        if not gpu_blocks:
            return None
        best = None
        max_freq = 0.0
        for b in gpu_blocks:
            f = self._freq(b)
            max_freq = max(max_freq, f)
            cand = (self._score(b), b.last_access, b.block_id)
            if best is None or cand < best:
                best = cand
        self.inflation = best[0] + self.origin
        if self.inflation - self.origin > max_freq:
            self.origin = self.inflation
            self.rebases += 1
        return best[2]

    def report(self):
        return {"inflation": round(self.inflation - self.origin, 2),
                "inflation_rebases": self.rebases}


class AdaptivePolicy:
    """
    基于 trace 数据的混合策略 —— **C++ AdaptivePolicy 的镜像**。

    规则（三层，逐级松手而不是掉进一个装作 LRU 的 fallback）：
      1. GDSF 打分排序（连续量，永远有最小值 → 构造上不可能"选不出候选"）
      2. 年龄规则：now - alloc_time < young_age_ms 的块优先不动
         （解释是"刚分配 = 马上还会被读"，不是"短命" —— 生命周期在线不可判）
      3. 热块规则：access_count > threshold 的块优先不动。绝对阈值只做
         **次级保险**，保护率超过 protect_max_rate 就夹断到相应分位。

    ⚠️ 骨架版本（v1）长这样：
           if b.access_count > 50: continue
           ... 在剩下的里面挑 last_access 最小的
       三个问题实测都复现过：常数不可迁移（T=50 在 long-summarize 上保护
       98.9%、在 legacy-c1 上只保护 27.0%）、候选集空时静默退化成 LRU、
       把"访问多"和"换出贵"混成一个数。
    """

    def __init__(self,
                 hot_threshold=TH["GSM_HOT_ACCESS_THRESHOLD"],
                 young_age_ms=TH["GSM_YOUNG_AGE_MS"],
                 protect_max_rate=TH["GSM_MAX_PROTECTION_RATE"]):
        self.hot_threshold = hot_threshold
        self.young_age_ms = young_age_ms
        self.protect_max_rate = protect_max_rate
        self.core = GDSFPolicy()
        self.now = 0
        # 统计：退化要**看得见**
        self.decisions = 0
        self.clamped = 0
        self.relaxed_hot = 0
        self.relaxed_young = 0
        self.protect_rates = []
        self.young_block_hits = 0
        self.hot_block_hits = 0
        self.resident_seen = 0
        self.cutoff_last = hot_threshold
        self.cutoff_clamped_last = hot_threshold

    def on_alloc(self, block_id):
        self.core.on_alloc(block_id)

    def on_free(self, block_id):
        self.core.on_free(block_id)

    def set_clock(self, now_ms):
        self.now = now_ms

    def select_victim(self, gpu_blocks):
        if not gpu_blocks:
            return None
        n = len(gpu_blocks)

        # ---- 热块阈值：绝对阈值 + 夹断 ----
        # 目标是"保护不超过 protect_max_rate"，cutoff 越大保护越少，
        # 所以夹断 = 把 cutoff 抬到计数分布的第 (1-q) 分位。
        cutoff = self.hot_threshold
        over = sum(1 for b in gpu_blocks if b.access_count > cutoff)
        if over > self.protect_max_rate * n:
            counts = sorted(b.access_count for b in gpu_blocks)
            k = min(int((1.0 - self.protect_max_rate) * n), n - 1)
            cutoff = counts[k]
            self.clamped += 1
        self.cutoff_last = self.hot_threshold
        self.cutoff_clamped_last = cutoff

        # ---- 逐块分类 ----
        #
        # ⚠️ 这里曾经写错，而且**从输出上看不出来**：本来的写法是
        #     fresh = [...]; young_hits = len(fresh); candidates = [年轻的那批]
        # 于是 `young_hits` 装的是"非热块的**总数**"，不是"年轻的个数" ——
        # 名字和内容对不上。上报的保护率因此变成 `1 - 热块占比`，
        # 和 C++ 侧的 `(young + hot)/n` 是两回事，两者差了 47 个百分点
        # （实测 0.368 vs 0.8385）却都"看起来合理"。
        # 真库跑出 0.368、模拟器跑出 0.8385 才露馅。
        fresh = [b for b in gpu_blocks if b.access_count <= cutoff]
        hot_hits = n - len(fresh)
        candidates = [b for b in fresh
                      if self.now - b.alloc_time >= self.young_age_ms]
        young_hits = len(fresh) - len(candidates)

        # ---- 逐级松手 ----
        if not candidates:
            self.relaxed_hot += 1
            candidates = fresh                      # 放开热块规则
        if not candidates:
            self.relaxed_young += 1                 # 连年龄也放开
            candidates = list(gpu_blocks)

        self.decisions += 1
        self.young_block_hits += young_hits
        self.hot_block_hits += hot_hits
        self.resident_seen += n
        self.protect_rates.append((young_hits + hot_hits) / n)
        return self.core.select_victim(candidates)

    def report(self):
        r = {"decisions": self.decisions, "clamped": self.clamped,
             "relaxed_hot": self.relaxed_hot,
             "relaxed_young": self.relaxed_young,
             "young_block_hits": self.young_block_hits,
             "hot_block_hits": self.hot_block_hits,
             "resident_seen": self.resident_seen,
             "inflation": round(self.core.inflation - self.core.origin, 2),
             "inflation_rebases": self.core.rebases}
        if self.protect_rates:
            s = self.protect_rates          # **时间序**，绝不排序
            r["protect_mean"] = round(sum(s) / len(s), 4)
            r["protect_min"] = round(min(s), 4)
            r["protect_max"] = round(max(s), 4)
            # 分位按**时间**切，跟 C++ 的 quartile_mean 逐字对齐。
            # 之前这里先 sorted() 再切 —— 那切的是"取值分布"，报出来是
            # 0.7576 / 0.9569；时间序切出来是 0.3976 / 0.1869。
            # 一个在上行一个在下行，**结论正好相反**，而两边的数字都自洽。
            def qmean(lo, hi):
                a = int(lo * len(s))
                b = min(max(a + 1, int(hi * len(s))), len(s))
                return round(sum(s[a:b]) / max(1, b - a), 4)
            r["protect_first_quartile"] = qmean(0.00, 0.25)
            r["protect_last_quartile"] = qmean(0.75, 1.00)
        return r


POLICIES = {"lru": LRUPolicy, "lfu": LFUPolicy,
            "gdsf": GDSFPolicy, "adaptive": AdaptivePolicy}


# ================================================================
# 后端 1：模拟器（不碰 CUDA）
# ================================================================

# 实测值（2026-10，RTX 4090 + EPYC 7B13）。测量与复现见
# kvcache-tracer/measurements/pcie_link_ceiling.txt
# （kvcache-tracer/scripts/measure_pcie.py）。
#
# 256 KB 作为 1 个描述符：262,144 / 23.4 GB/s + 0.44 µs ≈ 12 µs
#
# 旧值 25 µs 反推只有 10.5 GB/s，且注释声称来自"你的 Step 0 PCIe
# benchmark" —— 那个 benchmark 在本仓里并不存在，数字无处可考。
#
# ⚠️ 这个模型是**按字节**线性计价的，漏掉了每个描述符约 0.44 µs 的固定开销。
# 真实 KV 迁移是每层一个描述符（vLLM 在 Qwen2.5-7B @ block-size 16 下
# 一个 917,504 B 的块 = 28 个 32,768 B 描述符），实测 52.4 µs，
# 比按带宽算出来的 39 µs 高 33%。描述符一多，误差就跟着涨。
DEFAULT_PCIE_LATENCY_US = 12   # D→H 一次 256KB


class _Block:
    __slots__ = ("block_id", "size", "in_gpu", "last_access",
                 "access_count", "alloc_time")

    def __init__(self, block_id, size, clock):
        self.block_id = block_id
        self.size = size
        self.in_gpu = True
        self.last_access = clock
        self.access_count = 0
        self.alloc_time = clock


class MockSwapManager:
    """模拟的 GPU Swap 管理器——逻辑和 C++ 版本一致，但不依赖 CUDA。

    用途只有两个：没有 GPU 的机器上冒烟、以及快速迭代策略设计。
    **它测不到真实迁移的任何一种失效模式**，所以结论不能只靠它。
    """

    def __init__(self, hot_pool_bytes, cold_pool_bytes, policy,
                 pcie_latency_us=DEFAULT_PCIE_LATENCY_US):
        self.hot_capacity = hot_pool_bytes
        self.cold_capacity = cold_pool_bytes
        self.hot_used = 0
        self.cold_used = 0
        self.hot_peak = 0
        self.policy = policy
        self.pcie_latency_us = pcie_latency_us
        self.blocks = {}
        self.clock = 0

        self.total_evictions = 0
        self.total_prefetches = 0
        self.total_eviction_us = 0
        self.total_prefetch_us = 0
        self.miss_count = 0
        self.hit_count = 0

    def set_clock(self, now_ms):
        """注入式时钟（毫秒）。**不是**每个事件加 1 —— 那样 young_age_ms
        这个以毫秒为单位的阈值会退化成"0 个 tick"，年龄规则直接失效。"""
        self.clock = now_ms
        if hasattr(self.policy, "set_clock"):
            self.policy.set_clock(now_ms)

    def allocate(self, block_id, size):
        while self.hot_used + size > self.hot_capacity:
            victim_id = self.policy.select_victim(self._evictable())
            if victim_id is None:
                return None
            self._evict(victim_id)
        blk = _Block(block_id, size, self.clock)
        self.blocks[block_id] = blk
        self.hot_used += size
        self.hot_peak = max(self.hot_peak, self.hot_used)
        if hasattr(self.policy, "on_alloc"):
            self.policy.on_alloc(block_id)
        return blk

    def access(self, block_id):
        blk = self.blocks.get(block_id)
        if blk is None:
            return
        blk.last_access = self.clock
        blk.access_count += 1
        if blk.in_gpu:
            self.hit_count += 1
        else:
            self.miss_count += 1
            self._prefetch(block_id)

    def free(self, block_id):
        blk = self.blocks.pop(block_id, None)
        if blk is None:
            return
        if blk.in_gpu:
            self.hot_used -= blk.size
        else:
            self.cold_used -= blk.size
        if hasattr(self.policy, "on_free"):
            self.policy.on_free(block_id)

    def _evictable(self):
        return [b for b in self.blocks.values() if b.in_gpu]

    def _evict(self, block_id):
        blk = self.blocks[block_id]
        if not blk.in_gpu:
            return
        blk.in_gpu = False
        self.hot_used -= blk.size
        self.cold_used += blk.size
        self.total_evictions += 1
        self.total_eviction_us += self.pcie_latency_us * max(
            1, blk.size // (256 * 1024))

    def _prefetch(self, block_id):
        blk = self.blocks[block_id]
        if blk.in_gpu:
            return
        while self.hot_used + blk.size > self.hot_capacity:
            others = [b for b in self._evictable() if b.block_id != block_id]
            victim_id = self.policy.select_victim(others)
            if victim_id is None:
                return
            self._evict(victim_id)
        blk.in_gpu = True
        self.hot_used += blk.size
        self.hot_peak = max(self.hot_peak, self.hot_used)
        self.cold_used -= blk.size
        self.total_prefetches += 1
        self.total_prefetch_us += self.pcie_latency_us * max(
            1, blk.size // (256 * 1024))

    def report(self):
        total = self.hit_count + self.miss_count
        r = {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "miss_rate_pct": round(self.miss_count / total * 100, 2) if total else 0.0,
            "total_evictions": self.total_evictions,
            "total_prefetches": self.total_prefetches,
            "total_eviction_ms": round(self.total_eviction_us / 1000, 2),
            "total_prefetch_ms": round(self.total_prefetch_us / 1000, 2),
            # **峰值**利用率。取结束时刻的瞬时值恒为 0 —— 请求跑完时
            # 所有块都已释放，于是这个指标永远显示 0，有逐出也看不出来。
            "hot_peak_util_pct": round(
                self.hot_peak / self.hot_capacity * 100, 1) if self.hot_capacity else 0.0,
            "hot_peak_bytes": self.hot_peak,
        }
        if hasattr(self.policy, "report"):
            r["policy"] = self.policy.report()
        return r


# ================================================================
# 后端 2：真库（ctypes → libgpuswap.so）
# ================================================================

class RealSwapManager:
    """GpuSwapManager 的适配器，接口与 MockSwapManager 一致。

    ⚠️ 真库的槽位是**定长**的，所以这里强制 block 大小 = slot_bytes。
    模拟器那边也必须用同一个 block 大小，否则两个后端的"容量"不是一回事
    （256 KB 的块在同一个字节预算下能装 8 倍的数量）——那就不是对比，
    是两个不同的实验被放在同一张表里。
    """

    def __init__(self, hot_pool_bytes, cold_pool_bytes, policy_name,
                 young_age_ms=-1, hot_threshold=-1, protect_max_rate=-1.0,
                 reserve_slots=None):
        self._m = GpuSwap(hot_pool_bytes, cold_pool_bytes)
        self.slot = int(self._m.slot_bytes)
        self._m.set_policy(policy_name, young_age_ms, hot_threshold,
                           protect_max_rate)
        # 预热冷池：**必须**在任何 CUDA 工作之前。cudaMallocHost 是隐式
        # 同步点，留在窗口里会把"异步"变成"等设备空闲"。
        #
        # 预热数量 = 同一时刻可能同时身处冷池的块数上界。别按冷池**容量**
        # 预热：默认 64 GB 的容量意味着 3 万多个槽位、几十 GB 的 pinned
        # 内存（驱动文档明说 page-lock 过量会拖垮系统）。
        if reserve_slots is None:
            reserve_slots = max(1, cold_pool_bytes // self.slot)
        self._m.reserve_cold(min(reserve_slots, cold_pool_bytes // self.slot))
        self.policy_name = policy_name
        self.hot_capacity = int(self._m.hot_capacity)
        self.cold_capacity = int(self._m.cold_capacity)
        self.hot_peak = 0
        self.total_prefetches = 0
        self.miss_count = 0
        self.hit_count = 0
        self.alloc_fail = 0

    @property
    def block_size(self):
        return self.slot

    def set_clock(self, now_ms):
        self._m.set_clock(now_ms)

    def allocate(self, block_id, size):
        if size > self.slot:
            raise SystemExit(
                f"real 后端只支持定长槽位：块 {size} B > 槽位 {self.slot} B。"
                f" 用 --block-kb {self.slot // 1024} 保持一致。")
        if not self._m.allocate(block_id, size):
            self.alloc_fail += 1
            return None
        self.hot_peak = max(self.hot_peak, int(self._m.hot_used))
        return block_id

    def access(self, block_id):
        info = self._m.block_info(block_id)
        if info is None:
            return
        self._m.touch(block_id)
        if info["in_gpu"]:
            self.hit_count += 1
            return
        self.miss_count += 1
        # 真系统里是"读方在迁移事件上等"，这里同步等 —— 等的是同一件事，
        # 只是把等待点放在宿主里。
        code = self._m.prefetch(block_id)
        if code != 0:
            self.miss_fail = getattr(self, "miss_fail", 0) + 1
        self._m.sync()
        self.total_prefetches += 1

    def free(self, block_id):
        self._m.free(block_id)

    def close(self):
        self._m.close()

    def report(self):
        total = self.hit_count + self.miss_count
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "miss_rate_pct": round(self.miss_count / total * 100, 2) if total else 0.0,
            "total_evictions": int(self._m.eviction_count),
            "total_prefetches": self.total_prefetches,
            "hot_peak_util_pct": round(
                self.hot_peak / self.hot_capacity * 100, 1) if self.hot_capacity else 0.0,
            "hot_peak_bytes": self.hot_peak,
            "eviction_fingerprint": hex(self._m.eviction_fingerprint),
            "alloc_fail": self.alloc_fail,
            "slots": self.hot_capacity // self.slot,
            "policy": self._m.policy_report(),
            "backend": "real",
        }


# ================================================================
# 工作负载生成器（两个后端共用）
# ================================================================

# 实测的单请求 decode 步长基线 = 15.404 ms，噪声底 0.020 ms
# （kvcache-tracer/measurements/pcie_interference_redline.txt 的 baseline 行）。
DECODE_STEP_MS = 15.404

# 但生成器里**一轮 ≠ 一个 decode step**：一轮 = 所有活跃请求各前进一步
# **加上**一个新请求到达。峰值并发约 70，所以一轮比单请求的 15.4 ms 长。
#
# 这个常数是**显式**的旋钮，不许再藏进代码里。取值标准不是"看着差不多"，
# 是让块的生命周期落进项目一实测的区间 —— 因为 policy_thresholds.h 里的
# 年龄阈值（710 ms）是从真实 trace 的**生命周期 P25** 导出的，
# 只有时间轴对齐了，那个阈值才可比。
#
# 60 ms/轮 下的对账（round_ms=60）：
#   short   8-32 步  → 480-1920 ms   实测 short-qa     P10-P90 = 472-1908 ms
#   medium 32-120 步 → 1920-7200 ms  实测 code-gen     P10-P90 = 1921-7315 ms
#   long   75-130 步 → 4500-7800 ms  实测 long-summarize P10-P90 = 4429-7808 ms
#
# ⚠️ 历史：骨架版本每轮推进 0.5 s（expovariate(1/0.5)），而那时步数是
# 10-300，于是**一个块的整个生命周期都落在 710 ms 的"年轻"窗口里** ——
# 年龄规则保护的不是"最近几步刚分配的块"，是**全体**，然后在保护层
# 逐级松手的最后一级上退化成 GDSF。测出来的"adaptive 比 gdsf 差"
# 是整个工作负载的时间轴不对，不是策略的结论。
DEFAULT_ROUND_MS = 60.0


def generate_workload(num_requests=20, seed=42, round_ms=DEFAULT_ROUND_MS,
                      arrival_every=3):
    """
    生成模拟的 KV Cache block 访问序列。

    产出的事件流模拟了 LLM 推理的 block 分配/访问/释放模式：
    - alloc: 请求开始时分配一组 block（模拟 prefill）
    - access: decode 的每一步访问所有 block
    - free: 请求结束时释放所有 block

    三种请求类型（权重 50/30/20）：
    - short:  2-8 blocks,     8-32 steps  (问答)
    - medium: 8-32 blocks,   32-120 steps (摘要)
    - long:   32-128 blocks, 75-130 steps (长文档分析)

    `time` 的单位是**秒** —— 驱动侧乘 1000 变成管理器的毫秒时钟。

    ⚠️ **轮次和到达必须解耦。** 骨架版本把"一轮"同时定义成"所有活跃请求
    前进一步"**和**"一个新请求到达"，于是 N 个请求就只有 N 轮，块的
    生命周期被硬性截断在 N × 轮长 —— 步数写 75-130 也没用，永远走不到。
    那个 0.5 s/轮的怪值正是为了在 30 轮里凑出"看起来像样"的生命周期，
    属于用错误的参数掩盖结构错误。

    现在：每轮所有活跃请求前进一步；平均每 arrival_every 轮到达一个新请求
    （泊松抖动）。请求跑完自己的步数才结束，生命周期由步数决定，
    与请求总数无关。
    """
    import random
    random.seed(seed)
    events = []
    round_s = round_ms / 1000.0
    t = 0.0
    active = []
    pending = num_requests
    next_arrival = 0.0
    guard = 0

    def emit(now, kind, rid, nblk, steps=0, subtype=""):
        events.append({"time": now, "type": kind, "req_id": rid,
                       "num_blocks": nblk, "num_steps": steps,
                       "subtype": subtype})

    while (active or pending > 0) and guard < 200000:
        guard += 1
        t += round_s

        if pending > 0 and t >= next_arrival:
            rtype = random.choices(
                ["short", "medium", "long"], weights=[0.5, 0.3, 0.2])[0]
            if rtype == "short":
                nb, ns = random.randint(2, 8), random.randint(8, 32)
            elif rtype == "medium":
                nb, ns = random.randint(8, 32), random.randint(32, 120)
            else:
                nb, ns = random.randint(32, 128), random.randint(75, 130)
            rid = num_requests - pending
            emit(t, "alloc", rid, nb, ns, rtype)
            active.append([rid, nb, ns])
            pending -= 1
            next_arrival = t + round_s * random.expovariate(1.0) * arrival_every

        still_active = []
        for entry in active:
            rid, nblk, steps_left = entry
            emit(t, "access", rid, nblk)
            entry[2] -= 1
            if entry[2] > 0:
                still_active.append(entry)
            else:
                emit(t, "free", rid, nblk)
        active = still_active

    events.sort(key=lambda e: (e["time"],
                               {"alloc": 0, "access": 1, "free": 2}[e["type"]]))
    return events


def peak_footprint_bytes(events, block_bytes):
    """跑一遍事件流，量**峰值并发足迹**（字节）。

    预算要以它为基准，而不是拍一个绝对 GB 数。骨架版本用的是
    [8,4,2,1] GB，而整个 workload 的峰值只有 192 MiB —— 每一档预算
    都比全部数据还大，于是四个策略全是 0% miss、0 次逐出，
    对比表看着很整齐，实际上什么都没测。
    """
    live = {}
    peak = 0
    held = 0
    for ev in events:
        rid = ev["req_id"]
        if ev["type"] == "alloc":
            n = ev["num_blocks"] * block_bytes
            live[rid] = n
            held += n
            peak = max(peak, held)
        elif ev["type"] == "free":
            held -= live.pop(rid, 0)
    return peak


# ================================================================
# 主逻辑
# ================================================================

def make_manager(backend, strategy, hot_bytes, cold_bytes, reserve_slots=None,
                 **policy_knobs):
    if backend == "mock":
        policy = POLICIES[strategy]()
        return MockSwapManager(hot_bytes, cold_bytes, policy)
    return RealSwapManager(hot_bytes, cold_bytes, strategy,
                           reserve_slots=reserve_slots, **policy_knobs)


def run_test(strategy_name, hot_pool_bytes, num_requests=30, seed=42,
             backend="mock", block_bytes=2 << 20, cold_pool_bytes=None,
             hot_threshold=-1, young_age_ms=-1, protect_max_rate=-1.0,
             verbose=False):
    """驱动一次完整 workload，返回 report dict（两个后端字段一致）。

    后三个参数覆盖 adaptive 的规则旋钮（<0 = 用 policy_thresholds.h 的
    默认值），阈值扫描靠它们。**mock 后端不吃这些旋钮** ——
    MockSwapManager 直接用模块级的默认常数。扫描一律走 real。
    """
    if strategy_name not in POLICIES:
        raise SystemExit(f"未知策略 {strategy_name}，可选 {list(POLICIES)}")
    if backend == "mock" and (hot_threshold >= 0 or young_age_ms >= 0
                              or protect_max_rate >= 0):
        raise SystemExit("阈值旋钮只在 real 后端有效（mock 用默认常数）")

    events = generate_workload(num_requests, seed)
    peak = peak_footprint_bytes(events, block_bytes)

    reserve_slots = None
    if backend == "real":
        # 真库要先知道槽位大小才能规划冷池。冷池容量取"峰值足迹的两倍 +
        # 富余"，预热数量取**峰值并发块数**（同一时刻最多这么多块可能
        # 身处冷池）—— 按容量预热会变成几十 GB 的 pinned 内存。
        probe = GpuSwap(1 << 20, 1 << 20)
        block_bytes = int(probe.slot_bytes)
        probe.close()
        peak_blocks = peak // block_bytes
        reserve_slots = peak_blocks + 64
        # 真库只认整数个槽位
        hot_pool_bytes = max(block_bytes,
                             hot_pool_bytes // block_bytes * block_bytes)
        cold_pool_bytes = max(hot_pool_bytes * 2,
                              reserve_slots * block_bytes)
    elif cold_pool_bytes is None:
        cold_pool_bytes = 64 << 30

    mgr = make_manager(backend, strategy_name, hot_pool_bytes,
                       cold_pool_bytes, reserve_slots,
                       hot_threshold=hot_threshold,
                       young_age_ms=young_age_ms,
                       protect_max_rate=protect_max_rate)
    req_blocks = {}

    try:
        for ev in events:
            mgr.set_clock(int(ev["time"] * 1000))   # 秒 → 毫秒
            if ev["type"] == "alloc":
                rid = ev["req_id"]
                req_blocks[rid] = []
                for i in range(ev["num_blocks"]):
                    bid = rid * 100000 + i
                    if mgr.allocate(bid, block_bytes):
                        req_blocks[rid].append(bid)
                    # 分配失败 = 真系统里请求排队。计数在 report 里。
            elif ev["type"] == "access":
                for bid in req_blocks.get(ev["req_id"], ()):
                    mgr.access(bid)
            else:
                for bid in req_blocks.pop(ev["req_id"], ()):
                    mgr.free(bid)
        report = mgr.report()          # 必须在 close() 之前 —— 统计全在句柄里
    finally:
        if backend == "real":
            mgr.close()

    report["backend"] = backend
    report["block_kb"] = block_bytes / 1024
    report["hot_pool_mb"] = round(hot_pool_bytes / (1 << 20), 1)
    return report


def main():
    p = argparse.ArgumentParser(
        description="gpu-swap end-to-end test",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", default="lru",
                   choices=sorted(POLICIES), help="换出策略")
    mode = p.add_mutually_exclusive_group()
    # 骨架版本写的是 action="store_true", default=True —— 于是 --mock
    # 永远为 True，--real 没有开关，真库这条路根本走不到。
    mode.add_argument("--real", action="store_const", dest="backend", const="real",
                      help="走 ctypes 真库（权威结果，需要 CUDA + 先 make）")
    mode.add_argument("--mock", action="store_const", dest="backend", const="mock",
                      help="纯 Python 模拟（无需 CUDA，仅供冒烟）")
    p.set_defaults(backend="mock")
    p.add_argument("--hot-pool-mb", type=float, default=None,
                   help="热池大小（MB）。默认 = 峰值足迹的 50%%")
    p.add_argument("--budget-frac", type=float, default=0.5,
                   help="不给 --hot-pool-mb 时，热池 = 峰值足迹 × 这个比例")
    p.add_argument("--num-requests", type=int, default=30)
    p.add_argument("--block-kb", type=int, default=2048,
                   help="块大小 KB。默认 2048 = 真库的槽位粒度；"
                        "real 后端会强制成槽位大小")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    block_bytes = args.block_kb * 1024
    events = generate_workload(args.num_requests, args.seed)

    # real 后端要先知道槽位大小才能算预算，所以先探一次。
    if args.backend == "real":
        probe = GpuSwap(1 << 20, 1 << 20)
        block_bytes = int(probe.slot_bytes)
        probe.close()

    peak = peak_footprint_bytes(events, block_bytes)
    hot_bytes = (int(args.hot_pool_mb * (1 << 20)) if args.hot_pool_mb
                 else int(peak * args.budget_frac))
    if args.backend == "real":
        # 真库只认整数个槽位
        hot_bytes = max(block_bytes, hot_bytes // block_bytes * block_bytes)

    print("=== gpu-swap E2E Test ===")
    print(f"后端:       {args.backend}")
    print(f"策略:       {args.strategy}")
    print(f"块大小:     {block_bytes / 1024:.0f} KB")
    print(f"峰值足迹:   {peak / (1 << 20):.1f} MB")
    print(f"热池预算:   {hot_bytes / (1 << 20):.1f} MB "
          f"（峰值足迹的 {hot_bytes / peak * 100:.1f}%）")
    print(f"请求数:     {args.num_requests}   种子: {args.seed}")
    print()

    try:
        report = run_test(
            strategy_name=args.strategy,
            hot_pool_bytes=hot_bytes,
            num_requests=args.num_requests,
            seed=args.seed,
            backend=args.backend,
            block_bytes=block_bytes,
        )
    except GpuSwapError as e:
        print(f"真库报错: {e}", file=sys.stderr)
        return 1

    print("--- Results ---")
    for k, v in report.items():
        print(f"  {k}: {v}")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/test_{args.backend}_{args.strategy}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "report": report}, f,
                  indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
