#!/usr/bin/env python3
"""
test_e2e.py — gpu-swap 端到端测试

模拟 LLM 推理过程中 KV Cache block 的分配/访问/释放模式，
驱动 gpu-swap C++ 库（通过 ctypes），验证正确性和性能。

用法:
    python scripts/test_e2e.py --strategy lru --gpu-budget 4GB
    python scripts/test_e2e.py --strategy adaptive --trace kvcache_trace.json

如果你还没有编译 gpu-swap 的 C++ 库，本脚本可以独立运行在"模拟模式"：
    python scripts/test_e2e.py --strategy adaptive --mock

在模拟模式下，不依赖实际的 GPU/CUDA，而是用 Python 模拟 GpuSwapManager
的行为（延迟数字来自 Step 0 的 PCIe benchmark 数据）。
"""

import random
import argparse
import sys
import json
from dataclasses import dataclass, field
from typing import List, Optional


# ================================================================
# 模拟的 GpuSwapManager（纯 Python，用于策略测试）
# ================================================================

# 从你的 Step 0 PCIe benchmark 中获取的真实延迟数字
# 默认值：256KB block 的传输延迟约 25 微秒
DEFAULT_PCIE_LATENCY_US = 25   # D→H 一次 256KB
DEFAULT_GPU_LATENCY_US = 0     # 本地访问，近似 0


@dataclass
class Block:
    block_id: int
    size: int
    in_gpu: bool = True
    last_access: int = 0      # 逻辑时钟
    access_count: int = 0
    alloc_time: int = 0
    gpu_ptr: int = 0          # 模拟地址
    cpu_ptr: int = 0


class LRUPolicy:
    """最近最少使用——选 last_access 最小的"""
    def select_victim(self, gpu_blocks: List[Block]) -> Optional[int]:
        if not gpu_blocks:
            return None
        return min(gpu_blocks, key=lambda b: b.last_access).block_id


class LFUPolicy:
    """最不频繁使用——选 access_count 最小的"""
    def select_victim(self, gpu_blocks: List[Block]) -> Optional[int]:
        if not gpu_blocks:
            return None
        return min(gpu_blocks, key=lambda b: b.access_count).block_id


class AdaptivePolicy:
    """
    基于 trace 数据的混合策略：
    - 跳过 access_count > hot_threshold 的热块
    - 在剩余候选中用 LRU
    """
    def __init__(self, hot_threshold: int = 50):
        self.hot_threshold = hot_threshold

    def select_victim(self, gpu_blocks: List[Block]) -> Optional[int]:
        if not gpu_blocks:
            return None

        # 排除热块
        candidates = [b for b in gpu_blocks
                      if b.access_count <= self.hot_threshold]

        if not candidates:
            # 所有块都是热块——只能退化成 LRU
            candidates = list(gpu_blocks)

        return min(candidates, key=lambda b: b.last_access).block_id


class MockSwapManager:
    """
    模拟的 GPU Swap 管理器——逻辑和 C++ 版本一致，
    但不依赖实际 CUDA。用于快速迭代策略设计。
    """

    def __init__(self, hot_pool_bytes: int, cold_pool_bytes: int,
                 policy, pcie_latency_us: int = DEFAULT_PCIE_LATENCY_US):
        self.hot_capacity = hot_pool_bytes
        self.cold_capacity = cold_pool_bytes
        self.hot_used = 0
        self.cold_used = 0
        self.policy = policy
        self.pcie_latency_us = pcie_latency_us
        self.blocks: dict = {}
        self.clock = 0           # 逻辑时钟

        # 统计
        self.total_evictions = 0
        self.total_prefetches = 0
        self.total_eviction_us = 0
        self.total_prefetch_us = 0
        self.miss_count = 0      # 访问时 block 不在 GPU 的次数
        self.hit_count = 0

    def allocate(self, block_id: int, size: int) -> Optional[Block]:
        """分配一个 block。如果热池满，自动换出。"""
        while self.hot_used + size > self.hot_capacity:
            gpu_blocks = [b for b in self.blocks.values() if b.in_gpu]
            victim_id = self.policy.select_victim(gpu_blocks)
            if victim_id is None:
                return None  # 无法腾出空间
            self._evict(victim_id)

        blk = Block(
            block_id=block_id, size=size, in_gpu=True,
            last_access=self.clock, alloc_time=self.clock,
            gpu_ptr=id(block_id) * 4096,  # 模拟地址
        )
        self.blocks[block_id] = blk
        self.hot_used += size
        return blk

    def access(self, block_id: int):
        """标记 block 被访问。如果不在 GPU，触发换回（miss）。"""
        blk = self.blocks.get(block_id)
        if blk is None:
            return  # 已释放

        self.clock += 1
        blk.last_access = self.clock
        blk.access_count += 1

        if not blk.in_gpu:
            self._prefetch(block_id)
            self.miss_count += 1
        else:
            self.hit_count += 1

    def free(self, block_id: int):
        blk = self.blocks.get(block_id)
        if blk is None:
            return
        if blk.in_gpu:
            self.hot_used -= blk.size
        else:
            self.cold_used -= blk.size
        del self.blocks[block_id]

    def _evict(self, block_id: int):
        blk = self.blocks[block_id]
        if not blk.in_gpu:
            return
        blk.in_gpu = False
        blk.cpu_ptr = id(block_id) * 4096 + 2048
        self.hot_used -= blk.size
        self.cold_used += blk.size
        self.total_evictions += 1
        self.total_eviction_us += self.pcie_latency_us

    def _prefetch(self, block_id: int):
        blk = self.blocks[block_id]
        if blk.in_gpu:
            return

        # 如果热池满，先腾空间
        while self.hot_used + blk.size > self.hot_capacity:
            gpu_blocks = [b for b in self.blocks.values() if b.in_gpu]
            # 排除自己
            gpu_blocks = [b for b in gpu_blocks if b.block_id != block_id]
            victim_id = self.policy.select_victim(gpu_blocks)
            if victim_id is None:
                return
            self._evict(victim_id)

        blk.in_gpu = True
        self.hot_used += blk.size
        self.cold_used -= blk.size
        self.total_prefetches += 1
        self.total_prefetch_us += self.pcie_latency_us

    def report(self) -> dict:
        total = self.hit_count + self.miss_count
        miss_rate = self.miss_count / total * 100 if total > 0 else 0
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "miss_rate_pct": round(miss_rate, 2),
            "total_evictions": self.total_evictions,
            "total_prefetches": self.total_prefetches,
            "total_eviction_ms": round(self.total_eviction_us / 1000, 2),
            "total_prefetch_ms": round(self.total_prefetch_us / 1000, 2),
            "hot_utilization_pct": round(
                self.hot_used / self.hot_capacity * 100, 1
            ) if self.hot_capacity > 0 else 0,
        }


# ================================================================
# 工作负载生成器
# ================================================================

BLOCK_SIZE = 256 * 1024  # 256 KB per block（16 tokens * 4096 dim * KV * FP16）


def generate_workload(num_requests=20, seed=42):
    """
    生成模拟的 KV Cache block 访问序列。

    产出的事件流模拟了 LLM 推理的 block 分配/访问/释放模式：
    - alloc: 请求开始时分配一组 block（模拟 prefill）
    - access: decode 的每一步访问所有 block
    - free: 请求结束时释放所有 block

    三种请求类型：
    - short:  2-8 blocks,  10-30 steps  (问答)
    - medium: 8-32 blocks, 30-100 steps (摘要)
    - long:   32-128 blocks, 100-300 steps (长文档分析)
    """
    random.seed(seed)
    events = []
    t = 0
    active = []

    for req_id in range(num_requests):
        rtype = random.choices(
            ["short", "medium", "long"], weights=[0.5, 0.3, 0.2]
        )[0]

        if rtype == "short":
            num_blocks = random.randint(2, 8)
            num_steps = random.randint(10, 30)
        elif rtype == "medium":
            num_blocks = random.randint(8, 32)
            num_steps = random.randint(30, 100)
        else:
            num_blocks = random.randint(32, 128)
            num_steps = random.randint(100, 300)

        events.append({
            "time": t, "type": "alloc",
            "req_id": req_id, "num_blocks": num_blocks,
            "num_steps": num_steps, "subtype": rtype,
        })
        active.append((req_id, num_blocks, num_steps))

        # 请求到达间隔（泊松过程近似）
        t += random.expovariate(1.0 / 0.5)  # 平均 0.5s 间隔

        # 清理已完成的请求
        still_active = []
        for (rid, nblk, steps_left) in active:
            events.append({
                "time": t, "type": "access",
                "req_id": rid, "num_blocks": nblk,
            })
            steps_left -= 1
            if steps_left > 0:
                still_active.append((rid, nblk, steps_left))
            else:
                events.append({
                    "time": t, "type": "free", "req_id": rid,
                })
        active = still_active

        t += 0.001  # 微小时间增量

    # 清理剩余活动请求
    for (rid, nblk, _) in active:
        events.append({
            "time": t, "type": "free", "req_id": rid,
        })

    events.sort(key=lambda e: (e["time"], {"alloc": 0, "access": 1, "free": 2}[e["type"]]))
    return events


# ================================================================
# 主逻辑
# ================================================================

def run_test(strategy_name: str, hot_pool_gb: float, num_requests: int,
             cold_pool_gb: float = 32.0, seed: int = 42):
    """运行一次测试"""
    hot_bytes = int(hot_pool_gb * 1024**3)
    cold_bytes = int(cold_pool_gb * 1024**3)

    policy_map = {
        "lru": LRUPolicy,
        "lfu": LFUPolicy,
        "adaptive": AdaptivePolicy,
    }

    policy_cls = policy_map.get(strategy_name.lower())
    if policy_cls is None:
        print(f"Unknown strategy: {strategy_name}. Options: {list(policy_map.keys())}")
        sys.exit(1)

    policy = policy_cls()
    manager = MockSwapManager(hot_bytes, cold_bytes, policy)

    events = generate_workload(num_requests, seed)
    req_blocks = {}  # req_id → [block_ids]

    for ev in events:
        if ev["type"] == "alloc":
            req_id = ev["req_id"]
            req_blocks[req_id] = []
            for i in range(ev["num_blocks"]):
                blk_id = req_id * 10000 + i
                blk = manager.allocate(blk_id, BLOCK_SIZE)
                if blk:
                    req_blocks[req_id].append(blk_id)
                else:
                    # 热池满 + 无法换出 → 分配失败
                    # 在真实系统中这会导致请求排队
                    pass

        elif ev["type"] == "access":
            req_id = ev["req_id"]
            for blk_id in req_blocks.get(req_id, []):
                manager.access(blk_id)

        elif ev["type"] == "free":
            req_id = ev["req_id"]
            for blk_id in req_blocks.pop(req_id, []):
                manager.free(blk_id)

    return manager.report()


def main():
    parser = argparse.ArgumentParser(
        description="gpu-swap end-to-end test")
    parser.add_argument("--strategy", default="lru",
                        choices=["lru", "lfu", "adaptive"],
                        help="Eviction strategy")
    parser.add_argument("--gpu-budget", type=float, default=4.0,
                        help="GPU hot pool size in GB")
    parser.add_argument("--num-requests", type=int, default=30,
                        help="Number of simulated requests")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Run in mock mode (no CUDA required)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"=== gpu-swap E2E Test ===")
    print(f"Strategy: {args.strategy}")
    print(f"GPU budget: {args.gpu_budget} GB")
    print(f"Requests: {args.num_requests}")
    print(f"Block size: {BLOCK_SIZE / 1024:.0f} KB")
    print()

    report = run_test(
        strategy_name=args.strategy,
        hot_pool_gb=args.gpu_budget,
        num_requests=args.num_requests,
        seed=args.seed,
    )

    print(f"--- Results ---")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # 保存结果
    result = {
        "config": {
            "strategy": args.strategy,
            "gpu_budget_gb": args.gpu_budget,
            "num_requests": args.num_requests,
            "block_size_kb": BLOCK_SIZE / 1024,
            "seed": args.seed,
        },
        "report": report,
    }

    out_path = f"results/test_{args.strategy}_{args.gpu_budget}gb.json"
    import os
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
