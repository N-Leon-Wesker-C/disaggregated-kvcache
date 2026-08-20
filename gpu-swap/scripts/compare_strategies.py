#!/usr/bin/env python3
"""
compare_strategies.py — 横向对比不同换出策略

运行所有策略 × 不同 GPU 预算的组合，生成对比报告。

这是申请材料里最有说服力的产出：
  不是"我实现了 LRU"，
  而是"在 GPU 预算紧张时，Adaptive 的 miss rate 比 LRU 低 X%"。

用法:
    python scripts/compare_strategies.py
"""

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.test_e2e import run_test


def compare():
    strategies = ["lru", "lfu", "adaptive"]
    # GPU 预算：从"刚好够"到"紧张"再到"非常紧张"
    # 对于 30 个混合请求，总 KV Cache 峰值约 4-6GB
    gpu_budgets = [8.0, 4.0, 2.0, 1.0]  # GB
    num_requests = 30

    print("=" * 75)
    print("GPU-SWAP STRATEGY COMPARISON")
    print(f"{'=' * 75}")
    print(f"{'Strategy':<12} {'Budget':>8} {'Miss%':>8} "
          f"{'Evicts':>8} {'Prefetch':>9} {'HotUtil%':>9}")
    print("-" * 75)

    all_results = {}

    for strategy in strategies:
        for budget in gpu_budgets:
            report = run_test(
                strategy_name=strategy,
                hot_pool_gb=budget,
                num_requests=num_requests,
                seed=42,  # 固定种子，可复现
            )

            key = f"{strategy}_{budget}gb"
            all_results[key] = report

            print(f"{strategy:<12} {budget:>6.1f}GB "
                  f"{report['miss_rate_pct']:>7.1f}% "
                  f"{report['total_evictions']:>7} "
                  f"{report['total_prefetches']:>8} "
                  f"{report['hot_utilization_pct']:>8.1f}%")

    print("-" * 75)
    print()

    # 找出每种策略在紧张预算下的表现
    print("=== Key Finding: 2GB Budget (Tight) ===")
    for strategy in strategies:
        key = f"{strategy}_2.0gb"
        if key in all_results:
            r = all_results[key]
            print(f"  {strategy:<12} miss_rate={r['miss_rate_pct']:.1f}%  "
                  f"evictions={r['total_evictions']}")

    # 保存完整结果
    os.makedirs("results", exist_ok=True)
    with open("results/strategy_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to results/strategy_comparison.json")

    # 生成简单的文本报告
    report_path = "results/COMPARISON_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# GPU Swap Strategy Comparison\n\n")
        f.write(f"Workload: {num_requests} requests (50% short, 30% medium, 20% long)\n\n")
        f.write("| Strategy | GPU Budget | Miss Rate | Evictions | Prefetches | Hot Util |\n")
        f.write("|----------|-----------|-----------|-----------|------------|----------|\n")
        for key, r in all_results.items():
            strategy, budget = key.rsplit("_", 1)
            f.write(f"| {strategy} | {budget} | {r['miss_rate_pct']:.1f}% | "
                    f"{r['total_evictions']} | {r['total_prefetches']} | "
                    f"{r['hot_utilization_pct']:.1f}% |\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    compare()
