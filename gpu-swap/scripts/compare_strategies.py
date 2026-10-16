#!/usr/bin/env python3
"""
compare_strategies.py — 横向对比不同换出策略

跑所有策略 × 不同热池预算的组合，生成对比报告。

这是最有说服力的产出：不是"我实现了 LRU"，而是"在热池装不下工作集
时，Adaptive 的 miss 率比 LRU 低 X% —— 而且这个 X 是可复现的"。

用法:
    python scripts/compare_strategies.py                 # 真库，默认
    python scripts/compare_strategies.py --backend both  # 真库 + 模拟器对照
    python scripts/compare_strategies.py --sweep         # 加阈值扫描

⚠️ 骨架版本有两个坑，都在这里修掉了：

1. **预算用的是绝对 GB**（[8,4,2,1] GB），而整个 workload 的峰值足迹
   只有 192 MiB —— 每一档预算都比全部数据还大，四个策略全是 0% miss、
   0 次逐出。表很整齐，什么都没测。现在预算按**峰值足迹的比例**给。

2. **`hot_utilization_pct` 取的是结束时刻的瞬时值**，而请求跑完时所有块
   都已释放 → 恒为 0.0%，即使真有逐出也显示 0。现在报**峰值**利用率。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_e2e import (TH, generate_workload, peak_footprint_bytes,  # noqa: E402
                      run_test)
from gsm import GpuSwap                                           # noqa: E402

SLOT_BYTES = 2 << 20
STRATEGIES = ["lru", "lfu", "gdsf", "adaptive"]
BUDGET_FRACS = [0.125, 0.25, 0.50, 1.00]


def fmt_pct(x):
    return f"{x:.2f}%"


def run_matrix(backend, num_requests, seed, block_bytes, budgets_abs,
               strategies, extra_policy=None, verbose=True):
    """跑 strategy × budget 全矩阵。返回 {(strategy, frac): report}。"""
    results = {}
    for strategy in strategies:
        for frac, hot_bytes in budgets_abs:
            kw = dict(extra_policy or {})
            report = run_test(
                strategy_name=strategy,
                hot_pool_bytes=hot_bytes,
                num_requests=num_requests,
                seed=seed,
                backend=backend,
                block_bytes=block_bytes,
                **kw)
            results[(strategy, frac)] = report
            if verbose:
                tag = ""
                n_acc = report["hit_count"] + report["miss_count"]
                if report["total_evictions"] == 0:
                    tag = "  ← 本档无效（0 次逐出，什么都没测到）"
                elif report["total_evictions"] < 0.01 * max(n_acc, 1):
                    tag = "  ← 本档无效（逐出次数可忽略）"
                print(f"  {backend:<5} {strategy:<9} {frac*100:>5.1f}%  "
                      f"miss={fmt_pct(report['miss_rate_pct']):>8} "
                      f"逐出={report['total_evictions']:>7} "
                      f"换回={report['total_prefetches']:>7} "
                      f"峰值利用={report['hot_peak_util_pct']:>5.1f}%{tag}",
                      flush=True)
    return results


def load_adaptive_thresholds():
    return {
        "hot_access_threshold": TH["GSM_HOT_ACCESS_THRESHOLD"],
        "young_age_ms": TH["GSM_YOUNG_AGE_MS"],
        "protect_max_rate": TH["GSM_MAX_PROTECTION_RATE"],
    }


def main():
    p = argparse.ArgumentParser(description="gpu-swap 策略对比")
    p.add_argument("--backend", default="real",
                   choices=["real", "mock", "both"])
    p.add_argument("--num-requests", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sweep", action="store_true",
                   help="加跑一轮热块阈值扫描（看绝对阈值如何退化）")
    p.add_argument("--out", default="results/COMPARISON_REPORT.md")
    args = p.parse_args()

    backends = ["real", "mock"] if args.backend == "both" else [args.backend]

    # 真库的槽位是定长的，块大小必须 = 槽位大小，否则两个后端的"容量"
    # 不是一回事（256KB 的块在同一字节预算下能装 8 倍的数量）。
    block_bytes = SLOT_BYTES
    if "real" in backends:
        with GpuSwap(1 << 20, 1 << 20) as probe:
            block_bytes = int(probe.slot_bytes)

    events = generate_workload(args.num_requests, args.seed)
    peak = peak_footprint_bytes(events, block_bytes)

    # 预算 = 峰值足迹的比例，**再对齐到整数个槽位**。
    budgets = []
    for frac in BUDGET_FRACS:
        b = int(peak * frac) // block_bytes * block_bytes
        budgets.append((frac, max(block_bytes, b)))

    print("=" * 88)
    print("GPU-SWAP STRATEGY COMPARISON")
    print(f"  请求数={args.num_requests} 种子={args.seed} "
          f"块={block_bytes // 1024}KB 峰值足迹={peak / (1 << 20):.0f}MB")
    print(f"  预算档位（峰值足迹的百分比）: "
          f"{', '.join(f'{f*100:.1f}%={b//block_bytes}槽' for f, b in budgets)}")
    print(f"  adaptive 常数（来自 policy_thresholds.h，由项目一 trace 现算）: "
          f"T={TH['GSM_HOT_ACCESS_THRESHOLD']} "
          f"young={TH['GSM_YOUNG_AGE_MS']}ms "
          f"protect_max={TH['GSM_MAX_PROTECTION_RATE']}")
    print("=" * 88, flush=True)

    all_results = {}
    for backend in backends:
        print(f"\n── 后端: {backend} ──", flush=True)
        res = run_matrix(backend, args.num_requests, args.seed, block_bytes,
                         budgets, STRATEGIES)
        for (s, f), r in res.items():
            all_results[f"{backend}/{s}/{f}"] = r

    # ---- 阈值扫描 ----
    sweep = []
    if args.sweep:
        print("\n── 热块阈值扫描（真库）—— 绝对阈值为什么会退化 ──", flush=True)
        frac, hot_bytes = budgets[1]      # 25% 那档：逐出足够多，能分出差别
        base = load_adaptive_thresholds()
        # 三组：① 只有绝对阈值（= 骨架版规则）② 加年龄规则 ③ 再加夹断（现版）
        variants = [
            ("骨架版: 只有绝对阈值 T",
             dict(hot_threshold=None, young_age_ms=0, protect_max_rate=1.0)),
            ("+ 年龄规则",
             dict(hot_threshold=None, young_age_ms=base["young_age_ms"],
                  protect_max_rate=1.0)),
            ("+ 夹断（现版）",
             dict(hot_threshold=None, young_age_ms=base["young_age_ms"],
                  protect_max_rate=base["protect_max_rate"])),
        ]
        # T=0 是退化到极点：任何被碰过的块（access_count > 0）都算"热"，
        # 保护率 100%，保护层三级松手全程触发。留着当上界看。
        # 最后一档 = 关掉绝对阈值，只靠年龄规则 + 打分内核。
        thresholds = [0, 1, 50, 121, 456, 1 << 30]
        for t in thresholds:
            for label, cfg in variants:
                r = run_test("adaptive", hot_bytes, args.num_requests, args.seed,
                             backend="real", block_bytes=block_bytes,
                             hot_threshold=t,
                             young_age_ms=(cfg["young_age_ms"] if cfg["young_age_ms"] is not None else -1),
                             protect_max_rate=cfg["protect_max_rate"])
                pr = r.get("policy") or {}
                rate = pr.get("protect_mean", 0.0) if isinstance(pr, dict) else 0.0
                sweep.append({"threshold": t, "variant": label,
                              "miss_rate_pct": r["miss_rate_pct"],
                              "evictions": r["total_evictions"],
                              "protect_mean": rate,
                              "clamped": pr.get("clamped", 0),
                              "relaxed": (pr.get("relaxed_hot", 0)
                                          + pr.get("relaxed_young", 0))})
                print(f"  T={t:<10} {label:<22} "
                      f"保护率={rate*100:>5.1f}% 夹断={sweep[-1]['clamped']:>6} "
                      f"松手={sweep[-1]['relaxed']:>6} "
                      f"miss={fmt_pct(r['miss_rate_pct']):>8}", flush=True)

    # ---- 报告 ----
    os.makedirs("results", exist_ok=True)
    with open("results/strategy_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# GPU Swap 策略对比\n")
    lines.append(f"- workload: {args.num_requests} 个请求"
                 f"（50% 短问答 / 30% 摘要 / 20% 长文档），种子 {args.seed}")
    lines.append(f"- 块大小 {block_bytes // 1024} KB（= 真库槽位粒度，定长）")
    lines.append(f"- 峰值并发足迹 **{peak / (1 << 20):.0f} MB**"
                 f"（{peak // block_bytes} 块）—— 预算按它的比例给")
    lines.append(f"- adaptive 常数来自 `include/policy_thresholds.h`"
                 f"（由项目一 trace 现算）：T={TH['GSM_HOT_ACCESS_THRESHOLD']}、"
                 f"young={TH['GSM_YOUNG_AGE_MS']}ms、"
                 f"protect_max={TH['GSM_MAX_PROTECTION_RATE']}\n")

    for backend in backends:
        lines.append(f"## 后端 `{backend}`\n")
        lines.append("| 策略 | 预算（峰值足迹） | 槽位 | miss 率 | 逐出 | 换回 | 峰值利用率 | 有效？ |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for strategy in STRATEGIES:
            for frac, hot_bytes in budgets:
                r = all_results.get(f"{backend}/{strategy}/{frac}")
                if r is None:
                    continue
                # 口径：本档有没有真的测到东西。骨架版四档预算全部 0 次逐出
                # 却在表里显示 0.0% miss，看着像"四个策略都完美"。
                # 现在逐出次数微不足道（< 块次访问的 1%）也一律标无效。
                n_access = r["hit_count"] + r["miss_count"]
                if r["total_evictions"] == 0:
                    valid = "**✗ 0 次逐出**"
                elif r["total_evictions"] < 0.01 * max(n_access, 1):
                    valid = "**✗ 逐出可忽略**"
                else:
                    valid = "✓"
                lines.append(
                    f"| {strategy} | {frac*100:.1f}% | {hot_bytes // block_bytes} "
                    f"| {fmt_pct(r['miss_rate_pct'])} | {r['total_evictions']} "
                    f"| {r['total_prefetches']} | {r['hot_peak_util_pct']:.1f}% "
                    f"| {valid} |")
        lines.append("")

    if sweep:
        lines.append("## 热块阈值扫描（为什么绝对阈值不可迁移）\n")
        lines.append("| T | 变体 | 平均保护率 | 夹断次数 | 保护层松手次数 | miss 率 |")
        lines.append("|---|---|---|---|---|---|")
        for s in sweep:
            t = s["threshold"]
            t = "无（只靠年龄+打分）" if t >= (1 << 30) else str(t)
            lines.append(f"| {t} | {s['variant']} | {s['protect_mean']*100:.1f}% "
                         f"| {s['clamped']} | {s['relaxed']} "
                         f"| {s['miss_rate_pct']:.2f}% |")
        lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n报告: {args.out}")
    print(f"原始数据: results/strategy_comparison.json")


if __name__ == "__main__":
    main()
