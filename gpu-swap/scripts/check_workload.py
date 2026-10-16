#!/usr/bin/env python3
"""
check_workload.py — 合成 workload 的保真度对账

合成的负载是**近似**，但"近似"必须可检验，否则时间轴一歪，
从真实 trace 导出的阈值（policy_thresholds.h 里的 710 ms 年龄规则）
就成了另一个意思上的数字，而且看不出来。

这个脚本把合成 workload 的块生命周期分位数和项目一实测的放在一起。
对不上就说明 generate_workload 的时间参数该调。

用法:
    python scripts/check_workload.py [--requests 30] [--round-ms 60] [--seed 42]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_e2e import (DEFAULT_ROUND_MS, generate_workload,  # noqa: E402
                      peak_footprint_bytes)

# 项目一跨 workload 实测的块生命周期分位（毫秒）
# 出处: include/policy_thresholds.h 的注释，由
#       scripts/derive_thresholds.py 从 kvcache-tracer/traces/*.jsonl 现算。
REAL_LIFETIMES = {
    "short-qa":       {"P10": 472,  "P25": 1056, "P50": 1403, "P90": 1908},
    "code-gen":       {"P10": 1921, "P25": 4204, "P50": 5263, "P90": 7315},
    "long-summarize": {"P10": 4429, "P25": 5192, "P50": 6138, "P90": 7808},
}
# 生成器的三个请求类型对应到哪一类真实 workload
SYNTH_TO_REAL = {"short": "short-qa", "medium": "code-gen",
                 "long": "long-summarize"}


def quantile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(int(p * len(sorted_vals)), len(sorted_vals) - 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--requests", type=int, default=30)
    p.add_argument("--round-ms", type=float, default=DEFAULT_ROUND_MS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--block-kb", type=int, default=2048)
    args = p.parse_args()

    events = generate_workload(args.requests, args.seed, args.round_ms)

    allocs = {e["req_id"]: e for e in events if e["type"] == "alloc"}
    freed = {e["req_id"]: e["time"] for e in events if e["type"] == "free"}
    by_type = {}
    for rid, a in allocs.items():
        if rid in freed:
            by_type.setdefault(a["subtype"], []).append(
                (freed[rid] - a["time"]) * 1000.0)

    # 注意口径：一次 access **事件**会碰 num_blocks 个块，miss 率的分母
    # 是块次（block-accesses）不是事件数。两个数差一个量级。
    n_events = sum(1 for e in events if e["type"] == "access")
    n_touch = sum(e["num_blocks"] for e in events if e["type"] == "access")
    peak_bytes = peak_footprint_bytes(events, args.block_kb * 1024)

    print(f"轮长 {args.round_ms} ms   请求 {args.requests}   种子 {args.seed}")
    print(f"access 事件 {n_events}   块次访问 {n_touch}   "
          f"峰值足迹 {peak_bytes / (1 << 20):.0f} MB "
          f"({peak_bytes // (args.block_kb * 1024)} 块)\n")

    print(f"{'类型':<8} {'n':>3} | {'合成 P10':>8} {'P25':>7} {'P50':>7} {'P90':>7}"
          f" | {'实测 P10':>8} {'P25':>7} {'P50':>7} {'P90':>7}")
    print("-" * 88)
    for synth, real in SYNTH_TO_REAL.items():
        vals = sorted(by_type.get(synth, []))
        if not vals:
            continue
        r = REAL_LIFETIMES[real]
        print(f"{synth:<8} {len(vals):>3} | "
              f"{quantile(vals, .1):>8.0f} {quantile(vals, .25):>7.0f} "
              f"{quantile(vals, .5):>7.0f} {quantile(vals, .9):>7.0f} | "
              f"{r['P10']:>8} {r['P25']:>7} {r['P50']:>7} {r['P90']:>7}   "
              f"({real})")
    print("-" * 88)

    # "年轻"窗口覆盖了多大比例的块生命周期 —— 这才是年龄规则的真正作用面。
    # 骨架版（0.5 s/轮 + 只有 30 轮）下这个比例接近 100%，年龄规则等于没有。
    young_ms = 710.0
    covered = sorted(
        min(1.0, young_ms / max((freed[r] - allocs[r]["time"]) * 1000.0, 1e-9))
        for r in allocs if r in freed)
    print(f"年龄规则（young < {young_ms:.0f} ms）覆盖的生命周期比例: "
          f"中位数 {quantile(covered, .5)*100:.0f}%  "
          f"P90 {quantile(covered, .9)*100:.0f}%")
    print("  （接近 100% = 块**终身**都被当成'刚分配'，年龄规则失去区分度；"
          "这正是骨架 workload 的问题）")


if __name__ == "__main__":
    main()
