#!/usr/bin/env python3
"""从项目一的 trace 现算 gpu-swap 策略阈值，生成 include/policy_thresholds.h。

为什么要生成而不是手抄
------------------------------------------------------------
`eviction_policy.h` 里的 `hot_access_threshold_ = 50` 是个占位符，注释写着
"这些值应该从项目一的 trace 数据中推导"。手抄过一次，抄错了：同一个量
（long-summarize 的读次数 P90）在不同来源之间是 828 和 491 两个数。手抄
常数无法复核，脚本可以重跑。

所以本脚本：
  1. **直接 import `kvcache-tracer/scripts/analyze.py` 的函数**，不重新实现
     "什么算一次读"。口径只有一份，不可能漂移。
  2. 把结果写成 `include/policy_thresholds.h`，**并把这个头文件提交进仓库** ——
     构建不该依赖另一个项目的 trace 目录。头文件里带出处（读了哪些文件、
     它们的 md5），`--check` 可以验证它和当前 trace 是否还对得上。

核心结论（脚本每次都会重算并打印）
------------------------------------------------------------
**任何单一绝对阈值都不可迁移。** 三个 workload 的读次数分布差好几倍，
T=50 在 short-qa 上保护 82% 的块，在 long-summarize 上保护 99%。所以
AdaptivePolicy 的主规则必须是**相对分位**；绝对阈值只作为次级保险，
并且要被夹断（见 eviction_policy.h 的 AdaptivePolicy::Config）。

用法:
    python scripts/derive_thresholds.py            # 生成头文件
    python scripts/derive_thresholds.py --check    # 只验证，不写，漂移则退出码 1
"""

import argparse
import glob
import hashlib
import importlib.util
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # gpu-swap/
REPO = os.path.dirname(ROOT)                       # disaggregated-kvcache/
ANALYZE = os.path.join(REPO, "kvcache-tracer", "scripts", "analyze.py")
TRACE_GLOB = os.path.join(REPO, "kvcache-tracer", "traces", "*.jsonl")
OUT = os.path.join(ROOT, "include", "policy_thresholds.h")

# 报告里要摆出来的候选绝对阈值。50 是原占位符；100 是 kvcache-tracer 的
# compare_runs.py 用的口径；121 是 short-qa 的读次数 P90；456 是 code-gen 的 P90。
CANDIDATE_THRESHOLDS = (50, 100, 121, 456)


def load_analyze():
    """把 analyze.py 当模块加载。**同一份口径，不复制代码。**"""
    spec = importlib.util.spec_from_file_location("kt_analyze", ANALYZE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def workload_of(path):
    """`code-gen.c4.55701.jsonl` → `code-gen`；`c4.14108.jsonl` → `legacy-c4`。

    早期那批只有一个并发度数字的 trace 属于旧 workload，单独归一组，
    免得被算进 short-qa / code-gen / long-summarize 里污染统计。
    """
    stem = os.path.basename(path).removesuffix(".jsonl")
    m = re.match(r"^([a-z0-9-]+)\.c\d+\.\d+$", stem)
    if m:
        return m.group(1)
    m = re.match(r"^(c\d+)\.\d+$", stem)
    if m:
        return "legacy-" + m.group(1)
    return stem


def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(int(q * len(sorted_vals)), len(sorted_vals) - 1)]


def protection_rate(counts, t):
    if not counts:
        return float("nan")
    return sum(1 for c in counts if c > t) / len(counts)


def analyse_file(analyze, path):
    events = analyze.load(path)["events"]
    lifetimes = analyze.compute_block_lifetimes(events)
    counts = analyze.compute_access_counts(events)
    return lifetimes, counts


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect():
    analyze = load_analyze()
    paths = sorted(glob.glob(TRACE_GLOB))
    if not paths:
        print(f"没有 trace：{TRACE_GLOB}", file=sys.stderr)
        return None

    per_wl = defaultdict(lambda: {"lifetimes": [], "counts": defaultdict(int),
                                  "files": []})
    for p in paths:
        wl = workload_of(p)
        lifetimes, counts = analyse_file(analyze, p)
        per_wl[wl]["files"].append(os.path.basename(p))
        per_wl[wl]["lifetimes"] += [d["lifetime"] for d in lifetimes]
        # 并发度不同的运行是**同一 workload 的不同次重复**，读次数按
        # block_id 直接相加没有意义（block_id 会在运行之间复用）。
        # 所以每个文件单独算一组读次数，聚合时对"分布"取统计量，而不是
        # 把不同运行的计数混进一个数组里 —— 那样会把同一块稀释掉。
        per_wl[wl].setdefault("per_file_counts", []).append(list(counts.values()))

    out = {}
    for wl, d in per_wl.items():
        # 每个文件一份读次数分布；跨并发度聚合时取**逐文件的保护率的
        # 中位数**，这是"同一 workload 多次重复"的正确聚合方式。
        per_file = d["per_file_counts"]
        merged = [c for lst in per_file for c in lst]   # 仅用于看整体形状

        lt = sorted(d["lifetimes"])
        out[wl] = {
            "files": sorted(d["files"]),
            "n_files": len(d["files"]),
            # 每个文件里有多少个不同的 block（读次数表的长度）。**不要把它们
            # 相加**当作"这个 workload 有多少块" —— 并发度不同的运行之间
            # block_id 会复用，相加得到的是运行数×工作集，没有意义。
            "n_blocks_per_file": [len(lst) for lst in per_file],
            "count_p50": pct(sorted(merged), .50),
            "count_p90": pct(sorted(merged), .90),
            "count_max": max(merged) if merged else 0,
            "protect": {t: [protection_rate(lst, t) for lst in per_file]
                        for t in CANDIDATE_THRESHOLDS},
            "life_p10": pct(lt, .10), "life_p25": pct(lt, .25),
            "life_p50": pct(lt, .50), "life_p90": pct(lt, .90),
            "life_lt1s": (sum(1 for t in lt if t < 1.0) / len(lt)) if lt else 0.0,
            "n_intervals": len(lt),
        }
    return out


def median(vals):
    vals = sorted(vals)
    n = len(vals)
    if not n:
        return float("nan")
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def summarize(stats):
    """把逐工作负载的统计压成要写进头文件的几个常数。"""
    wls = sorted(stats)
    hot50 = {w: median(stats[w]["protect"][50]) for w in wls}
    # 同一个阈值在不同 workload 上的保护率极差 —— 这就是"绝对阈值不可迁移"
    # 的量化形式：极差越大，越说明这个常数只是在拟合某一个 workload。
    spread50 = max(hot50.values()) - min(hot50.values())
    # 年龄阈值：取跨 workload 的**生命周期 P25 的最小值**。
    # 解释不是"短命所以别换"，而是"**刚分配 = 马上还会被读**"——
    # 一个块刚被分配，说明它的请求还在跑，计数低只是因为还没轮到它被读。
    # 取最小值是保守的一侧：宁可在少数 workload 上保护范围过窄，
    # 也不要在某个 workload 上把活跃块全保护住导致候选集为空。
    young_ms = int(min(stats[w]["life_p25"] for w in wls) * 1000)
    return {
        "wls": wls,
        "hot50": hot50,
        "spread50": spread50,
        "young_ms": young_ms,
        # 夹断线：取"绝对阈值诱导的保护率"跨 workload 的上界再放宽一点，
        # 实际含义是"绝对规则最多只许保护这么多个块，超了就退回相对分位"。
        "max_protect": min(0.30, max(hot50.values())),
        "lt1s": max(stats[w]["life_lt1s"] for w in wls),
    }


def render(stats, s):
    files = sorted(f for w in stats for f in stats[w]["files"])
    lines = [
        "/* policy_thresholds.h — **自动生成，不要手改**",
        " *",
        f" * 生成器: gpu-swap/scripts/derive_thresholds.py",
        f" * 数据源: kvcache-tracer/traces/ 下 {len(files)} 个 jsonl",
        " * 校验:   python scripts/derive_thresholds.py --check",
        " *",
        " * 口径来自 kvcache-tracer/scripts/analyze.py（脚本直接 import 它，",
        " * 不重新实现）：读次数取 free 事件里的 reads 字段，生命周期按 FIFO",
        " * 配对 alloc/free 得到**每一段**获取区间，不是每个 block 一条。",
        " *",
        " * 测量条件：这批运行里**没有发生任何逐出**（GPU 池装得下工作集），",
        " * 所以它们刻画的是 block 的自然行为，**不是逐出行为**。",
        " * 拿这些常数去预测「换出之后会怎样」是超纲的。",
        " */",
        "",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "// ---- 跨 workload 的保护率（读次数 > T 的块占比，逐文件取中位数）----",
    ]
    for t in CANDIDATE_THRESHOLDS:
        lines.append(f"//   T={t:>4}: " + "  ".join(
            f"{w}={median(stats[w]['protect'][t]) * 100:.1f}%"
            for w in s["wls"]))
    lines += [
        "//",
        f"// T=50 的跨 workload 极差 = {s['spread50'] * 100:.1f} 个百分点。",
        "// 这个极差就是\"绝对阈值不可迁移\"的量化形式：同一个常数在 legacy-c1 上",
        f"// 放行 {100 - median(stats['legacy-c1']['protect'][50]) * 100:.0f}% 的块，"
        f"在 long-summarize 上只放行 "
        f"{100 - median(stats['long-summarize']['protect'][50]) * 100:.0f}%。"
        "候选集大小差一个量级，",
        "// 策略的实际行为就不是同一个策略了。项目一在 tau 上栽过一次同一类",
        "// 的坑（tau 取 63ms 时 adaptive 逐位等于 LRU）。",
        "// 因此 **主规则必须是相对分位**，下面这个常数只做次级保险 + 夹断。",
        "",
        f"static constexpr int64_t GSM_HOT_ACCESS_THRESHOLD = 50;",
        f"static constexpr double  GSM_MAX_PROTECTION_RATE = {s['max_protect']:.4f};",
        "",
        "// ---- 年龄规则（毫秒）----",
        "//",
        "// 解释不是\"生命周期 < 1s 的块短命所以别换\" —— **生命周期在线不可判**，",
        "// 要等 free 才知道。在线可判的是**年龄**，正确解释是\"刚分配 = 马上",
        "// 还会被读\"：块刚分配说明它的请求还在跑，计数低只是因为还没轮到它。",
        "//",
        f"// 取值 = 跨 workload 的生命周期 P25 最小值 = {s['young_ms']} ms。",
        "// 各 workload 的 <1s 区间占比（最高者）："
        f"{s['lt1s'] * 100:.1f}%",
        "// 逐 workload 生命周期分位：",
    ]
    for w in s["wls"]:
        st = stats[w]
        lines.append(f"//   {w:<16} P10={st['life_p10'] * 1000:.0f}ms "
                     f"P25={st['life_p25'] * 1000:.0f}ms "
                     f"P50={st['life_p50'] * 1000:.0f}ms "
                     f"P90={st['life_p90'] * 1000:.0f}ms "
                     f"<1s={st['life_lt1s'] * 100:.1f}%")
    lines += [
        f"static constexpr int64_t GSM_YOUNG_AGE_MS = {s['young_ms']};",
        "",
        "// ---- 换出代价比（来自项目一 PCIe 实测，不是 trace）----",
        "//",
        "// 重算一个块 1620 us vs 换回一个块 52.4 us = **31x**。",
        "// 出处: kvcache-tracer/GUIDE.md 的 prefill 重算实测与 PCIe 实测表。",
        "// 这个比值是 AdaptivePolicy 里 GDSF 打分的 cost 项。",
        "static constexpr double GSM_RECOMPUTE_US_PER_BLOCK = 1620.0;",
        "static constexpr double GSM_SWAP_IN_US_PER_BLOCK = 52.4;",
        "",
        "// ---- 数据源指纹（--check 用）----",
        "// 生成时的 trace 文件 md5，任一个变了就说明阈值该重算。",
    ]
    for f in files:
        lines.append(f'//   {f}  {md5(os.path.join(REPO, "kvcache-tracer", "traces", f))}')
    lines.append("")
    return "\n".join(lines)


def render_digest(stats):
    """--check 比对用的稳定摘要：只含会进头文件的数字。"""
    s = summarize(stats)
    parts = [f"young_ms={s['young_ms']}", f"max_protect={s['max_protect']:.4f}"]
    for w in s["wls"]:
        parts.append(f"{w}:p50={stats[w]['count_p50']:.0f},"
                     f"p90={stats[w]['count_p90']:.0f},"
                     + ",".join(f"t{t}={median(stats[w]['protect'][t]) * 100:.1f}"
                                for t in CANDIDATE_THRESHOLDS))
    return "\n".join(parts)


def print_table(stats):
    wls = sorted(stats)
    print("=" * 78)
    print("逐 workload 的读次数分布与保护率（口径 = analyze.py，逐文件取中位数）")
    print("=" * 78)
    hdr = f"{'workload':<16} {'文件':>4} {'块/文件':>8} {'P50':>6} {'P90':>6} " + \
          " ".join(f"{'T=' + str(t):>7}" for t in CANDIDATE_THRESHOLDS)
    print(hdr)
    for w in wls:
        st = stats[w]
        nbf = st["n_blocks_per_file"]
        row = f"{w:<16} {st['n_files']:>4} {min(nbf):>8} {st['count_p50']:>6.0f} " \
              f"{st['count_p90']:>6.0f} "
        row += " ".join(
            f"{median(st['protect'][t]) * 100:>6.1f}%" for t in CANDIDATE_THRESHOLDS)
        print(row)
        # 逐文件的原值。**这行必须打出来** —— 上表是中位数，看不出重复之间的
        # 离散度。项目一就是在这里栽的：同一个量在不同聚合方式下是 828 和 491
        # 两个数，谁也说不清哪个对。把原值摆出来，对不上时能当场看出来。
        print(f"{'':16} 逐文件 T=50: " + "  ".join(
            f"{p * 100:.1f}%" for p in st["protect"][50]))

    s = summarize(stats)
    print(f"\nT=50 的跨 workload 极差: {s['spread50'] * 100:.1f} 个百分点")
    print("\n生命周期分位（在线**不可判**，只能用于离线解释；在线用年龄）：")
    for w in wls:
        st = stats[w]
        print(f"  {w:<16} P10={st['life_p10'] * 1000:>7.0f}ms "
              f"P25={st['life_p25'] * 1000:>7.0f}ms "
              f"P50={st['life_p50'] * 1000:>7.0f}ms "
              f"P90={st['life_p90'] * 1000:>8.0f}ms "
              f"<1s={st['life_lt1s'] * 100:>5.1f}%")
    print(f"\n→ GSM_YOUNG_AGE_MS = {s['young_ms']}")
    print(f"→ GSM_MAX_PROTECTION_RATE = {s['max_protect']:.4f}")
    print(f"\n⚠️  任何单一绝对阈值都会对某个 workload 保护 ~100%、对另一个 ~0%。")
    print(f"    AdaptivePolicy 的主规则必须是相对分位，绝对阈值只做夹断。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="只验证已提交的头文件是否与当前 trace 一致")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    stats = collect()
    if stats is None:
        return 1
    print_table(stats)

    text = render(stats, summarize(stats))
    if a.check:
        if not os.path.exists(a.out):
            print(f"\n✗ {os.path.relpath(a.out, ROOT)} 不存在，先不带 --check 跑一次")
            return 1
        with open(a.out) as f:
            cur = f.read()
        if cur == text:
            print(f"\n✓ {os.path.relpath(a.out, ROOT)} 与当前 trace 一致")
            return 0
        # 给出差异定位，别只说"不一致"
        cur_l = cur.splitlines()
        new_l = text.splitlines()
        import difflib
        diff = list(difflib.unified_diff(cur_l, new_l, "已提交", "重算",
                                         lineterm="", n=1))
        print(f"\n✗ {os.path.relpath(a.out, ROOT)} 已过期，"
              f"trace 变过了。重跑不带 --check 的版本更新它。")
        print("\n".join(diff[:60]))
        return 1

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(text)
    print(f"\n已写 {os.path.relpath(a.out, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
