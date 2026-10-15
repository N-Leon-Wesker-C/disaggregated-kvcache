"""
compare_runs.py — 横向对比多次运行（典型用途：并发度 1 / 4 / 8）

analyze.py 回答「这一次运行的 block 长什么样」；本脚本回答「换一个并发度，
什么地方变了」。两者共用同一套统计实现（直接 import analyze），避免两份
口径各自漂移。

最该看的一个指标是 **物理 block 数 vs 池容量**：

  - 物理 block 数 < 池容量  → 空闲链从来没被耗尽，逐出不可能发生。
    这时不管生命周期统计多漂亮，都和 swap 策略无关 —— 没有东西被换出去过。
  - 物理 block 数 ≈ 池容量  → 池子被占满并开始回收，逐出发生了。

并发度越高，同样的 workload 会越早把池子填满，这个指标的差异就是"并发压力"
最直接的证据。

用法:
    python scripts/compare_runs.py c1=traces/c1.*.jsonl \
                                    c4=traces/c4.*.jsonl \
                                    c8=traces/c8.*.jsonl

    # 池容量没给的话会试着从 logs/<label>.log 里读 vLLM 的启动日志
    python scripts/compare_runs.py --pool 110416 c1=... c4=...
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze  # noqa: E402

#: workload 的四类请求。trace 里的 request_id 形如
#: "cmpl-long-prefill-07-0-957e927f"，按子串匹配归类即可（见 workload.py）。
CATEGORY_NAMES = ["short-chat", "long-prefill", "long-decode", "shared-prefix"]


def category_of(request_id):
    """从 request_id 反推类别；认不出来归到 "(未知)"。"""
    rid = request_id or ""
    for cat in CATEGORY_NAMES:
        if cat in rid:
            return cat
    return "(未知)"


def pct(vals, q):
    """分位数。vals 会被就地排序 —— 调用方给的都是一次性列表。"""
    if not vals:
        return 0.0
    vals.sort()
    return vals[min(len(vals) - 1, int(len(vals) * q))]


def run_stats(paths, pool_tokens, block_size):
    events = []
    for p in paths:
        events.extend(analyze.load(p).get("events", []))
    events.sort(key=lambda e: e.get("timestamp", 0))

    lifetimes = analyze.compute_block_lifetimes(events)
    reads = analyze.compute_access_counts(events)

    n_alloc = sum(1 for e in events if e["event"] == "alloc")
    n_free = sum(1 for e in events if e["event"] == "free")
    n_new = sum(1 for e in events if e["event"] == "alloc"
                and e.get("source", "new") == "new")
    n_cached = n_alloc - n_new

    # 同时存活（被请求持有）的 block 数峰值。注意它**不等于**池占用 ——
    # 请求释放掉的 block 会回到池子里的空闲链（可能仍带缓存），
    # 本 trace 看不到那部分。所以峰值存活低不代表池子空。
    live = peak_live = 0
    for e in events:
        if e["event"] == "alloc":
            live += 1
            peak_live = max(peak_live, live)
        elif e["event"] == "free":
            live -= 1

    unique_blocks = len({e["block_id"] for e in events})
    pool_blocks = (pool_tokens // block_size) if pool_tokens else None

    lt = [d["lifetime"] for d in lifetimes]
    rd = list(reads.values())

    # 按类别聚合
    per_cat_lifetime = defaultdict(list)
    per_cat_reads = defaultdict(list)
    per_cat_alloc = defaultdict(int)
    per_cat_cached = defaultdict(int)
    per_cat_requests = defaultdict(set)
    for d in lifetimes:
        cat = category_of(d["request_id"])
        per_cat_lifetime[cat].append(d["lifetime"])
        per_cat_requests[cat].add(d["request_id"])
    for e in events:
        if e["event"] == "alloc":
            cat = category_of(e.get("request_id"))
            per_cat_alloc[cat] += 1
            if e.get("source") == "cache":
                per_cat_cached[cat] += 1
    # reads 的归属：free 事件带 request_id，借它把读取次数划到类别上
    for e in events:
        if e["event"] == "free" and "reads" in e:
            per_cat_reads[category_of(e.get("request_id"))].append(e["reads"])

    span = (events[-1]["timestamp"] - events[0]["timestamp"]) if events else 0.0

    return {
        "n_events": len(events),
        "n_alloc": n_alloc, "n_free": n_free,
        "n_new": n_new, "n_cached": n_cached,
        "n_unclosed": n_alloc - len(lifetimes),
        "unique_blocks": unique_blocks,
        "pool_blocks": pool_blocks,
        "peak_live": peak_live,
        "lifetimes": lt,
        "reads": rd,
        "span": span,
        "per_cat": {
            "lifetime": per_cat_lifetime,
            "reads": per_cat_reads,
            "alloc": per_cat_alloc,
            "cached": per_cat_cached,
            "requests": per_cat_requests,
        },
    }


def fmt_row(label, values, width=13):
    return f"{label:<30}" + "".join(f"{v:>{width}}" for v in values)


def print_comparison(runs):
    labels = [lbl for lbl, _ in runs]
    st = [s for _, s in runs]
    W = 15

    print("=" * (30 + W * len(runs)))
    print("跨运行对比")
    print("=" * (30 + W * len(runs)))
    print(fmt_row("指标", labels, W))
    print("-" * (30 + W * len(runs)))

    def row(name, fn, fmt="{:,.0f}"):
        print(fmt_row(name, [fmt.format(fn(s)) for s in st], W))

    row("事件总数", lambda s: s["n_events"])
    print(fmt_row("alloc / free",
                  [f"{s['n_alloc']:,}/{s['n_free']:,}" for s in st], W))
    row("未闭合区间", lambda s: s["n_unclosed"])
    print()
    row("物理 block 数(unique)", lambda s: s["unique_blocks"])
    row("KV 池容量(blocks)", lambda s: s["pool_blocks"] or 0)
    print(fmt_row("池利用率 unique/pool",
                  [f"{s['unique_blocks']/s['pool_blocks']*100:.1f}%"
                   if s["pool_blocks"] else "n/a" for s in st], W))
    row("峰值同时存活 block", lambda s: s["peak_live"])
    print(fmt_row("逐出是否可能发生",
                  ["是" if (s["pool_blocks"] and
                            s["unique_blocks"] >= s["pool_blocks"] * 0.95)
                   else "否(池子没满)" if s["pool_blocks"] else "n/a"
                   for s in st], W))
    print()
    row("alloc 新分配", lambda s: s["n_new"])
    row("alloc 缓存命中", lambda s: s["n_cached"])
    print(fmt_row("缓存命中率",
                  [f"{s['n_cached']/s['n_alloc']*100:.1f}%" if s["n_alloc"]
                   else "n/a" for s in st], W))
    print(fmt_row("block 平均复用次数",
                  [f"{s['n_alloc']/s['unique_blocks']:.2f}x"
                   if s["unique_blocks"] else "n/a" for s in st], W))
    print()
    print(fmt_row("生命周期 P50",
                  [f"{pct(list(s['lifetimes']), .5):.3f}s" if s["lifetimes"]
                   else "n/a" for s in st], W))
    print(fmt_row("生命周期 P90",
                  [f"{pct(list(s['lifetimes']), .9):.3f}s" if s["lifetimes"]
                   else "n/a" for s in st], W))
    print(fmt_row("生命周期 max",
                  [f"{max(s['lifetimes']):.2f}s" if s["lifetimes"] else "n/a"
                   for s in st], W))
    print(fmt_row("存活 <1s 占比",
                  [f"{sum(1 for t in s['lifetimes'] if t < 1.0)/len(s['lifetimes'])*100:.1f}%"
                   if s["lifetimes"] else "n/a" for s in st], W))
    print()
    print(fmt_row("读取次数 P50",
                  [f"{pct(list(s['reads']), .5):.0f}" if s["reads"] else "n/a"
                   for s in st], W))
    print(fmt_row("读取次数 P90",
                  [f"{pct(list(s['reads']), .9):.0f}" if s["reads"] else "n/a"
                   for s in st], W))
    print(fmt_row("读取次数 max",
                  [f"{max(s['reads']):,}" if s["reads"] else "n/a" for s in st], W))
    print(fmt_row("热 block (>100 次读)",
                  [f"{sum(1 for c in s['reads'] if c > 100)/len(s['reads'])*100:.1f}%"
                   if s["reads"] else "n/a" for s in st], W))
    print(fmt_row("trace 时间跨度",
                  [f"{s['span']:.1f}s" for s in st], W))


def print_per_category(runs):
    for lbl, s in runs:
        pc = s["per_cat"]
        print()
        print("=" * 92)
        print(f"按类别拆解 — {lbl}")
        print("=" * 92)
        print(f"{'类别':<16}{'请求数':>7}{'alloc':>9}{'其中缓存命中':>13}"
              f"{'生命周期P50':>13}{'读取P50':>10}{'读取max':>10}")
        print("-" * 92)
        cats = [c for c in CATEGORY_NAMES if c in pc["alloc"]]
        cats += [c for c in pc["alloc"] if c not in CATEGORY_NAMES]
        for cat in cats:
            lt = pc["lifetime"].get(cat, [])
            rd = pc["reads"].get(cat, [])
            alloc = pc["alloc"].get(cat, 0)
            cached = pc["cached"].get(cat, 0)
            hit = f"{cached}({cached/alloc*100:.0f}%)" if alloc else "-"
            print(f"{cat:<16}{len(pc['requests'].get(cat, ())):>7}{alloc:>9}"
                  f"{hit:>13}"
                  f"{(f'{pct(list(lt), .5):.3f}s' if lt else '-'):>13}"
                  f"{(f'{pct(list(rd), .5):.0f}' if rd else '-'):>10}"
                  f"{(f'{max(rd):,}' if rd else '-'):>10}")


def main():
    ap = argparse.ArgumentParser(description="跨并发度对比多个 trace")
    ap.add_argument("runs", nargs="+", metavar="标签=trace文件(可带glob)",
                    help="例: c1=traces/c1.*.jsonl")
    ap.add_argument("--pool", type=int, default=None,
                    help="KV 池容量(token 数)。不给则尝试从 logs/<标签>.log 读")
    ap.add_argument("--block-size", type=int, default=16)
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        if "=" not in spec:
            ap.error(f"格式应为 标签=文件: {spec}")
        label, pattern = spec.split("=", 1)
        paths = sorted(glob.glob(pattern)) or [pattern]

        pool = args.pool
        if pool is None:
            log = f"logs/{label}.log"
            if os.path.exists(log):
                m = re.search(r"GPU KV cache size: ([\d,]+)", open(log).read())
                if m:
                    pool = int(m.group(1).replace(",", ""))

        stats = run_stats(paths, pool, args.block_size)
        runs.append((label, stats))
        print(f"  {label}: {len(paths)} 个文件, {stats['n_events']:,} 事件, "
              f"池={pool or '未知'} tokens")

    print()
    print_comparison(runs)
    print_per_category(runs)


if __name__ == "__main__":
    main()
