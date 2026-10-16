"""
report.py — 把正式实验的多次运行聚合成可写进 README 的数字

读 results/ 下每次运行的元数据（完整实验条件 + 客户端统计）和 traces/ 下对应的
block 级事件，按 (workload, 并发度) 把多次重复聚合成 **均值 ± 标准差**。

为什么必须给出方差：单次运行的数字无法区分「这是 workload 的性质」还是
「这一次调度的运气」。三次重复之间只做 prefix cache 重置、不重启服务，
所以同一组合的差异确实只来自调度时序 —— 标准差小，才说明观测到的差异
是 workload 之间真实的差异。

统计口径和 analyze.py / compare_runs.py 一致（直接 import analyze），
避免三份实现各自漂移：

  * block 生命周期 = free.timestamp - alloc.timestamp，按 (block_id, request_id)
    配对（一个 block 可以先后/同时被多个请求持有，每次持有是一个区间）。
  * block 访问次数 = 同一个 block 的各条 free 事件的 reads **累加**
    （不是每个 free 事件的 reads，那是"本次持有的读数"，另一个量）。

用法:
    python scripts/report.py                    # 全部
    python scripts/report.py --workload short-qa --concurrency 1
    python scripts/report.py --quiet            # 只输出 README 结论句
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: vLLM 会把 X-Request-Id 包成 "cmpl-short-qa-098-r3-0-957e927f"，
#: 重复序号藏在中间，用 -r<N>- 精确取出（不能用子串匹配，-r1 会撞上 -r1x）。
REP_RE = re.compile(r"-r(\d+)-")


def rep_of(request_id):
    m = REP_RE.search(request_id or "")
    return int(m.group(1)) if m else None


def mean_std(vals, fmt="{:,.2f}"):
    """均值 ± 标准差。n==1 时不报标准差（报 0 会假装很确定）。"""
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return fmt.format(vals[0])
    return f"{fmt.format(statistics.mean(vals))} ± {fmt.format(statistics.stdev(vals))}"


def pct(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(len(vals) * q))]


def load_run(result_path):
    """一次运行 → 事件 + 客户端统计 + 实验条件。"""
    with open(result_path) as f:
        meta = json.load(f)

    rep = meta["run"]["rep"]
    # trace 文件是整个会话（同一 workload+并发度的多次重复）共用的，
    # 靠 request_id 里的 -rN- 切出属于本次重复的事件。
    # trace_glob 是 run_experiment.sh 在 ROOT 下起的相对路径（traces/xxx.*.jsonl）。
    # 不锚到 ROOT 的话，换个工作目录就静默 glob 不到 —— 静默是最糟的失败方式，
    # 因为报告会照常打印，只是所有 block 级数字都变成 0。
    pattern = meta.get("trace_glob") or ""
    if pattern and not os.path.isabs(pattern):
        pattern = os.path.join(ROOT, pattern)
        if not glob.glob(pattern):                  # 兼容从别处生成的结果
            pattern = meta["trace_glob"]
    events = []
    for p in sorted(glob.glob(pattern)):
        for e in analyze.load(p).get("events", []):
            if rep_of(e.get("request_id")) == rep:
                events.append(e)
    events.sort(key=lambda e: e.get("timestamp", 0))
    return meta, events


def run_stats(meta, events):
    """单次运行的 block 级统计。"""
    lifetimes = analyze.compute_block_lifetimes(events)
    reads = analyze.compute_access_counts(events)

    n_alloc = sum(1 for e in events if e["event"] == "alloc")
    n_cached = sum(1 for e in events if e["event"] == "alloc"
                   and e.get("source") == "cache")
    n_free = sum(1 for e in events if e["event"] == "free")

    live = peak_live = 0
    for e in events:
        if e["event"] == "alloc":
            live += 1
            peak_live = max(peak_live, live)
        elif e["event"] == "free":
            live -= 1

    lt = [d["lifetime"] for d in lifetimes]
    rd = list(reads.values())
    pool_blocks = meta["server"].get("kv_pool_blocks")
    unique = len({e["block_id"] for e in events})

    client = meta.get("client", {})
    return {
        "workload": meta["run"]["workload"],
        "concurrency": meta["run"]["concurrency"],
        "rep": meta["run"]["rep"],
        "wall_s": meta["run"]["wall_s"],
        "n_requests": client.get("n_ok", 0),
        "n_failed": client.get("n_failed", 0),
        # 逻辑 block 获取：调度器视角一共"要"了多少个 block，
        # 其中有多少直接来自 prefix cache（add_local_computed_blocks）
        "n_alloc": n_alloc,
        "n_cached": n_cached,
        "cache_frac": (n_cached / n_alloc) if n_alloc else 0.0,
        "n_free": n_free,
        "n_unclosed": n_alloc - len(lifetimes),
        "unique_blocks": unique,
        "pool_blocks": pool_blocks,
        "pool_util": (unique / pool_blocks) if pool_blocks else None,
        "peak_live": peak_live,
        "lifetime_p50": pct(lt, .5) if lt else None,
        "lifetime_p90": pct(lt, .9) if lt else None,
        "lifetime_max": max(lt) if lt else None,
        "lifetime_mean": statistics.mean(lt) if lt else None,
        "short_frac": (sum(1 for t in lt if t < 1.0) / len(lt)) if lt else None,
        "reads_p50": pct(rd, .5) if rd else None,
        "reads_p90": pct(rd, .9) if rd else None,
        "reads_max": max(rd) if rd else None,
        "reads_mean": statistics.mean(rd) if rd else None,
        "reads_total": sum(rd),
        "n_blocks_sampled": len(rd),
        "total_tok_per_s": client.get("total_tok_per_s"),
        "ttft_p90": (client.get("ttft_s") or {}).get("p90"),
        "lat_p90": (client.get("latency_s") or {}).get("p90"),
    }


#: 聚合时对每个字段取均值/标准差
AGG_FIELDS = [
    ("n_alloc", "{:,.0f}"), ("n_cached", "{:,.0f}"), ("cache_frac", "{:.3f}"),
    ("unique_blocks", "{:,.0f}"), ("pool_util", "{:.4f}"), ("peak_live", "{:,.0f}"),
    ("lifetime_p50", "{:.3f}"), ("lifetime_p90", "{:.3f}"),
    ("lifetime_mean", "{:.3f}"), ("short_frac", "{:.3f}"),
    ("reads_p50", "{:,.0f}"), ("reads_p90", "{:,.0f}"), ("reads_max", "{:,.0f}"),
    ("wall_s", "{:.1f}"), ("total_tok_per_s", "{:,.0f}"),
]


def aggregate(stats_list):
    return {k: mean_std([s[k] for s in stats_list], fmt)
            for k, fmt in AGG_FIELDS}


def load_all(workloads=None, concurrencies=None, results_dir=None):
    results_dir = results_dir or os.path.join(ROOT, "results")
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "*.c*.r*.json"))):
        meta, events = load_run(path)
        w, c = meta["run"]["workload"], meta["run"]["concurrency"]
        if workloads and w not in workloads:
            continue
        if concurrencies and c not in concurrencies:
            continue
        if not events:
            print(f"  警告: {os.path.basename(path)} 切不出任何事件 "
                  f"(rep={meta['run']['rep']}, glob={meta.get('trace_glob')})"
                  f" —— 这一轮会被算成 0，别当成真实结果", file=sys.stderr)
        groups[(w, c)].append((run_stats(meta, events), meta, events))
    return groups


def print_conditions(groups):
    """实验条件 —— 协议要求逐项记录，也是复现的前提。"""
    any_meta = next(iter(groups.values()))[0][1]
    s = any_meta["server"]
    print("=" * 78)
    print("实验条件")
    print("=" * 78)
    print(f"  模型            {any_meta['model']['path']}")
    print(f"  vLLM 版本       {any_meta['vllm_version']}")
    hw = any_meta.get("hardware") or {}
    print(f"  GPU             {hw.get('name')} "
          f"({hw.get('memory_total_mib')} MiB)")
    print(f"  GPU 热池预算    --gpu-memory-utilization "
          f"{s['gpu_memory_utilization']}  →  KV 池 {s['kv_pool_tokens']:,} tokens"
          f" = {s['kv_pool_blocks']:,} blocks")
    print(f"  block size      {s['block_size']}")
    print(f"  max model len   {s['max_model_len']}")
    print(f"  prefix cache    {'开' if s['prefix_caching'] else '关'}"
          f"（--enable-prefix-caching）")
    print(f"  随机种子        {any_meta['workload_spec']['seed']}")
    print(f"  总运行次数      {sum(len(v) for v in groups.values())}"
          f"（{len(groups)} 个组合 × 每组合多次重复）")


def print_workload_specs(groups):
    print()
    print("=" * 78)
    print("workload 规格与实测 token 分布")
    print("=" * 78)
    seen = {}
    for (w, _c), runs in groups.items():
        if w in seen:
            continue
        seen[w] = runs[0][1]["workload_spec"]

    print(f"{'workload':<17}{'请求数':>7}   "
          f"{'prompt tokens':<34}{'output tokens':<34}")
    print(f"{'':<17}{'':>7}   "
          f"{'min    p50    max    合计':<34}{'min    p50    max    合计':<34}")
    print("-" * 96)
    for w, spec in seen.items():
        p, o = spec["prompt_tokens"], spec["output_tokens"]
        print(f"{w:<17}{spec['n_requests']:>7}   "
              f"{p['min']:>5}{p['p50']:>7}{p['max']:>7}{p['total']:>10,}   "
              f"{o['min']:>5}{o['p50']:>7}{o['max']:>7}{o['total']:>10,}")


def print_core(groups, pool_tokens):
    """核心论点：逻辑 block 获取中来自 prefix cache 的比例。"""
    print()
    print("=" * 78)
    print("逻辑 block 获取中来自 prefix cache 的比例")
    print("=" * 78)
    ws = sorted({w for w, _ in groups})
    cs = sorted({c for _, c in groups})
    print(f"{'workload':<17}" + "".join(f"{'并发 ' + str(c):>24}" for c in cs))
    print("-" * (17 + 24 * len(cs)))
    for w in ws:
        cells = []
        for c in cs:
            runs = groups.get((w, c))
            if not runs:
                cells.append(f"{'-':>24}")
                continue
            agg = aggregate([s for s, _, _ in runs])
            cells.append(f"{agg['cache_frac']:>24}")
        print(f"{w:<17}" + "".join(cells))

    print()
    print("同一张表，换成「缓存命中数 / 逻辑获取总数」的绝对值：")
    print(f"{'workload':<17}" + "".join(f"{'并发 ' + str(c):>24}" for c in cs))
    print("-" * (17 + 24 * len(cs)))
    for w in ws:
        cells = []
        for c in cs:
            runs = groups.get((w, c))
            if not runs:
                cells.append(f"{'-':>24}")
                continue
            agg = aggregate([s for s, _, _ in runs])
            cells.append(f"{agg['n_cached'] + ' / ' + agg['n_alloc']:>24}")
        print(f"{w:<17}" + "".join(cells))


def print_lifetime_reads(groups):
    print()
    print("=" * 78)
    print("block 生命周期与访问次数（按 workload × 并发度）")
    print("=" * 78)
    hdr = (f"{'workload':<17}{'并发':>5}{'生命周期P50':>16}{'生命周期P90':>16}"
           f"{'存活<1s':>14}{'访问P50':>14}{'访问max':>14}{'访问总量':>16}")
    print(hdr)
    print("-" * len(hdr))
    for (w, c) in sorted(groups):
        st = [s for s, _, _ in groups[(w, c)]]
        agg = aggregate(st)
        reads_total = mean_std([s["reads_total"] for s in st], "{:,.0f}")
        print(f"{w:<17}{c:>5}{agg['lifetime_p50']:>16}{agg['lifetime_p90']:>16}"
              f"{agg['short_frac']:>14}{agg['reads_p50']:>14}"
              f"{agg['reads_max']:>14}{reads_total:>16}")


def print_pool(groups):
    print()
    print("=" * 78)
    print("池占用与逐出判定")
    print("=" * 78)
    hdr = (f"{'workload':<17}{'并发':>5}{'物理block数':>18}{'池容量':>9}"
           f"{'池利用率':>18}{'峰值同时存活':>16}{'墙钟(s)':>16}{'吞吐(tok/s)':>16}")
    print(hdr)
    print("-" * len(hdr))
    for (w, c) in sorted(groups):
        st = [s for s, _, _ in groups[(w, c)]]
        agg = aggregate(st)
        pool = st[0]["pool_blocks"]
        print(f"{w:<17}{c:>5}{agg['unique_blocks']:>18}{pool:>9}"
              f"{agg['pool_util']:>18}{agg['peak_live']:>16}"
              f"{agg['wall_s']:>16}{agg['total_tok_per_s']:>16}")


def print_repeatability(groups):
    """重复之间稳不稳 —— 这是「方差」这一栏可信与否的前提。"""
    print()
    print("=" * 78)
    print("重复一致性（同一组合内部，各次运行之间）")
    print("=" * 78)
    hdr = (f"{'workload':<17}{'并发':>5}{'重复次数':>9}"
           f"{'逻辑获取(各次)':>34}{'缓存比例(各次)':>40}")
    print(hdr)
    print("-" * len(hdr))
    for (w, c) in sorted(groups):
        st = [s for s, _, _ in groups[(w, c)]]
        allocs = ", ".join(f"{s['n_alloc']:,}" for s in st)
        fracs = ", ".join(f"{s['cache_frac']:.3f}" for s in st)
        print(f"{w:<17}{c:>5}{len(st):>9}{allocs:>34}{fracs:>40}")


def print_conclusions(groups):
    """README 可直接引用的句子。数字全部来自上面的聚合，不手写。"""
    print()
    print("=" * 78)
    print("结论句（数字由本脚本从数据算出，可直接引用）")
    print("=" * 78)
    any_meta = next(iter(groups.values()))[0][1]
    s = any_meta["server"]
    model = os.path.basename(any_meta["model"]["path"])
    n_reps = len(next(iter(groups.values())))

    def avg(st, key):
        vals = [x[key] for x in st if x[key] is not None]
        return statistics.mean(vals) if vals else float("nan")

    for (w, c) in sorted(groups):
        st = [x for x, _, _ in groups[(w, c)]]
        agg = aggregate(st)
        print(f"  [{w} / 并发 {c}]")
        print(f"    在 {model}、max-model-len={s['max_model_len']}、KV 池 "
              f"{s['kv_pool_tokens']:,} tokens、prefix cache 开启、"
              f"并发 {c} 下，{n_reps} 次重复取均值 —")
        print(f"    逻辑 block 获取 {agg['n_alloc']} 次，其中来自 prefix cache "
              f"{agg['n_cached']} 次，占比 {avg(st, 'cache_frac') * 100:.1f}%"
              f"（各次 {agg['cache_frac']}）；")
        print(f"    block 生命周期 P50 {agg['lifetime_p50']}s / P90 "
              f"{agg['lifetime_p90']}s，存活 <1s 占 {agg['short_frac']}；")
        print(f"    单 block 访问次数 P50 {agg['reads_p50']} / max {agg['reads_max']}；")
        print(f"    池利用率 {agg['pool_util']}，峰值同时存活 block {agg['peak_live']}。")
        print()


def main():
    ap = argparse.ArgumentParser(description="正式实验聚合报告")
    ap.add_argument("--workload", action="append", default=None)
    ap.add_argument("--concurrency", action="append", type=int, default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--quiet", action="store_true", help="只输出结论句")
    args = ap.parse_args()

    groups = load_all(args.workload, args.concurrency, args.results_dir)
    if not groups:
        print("没有找到 results/*.c*.r*.json —— 先跑 workload/run_experiment.sh")
        return 1

    any_meta = next(iter(groups.values()))[0][1]
    if args.quiet:
        print_conclusions(groups)
        return 0

    print_conditions(groups)
    print_workload_specs(groups)
    print_core(groups, any_meta["server"]["kv_pool_tokens"])
    print_lifetime_reads(groups)
    print_pool(groups)
    print_repeatability(groups)
    print_conclusions(groups)
    return 0


if __name__ == "__main__":
    sys.exit(main())
