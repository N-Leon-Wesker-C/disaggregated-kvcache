"""
analyze.py — 分析 KV Cache trace，提取 block 的生命周期和访问模式

支持两种输入：
  - kvcache_trace.<pid>.jsonl  增量格式（每行一条事件），tracer 的正式产出。
                               EngineCore 是 fork 出的独立进程，事件由子进程
                               边产生边落盘，所以一个 pid 一个文件，可以传多个。
  - kvcache_trace.json         旧的汇总格式（一个 JSON 对象），兼容保留。

用法:
    python scripts/analyze.py kvcache_trace.*.jsonl
    python scripts/analyze.py kvcache_trace.1234.jsonl kvcache_trace.5678.jsonl
"""

import glob
import json
import sys
from collections import defaultdict


def load(path):
    """读单个 trace 文件，返回 {"events": [...]}。

    .jsonl 每行一条事件；.json（旧格式）整个文件是一个对象。
    """
    if path.endswith(".jsonl"):
        events = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return {"events": events}
    with open(path) as f:
        return json.load(f)


def compute_block_lifetimes(events):
    """返回**每一次**获取→释放区间（每段一个 dict），不是每个 block 一条。

    一个 block 会被获取/释放多次：prefix cache 命中的 block 会先后服务
    多个请求，block pool 里的物理块也会被反复复用。所以不能按 block_id
    覆盖着记 —— 那样会丢掉绝大部分区间（实测 784 次 alloc 只对应 220 个
    block，覆盖写法扔掉 72% 的数据），而且还会把不同请求的时间戳错配成
    一段根本不存在的"生命周期"：

        R1 获取 @t=1, R2 获取 @t=2, R1 释放 @t=3, R2 释放 @t=5
        覆盖写法 → {alloc:2, free:5, lifetime:3}   ← 对应不到任何真实区间

    配对规则用 FIFO：释放事件闭合「最早那次尚未闭合的获取」。block 是
    ref-count 的，谁先获取谁先释放，FIFO 才对应真实的持有关系。
    """
    open_acquisitions = defaultdict(list)   # block_id -> [(t, rid, source), ...]
    lifetimes = []

    for e in events:
        blk = e["block_id"]
        if e["event"] == "alloc":
            open_acquisitions[blk].append(
                (e["timestamp"], e.get("request_id", "?"), e.get("source", "new")))
        elif e["event"] == "free":
            pending = open_acquisitions.get(blk)
            if pending:
                alloc_time, request_id, source = pending.pop(0)
                lifetimes.append({
                    "block_id": blk,
                    "alloc_time": alloc_time,
                    "free_time": e["timestamp"],
                    "lifetime": e["timestamp"] - alloc_time,
                    "request_id": request_id,
                    "source": source,
                })
    return lifetimes


def compute_access_counts(events):
    """每个 block 被读取的次数 —— hot/cold skew 的数据源。

    首选：free 事件里的 "reads" 字段。tracer 在每个 decode step 通过
    allocate_slots 累加计数，block 释放时一次性写出。这才是真实的
    「这个 block 被读了几次」。

    退化：数 access 事件。它来自 cache_blocks，只在 block 写满时触发
    一次，所以每个 block 恒为 1 —— 那回答不了 hot/cold 问题，属于
    没有 reads 字段时的兜底。
    """
    reads = defaultdict(int)
    access = defaultdict(int)
    for e in events:
        if e["event"] == "free" and "reads" in e:
            # 累加而不是覆盖：一个 block 在生命周期里可能被获取/释放多次
            # （prefix cache 命中会让同一个 block 服务多个请求），tracer 里
            # 的计数器在每次释放时清零，所以这里要把各段相加才是总读取次数。
            reads[e["block_id"]] += e["reads"]
        elif e["event"] == "access":
            access[e["block_id"]] += 1
    return dict(reads) if reads else dict(access)


def compute_blocks_per_request(events):
    req_blocks = defaultdict(set)
    for e in events:
        if "request_id" not in e:
            continue
        req_blocks[e["request_id"]].add(e["block_id"])
    return {rid: len(blocks) for rid, blocks in req_blocks.items()}


def compute_request_lifetimes(events):
    req_times = defaultdict(lambda: {"first_alloc": float("inf"), "last_free": 0})
    for e in events:
        rid = e.get("request_id")
        if not rid:
            continue
        if e["event"] == "alloc":
            req_times[rid]["first_alloc"] = min(
                req_times[rid]["first_alloc"], e["timestamp"])
        elif e["event"] in ("free", "free_request"):
            req_times[rid]["last_free"] = max(
                req_times[rid]["last_free"], e["timestamp"])
    return {
        rid: {
            "first_alloc": t["first_alloc"],
            "last_free": t["last_free"],
            "lifetime": max(0, t["last_free"] - t["first_alloc"]),
        }
        for rid, t in req_times.items()
    }


def print_statistics(lifetimes, access_counts, blocks_per_req, req_lifetimes):
    print("=" * 60)
    print("KV Cache Trace Analysis")
    print("=" * 60)

    # lifetimes 现在是「获取区间」列表，不是「每个 block 一条」
    valid_lt = [d["lifetime"] for d in lifetimes]
    if valid_lt:
        valid_lt.sort()
        n = len(valid_lt)
        n_blocks = len({d["block_id"] for d in lifetimes})
        n_cached = sum(1 for d in lifetimes if d["source"] == "cache")
        print(f"\n--- Block Lifetimes ({n} 段区间, "
              f"覆盖 {n_blocks} 个物理 block) ---")
        print(f"  来源: 新分配 {n - n_cached}, 缓存命中 {n_cached}")
        print(f"  Avg:   {sum(valid_lt)/n:.4f}s")
        print(f"  Median:{valid_lt[n//2]:.4f}s")
        print(f"  P10:   {valid_lt[n//10]:.4f}s")
        print(f"  P90:   {valid_lt[n*9//10]:.4f}s")
        print(f"  Max:   {valid_lt[-1]:.4f}s")
        short = sum(1 for t in valid_lt if t < 1.0)
        medium = sum(1 for t in valid_lt if 1.0 <= t < 10.0)
        long = sum(1 for t in valid_lt if t >= 10.0)
        print(f"  <1s: {short} ({short/n*100:.1f}%)  "
              f"1-10s: {medium} ({medium/n*100:.1f}%)  "
              f">10s: {long} ({long/n*100:.1f}%)")

    if access_counts:
        counts = sorted(access_counts.values())
        n = len(counts)
        print(f"\n--- Block Access Frequency ({n} blocks) ---")
        print(f"  Avg:   {sum(counts)/n:.1f}")
        print(f"  Median:{counts[n//2]:.0f}")
        print(f"  P90:   {counts[n*9//10]:.0f}")
        print(f"  Max:   {counts[-1]:.0f}")
        cold = sum(1 for c in counts if c <= 5)
        warm = sum(1 for c in counts if 5 < c <= 50)
        hot = sum(1 for c in counts if c > 50)
        print(f"  Cold (≤5): {cold} ({cold/n*100:.1f}%)  "
              f"Warm (6-50): {warm} ({warm/n*100:.1f}%)  "
              f"Hot (>50): {hot} ({hot/n*100:.1f}%)")

    if blocks_per_req:
        vals = sorted(blocks_per_req.values())
        n = len(vals)
        print(f"\n--- Blocks Per Request ({n} requests) ---")
        print(f"  Avg:   {sum(vals)/n:.1f}")
        print(f"  Median:{vals[n//2]:.0f}")
        print(f"  Min/Max: {vals[0]}/{vals[-1]}")

    if req_lifetimes:
        vals = sorted(d["lifetime"] for d in req_lifetimes.values())
        n = len(vals)
        print(f"\n--- Request Lifetimes ({n} requests) ---")
        print(f"  Avg:   {sum(vals)/n:.2f}s")
        print(f"  Median:{vals[n//2]:.2f}s")
        print(f"  Max:   {vals[-1]:.2f}s")


def print_swap_insights(lifetimes, access_counts, blocks_per_req):
    print("\n" + "=" * 60)
    print("INSIGHTS FOR GPU-SWAP POLICY DESIGN")
    print("=" * 60)

    valid_lt = [d["lifetime"] for d in lifetimes]
    if valid_lt:
        short_ratio = sum(1 for t in valid_lt if t < 1.0) / len(valid_lt)
        print(f"\n[Insight 1] {short_ratio*100:.0f}% of blocks live < 1 second.")
        print(f"  → Swapping them to CPU is wasted work.")
        print(f"  → Policy rule: don't swap blocks from short requests.")

    if access_counts:
        counts = sorted(access_counts.values())
        hot_ratio = sum(1 for c in counts if c > 50) / len(counts) if counts else 0
        print(f"\n[Insight 2] {hot_ratio*100:.0f}% of blocks are accessed >50 times.")
        print(f"  → Protect hot blocks from eviction.")

    if blocks_per_req:
        vals = list(blocks_per_req.values())
        if len(vals) > 1 and max(vals) > min(vals) * 3:
            print(f"\n[Insight 3] Wide variance in blocks per request: "
                  f"{min(vals)}–{max(vals)}.")
            print(f"  → Request-type-aware swap strategy is justified.")
        else:
            print(f"\n[Insight 3] Blocks per request is uniform "
                  f"({min(vals)}–{max(vals)}).")
            print(f"  → Focus on block-level heuristics instead.")

    print(f"\n[Insight 4] For fragmentation analysis: track allocated vs free "
          f"block count over time, compute fragmentation ratio.")
    print()


def main():
    patterns = sys.argv[1:] or ["kvcache_trace.json"]

    # 展开 glob（shell 没展开时也照常工作），一个 pid 一个文件
    paths = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        paths.extend(hits if hits else [pat])

    events = []
    for path in paths:
        try:
            data = load(path)
        except FileNotFoundError:
            print(f"Error: {path} not found. Run the tracer first.")
            sys.exit(1)
        got = data.get("events", data)
        events.extend(got)
        print(f"  {path}: {len(got)} events")

    if not events:
        print("Error: no events in trace file.")
        sys.exit(1)

    # 各进程的时间戳都基于 fork 时继承的同一个 start_time，可直接混合排序
    events.sort(key=lambda e: e.get("timestamp", 0))
    print(f"Loaded {len(events)} events from {len(paths)} file(s)")

    lifetimes = compute_block_lifetimes(events)

    # 获取了但没等到释放的区间会被丢掉（trace 结束时请求还在跑，或被中断）。
    # 不吭声地丢数据容易让人误以为统计是完整的，所以显式报出来。
    n_alloc = sum(1 for e in events if e["event"] == "alloc")
    n_unclosed = n_alloc - len(lifetimes)
    if n_unclosed:
        print(f"  注意: {n_unclosed} 次获取没有配对的释放事件，"
              f"未计入生命周期统计（trace 截断时正常）")

    access_counts = compute_access_counts(events)
    blocks_per_req = compute_blocks_per_request(events)
    req_lifetimes = compute_request_lifetimes(events)

    print_statistics(lifetimes, access_counts, blocks_per_req, req_lifetimes)
    print_swap_insights(lifetimes, access_counts, blocks_per_req)


if __name__ == "__main__":
    main()
