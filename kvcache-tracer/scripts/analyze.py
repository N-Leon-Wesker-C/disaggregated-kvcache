"""
analyze.py — 分析 kvcache_trace.json，提取 KV Cache block 的生命周期和访问模式

用法:
    python scripts/analyze.py kvcache_trace.json
"""

import json
import sys
from collections import defaultdict


def load(path):
    with open(path) as f:
        return json.load(f)


def compute_block_lifetimes(events):
    lifetimes = {}
    for e in events:
        blk = e["block_id"]
        if e["event"] == "alloc":
            lifetimes[blk] = {
                "alloc_time": e["timestamp"],
                "free_time": None,
                "lifetime": None,
                "request_id": e.get("request_id", "?"),
            }
        elif e["event"] == "free":
            if blk in lifetimes:
                lifetimes[blk]["free_time"] = e["timestamp"]
                lifetimes[blk]["lifetime"] = (
                    e["timestamp"] - lifetimes[blk]["alloc_time"])
    return lifetimes


def compute_access_counts(events):
    counts = defaultdict(int)
    for e in events:
        if e["event"] == "access":
            counts[e["block_id"]] += 1
    return dict(counts)


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

    valid_lt = [d["lifetime"] for d in lifetimes.values() if d["lifetime"] is not None]
    if valid_lt:
        valid_lt.sort()
        n = len(valid_lt)
        print(f"\n--- Block Lifetimes ({n} blocks) ---")
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

    valid_lt = [d["lifetime"] for d in lifetimes.values() if d["lifetime"] is not None]
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
    path = sys.argv[1] if len(sys.argv) > 1 else "kvcache_trace.json"

    try:
        data = load(path)
    except FileNotFoundError:
        print(f"Error: {path} not found. Run the tracer first.")
        sys.exit(1)

    events = data.get("events", data)
    if not events:
        print("Error: no events in trace file.")
        sys.exit(1)

    print(f"Loaded {len(events)} events")

    lifetimes = compute_block_lifetimes(events)
    access_counts = compute_access_counts(events)
    blocks_per_req = compute_blocks_per_request(events)
    req_lifetimes = compute_request_lifetimes(events)

    print_statistics(lifetimes, access_counts, blocks_per_req, req_lifetimes)
    print_swap_insights(lifetimes, access_counts, blocks_per_req)


if __name__ == "__main__":
    main()
