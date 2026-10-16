"""
measure_pcie.py — Step 1：链路上限微基准

Step 0 已确认链路不是卡在 Gen1（空闲 2.5 GT/s → 负载下 100% 16.0 GT/s，4 ms 升速）。
本步要回答的是**生产粒度下到底能跑多快**。

为什么「生产形状」必须是默认形状
============================================================

我原先的假设是「一个 block 拷贝 = 1 次 875 KB memcpy」。逐条读源码后这是错的：

  * `_select_swap_blocks_fn` 比较的是**逐 data_ref** 的 `page_size_bytes`
    （`kv_offload/cpu/gpu_worker.py:49-58`），即**单层**大小 = 32,768 B，
    不是整块的 917,504 B。
  * 阈值 `THRESHOLD_BYTES = 28,672`（`swap_blocks_triton.py:20`）
    —— 余量**只有 14.3%**，不是 14 倍。
  * 描述符按 `(group, data_ref)` 展开（`gpu_worker.py:283-357`），
    于是 **1 个 block = 28 个 32,768 B 的描述符**，装在一次 `cuMemcpyBatchAsync` 里。

32,768 B ÷ 24 GB/s = 1.37 µs，**低于 DMA 启动开销**（~2–5 µs）。所以渐近上限
（64 MB 连续）很可能在生产粒度上**根本达不到**，而按 875 KB 连续测出来的数会
系统性高估 —— 高估的方向正好是让迁移层显得划算的方向。

本脚本因此把生产形状作第一公民，连续大块只作上界参照，并单独做一组
**粒度隔离**（总字节数相同、描述符大小不同），把「每 op 开销」从「链路带宽」里分出来。

用法:
    python scripts/measure_pcie.py
    python scripts/measure_pcie.py --sm-hog          # 加 SM 占用对照
    python scripts/measure_pcie.py --out measurements/pcie_link_ceiling.txt
"""

import argparse
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from pcie_common import (  # noqa: E402
    BLOCK_BYTES, LAYERS, PER_LAYER_BLOCK_BYTES, KVBlockStore, LinkSampler,
    gpu_pci_path, issue_batch, numa_placement,
)

#: 每组统计跑多少字节。太小则计时噪声占比高，太大则总时长失控。
TARGET_BYTES_PER_GROUP = 48 * 2**20
GROUPS = 5
WARMUP_CYCLES = 8


def _store_cache(n_blocks, n_layers, per_layer_bytes):
    """按形状缓存 store。重建一次要重新锁页 900 MB，不能每组都建。"""
    key = (n_blocks, n_layers, per_layer_bytes)
    if key not in _store_cache.pool:
        _store_cache.pool[key] = KVBlockStore(n_blocks, n_layers, per_layer_bytes)
    return _store_cache.pool[key]


_store_cache.pool = {}


def bench_one(store, direction: str, n_blocks: int, stream,
              groups: int = GROUPS, sampler: LinkSampler | None = None) -> dict:
    """跑 shape 相同的一串 batch，返回吞吐与每单位延迟。

    block id 每组**轮换**，不重复用同一段缓冲：真实 workload 每次搬的是不同的
    block，而反复搬同一段会命中 GPU 侧 L2、主机侧 cache，把带宽抬高。
    """
    block_bytes = store.block_bytes
    per_group = max(1, math.ceil(TARGET_BYTES_PER_GROUP / (n_blocks * block_bytes)))
    # 大 shape 时 per_group 会很小；小 shape 时别让它爆掉
    per_group = min(per_group, 4096)

    n_dirs = 2 if direction == "both" else 1
    params = {}
    streams = {}
    if direction == "both":
        s2 = torch.cuda.Stream()
        streams = {"a": stream, "b": s2}
        params = {"a": store.params_d2h(stream), "b": store.params_h2d(s2)}
    else:
        streams = {"a": stream}
        params = {"a": store.params(direction, stream)}

    # 轮换窗口：base 每次前进 n_blocks，到头回绕
    stride = store.n - n_blocks + 1 if store.n > n_blocks else 1
    base = [0]

    def one_cycle():
        b0 = base[0]
        ids = list(range(b0, b0 + n_blocks))
        for key in streams:
            issue_batch(store, params[key], ids)
        base[0] = (b0 + n_blocks) % max(stride, 1)

    for _ in range(WARMUP_CYCLES):
        one_cycle()
    for st in streams.values():
        st.synchronize()

    ev = {k: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for k in streams}
    per_group_ms = []
    for _ in range(groups):
        for k, st in streams.items():
            ev[k][0].record(st)
        for _ in range(per_group):
            one_cycle()
        for k, st in streams.items():
            ev[k][1].record(st)
        for k, st in streams.items():
            ev[k][1].synchronize()
        ms = max(ev[k][0].elapsed_time(ev[k][1]) for k in streams)
        per_group_ms.append(ms)

    ms = statistics.median(per_group_ms)
    n_transfers = per_group * n_blocks * n_dirs
    total_bytes = n_transfers * block_bytes
    seconds = ms / 1000.0

    return {
        "n_blocks": n_blocks,
        "n_layers": store.n_layers,
        "per_layer_bytes": store.per_layer_bytes,
        "n_descriptors": n_transfers * store.n_layers,
        "direction": direction,
        "cycles_per_group": per_group,
        "ms": ms,
        "ms_min": min(per_group_ms),
        "ms_max": max(per_group_ms),
        "gbps": total_bytes / seconds / 1e9 if seconds else 0.0,
        "blocks_per_s": n_transfers / seconds if seconds else 0.0,
        "us_per_block": seconds * 1e6 / n_transfers if n_transfers else 0.0,
        "us_per_descriptor": (seconds * 1e6 / n_transfers / store.n_layers
                              if n_transfers else 0.0),
        "block_bytes": block_bytes,
    }


def measure_submit_cost(store, stream, direction: str, n: int = 300) -> dict:
    """量纯提交开销（描述符构建 + 调用），不含 DMA 等待。

    做法：一次只提交 1 个 block 的 batch（875 KB，约 53 µs 的 DMA），
    连着提交 n 次不同步。队列足够深时，这段墙钟时间就是纯 CPU 成本。
    若 submit 与单次传输时间可比，**整份测量都不可信** —— 测到的是
    Python 的上限，不是链路的上限。
    """
    params = store.params(direction, stream)
    for _ in range(10):
        issue_batch(store, params, [0])
    stream.synchronize()

    t0 = time.perf_counter()
    for i in range(n):
        issue_batch(store, params, [i % store.n])
    t1 = time.perf_counter()
    stream.synchronize()
    return {"us_per_submit": (t1 - t0) * 1e6 / n, "n": n}


def main() -> int:
    ap = argparse.ArgumentParser(description="PCIe 链路上限微基准")
    ap.add_argument("--out", default="measurements/pcie_link_ceiling.txt")
    ap.add_argument("--sm-hog", action="store_true",
                    help="加对照：并行跑满占用 kernel，看吞吐是否掉（掉=走的是 SM 不是 CE）")
    args = ap.parse_args()

    pci = gpu_pci_path()
    out: list[str] = []
    say = out.append

    say("=" * 78)
    say("Step 1 — PCIe 链路上限微基准")
    say("=" * 78)
    say(f"  生产形状  1 block = {LAYERS} 描述符 × {PER_LAYER_BLOCK_BYTES:,} B"
        f" = {BLOCK_BYTES:,} B = {BLOCK_BYTES / 2**20:.3f} MiB")
    say(f"  拷贝原语  vllm.v1.simple_kv_offload.cuda_mem_ops.copy_blocks")
    say(f"            （cuGetProcAddress(b'cuMemcpyBatchAsync', 12080) 的 v1 符号，")
    say(f"              numAttrs=1 / srcAccessOrder=ANY —— 与生产路径逐字相同）")
    say(f"  计时      CUDA event elapsed_time；每组 {GROUPS} 次取中位数")
    say(f"  block id  每组轮换，避免反复搬同一段命中 L2 / 主机 cache")
    say("")

    sampler = LinkSampler(pci, interval=0.02)
    sampler.start()

    stream = torch.cuda.Stream()
    rows: list[dict] = []

    # ---------------- A. 生产形状扫描 ----------------
    say("A. 生产形状批量扫描（1 block = 28 × 32,768 B）")
    say("-" * 78)
    say(f"  {'方向':<5} {'blocks':>7} {'描述符':>8} {'GB/s':>8} {'block/s':>10} "
        f"{'µs/block':>10} {'µs/desc':>9}")
    A_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for direction in ("d2h", "h2d"):
        for n in A_SIZES:
            store = _store_cache(max(n, 512), LAYERS, PER_LAYER_BLOCK_BYTES)
            r = bench_one(store, direction, n, stream, sampler=sampler)
            rows.append(r)
            say(f"  {direction:<5} {n:>7} {r['n_descriptors']:>8} {r['gbps']:>8.2f} "
                f"{r['blocks_per_s']:>10,.0f} {r['us_per_block']:>10.2f} "
                f"{r['us_per_descriptor']:>9.2f}")
        say("")

    # ---------------- B. 粒度隔离 ----------------
    say("B. 粒度隔离（总字节数固定 = 128 block × 917,504 B，只改描述符大小）")
    say("-" * 78)
    say(f"  {'描述符布局':<22} {'描述符数':>10} {'GB/s':>8} {'per-op 时延':>12}")
    TOTAL_BLOCKS = 128
    for n_layers, plb in ((1, BLOCK_BYTES), (14, BLOCK_BYTES // 14),
                          (28, PER_LAYER_BLOCK_BYTES), (56, PER_LAYER_BLOCK_BYTES // 2),
                          (112, PER_LAYER_BLOCK_BYTES // 4)):
        if n_layers * plb != BLOCK_BYTES:
            continue
        store = _store_cache(TOTAL_BLOCKS, n_layers, plb)
        r = bench_one(store, "d2h", TOTAL_BLOCKS, stream, sampler=sampler)
        r["case"] = "granularity"
        rows.append(r)
        say(f"  {f'{n_layers} × {plb:,} B':<22} {r['n_descriptors']:>10,} "
            f"{r['gbps']:>8.2f} {r['us_per_descriptor']:>11.2f}µs")
    say("")

    # ---------------- C. 渐近上限 ----------------
    say("C. 渐近上限（大块连续，单方向）")
    say("-" * 78)
    say(f"  {'布局':<24} {'GB/s':>8} {'备注':<30}")
    for label, n_blocks, n_layers, plb, note in (
        ("1 × 8 MiB × 8", 8, 1, 8 * 2**20, "8 个 8 MiB 描述符"),
        ("1 × 64 MiB × 1", 1, 1, 64 * 2**20, "单个 64 MiB 描述符"),
        ("28 × 2 MiB × 4", 4, 28, 2 * 2**20, "112 个 2 MiB 描述符"),
    ):
        store = _store_cache(n_blocks, n_layers, plb)
        r = bench_one(store, "d2h", n_blocks, stream, sampler=sampler)
        r["case"] = "asymptotic"
        rows.append(r)
        say(f"  {label:<24} {r['gbps']:>8.2f}  {note}")

    # 生产形状在最大 batch 下的数，紧挨着放，方便对比
    store = _store_cache(512, LAYERS, PER_LAYER_BLOCK_BYTES)
    r_max = bench_one(store, "d2h", 512, stream, sampler=sampler)
    say(f"  {'生产形状 (512 block)':<24} {r_max['gbps']:>8.2f}  "
        f"512 × 28 = {r_max['n_descriptors']:,} 个 32 KB 描述符")
    say("")

    # ---------------- D. 双向可加性 ----------------
    say("D. 双向可加性（生产形状，512 block/cycle）")
    say("-" * 78)
    r_both = bench_one(store, "both", 512, stream, sampler=sampler)
    rows.append(r_both)
    d2h = r_max["gbps"]
    h2d = next((r["gbps"] for r in rows
                if r.get("n_blocks") == 512 and r.get("direction") == "h2d"), None)
    say(f"  d2h 单独        {d2h:>7.2f} GB/s")
    if h2d:
        say(f"  h2d 单独        {h2d:>7.2f} GB/s")
        say(f"  两者之和        {d2h + h2d:>7.2f} GB/s   （若真可加，双向应接近此值）")
    say(f"  同时跑（聚合）  {r_both['gbps']:>7.2f} GB/s")
    if h2d:
        say(f"  → 同时/单向max = {r_both['gbps'] / max(d2h, h2d):.2f}，"
            f"同时/单向和 = {r_both['gbps'] / (d2h + h2d):.2f}")
        if r_both["gbps"] < 1.25 * max(d2h, h2d):
            say("  → **两个方向不可加，共享同一个上限**。曲线横轴必须用聚合字节/s。")
        else:
            say("  → 两个方向部分可加。")
    say("")

    # ---------------- 控制项 ----------------
    say("E. 控制项")
    say("-" * 78)
    sc = measure_submit_cost(store, stream, "d2h")
    prod = next(r for r in rows if r.get("n_blocks") == 1 and r.get("direction") == "d2h")
    ratio = prod["us_per_block"] / sc["us_per_submit"] if sc["us_per_submit"] else float("inf")
    say(f"  纯提交开销           {sc['us_per_submit']:.2f} µs/batch（{sc['n']} 次）")
    say(f"  1 block 的传输时间   {prod['us_per_block']:.2f} µs")
    say(f"  提交/传输            {1 / ratio:.3f}"
        f"  → 提交占传输的 {1 / ratio * 100:.1f}%")
    if ratio < 2:
        say("  ⚠️ 提交开销与传输时间可比 —— 测到的是 Python 的上限，不是链路的上限！")
    else:
        say("  ✓ 提交开销不是瓶颈（传输时间是提交的 %.1f 倍）" % ratio)

    if args.sm_hog:
        say("")
        say("  跑 SM 占用对照（满占用 matmul 循环 + 同时做 DMA）...")
        stop = {"v": False}
        import threading

        def hog():
            a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
            while not stop["v"]:
                for _ in range(20):
                    a @ b

        th = threading.Thread(target=hog, daemon=True)
        th.start()
        torch.cuda.synchronize()
        r_hog = bench_one(store, "d2h", 512, stream, sampler=sampler)
        stop["v"] = True
        th.join(timeout=5)
        say(f"  无 SM 争用  {r_max['gbps']:.2f} GB/s")
        say(f"  有 SM 争用  {r_hog['gbps']:.2f} GB/s")
        say(f"  比值        {r_hog['gbps'] / r_max['gbps']:.2f}"
            f"   （≈1 → 走 CE；≪1 → 走的是 SM kernel）")

    say("")
    sampler.stop()
    say("F. 链路代际（负载期间采样）")
    say("-" * 78)
    for s, n in sampler.histogram().items():
        say(f"  {s:<20} {n:>6} 次   {n / max(len(sampler.samples), 1):6.1%}")

    say("")
    say("=" * 78)
    say("结论用数")
    say("=" * 78)
    say(f"  生产形状 (512 block, d2h)  {r_max['gbps']:.2f} GB/s"
        f"  = {r_max['blocks_per_s']:,.0f} block/s"
        f"  = {r_max['us_per_block']:.2f} µs/block")

    text = "\n".join(out) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
