"""
probe_pcie_link.py — Step 0 闸门：链路到底跑在几代？

为什么这是个闸门而不是一个数
============================================================

实测空闲态是 `2.5 GT/s`（Gen1），而设备与主机都支持 `16.0 GT/s`（Gen4）。
Gen1×16 有效带宽约 3.2 GB/s，比 replay 里假定的 24 GB/s **低 7 倍**。

两种可能，处理方式完全相反：

  空闲降速（ASPM / P-state）  负载一到就升回 Gen4   → 测量时预热即可，正常
  真的卡住（直通配置/驱动）    一直停在 Gen1        → 那是硬事实，结论要按 3.2 GB/s 重写

分不清就会写出一份**自信但错 7 倍**的报告。所以先测这个，通过了才往下做。

判据（收紧过）：升速窗口内 **≥95% 的采样点是 16.0 GT/s** 才算通过。

负载用**生产形状**（28 × 32,768 B 的 batch），不是 875 KB 连续拷贝 ——
否则测的是另一种负载下的链路状态，而链路升速行为恰恰可能依赖请求的形状。

用法:
    python scripts/probe_pcie_link.py                    # 默认跑 10 秒负载
    python scripts/probe_pcie_link.py --duration 20
    python scripts/probe_pcie_link.py --out measurements/pcie_link_probe.txt
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from pcie_common import (  # noqa: E402
    BLOCK_BYTES, LAYERS, PER_LAYER_BLOCK_BYTES, TRITON_THRESHOLD_BYTES,
    KVBlockStore, LinkSampler, gpu_pci_path, numa_placement, run_load,
)

GEN1_GTPS = 2.5
GEN4_GTPS = 16.0
#: 判据阈值：窗口内 Gen4 的时间占比
PASS_SHARE = 0.95

#: 每 cycle 拷多少个 block。1024 块 ≈ 940 MB，@24GB/s 约 39 ms —— 够细，
#: 能看清升速过程，又足够长到让队列一直有活干。
BLOCKS_PER_CYCLE = 1024


def env_report(pci: str) -> list[str]:
    out = ["环境", "-" * 70]

    def cat(name, path):
        try:
            with open(os.path.join(pci, path)) as f:
                return f.read().strip()
        except OSError:
            return "?"

    out.append(f"  GPU PCIe 地址        {os.path.basename(pci)}")
    out.append(f"  numa_node            {cat('', 'numa_node')}")
    out.append(f"  local_cpulist        {cat('', 'local_cpulist')}")
    out.append(f"  current_link_speed   {cat('', 'current_link_speed')}")
    out.append(f"  current_link_width   {cat('', 'current_link_width')}")
    out.append(f"  max_link_speed       {cat('', 'max_link_speed')}")
    out.append(f"  max_link_width       {cat('', 'max_link_width')}")

    try:
        cs = subprocess.run(["cat", "/sys/fs/cgroup/cpuset.cpus.effective"],
                            capture_output=True, text=True).stdout.strip()
    except OSError:
        cs = "?"
    out.append(f"  cgroup cpuset        {cs}   (nproc={os.cpu_count()})")

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=pstate,clocks.sm,power.draw,power.limit,"
                            "temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        smi = "?"
    out.append(f"  GPU 状态             {smi}")

    out.append(f"  生产形状             1 block = {LAYERS} 个描述符 × "
               f"{PER_LAYER_BLOCK_BYTES:,} B = {BLOCK_BYTES:,} B")
    out.append(f"  Triton 阈值          {TRITON_THRESHOLD_BYTES:,} B"
               f"（比较对象是**每层** {PER_LAYER_BLOCK_BYTES:,} B，"
               f"余量 {(PER_LAYER_BLOCK_BYTES / TRITON_THRESHOLD_BYTES - 1) * 100:.1f}%）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="PCIe 链路代际诊断（Step 0 闸门）")
    ap.add_argument("--duration", type=float, default=10.0, help="负载持续秒数")
    ap.add_argument("--blocks-per-cycle", type=int, default=BLOCKS_PER_CYCLE)
    ap.add_argument("--direction", default="d2h", choices=["d2h", "h2d", "both"])
    ap.add_argument("--out", default=None, help="另存一份报告")
    args = ap.parse_args()

    pci = gpu_pci_path()
    lines: list[str] = []
    say = lines.append

    say("=" * 70)
    say("Step 0 — PCIe 链路代际诊断")
    say("=" * 70)
    lines.extend(env_report(pci))
    say("")

    sampler = LinkSampler(pci, interval=0.01)

    # --- 空闲基线 ---
    sampler.start()
    time.sleep(2.0)
    idle = sampler.histogram()
    say("空闲态（2 秒，无负载）")
    say("-" * 70)
    for s, n in idle.items():
        say(f"  {s:<20} {n:>5} 次")
    idle_gt = sampler.gtps(max(idle, key=idle.get)) if idle else -1.0
    say("")

    # --- 建缓冲并核验放置 ---
    n_blocks = max(args.blocks_per_cycle, 1024)
    store = KVBlockStore(n_blocks)
    torch.cuda.synchronize()

    host_ptr = store.host[0].data_ptr()
    placement = numa_placement(host_ptr)
    say("pinned 缓冲放置（/proc/self/numa_maps；numactl 未安装，只能事后核验）")
    say("-" * 70)
    say(f"  host[0] @ {hex(host_ptr)}  {store.host_bytes / 2**20:.0f} MiB/层")
    say(f"  {placement or '（没找到映射，可能已被 register 拆散）'}")
    say("")

    # --- 带负载 ---
    stream = torch.cuda.Stream()
    say(f"施加负载：{args.direction} 方向，生产形状 "
        f"{args.blocks_per_cycle} block/cycle × {LAYERS} 描述符，{args.duration} 秒")
    say("-" * 70)

    t_load = time.monotonic()
    cycles, elapsed_ms, total_bytes = run_load(
        store, args.direction, args.duration, args.blocks_per_cycle, stream,
        deadline=t_load + args.duration,
    )
    t_load_end = time.monotonic()
    sampler.stop()

    gbps = total_bytes / (elapsed_ms / 1000) / 1e9 if elapsed_ms > 0 else 0.0
    blocks_per_s = cycles * args.blocks_per_cycle / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0
    say(f"  {cycles} cycle，{elapsed_ms / 1000:.2f} s")
    say(f"  {total_bytes / 1e9:.2f} GB  →  {gbps:.2f} GB/s（十进制）"
        f" = {gbps * 1e9 / 2**30:.2f} GiB/s")
    say(f"  {blocks_per_s:,.0f} block/s   （单块 {1e6 / blocks_per_s:.2f} µs）" if blocks_per_s
        else "  （无有效吞吐）")
    say("")

    # --- 升速过程 ---
    loaded = [(t, s, w) for t, s, w in sampler.samples if t >= t_load]
    say("负载期间链路速度（相对负载起点）")
    say("-" * 70)
    hist: dict[str, int] = {}
    for _, s, _ in loaded:
        hist[s] = hist.get(s, 0) + 1
    for s, n in sorted(hist.items(), key=lambda kv: -kv[1]):
        share = n / len(loaded) if loaded else 0
        say(f"  {s:<20} {n:>5} 次   {share:6.1%}")

    upshift_t = None
    for t, s, _ in loaded:
        if sampler.gtps(s) >= GEN4_GTPS - 0.01:
            upshift_t = t - t_load
            break
    if upshift_t is not None:
        say(f"\n  首次达到 {GEN4_GTPS} GT/s：负载开始后 {upshift_t * 1000:.0f} ms")

    gen4_share = (
        sum(1 for _, s, _ in loaded if sampler.gtps(s) >= GEN4_GTPS - 0.01) / len(loaded)
        if loaded else 0.0
    )

    say("")
    say("=" * 70)
    say("判据")
    say("=" * 70)
    say(f"  空闲 {idle_gt:.1f} GT/s  →  负载期 Gen4 占比 {gen4_share:.1%}")

    passed = gen4_share >= PASS_SHARE
    if passed:
        say("  ✅ 通过 —— 空闲降速，负载下能升到 Gen4。后续所有测量加预热即可。")
        rc = 0
    elif gen4_share <= 0.02:
        say("  ❌ 未通过 —— 负载下**仍停在低代际**。这不是空闲降速，是真卡住。")
        say("     先查：驱动版本 / 虚机 PCI 直通配置 / 宿主 ASPM 策略 / BIOS 里的")
        say("     链路宽度与代际设置。查清之前不要往下做 —— 后面所有数字都会错 7 倍。")
        rc = 2
    else:
        say(f"  ⚠️ 未通过 —— Gen4 占比只有 {gen4_share:.1%}（判据 ≥{PASS_SHARE:.0%}）。")
        say("     链路在负载期间**上下跳**。这比停在 Gen1 更麻烦：带宽会随时间变化，")
        say("     测量必须逐块记录代际并分层统计，不能给一个单一的数。")
        rc = 3

    say("")
    say(f"  参照：Gen1×16 ≈ 3.2 GB/s，Gen4×16 ≈ 32 GB/s（均为有效值量级）")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"报告已写入 {args.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
