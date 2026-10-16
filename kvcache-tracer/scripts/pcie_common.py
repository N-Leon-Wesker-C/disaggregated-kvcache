"""
pcie_common.py — PCIe 测量的共用件

拷贝原语**直接复用 vLLM 自己的** `vllm.v1.simple_kv_offload.cuda_mem_ops`，
不自己重新实现。理由不是省事，是保真：

  * 它经 `cuGetProcAddress(b"cuMemcpyBatchAsync", 12080, 0)` 拿的是 **v1 符号**
    （带 `failIdx`，9 个参数），而 `cuda.bindings.driver.cuMemcpyBatchAsync`
    是 **v2**（无 failIdx，8 个参数）。两者签名不同，自己写就要赌哪个对上。
  * 它的描述符布局是 **layer-major**：
    `src_bases[:, None] + src_ids[None, :] * bpb[:, None]` 再 `ravel()`
    —— 即「每个 layer 内跑一遍 block」。顺序会影响 DMA 合并行为，照抄。
  * 属性走 `CU_MEMCPY_SRC_ACCESS_ORDER_ANY`、`numAttrs=1`（ROCm 才是 0）。

**形状**（这才是本实验最容易搞错的地方）：一次 block 拷贝在 vLLM 里
不是「1 次 875 KB memcpy」，而是 **28 个 32,768 B 的描述符装在同一个
batch 调用里**（每层一个 data_ref，见 offloading/worker.py:206 与
kv_offload/cpu/gpu_worker.py:283-357）。所以这里 host / device 两侧都按
「每层一个张量」建，`stride(0)*element_size()` 自然就是 32,768 B。

32,768 B ÷ 24 GB/s = 1.37 µs，**低于 DMA 启动开销**。按「875 KB 连续」测出来的
带宽会显著高估生产可达带宽 —— 而高估的方向正好是让迁移层显得划算的方向。

用法:
    from pcie_common import KVBlockStore, LinkSampler, BLOCK_BYTES
"""

import os
import subprocess
import threading
import time

import torch

# ---- 生产形状常量（Qwen2.5-7B, --block-size 16）----------------------------
# 每 token 每层 = 2(K/V) × 4(kv_head) × 128(head_dim) × 2 B(bf16) = 2,048 B
LAYERS = 28
PER_LAYER_BLOCK_BYTES = 16 * 2 * 4 * 128 * 2        # 32,768
BLOCK_BYTES = LAYERS * PER_LAYER_BLOCK_BYTES         # 917,504 = 0.875 MiB

#: vLLM 的 Triton/DMA 分界（swap_blocks_triton.py:20）。
#: 注意比较对象是**每层**的 32,768，不是整块的 917,504 —— 余量只有 14.3%。
TRITON_THRESHOLD_BYTES = 28 * 1024

TENSOR_NAMES = [f"layer_{i}" for i in range(LAYERS)]


# ---- 链路状态采样 ----------------------------------------------------------

def gpu_pci_path(bdf: str | None = None) -> str:
    """返回 GPU 的 sysfs 路径。默认从 nvidia-smi 反查第一块卡。"""
    if bdf is None:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
        # nvidia-smi 给 "00000000:01:00.0"，sysfs 用 "0000:01:00.0"
        bdf = out[-12:] if len(out) > 12 else out
    return f"/sys/bus/pci/devices/{bdf}"


class LinkSampler(threading.Thread):
    """持续采 `current_link_speed`。

    只看一眼是不够的：突发负载可能让链路停在 Gen1，也可能在一个条件内的不同
    时刻上下跳。必须按块采，事后看的是**时间占比**而不是某一瞬间的值。
    """

    def __init__(self, pci_path: str, interval: float = 0.01):
        super().__init__(daemon=True)
        self.speed_file = os.path.join(pci_path, "current_link_speed")
        self.width_file = os.path.join(pci_path, "current_link_width")
        self.interval = interval
        self.samples: list[tuple[float, str, int]] = []
        self._halt = threading.Event()

    @staticmethod
    def _read(path: str) -> str:
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            return "?"

    @staticmethod
    def gtps(speed: str) -> float:
        """'16.0 GT/s PCIe' -> 16.0；读不到就给 -1。"""
        try:
            return float(speed.split()[0])
        except (ValueError, IndexError):
            return -1.0

    def run(self):
        while not self._halt.is_set():
            try:
                w = int(self._read(self.width_file))
            except ValueError:
                w = -1
            self.samples.append((time.monotonic(), self._read(self.speed_file), w))
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=2.0)

    def histogram(self) -> dict[str, int]:
        """速度字符串 -> 采样点数，按出现次数降序。"""
        h: dict[str, int] = {}
        for _, s, _ in self.samples:
            h[s] = h.get(s, 0) + 1
        return dict(sorted(h.items(), key=lambda kv: -kv[1]))

    def share_at(self, gtps: float) -> float:
        if not self.samples:
            return 0.0
        n = sum(1 for _, s, _ in self.samples if abs(self.gtps(s) - gtps) < 0.01)
        return n / len(self.samples)


# ---- 缓冲区 -----------------------------------------------------------------

def numa_placement(ptr: int) -> str | None:
    """从 /proc/self/numa_maps 里找出包含 ptr 的映射。

    `numactl` 这个容器里**没装**（实测），所以放置只能事后核验。
    """
    page = os.sysconf("SC_PAGE_SIZE")
    try:
        with open("/proc/self/numa_maps") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    start = int(parts[0], 16)
                except ValueError:
                    continue
                pages = 0
                for p in parts[2:]:
                    if p.startswith("N") and "=" in p:
                        try:
                            pages += int(p.split("=")[1])
                        except ValueError:
                            pass
                if pages and start <= ptr < start + pages * page:
                    return line.strip()
    except OSError:
        return None
    return None


class KVBlockStore:
    """按生产形状建的一对池子：28 个 per-layer host 张量 + 28 个 per-layer device 张量。

    每个张量形状 [n_blocks, 32,768] uint8 —— 于是 `stride(0)*element_size()`
    正好是 32,768，和 vLLM 从 `unpadded_page_size_bytes` 拿到的 bpb 一致。

    所有内存在循环外分配完。循环里**绝不能**再分配：caching allocator 会同步，
    首次 `cudaMalloc` 还是设备级串行的，那样测出来的抖动不是 PCIe 造成的。
    """

    def __init__(self, n_blocks: int, n_layers: int = LAYERS,
                 per_layer_bytes: int = PER_LAYER_BLOCK_BYTES,
                 device: str = "cuda", pin: bool = True):
        """`n_layers × per_layer_bytes` 就是「一个 block」的形状。

        默认是生产形状 (28, 32,768)。**改这两个数可以在总字节数不变的前提下
        改描述符粒度** —— 这正是隔离「每 op 开销」与「链路带宽」的手段：

            1 × 917,504 B      一个 block = 1 个描述符
            28 × 32,768 B      生产形状
            56 × 16,384 B      一个 block = 56 个描述符
        """
        from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor

        self.n = n_blocks
        self.n_layers = n_layers
        self.per_layer_bytes = per_layer_bytes
        self.block_bytes = n_layers * per_layer_bytes

        names = [f"slice_{i}" for i in range(n_layers)]
        self.host = [torch.empty((n_blocks, per_layer_bytes), dtype=torch.uint8)
                     for _ in range(n_layers)]
        if pin:
            # 先写一遍把页落实（首次触碰决定 NUMA 归属），再锁页
            for t in self.host:
                t.zero_()
            for t in self.host:
                pin_tensor(t)
        self.gpu = [torch.empty((n_blocks, per_layer_bytes), dtype=torch.uint8,
                                device=device) for _ in range(n_layers)]
        self.host_names = dict(zip(names, self.host))
        self.gpu_names = dict(zip(names, self.gpu))

    @property
    def host_bytes(self) -> int:
        return self.n * self.block_bytes

    def params_d2h(self, stream):
        """换出：device -> host。"""
        from vllm.v1.simple_kv_offload.cuda_mem_ops import build_params
        return build_params(self.gpu_names, self.host_names, stream)

    def params_h2d(self, stream):
        """换回：host -> device。"""
        from vllm.v1.simple_kv_offload.cuda_mem_ops import build_params
        return build_params(self.host_names, self.gpu_names, stream)

    def params(self, direction: str, stream):
        return self.params_d2h(stream) if direction == "d2h" else self.params_h2d(stream)


def copy_blocks(src_ids, dst_ids, params):
    """薄封装，只为把 vLLM 的调用点集中到一处。"""
    from vllm.v1.simple_kv_offload.cuda_mem_ops import copy_blocks as _cb
    _cb(src_ids, dst_ids, params)


def build_descriptor_arrays(src_ids, dst_ids, params):
    """和 vLLM `copy_blocks` 里那段**逐字相同**的 numpy 构造，但不发出去。

    给 sham 对照用。sham 的意义是「同样的循环、同样的 CPU 工作，零字节 DMA」——
    如果 sham 只空转 Python 循环，它和真负载的差异就同时包含「DMA」和
    「描述符构造的 CPU 成本」两样东西，测出来的差值归因不了。
    所以 sham 必须把这段构造也做了。
    """
    import numpy as np
    n = len(src_ids)
    src_ids = np.array(src_ids, dtype=np.uint64)
    dst_ids = np.array(dst_ids, dtype=np.uint64)
    src_all = (
        params.src_bases[:, None] + src_ids[None, :] * params.bpb[:, None]
    ).ravel()
    dst_all = (
        params.dst_bases[:, None] + dst_ids[None, :] * params.bpb[:, None]
    ).ravel()
    sz_all = np.repeat(params.bpb, n)
    return src_all, dst_all, sz_all


# ---- 负载循环 ---------------------------------------------------------------

def issue_batch(store: KVBlockStore, params, block_ids):
    """一次 `cuMemcpyBatchAsync`：len(block_ids) × 28 个描述符。"""
    copy_blocks(block_ids, block_ids, params)


def run_load(store: KVBlockStore, direction: str, duration_s: float,
             blocks_per_cycle: int, stream, sampler: LinkSampler | None = None,
             warmup_cycles: int = 20, deadline: float | None = None):
    """连续跑 batch 拷贝，返回 (cycles, elapsed_ms, bytes)。

    `direction`: "d2h" / "h2d" / "both"。both 会在两个 stream 上同时跑
    （用两个 stream 是为了让驱动有机会并行，但 CE 跨流是串行的 ——
    所以**不要**从流数推断饱和度，只看聚合吞吐）。
    """
    if blocks_per_cycle > store.n:
        raise ValueError(f"blocks_per_cycle={blocks_per_cycle} > n_blocks={store.n}")

    if direction == "both":
        # 两个方向各一个 stream + 各自的 params
        s2 = torch.cuda.Stream()
        p_a = store.params_d2h(stream)
        p_b = store.params_h2d(s2)
        targets = [(stream, p_a), (s2, p_b)]
    else:
        targets = [(stream, store.params(direction, stream))]

    ids = list(range(blocks_per_cycle))

    def cycle():
        for st, p in targets:
            issue_batch(store, p, ids)

    for _ in range(warmup_cycles):
        cycle()
    for st, _ in targets:
        st.synchronize()

    if deadline is None:
        deadline = time.perf_counter() + duration_s

    ev0 = [torch.cuda.Event(enable_timing=True) for _ in targets]
    ev1 = [torch.cuda.Event(enable_timing=True) for _ in targets]
    for (st, _), e in zip(targets, ev0):
        e.record(st)

    cycles = 0
    while time.perf_counter() < deadline:
        cycle()
        cycles += 1

    for st, _ in targets:
        st.synchronize()
    for (st, _), e in zip(targets, ev1):
        e.record(st)
    for e in ev1:
        e.synchronize()

    elapsed_ms = max(e0.elapsed_time(e1) for e0, e1 in zip(ev0, ev1))
    n_dirs = len(targets)
    total_bytes = cycles * blocks_per_cycle * BLOCK_BYTES * n_dirs
    return cycles, elapsed_ms, total_bytes
