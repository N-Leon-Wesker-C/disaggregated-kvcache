#!/usr/bin/env python3
"""
pcie_loadgen.py — 产生定速 PCIe 负载的独立进程

为什么是独立进程且常驻
============================================================

* **独立进程**：CUDA 上下文共享，但 Python 解释器和 GIL 分开。vLLM 的
  EngineCore 才能真的被抢到 CPU —— 如果把负载跑在同一个进程里，
  测出来的是自己和自己抢 GIL，不是 PCIe 对推理的影响。
* **常驻**：每次重启都要重新分配并锁页 ~900 MB（好几秒），而一个测量
  block 只有 8 秒。常驻之后只靠 stdin 发指令，分配开销只付一次。
  另一个好处是负载的进程身份在整个 session 里不变。

两种强度语义
============================================================

* `--rate-gbps` 定速：token bucket，按目标速率限流。**周期远小于一个
  decode step**，所以强度是连续的。
* 不定速 = 全速：能跑多快跑多快，作为该方向的 100% 档。

**为什么不用占空比（跑 X ms 停 Y ms）当强度**：那会让每步延迟变成双峰分布，
「平均 GB/s」是一个没有任何单独一步真正经历过的数，而混合分布的均值不是
合法的纵轴。占空比只在需要「明确突发」对照时才用。

`sham` 模式
============================================================

同样的循环、同样的 sleeps、同样的迭代次数，**但不做拷贝**。这是最关键的一个
对照：若 sham ≈ 真负载，说明效应来自 CPU 侧的循环本身而不是 DMA，结论
必须照此改写。

指令协议（走 stdin，逐行）
============================================================

    run <rate_gbps|max> <direction> <duration_s>   → 起一段负载
    quit

每段负载结束打印一行 `DONE {json}`，含实际达到的速率。
"""

import argparse
import json
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from pcie_common import (  # noqa: E402
    LAYERS, PER_LAYER_BLOCK_BYTES, KVBlockStore, build_descriptor_arrays,
    issue_batch,
)


class LoadGen:
    def __init__(self, n_blocks: int, chunk_blocks: int):
        self.store = KVBlockStore(n_blocks, LAYERS, PER_LAYER_BLOCK_BYTES)
        self.chunk_blocks = chunk_blocks
        self.stream = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()
        self._lock = threading.Lock()

    # ---- 一个 chunk：chunk_blocks × 28 个描述符 ----------------------------
    def _cycle(self, direction: str, base: int):
        ids = [(base + i) % self.store.n for i in range(self.chunk_blocks)]
        if direction == "sham":
            # 同样的循环、**同样的描述符构造 CPU 工作**，只是不把 batch 发出去。
            # 若这里直接 return，sham 与真负载的差值里就同时混进了「DMA」和
            # 「描述符构造」两样东西，归因不了。
            build_descriptor_arrays(ids, ids, self.params_d2h)
            return
        if direction == "d2h":
            issue_batch(self.store, self.params_d2h, ids)
        elif direction == "h2d":
            issue_batch(self.store, self.params_h2d, ids)
        else:                           # both：两个 stream 上同时发
            issue_batch(self.store, self.params_d2h, ids)
            issue_batch(self.store, self.params_h2d, ids)

    def prepare(self):
        self.params_d2h = self.store.params_d2h(self.stream)
        self.params_h2d = self.store.params_h2d(self.stream2)
        # 预热：既让页落实，也把链路从空闲态顶上去
        for _ in range(30):
            self._cycle("d2h", 0)
            self._cycle("h2d", 0)
        self.stream.synchronize()
        self.stream2.synchronize()

    def run(self, target_gbps: float, direction: str, duration_s: float) -> dict:
        """跑一段定速负载，返回实际达成情况。"""
        n_dirs = 2 if direction == "both" else 1
        # sham 也按同样的「假定字节数」限速 —— 只是这字节数不进账。
        # 若 sham 用 0 字节算 budget，它就变成不限速空转（实测 2 秒转 297 万次，
        # 而真负载只有约 2000 次），节奏完全不同，做不了对照。
        emulated_bytes = self.chunk_blocks * self.store.block_bytes * n_dirs
        bytes_per_chunk = 0 if direction == "sham" else emulated_bytes
        budget = (emulated_bytes / (target_gbps * 1e9)
                  if target_gbps and target_gbps > 0 else 0.0)

        deadline = time.perf_counter() + duration_s
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(n_dirs)]
        streams = [self.stream] if n_dirs == 1 else [self.stream, self.stream2]
        if direction != "sham":
            for s, (e0, _) in zip(streams, ev):
                e0.record(s)

        base = 0
        chunks = 0
        t_next = time.perf_counter()
        while time.perf_counter() < deadline:
            self._cycle(direction, base)
            base = (base + self.chunk_blocks) % max(self.store.n - self.chunk_blocks, 1)
            chunks += 1
            if budget > 0:
                t_next += budget
                slack = t_next - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    # 追不上目标 —— 说明目标已经超过这个方向的上限，
                    # 这段就退化成全速跑。DONE 里的 achieved 会如实反映。
                    t_next = time.perf_counter()

        elapsed_ms = 0.0
        if direction != "sham":
            for s, (_, e1) in zip(streams, ev):
                s.synchronize()
                e1.record(s)
                e1.synchronize()
            elapsed_ms = max(e0.elapsed_time(e1) for e0, e1 in ev)

        wall = duration_s
        total_bytes = chunks * bytes_per_chunk
        return {
            "direction": direction,
            "target_gbps": target_gbps,
            "chunks": chunks,
            "chunk_blocks": self.chunk_blocks,
            "duration_s": wall,
            "elapsed_ms": elapsed_ms,
            "bytes": total_bytes,
            "achieved_gbps": (total_bytes / (elapsed_ms / 1000) / 1e9
                              if elapsed_ms > 0 else 0.0),
            # 相对墙钟的速率：这是干扰真正按时间摊开的强度
            "wall_gbps": total_bytes / wall / 1e9,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="定速 PCIe 负载发生器（常驻，stdin 控制）")
    ap.add_argument("--blocks", type=int, default=1024, help="缓冲池的 block 数")
    ap.add_argument("--chunk-blocks", type=int, default=8,
                    help="每次限速迭代搬多少 block；周期需远小于一个 decode step")
    args = ap.parse_args()

    gen = LoadGen(args.blocks, args.chunk_blocks)
    gen.prepare()
    print(json.dumps({"ready": True, "block_bytes": gen.store.block_bytes,
                      "chunk_blocks": args.chunk_blocks}), flush=True)

    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "quit":
            break
        if parts[0] != "run":
            continue
        rate = 0.0 if parts[1] == "max" else float(parts[1])
        direction = parts[2]
        duration = float(parts[3])
        try:
            res = gen.run(rate, direction, duration)
            res["ok"] = True
        except Exception as e:                                   # noqa: BLE001
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print("DONE " + json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
