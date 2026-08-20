# 项目二：gpu-swap — GPU 显存到 CPU 内存的热度感知迁移层

## 你要构建什么

当 GPU 显存不够时，自动把冷数据换出到 CPU 内存，需要时再换回。策略不是 LRU——策略由项目一的 trace 数据驱动。

```
┌──────────────────────────────────────────┐
│  GPU 显存 (24GB on 4090)                  │
│  ┌────────────┐  ┌────────────────────┐   │
│  │ 模型权重     │  │  KV Cache 热池      │   │
│  │ ~14GB       │  │  (常驻 GPU)         │   │
│  └────────────┘  └────────┬───────────┘   │
│                           │               │
│                    gpu-swap 管理层         │
│                           │               │
│  ┌────────────────────────▼───────────┐   │
│  │  CPU 内存 (64-256GB)                │   │
│  │  KV Cache 冷池                      │   │
│  │  (按需换入/换出)                     │   │
│  └────────────────────────────────────┘   │
└──────────────────────────────────────────┘
```

## 核心学习目标

做完这个项目，你将深入理解：

1. GPU 显存管理的底层机制（CUDA Driver API 层面）
2. 异步数据迁移——如何让换出不阻塞推理
3. 缓存替换策略——LRU vs LFU vs 你的自适应策略
4. 为什么 CXL 是"硬件做这件事"，而你是"软件做这件事"

---

## Step 0：理解 GPU 和 CPU 之间的数据传输

### 概念

```
你的 4090 和 CPU 之间的数据通道：

  GPU 显存 (GDDR6X)
      │
      ▼
  PCIe 4.0 x16 (~32 GB/s 单向理论带宽)
      │
      ▼
  CPU 内存 (DDR4/DDR5)

关键数字（实测，非理论峰值）：
  - GPU→CPU 拷贝：~12-16 GB/s（取决于块大小）
  - CPU→GPU 拷贝：~12-16 GB/s
  - 延迟（4KB 小块）：~5-10 微秒
  - 延迟（1MB 大块）：~100-200 微秒
  
对比：
  - GPU 显存内部带宽：~1 TB/s（GDDR6X）
  - CXL 内存延迟：~200 纳秒（比 PCIe 快 25-50 倍）
  - 本地 DDR5 延迟：~80 纳秒
```

**关键 insight**：PCIe 传输的延迟对于 LLM 推理的 decode 循环（~5-10ms/token）不是瓶颈。但如果你在关键路径上同步传输，每次传输几十微秒累积起来就慢了。所以异步是关键。

### 动手：体验一次 GPU↔CPU 拷贝

```c
// learn/pcie_bench.cu — 测量单次拷贝的延迟
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    size_t sizes[] = {4096, 65536, 1048576, 16777216, 67108864}; // 4K 到 64M
    int n = sizeof(sizes) / sizeof(sizes[0]);

    void *gpu_buf, *cpu_buf;
    cudaMalloc(&gpu_buf, sizes[n-1]);
    cudaMallocHost(&cpu_buf, sizes[n-1]);  // pinned memory — 关键！

    for (int i = 0; i < n; i++) {
        size_t sz = sizes[i];

        // 测量 GPU→CPU
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cudaMemcpy(cpu_buf, gpu_buf, sz, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("D2H %8zu bytes: %8.3f ms  (%6.1f MB/s)\n",
               sz, ms, (sz / 1e6) / (ms / 1000));

        // 测量 CPU→GPU
        cudaEventRecord(start);
        cudaMemcpy(gpu_buf, cpu_buf, sz, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&ms, start, stop);
        printf("H2D %8zu bytes: %8.3f ms  (%6.1f MB/s)\n",
               sz, ms, (sz / 1e6) / (ms / 1000));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(gpu_buf);
    cudaFreeHost(cpu_buf);
    return 0;
}
```

**编译和运行**：
```bash
nvcc -o pcie_bench learn/pcie_bench.cu
./pcie_bench
```

**分析你的数据**：
```
Q1: 多大块时带宽接近理论峰值？小块为什么慢？
    答案：小于 64KB 时 DMA 启动开销占主导，带宽很低。

Q2: 为什么用 cudaMallocHost（pinned memory）？
    不用的话慢 2-3 倍——因为普通内存页可以被 OS 换出，
    GPU 无法 DMA 访问。pinned memory 是锁死的。

Q3: 如果用 cudaMemcpyAsync（异步），延迟会变吗？
    异步不改变总传输时间，但可以让 GPU 在传输期间干别的。

Q4: 一个 KV Cache block（比如 16 tokens * 4096 dim * 2 layers * 2 bytes
    = ~256KB）的传输延迟大约多少？
    从你的数据估计：256KB → ~20-30 微秒。
    这比一个 decode step（~5ms）小两个数量级。
    → PCIe 传输不是瓶颈！策略的智能程度才是。
```

**检查**：
- [ ] 你跑了 benchmark，有自己的数据（不是网上查的）
- [ ] 你理解 pinned memory 和 pageable memory 的区别
- [ ] 你能估算 KV Cache block 的传输延迟

---

## Step 1：设计 gpu-swap 的 API 和数据结构

### 概念

gpu-swap 不是完整的 GPU 内存管理器——它只做一件事：管理"热池"和"冷池"之间的迁移。

它的接口设计需要回答：

```
1. 谁来决定"该迁移哪个 block"？
   → 外部（调用者）决定。gpu-swap 只负责高效执行迁移。
   策略是调用者的责任——你可以换不同的策略（LRU / LFU / 自定义）。

2. 迁移是同步还是异步？
   → 异步的。同步会阻塞推理。但调用者需要知道迁移何时完成。

3. 多个 block 同时迁移怎么处理？
   → 用队列 + CUDA stream 排队。

4. 如果换回时发现"冷池里没有"怎么办？
   → 返回错误或阻塞等待。这是策略层的 bug，不是 swap 层的问题。
```

### 动手：定义数据结构

```cpp
// include/gpu_swap.h — 你应该自己定义，这是参考

// 一个 KV Cache block 的描述
struct BlockDesc {
    int64_t block_id;          // 全局唯一标识
    size_t  size;              // 这个 block 的大小（字节）
    int     gpu_dev;           // GPU 设备编号
    void*   gpu_ptr;           // 在 GPU 显存中的地址（如果当前在 GPU）
    void*   cpu_ptr;           // 在 CPU 内存中的地址（如果当前在 CPU）
    bool    in_gpu;            // 当前是否在 GPU
    int64_t last_access_time;  // 最后一次被访问的时间戳（用于 LRU）
    int64_t access_count;      // 访问次数（用于 LFU）
};

// 迁移结果
enum class MigrateResult {
    OK = 0,
    ERR_GPU_OOM,           // GPU 显存满，无法分配
    ERR_CPU_OOM,           // CPU 内存满
    ERR_BLOCK_NOT_FOUND,   // 要换回的 block 不在冷池
    ERR_IN_PROGRESS,       // 这个 block 正在迁移中
};

// 核心类
class GpuSwapManager {
public:
    // 初始化：指定热池大小（GPU 显存预算）和冷池大小（CPU 内存预算）
    GpuSwapManager(size_t hot_pool_bytes, size_t cold_pool_bytes);

    // 在 GPU 热池中分配一个 block
    BlockDesc* allocate(int64_t block_id, size_t size);

    // 释放一个 block（从热池或冷池）
    void free(int64_t block_id);

    // 换出：block 从 GPU 热池 → CPU 冷池（异步）
    MigrateResult evict_to_cold(int64_t block_id);

    // 换回：block 从 CPU 冷池 → GPU 热池（异步）
    // 如果热池已满，会先换出其他 block（根据策略）
    MigrateResult prefetch_to_hot(int64_t block_id);

    // 等待所有正在进行的异步迁移完成
    void sync();

    // 统计
    size_t hot_used_bytes() const;
    size_t cold_used_bytes() const;
    int    pending_migrations() const;
};
```

**思考**（写下来，这部分是申请材料的好内容）：

1. `evict_to_cold` 和 `prefetch_to_hot` 应该是什么语义？
   - 发起迁移 + 立刻返回（异步）？还是阻塞到迁移完成？
   - 如果是异步，调用者怎么知道完成了？callback？future？轮询？

2. 当热池满了，`prefetch_to_hot` 需要先腾出空间。谁来选"哪个 block 被换出"？
   - gpu-swap 内部可以根据 `last_access_time` 或 `access_count` 自动选
   - 也可以由外部传入一个 `evict_candidate` 参数

---

## Step 2：实现核心换出/换回逻辑

### 概念：异步迁移的流水线

同步换出（不好）：
```
  GPU 当前 step   │████████████│
                   │ 等待换出完成...  │████████████│
                                     │ 下一个 step │
  延迟 = GPU 执行时间 + 换出时间
```

异步换出（好）：
```
  GPU 当前 step   │████████████│
  PCIe 换出        │███████████│    ← 和 GPU 执行重叠
  GPU 下一个 step               │████████████│
  延迟 = MAX(GPU 执行时间, 换出时间)  ← 几乎不增加
```

CUDA stream 是实现异步的关键——不同 stream 上的操作可以并发执行。

### 动手：实现异步换出

```cpp
// src/swap_core.cu — 参考骨架

#include "gpu_swap.h"
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <queue>

class GpuSwapManager {
private:
    size_t hot_capacity_;   // GPU 热池预算
    size_t cold_capacity_;  // CPU 冷池预算
    size_t hot_used_ = 0;
    size_t cold_used_ = 0;

    // block_id → BlockDesc
    std::unordered_map<int64_t, BlockDesc> blocks_;

    // 迁移队列
    cudaStream_t swap_stream_;  // 专用 stream，不干扰推理 stream
    std::queue<int64_t> pending_evictions_;
    std::queue<int64_t> pending_prefetches_;

    // 换出：GPU → CPU
    MigrateResult do_evict(int64_t block_id) {
        auto& blk = blocks_[block_id];
        if (!blk.in_gpu) {
            return MigrateResult::ERR_BLOCK_NOT_FOUND;
        }

        // 1. 在 CPU 端分配 pinned memory
        //    （必须用 pinned，否则 GPU DMA 无法直接访问）
        cudaMallocHost(&blk.cpu_ptr, blk.size);

        // 2. 异步拷贝：GPU → CPU
        //    使用独立的 swap_stream_，不阻塞推理 stream
        cudaMemcpyAsync(
            blk.cpu_ptr,    // dst: CPU
            blk.gpu_ptr,    // src: GPU
            blk.size,
            cudaMemcpyDeviceToHost,
            swap_stream_
        );

        // 3. 在 swap_stream_ 上记录一个回调，
        //    拷贝完成后释放 GPU 显存
        //    （这里用 cudaStreamAddCallback 或简单的 event）
        cudaEvent_t done;
        cudaEventCreate(&done);
        cudaEventRecord(done, swap_stream_);

        // TODO: 在 event done 后：
        //   - cudaFree(blk.gpu_ptr)
        //   - hot_used_ -= blk.size
        //   - blk.in_gpu = false
        //   - cold_used_ += blk.size

        cudaEventDestroy(done);
        return MigrateResult::OK;
    }

    // 换回：CPU → GPU
    MigrateResult do_prefetch(int64_t block_id) {
        auto& blk = blocks_[block_id];
        if (blk.in_gpu) {
            return MigrateResult::OK;  // 已经在 GPU，不需要换回
        }
        if (!blk.cpu_ptr) {
            return MigrateResult::ERR_BLOCK_NOT_FOUND;
        }

        // 如果热池满了，先换出一个冷 block
        if (hot_used_ + blk.size > hot_capacity_) {
            int64_t victim = select_eviction_victim();
            if (victim < 0) {
                return MigrateResult::ERR_GPU_OOM;
            }
            do_evict(victim);
        }

        // 1. 在 GPU 分配显存
        cudaMalloc(&blk.gpu_ptr, blk.size);

        // 2. 异步拷贝：CPU → GPU
        cudaMemcpyAsync(
            blk.gpu_ptr,
            blk.cpu_ptr,
            blk.size,
            cudaMemcpyHostToDevice,
            swap_stream_
        );

        blk.in_gpu = true;
        hot_used_ += blk.size;
        cold_used_ -= blk.size;

        return MigrateResult::OK;
    }

    // 选择一个"最该被换出"的 block
    int64_t select_eviction_victim() {
        // 策略 A: LRU — 选最久没被访问的
        int64_t victim = -1;
        int64_t oldest_time = INT64_MAX;
        for (auto& [id, blk] : blocks_) {
            if (blk.in_gpu && blk.last_access_time < oldest_time) {
                oldest_time = blk.last_access_time;
                victim = id;
            }
        }
        return victim;

        // 策略 B: LFU — 选访问次数最少的
        // （你自己实现）

        // 策略 C: Hybrid — LRU 但排除最近刚被访问的（短时间内可能再被访问）
        // （你自己实现——这是项目一 trace 数据的用武之地）
    }

public:
    // ... 公开接口实现 ...
};
```

**关键设计决策**：

```
决策 1: 换出后是否保留 CPU 副本？
  保留 → 后续换回时不需要重新计算 KV Cache（但占 CPU 内存）
  不保留 → 省 CPU 内存，但换回时需要重新 prefill
  你选哪个？为什么？

决策 2: prefetch 是主动（预测性）还是被动（访问到才换回）？
  主动 → GPU 显存利用率高，但预测不准会浪费带宽
  被动 → 简单，但每次 miss 都会阻塞推理
  项目一的 trace 数据能帮你回答：访问模式足够规律吗？能预测吗？

决策 3: swap_stream 和推理 stream 之间需要同步吗？
  如果推理正在读一个 block，同时 swap 正在换出它 → 数据竞争
  你需要一种同步机制：换出前先"锁定"block，推理完成后才释放
  这个"锁定"怎么实现？
```

**检查**：
- [ ] 你在单测中验证：分配 → 换出 → 换回 → 数据正确
- [ ] 异步迁移真的不阻塞（在 swap_stream 上操作时推理 stream 不受影响）
- [ ] 热池满时能自动腾出空间

---

## Step 3：实现策略层

### 概念

gpu-swap 的核心代码是"怎么搬数据"。策略层是"搬什么、什么时候搬"。把这两层分开——策略层是可插拔的。

### 动手

```cpp
// include/eviction_policy.h

struct BlockStats {
    int64_t block_id;
    int64_t last_access;
    int64_t access_count;
    bool    in_gpu;
    size_t  size;
};

class IEvictionPolicy {
public:
    virtual ~IEvictionPolicy() = default;

    // 通知策略层：发生了什么事件
    virtual void on_alloc(int64_t block_id, size_t size) = 0;
    virtual void on_access(int64_t block_id) = 0;
    virtual void on_free(int64_t block_id) = 0;

    // 核心方法：选一个被换出的 block
    // 返回 block_id，或者 -1 表示"不需要换出"
    virtual int64_t select_victim(
        const std::vector<BlockStats>& gpu_blocks
    ) = 0;
};

// 三种策略实现：

class LRUPolicy : public IEvictionPolicy {
    // 选 last_access 最早的 block
};

class LFUPolicy : public IEvictionPolicy {
    // 选 access_count 最少的 block
};

class AdaptivePolicy : public IEvictionPolicy {
    // 基于项目一的 trace 数据：
    // - 识别"短命 block"（生命周期 < 1s）→ 不换出
    // - 识别"热 block"（访问次数 > 阈值）→ 不换出
    // - 其余用 LRU 排序
    //
    // 阈值怎么定？从项目一的 trace 数据里分析出来。
};
```

---

## Step 4：集成测试

### 概念

你不必真的和 vLLM 集成（改动太大）。但你需要一个端到端的测试来证明 gpu-swap 能工作。

### 动手

```python
# scripts/test_e2e.py
"""
模拟 LLM 推理过程中 KV Cache block 的分配/访问/释放模式，
验证 gpu-swap 的正确性和性能。

用法:
    python scripts/test_e2e.py --strategy lru
    python scripts/test_e2e.py --strategy adaptive
"""

import random

def generate_workload(num_requests=20, seed=42):
    """
    生成模拟的请求序列。

    不依赖真实的 LLM，而是生成类似 KV Cache 访问模式的事件流。
    事件类型：alloc_request, access, free_request
    """
    random.seed(seed)
    events = []
    t = 0.0
    active = []

    for req_id in range(num_requests):
        # 随机决定请求类型
        rtype = random.choice(["short", "medium", "long"])

        if rtype == "short":
            num_blocks = random.randint(2, 8)
            num_steps = random.randint(10, 30)
        elif rtype == "medium":
            num_blocks = random.randint(8, 32)
            num_steps = random.randint(30, 100)
        else:
            num_blocks = random.randint(32, 128)
            num_steps = random.randint(100, 300)

        # alloc
        events.append({
            "time": t, "type": "alloc",
            "req_id": req_id, "num_blocks": num_blocks
        })
        active.append((req_id, num_blocks, num_steps))

        # 模拟请求在 decode 阶段并发
        t += random.uniform(0.01, 0.5)

        # 清理已完成的请求
        still_active = []
        for (rid, nblk, steps_left) in active:
            events.append({
                "time": t, "type": "access",
                "req_id": rid, "num_blocks": nblk
            })
            steps_left -= 1
            if steps_left > 0:
                still_active.append((rid, nblk, steps_left))
            else:
                events.append({
                    "time": t, "type": "free",
                    "req_id": rid
                })
        active = still_active

    events.sort(key=lambda e: e["time"])
    return events
```

然后用这个 workload 驱动你的 gpu-swap C++ 库（通过 Python binding 或 subprocess）：

```
测试矩阵：
  workload 类型: short-heavy / mixed / long-heavy
  策略: LRU / LFU / Adaptive
  热池大小: 紧（刚好够用）/ 松（大量冗余）

每个组合记录：
  - 换出次数
  - 换回次数
  - 平均 block miss 延迟（需要换回才能访问的延迟）
  - 热池利用率
```

---

## Step 5：分析——你的策略比 LRU 好在哪里

### 核心问题

```
Q1: 在什么样的 workload 下，你的 Adaptive 策略显著优于 LRU？
    （短请求多？长请求多？混合？）

Q2: 在什么样的 workload 下，Adaptive 策略和 LRU 差不多？
    → 意味着 LRU 已经足够好了，不需要复杂策略

Q3: 策略的选择对"换回延迟"的影响有多大？
    每次 miss 的延迟 = PCIe 传输 + CUDA API 开销
    如果 miss 率从 5% 降到 1%，对于 P99 延迟的影响是多少？

Q4: 你的 Adaptive 策略的额外开销（计算热度、维护统计）有多大？
    是否超过了好处？

Q5: 如果 CXL 可用（延迟降低 25-50 倍），策略的选择还重要吗？
    当 miss 延迟足够低时，即使策略不好，影响也小。
    → 这意味着：硬件越快，软件策略越不重要？
    还是：硬件越快，策略可以更大胆（频繁迁移），整体效率更高？
```

---

## 学习检查清单

- [ ] 你能解释为什么 `cudaMallocHost` 是必需的（pinned vs pageable）
- [ ] 你理解 CUDA stream 的异步语义以及如何用它做流水线
- [ ] 你实现了至少两种换出策略并能对比
- [ ] 你的代码能通过端到端的正确性测试（换出→换回→数据一致）
- [ ] 你回答了 Step 5 中的 5 个问题
- [ ] 你的 README 里有 benchmark 数据和策略对比图表
