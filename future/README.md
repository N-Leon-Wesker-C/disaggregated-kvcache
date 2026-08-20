# 后续研究路径 Roadmap

四个方向按依赖关系排列。A 是最直接的延伸，D 是博士论文级别的选题。

```
路径 A（工程验证）       做完 gpu-swap 后直接可以做
    │
    ▼
路径 B（自适应策略）       需要路径 A 的真实数据驱动
    │
    ▼
路径 C（多 GPU 池化）     需要多一台 GPU 服务器
    │
    ▼
路径 D（通用框架/OS 抽象） 需要 A+B+C 的经验积累
                          博士论文级别的选题
```

---

## 路径 A：从模拟到真实集成

**状态**：gpu-swap 用 mock workload 验证策略。
**目标**：将 Adaptive Policy 集成到真实 vLLM serving，测量端到端效果。

```bash
# vLLM 的 swap 逻辑位置（v1 engine）
vllm/v1/kv_offload/cpu/manager.py
```

**做法**：
1. Hook `vllm/v1/kv_offload/cpu/manager.py` 的 CPU offload manager
2. 将内部 LRU 替换为 Adaptive Policy
3. 跑真实 serving benchmark（不同 arrival rate、prompt 分布）
4. 对比指标：TTFT P99、throughput、swap 次数

**产出**：从"模拟 miss rate 降 X%"升级为"真实 serving P99 延迟降 X%"。

---

## 路径 B：从手动策略到自动策略推导

**状态**：Adaptive 阈值（short_lifetime、hot_access）从 trace 手动观察硬编码。
**目标**：系统自动从 trace 中推导最优策略参数。

**设计**：
```
1. 参数空间：hot_threshold, short_lifetime_threshold, prefetch_window
2. 目标函数：miss_rate 或 P99 latency
3. 离线性：对每种 workload 在参数空间搜索最优解
4. 在线性：检测 workload 类型 → 动态切换参数
```

**与 Mortise (NSDI'26) 的方法论连接**：自动调优系统参数以适应变化的工作负载，是孟子立研究范式的直接延伸。

**产出**：publishable workshop paper 雏形（HotMobile / EuroSys poster）。

---

## 路径 C：多 GPU KV Cache 池

**状态**：单台 4090，溢出到本机 CPU 内存。
**目标**：多 GPU 共享 KV Cache 池。

```
┌─────────┐    ┌─────────┐
│ GPU 0    │    │ GPU 1    │
│ KV hot   │    │ KV hot   │
└────┬─────┘    └────┬─────┘
     └──────┬────────┘
            │ TCP/RDMA
            ▼
     ┌─────────────┐
     │ CPU 冷池（共享）│
     └─────────────┘
```

**关键问题**：
- Block 归属：哪个 GPU 拥有一个 block？
- 迁移代价：跨 GPU block 传输延迟？
- 一致性：GPU 0 修改的 block，GPU 1 如何感知？
- 请求路由：是否路由到"存了它 KV Cache 的 GPU"？

**产出**：disaggregated LLM inference 的核心问题探索。CXL 硬件未普及时，软件模拟提供设计经验。

---

## 路径 D：GPU 内存层级管理的通用框架

**状态**：策略与机制耦合在 gpu-swap 里。
**目标**：设计一个 GPU memory tiering runtime，提供通用抽象。

```
接口层（类比 mmap/msync/madvise）：
  gpu_mmap(addr, len, tier_policy)
  // tier_policy: GPU_ONLY | GPU_PREFERRED | CPU_AS_COLD | AUTO

  gpu_msync(addr, len)
  // 显式控制数据在层级间的迁移

  gpu_madvise(addr, len, advice)
  // WILL_ACCESS_SOON | WILL_NOT_ACCESS | DONT_SWAP
  // 应用给 hint，不强制

策略层（可插拔）：
  EvictionPolicy 接口 + 多实现 + workload auto-detection

机制层：
  CUDA stream 异步迁移 + pinned memory 池 + 碎片 compaction
```

**价值**：若设计得当，vLLM、SGLang、TensorRT-LLM、甚至非 LLM 的 CUDA 程序都可复用。这就是"为解耦硬件设计 OS 抽象"的具体形态。

**长期目标**：结合路径 C 的多 GPU 场景，这个框架演化为 disaggregated GPU memory 的通用管理层——博士论文的核心贡献。
