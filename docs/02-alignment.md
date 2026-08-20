# 项目与孟子立教授研究方向的契合度分析

## 孟老师的研究 DNA

三条线贯穿他的所有工作：

1. **Near-Bufferless 哲学**：系统性地消除每一层的缓冲，追求一致低延迟
2. **Adaptive Systems**：不是静态配置，而是根据网络/负载条件动态调整策略
3. **Resource Disaggregation**：WiFi 解耦 GPU 计算（WiCi）、网络解耦视频流（Hairpin/Zhuge）

## 两个项目卡在哪几条线上

| | Near-Bufferless | Adaptive | Disaggregation |
|------|:--:|:--:|:--:|
| kvcache-tracer | ✅ 分析缓冲来源 | ✅ 为自适应策略提供数据 | ✅ 内存层级解耦 |
| gpu-swap | ✅ 消除 swap 路径上的缓冲 | ✅ 基于热度的自适应换出 | ✅ GPU↔CPU 内存解耦 |

## 具体的契合点

**1. KV Cache 管理 = WiCi 服务端的核心问题**

WiCi 的 GPU 服务器同时服务多个客户端。每个客户端的请求都需要 KV Cache。24GB 显存的 4090 能存几个长上下文的 KV Cache？不够时怎么办？gpu-swap 提供了一种答案——冷 block 换出到 CPU 内存。这和 WiCi 的设计直接相关。

**2. 自适应策略是孟老师的标志性方法**

他的 MAE (NSDI '26) 做自适应视频编码，Mortise (NSDI '26) 做自适应拥塞控制。gpu-swap 的自适应换出策略（基于 block 热度动态调整）是同一方法论在 GPU 内存管理场景下的应用。你的申请材料里可以直接说："我借鉴了 MAE/Mortise 的自适应设计思路，应用到了 KV Cache 内存管理"。

**3. 内核/驱动级编程能力**

WiCi 需要 CUDA Driver API、NVIDIA 驱动、mac80211 等底层技能。gpu-swap 用 CUDA Driver API 做异步显存管理、用 CUDA stream 做流水线，展示了你在 GPU 底层编程上的能力。

**4. 方法论上的共鸣**

孟老师做研究的风格：先观测系统的真实行为、找到瓶颈、再设计策略。kvcache-tracer 就是"先观测"——理解 KV Cache block 的真实访问模式。gpu-swap 是"再设计"——基于观测数据优化迁移策略。这和他的研究范式高度一致。

## 申请时的一句话版本

> 我的研究兴趣是解耦架构下的资源管理：当 GPU 显存不再是本地独占资源，而是可以在内存层级间流动时，操作系统需要新的抽象和策略。我的两个项目——kvcache-tracer 和 gpu-swap——分别从"观测"和"设计"两个角度探索了这个问题的 LLM 推理场景。这与 SPARK 实验室在 WiCi 项目中面临的 GPU 资源管理挑战直接相关。
