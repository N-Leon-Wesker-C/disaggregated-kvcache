# 研究叙事：从 trace 数据到研究愿景

## 层级 1：Workload Characterization

**做了什么**：通过 instrumentation 捕获 vLLM KVCacheManager 中每个 block 的分配、访问、释放事件，构建 KV Cache block 的完整生命周期 trace。

**科学问题**：
1. Block 生命周期的分布特征是什么？短命的比例有多大？
2. Block 访问频率的分布特征是什么？是否存在显著的访问偏斜（hot/cold skew）？
3. 访问模式是否与请求类型（prompt length, output length）相关？

**为什么必要**：不了解 workload 特征的系统优化是盲目试错。这是体系结构领域 workload characterization 方法论的直接应用。

## 层级 2：Policy Design and Evaluation

**做了什么**：基于 trace 观测到的规律，设计块级热度感知的换出策略（Adaptive Eviction Policy），与 LRU、LFU 做对照实验。

**科学问题**：
1. GPU 显存受限时，Adaptive 策略的 miss rate 是否显著低于 LRU？
2. 策略优势对 workload mix 的敏感性：什么 workload 下优势最大？
3. 策略的额外开销（热度统计）是否小于其收益？

**核心 insight**：LRU 是 workload-agnostic 的——假设"最近访问的最可能再被访问"。但 KV Cache 的访问模式是 write-once-read-many：block 在 prefill 写一次、decode 每步读一次、请求结束即释放。一个刚刚被访问但请求即将结束的 block 恰恰是最不该保留的。LRU 的假设在此失效。

## 层级 3：Identifying the System Gap

**抽象出的更一般问题**：

当前 GPU 编程模型（CUDA）将 GPU 显存建模为静态的、进程独占的、物理紧耦合的地址空间。以下场景正在打破这一假设：

- 多租户推理服务：多个请求竞争一块 GPU 的显存
- 解耦内存池：GPU 显存通过 CXL 扩展到外部池，或经 RDMA 访问远端显存
- WiFi 远端 GPU（WiCi）：GPU 计算本身被卸载到远端设备

**Gap 的精确描述**：在 disaggregated memory hierarchy 中，缺少一个 workload-aware、policy-pluggable 的 GPU 内存迁移层——类似于 OS 虚拟内存的 page replacement 框架，但面向 GPU 的访问模式和数据粒度。

现有证据：每个推理框架（vLLM、SGLang、TensorRT-LLM）都各自实现了一套 swap 逻辑，策略与机制耦合，且策略粗糙（固定 LRU 或随机）。这正是"缺一层基础设施"的典型症状。

## 层级 4：Research Vision

**方向**：为解耦硬件设计操作系统抽象。

**Disaggregation spectrum**：硬件资源可以在不同延迟尺度上被解耦——CXL（~200ns）、RDMA（~5μs）、WiFi（~2ms）。每层的 trade-off 不同，但管理问题本质相同：**如何让上层应用无需关心资源物理位置？**

**Missing OS layer**：现有 OS 为单体主机设计了进程、虚拟内存、文件系统。当资源边界从"一台机器"变成"网络内的资源池"时，需要新的抽象：

- Resource discovery：应用如何发现可用资源池？
- Capability negotiation：能力如何协商？
- Transparent migration：数据如何在层级间透明迁移？
- QoS isolation：多租户下的性能隔离如何保证？

这些在内核层面目前都不存在。

**切入点**：GPU 显存管理是 disaggregation 在 AI 场景下最紧迫的瓶颈，KV Cache 是验证这一问题的具体载体。方法：trace-driven system design。

**与 WiCi 的精确连接**：WiCi 解耦 compute（CUDA 调用经 WiFi 卸载），本项目解耦 memory（KV Cache 在 GPU↔CPU↔Remote 间迁移）。同一愿景在不同资源维度上的投影。WiCi 的攻击面是网络传输层，本项目的攻击面是内存管理层。最终需要一个统一的 OS 抽象同时管理两者。
