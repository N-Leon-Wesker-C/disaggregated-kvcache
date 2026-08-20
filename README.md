# disaggregated-kvcache

**面向解耦架构的 LLM 推理 GPU 内存管理：从观测到策略，再到操作系统抽象**

这个仓库是我对一个问题层层递进的探索：

> 当 GPU 显存不再是紧耦合的本地资源，而是可以在内存层级（GPU ↔ CPU ↔ 远端内存池）之间流动时，系统软件应该如何管理它？

---

## 研究叙事：四个层级

```
层级 1: 观测（workload characterization）
  → 插桩 vLLM，追踪 KV Cache block 的完整生命周期
  → 回答：block 的真实访问模式是什么？

层级 2: 策略（policy design）
  → 基于 trace 数据设计热度感知的换出策略
  → 回答：知道了访问模式，能比 LRU 做得更好吗？

层级 3: 抽象（system gap）
  → 识别出问题本质：GPU 编程模型假设显存是静态、独占的，
    但多租户推理和解耦内存池打破了这一假设
  → 缺少一个 workload-aware 的 GPU 内存迁移层

层级 4: 愿景（research vision）
  → 为解耦硬件设计操作系统抽象
  → CXL 内存池化、WiFi 远端 GPU（WiCi）、RDMA 跨节点 KV Cache——
    是同一愿景在不同资源维度和延迟尺度上的投影
```

---

## 仓库结构

```
disaggregated-kvcache/
├── kvcache-tracer/          ← 层级 1：观测
│   ├── GUIDE.md             ← 5 步学习指南
│   └── scripts/
│       ├── tracer.py        ← vLLM KVCacheManager 插桩
│       └── analyze.py       ← trace 分析与 insight 提取
│
├── gpu-swap/                ← 层级 2：策略
│   ├── GUIDE.md             ← 6 步学习指南
│   ├── include/
│   │   ├── gpu_swap.h       ← 迁移层 C++ API
│   │   └── eviction_policy.h ← LRU / LFU / Adaptive 策略
│   └── scripts/
│       ├── test_e2e.py      ← 端到端模拟测试
│       └── compare_strategies.py ← 策略横向对比
│
├── future/                  ← 层级 3-4：后续研究路径
│   └── README.md            ← 路径 A/B/C/D 的 roadmap
│
└── docs/
    ├── 01-research-narrative.md  ← 研究叙事全文（四层级）
    └── 02-alignment.md           ← 与 WiCi / CXL 研究的契合分析
```

---

## 核心发现（随实验推进更新）

| 问题 | 方法 | 发现 |
|------|------|------|
| block 生命周期分布 | kvcache-tracer + analyze.py | （待填入实验数据） |
| block 访问偏斜 | 同上 | （待填入实验数据） |
| Adaptive vs LRU miss rate | gpu-swap + compare_strategies.py | （待填入实验数据） |
| PCIe 传输对 decode 的影响 | gpu-swap GUIDE Step 0 benchmark | （待填入实验数据） |

---

## 与解耦硬件研究的连接

- **CXL 内存池化**：本项目在软件层（PCIe, ~10μs）模拟了 CXL 在硬件层（~200ns）做的事——GPU 透明访问外部内存池。管理策略的算法设计与底层传输延迟正交。
- **WiCi（无线 GPU 计算）**：WiCi 在 WiFi 尺度（~2ms）解耦 GPU 计算；本项目在 PCIe 尺度解耦 GPU 显存。两者都是 disaggregated AI infrastructure 的组成部分，需要统一的 OS 抽象。
- **RDMA 跨节点 KV Cache**：多 GPU 共享 KV Cache 池是本项目的直接延伸（见 future/ 路径 C）。

---

## 环境

- RTX 4090 (24GB)，CUDA 13.2，vLLM 0.26，Qwen2.5-7B-Instruct
