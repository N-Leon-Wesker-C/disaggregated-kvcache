# gpu-swap

**GPU 显存到 CPU 内存的热度感知迁移层**

> 🎓 先读 [GUIDE.md](GUIDE.md) — 完整的 6 步学习指南

当 GPU 显存不够时，自动把冷 KV Cache block 换出到 CPU 内存，需要时换回。策略由 [kvcache-tracer](../kvcache-tracer/) 的 trace 数据驱动，不是拍脑袋的 LRU。

## 攻击的问题

LLM 推理的显存瓶颈：
- 模型权重占 14GB（Qwen-7B FP16）
- KV Cache 每个请求占 0.5-5GB
- RTX 4090 总共只有 24GB
- → 多用户并发或长上下文时显存不够

已有的 vLLM swap 策略是简单 LRU。本项目探索"如果知道 block 的真实访问模式，能不能做得更好？"

## 架构

```
GPU 热池 (受限)          CPU 冷池 (充裕)
┌──────────┐            ┌──────────────┐
│ hot block │            │ cold block   │
│ hot block │  ←──→     │ cold block   │
│ hot block │  PCIe     │ cold block   │
│   ...     │  异步      │   ...        │
└──────────┘            └──────────────┘
     ↑                        ↑
     └── 策略层 ──────────────┘
          LRU / LFU / Adaptive
          (基于 kvcache-tracer 的 trace 数据)
```

## 快速开始

```bash
# 策略对比（不需要 GPU）
python scripts/compare_strategies.py

# 单次测试
python scripts/test_e2e.py --strategy adaptive --gpu-budget 4
```

## 与 CXL 的关系

CXL 在硬件层做同样的事——让 GPU 透明访问外部内存池。本项目的"冷池"是 CPU DDR（通过 PCIe），延迟更高（~25us vs ~200ns），但管理逻辑是通用的。策略设计不依赖于底层传输延迟——无论 CXL 还是 PCIe，你都需要决定"换哪个 block"。

## 学习目标

做完这个项目，你将理解：
- GPU 显存管理的底层机制（CUDA Driver API, pinned memory）
- 异步数据迁移（CUDA stream 流水线）
- 缓存替换策略的设计和评估
- 为什么 CXL 被认为是"数据中心的下一件大事"
