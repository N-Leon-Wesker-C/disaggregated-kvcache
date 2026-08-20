# kvcache-tracer

**KV Cache 块级生命周期追踪器**

通过 monkey-patch vLLM (v0.26, v1 engine) 的 `SingleTypeKVCacheManager`，记录每个 KV Cache block 的分配、访问、释放事件。产出的 trace 数据直接驱动 [gpu-swap](../gpu-swap/) 的换出策略设计。

## Hook 点（针对 vLLM 0.26）

| 方法 | 事件 | 在生命周期中的位置 |
|------|------|-------------------|
| `allocate_new_blocks` | block 分配 | 出生 |
| `cache_blocks` | block 写入/访问 | 使用 |
| `free` | block 释放 | 死亡 |

## 快速开始

```bash
pip install vllm

python -c "
import tracer
tracer.install_hooks()
tracer.save_on_exit()
import subprocess, sys
subprocess.run([sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
    '--model', 'Qwen/Qwen2.5-7B-Instruct',
    '--gpu-memory-utilization', '0.85',
    '--port', '8000'])
"
# 发请求 → Ctrl+C → trace 自动保存到 kvcache_trace.json

python scripts/analyze.py kvcache_trace.json
```

## 分析产出

- block 生命周期分布（短命 vs 长命）
- block 访问频率分布（hot/cold skew）
- 每请求 block 数与请求生命周期
- 针对 swap 策略设计的 insights（见 `analyze.py` 输出）

## 学习目标

- vLLM PagedAttention 的 block 管理机制
- 无侵入式 instrumentation（monkey-patch）
- KV Cache 的 write-once-read-many 访问模式
- 为什么观测数据是策略设计的前提
