# kvcache-tracer 学习指南

## 你要回答的问题

LLM 推理时，显存里的 KV Cache block 到底是怎么被使用的？

- 哪些 block 频繁访问、哪些只用一次？
- 一个 block 从分配到释放，活多久？
- 不同请求类型的访问模式有什么不同？

**不知道这些，swap 策略就是盲猜。**

---

## Step 0：理解 KV Cache 的访问模式

### 概念

KV Cache 的访问模式是 **write-once, read-many**：

```
请求生命周期：
  ┌─ prefill ──┬────────── decode ──────────┐
  │ 一次性写入   │  每个 step 读取所有历史 block │
  │ 所有 block  │  每个 block 每 step 读一次    │
  └────────────┴─────────────────────────────┘
```

vLLM 的 PagedAttention 把 KV Cache 切成固定大小 block（每 block 16 tokens），类比 OS 分页——物理 block 可以不连续，消除碎片。

### 检查

- [ ] 能解释 prefill 是"写"、decode 是"读"
- [ ] 理解为什么 block 化（类比虚拟内存分页的动机）

---

## Step 1：定位 vLLM 0.26 的 BlockAllocator

vLLM 0.26（v1 engine）的核心在：

```bash
VLLM_PATH=$(python -c "import vllm; print(vllm.__path__[0])")
grep -n "class\|def " $VLLM_PATH/v1/core/single_type_kv_cache_manager.py | head -40
```

关键类：`SingleTypeKVCacheManager`（ABC），子类 `FullAttentionManager`。

关键方法：
- `allocate_new_blocks(321)` — 分配新 block
- `cache_blocks(403)` — block 写入/访问
- `free(495)` — 释放 block

**检查**：
- [ ] 找到这三个方法的签名
- [ ] 画出 block 生命周期草图：allocate → cache* → free

---

## Step 2：写第一个 hook

```python
import vllm.v1.core.single_type_kv_cache_manager as m

_orig = m.SingleTypeKVCacheManager.allocate_new_blocks

def _patched(self, request_id, num_blocks, *args, **kwargs):
    print(f"[HOOK] allocate_new_blocks: request={request_id}, num={num_blocks}")
    return _orig(self, request_id, num_blocks, *args, **kwargs)

m.SingleTypeKVCacheManager.allocate_new_blocks = _patched
```

原理：Python 方法即属性，运行时替换 = monkey-patch。vLLM 所有内部调用都会经过你的包装函数。

**检查**：
- [ ] hook 生效（启动 vLLM 后发请求能看到输出）
- [ ] vLLM 没有崩溃

---

## Step 3：收集结构化数据

不只是打印——写入 JSON trace。字段：

```json
{"event": "alloc", "timestamp": 0.034, "block_id": 1423, "request_id": "req_42"}
```

三个 hook 分别记录 alloc / access / free。难点：从调用栈中回溯 request_id（`free(request_id)` 直接有；`allocate_new_blocks(request_id, ...)` 直接有；`cache_blocks` 签名因版本而异，需要确认）。

**检查**：
- [ ] trace 包含三种事件
- [ ] 同一 block 的 alloc/access/free 用同一个 block_id

---

## Step 4：分析

用 `scripts/analyze.py` 回答：

1. block 生命周期分布——短命比例？
2. 访问频率分布——hot/cold skew？
3. 每请求 block 数——不同请求类型差异？
4. 请求生命周期分布？

**核心产出**：`INSIGHTS FOR GPU-SWAP POLICY DESIGN` 部分——trace 数据对策略设计的直接指导。

---

## Step 5：连接 gpu-swap

最终产出不是 trace 文件，而是三个 insight：

1. 短命 block 不应换出（换出时它已释放，白花 PCIe 带宽）
2. 热 block 不应换出（频繁换回的代价 > 收益）
3. 请求类型差异是否需要分类策略（取决于 blocks-per-request 的方差）

这三个 insight 就是 gpu-swap 的 Adaptive Policy 的设计依据。
