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

关键方法（行号按本机 vLLM 0.25.1，版本不同会漂移，以你 grep 出来的为准）：
- `allocate_new_blocks(279)` — 分配新 block
- `cache_blocks(319)` — block 写入/访问
- `pop_blocks_for_free(384)` — 返回待释放的 block（`free(402)` 内部调它）
- `KVCacheManager.allocate_slots`（另一个文件 `kv_cache_manager.py`）— 每步调度

注意 `free()` 不是好的 hook 点：它把 block 交给 block pool 之后就什么都不剩了，
拿不到 block 列表。`pop_blocks_for_free()` 才是——它返回请求持有的全部
block，而且是所有释放路径（正常结束 + scheduler 抢占）的交汇点。

**检查**：
- [ ] 找到这几个方法的签名
- [ ] 画出 block 生命周期草图：allocate → cache* → free
- [ ] 能说出为什么 hook `pop_blocks_for_free` 而不是 `free`

---

## Step 2：写第一个 hook

```python
import vllm.v1.core.single_type_kv_cache_manager as m

_orig = m.SingleTypeKVCacheManager.allocate_new_blocks

def _patched(self, *args, **kwargs):
    print(f"[HOOK] allocate_new_blocks: {args}")
    return _orig(self, *args, **kwargs)

m.SingleTypeKVCacheManager.allocate_new_blocks = _patched
```

原理：Python 方法即属性，运行时替换 = monkey-patch。

**但是**：patch 只对「执行这段代码的那个解释器」有效，而 KVCacheManager
不在你启动 server 的那个进程里 —— 见下面 Step 2.5。写包装函数时建议用
`(self, *args, **kwargs)` 透传而不是硬编码参数名：0.25 的 `cache_blocks`
第一个参数是 `Request` 对象而不是 `request_id`，硬编码参数名的写法在
版本间很容易踩空。

**检查**：
- [ ] 能说清 hook 装在了哪个进程、为什么单进程不够

---

## Step 2.5：让 hook 真正跑到 EngineCore 进程里

这是本项目最容易翻车的一步，值得单独练。先做个实验：

```bash
# 启动 server，同时打印每次 vLLM 决定 multiprocessing start method 的结果
python -c "
import os, sys
import vllm.utils.system_utils as su
_orig = su.get_mp_context
def probe():
    ctx = _orig()
    print(f'[pid={os.getpid()}] mp context -> {ctx.get_start_method()!r}', flush=True)
    return ctx
su.get_mp_context = probe
sys.argv = ['vllm', 'serve', 'Qwen/Qwen2.5-0.5B-Instruct', '--port', '8000']
from vllm.entrypoints.cli.main import main
main()
"
```

你会看到 **两个不同的答案**：前面是 `'fork'`，真正创建 EngineCore 那次是 `'spawn'`。

原因是 `vllm/utils/system_utils.py:_maybe_force_spawn()`：

```python
if cuda_is_initialized():
    reasons.append("CUDA is initialized")
...
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
```

frontend 刚起来时 CUDA 还没初始化 → `fork`；等 vLLM 真要建 EngineCore
时 CUDA 已经就绪 → 强制 `spawn`。

**spawn 是全新解释器，父进程的 monkey-patch 一律不存在。**

正确做法是用 vLLM 的 plugin 入口点（`kvcache-tracer/plugin/`），
它会被加载进所有进程：

```bash
cd plugin && pip install -e .
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
# 日志里应该看到两行（frontend + EngineCore 各一行）
#   [KVTrace][pid=...] plugin 已加载，hook 安装完成
```

**检查**：
- [ ] 亲眼看到 fork/spawn 两个不同答案
- [ ] 能解释为什么"我在启动脚本里 patch 一下不就行了"是错的
- [ ] plugin 装好后，日志里出现两行 plugin 加载记录

---

## Step 3：收集结构化数据

不只是打印——写入 JSON trace。字段：

```json
{"event": "alloc", "timestamp": 0.034, "block_id": 1423, "request_id": "req_42"}
```

hook 分别记录 alloc / access / free。难点：从调用栈中回溯 request_id
（`pop_blocks_for_free(request_id)` 直接有；`allocate_new_blocks(request_id, ...)`
直接有；`cache_blocks(request, ...)` 要先 `request.request_id`）。

### 这里有个陷阱：访问次数

`cache_blocks` 只在 block **写满那一次**触发，所以数 access 事件得到的
访问次数**恒为 1**。「哪些 block 是热的」这个问题它回答不了。

真正「每个 decode step 读一次」发生在 attention kernel 里，
KVCacheManager 这一层看不到。唯一可见的等价信号是
`KVCacheManager.allocate_slots()` —— scheduler 每个 step 对每个 running
request 调一次（sched/scheduler.py:535）。

```python
import vllm.v1.core.kv_cache_manager as kcm
_orig_alloc_slots = kcm.KVCacheManager.allocate_slots

def _patched(self, *args, **kwargs):
    result = _orig_alloc_slots(self, *args, **kwargs)
    request = args[0]
    if result is not None:                    # None = 分配失败，这步没跑
        blocks = []
        for mgr in self.coordinator.single_type_managers:
            blocks.extend(mgr.req_to_blocks.get(request.request_id, []))
        # 只做内存计数，别每步每 block 写一条事件（会撑爆文件）
        for b in blocks:
            tracer.reads[b.block_id] += 1
    return result
```

计数等 block 释放时一次性写进 free 事件的 `reads` 字段 —— I/O 是
**每个 block 一条**，而不是每个 block 每个 step 一条。

**检查**：
- [ ] trace 包含四种信号（alloc / access / free / reads）
- [ ] 同一 block 的 alloc/access/free 用同一个 block_id
- [ ] 长请求的 block `reads` 明显大于短请求（实测：104 vs 4）

---

## Step 4：分析

用 `scripts/analyze.py` 回答：

1. block 生命周期分布——短命比例？
2. 访问频率分布——hot/cold skew？（看 `reads`，不是 access 计数）
3. 每请求 block 数——不同请求类型差异？
4. 请求生命周期分布？

⚠️ **insights 的结论强依赖 workload**。4 个请求的玩具测试跑出来的
"100% of blocks live < 1 second" 是这个 workload 的性质，不是 KV Cache
的普遍性质 —— 要驱动换出策略，得先跑一个像样的 workload（并发请求、
更长的生成、混合长短 prompt），再拿数据说话。

**核心产出**：`INSIGHTS FOR GPU-SWAP POLICY DESIGN` 部分——trace 数据对策略设计的直接指导。

---

## Step 5：连接 gpu-swap

最终产出不是 trace 文件，而是三个 insight：

1. 短命 block 不应换出（换出时它已释放，白花 PCIe 带宽）
2. 热 block 不应换出（频繁换回的代价 > 收益）
3. 请求类型差异是否需要分类策略（取决于 blocks-per-request 的方差）

这三个 insight 就是 gpu-swap 的 Adaptive Policy 的设计依据。
