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

## Step 1：定位 vLLM 的 BlockAllocator

vLLM 0.25（v1 engine）的核心在：

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
            tracer.reads[(b.block_id, request.request_id)] += 1
    return result
```

计数等 block 释放时一次性写进 free 事件的 `reads` 字段 —— I/O 是
**每个 block 一条**，而不是每个 block 每个 step 一条。

**键必须带 request_id**：一个 block 会被多个请求持有（prefix cache 共享、
释放后被复用），只按 block_id 计数的话，先释放的请求会把所有持有者的读数
一并取走，后面的取到 0 —— 实测 8 并发下出现 128 条 `reads=0`。

**检查**：
- [ ] trace 包含四种信号（alloc / access / free / reads）
- [ ] 同一 block 的 alloc/access/free 用同一个 block_id
- [ ] 长请求的 block `reads` 明显大于短请求

---

## Step 4：分析

用 `scripts/analyze.py` 回答：

1. block 生命周期分布——短命比例？
2. 访问频率分布——hot/cold skew？（看 `reads`，不是 access 计数）
3. 每请求 block 数——不同请求类型差异？
4. 请求生命周期分布？

⚠️ **insights 的结论强依赖 workload**。4 个请求的玩具测试跑出来的
"100% of blocks live < 1 second" 是这个 workload 的性质，不是 KV Cache
的普遍性质 —— 要驱动换出策略，得先跑一个像样的 workload。**Step 4.5 就是这件事。**

**核心产出**：`INSIGHTS FOR GPU-SWAP POLICY DESIGN` 部分。

---

## Step 4.5：正式实验 —— 3 类 workload × 并发 1/4/8 × 3 次重复

```bash
python workload/run.py --workload code-gen --concurrency 1 --dry-run --out /tmp/x.json
bash workload/run_experiment.sh      # 27 次运行，约 70 分钟
python scripts/report.py             # 聚合成 均值±标准差 + 结论句
```

### 规格与实测 token 分布

| workload | prompt（规格 → 实测） | output（规格 → 实测） | 请求数 | 共享前导 |
|---|---|---|---|---|
| short-qa | 64–256 → 71–256 | 64–128 → 65–128 | 144 | 64 tok（4 block） |
| code-gen | 512–1024 → 520–1024 | 256–512 → 256–507 | 36 | 512 tok（32 block） |
| long-summarize | 2048–4096 → 2257–4021 | 256–512 → 261–507 | 40 | 2048 tok（128 block） |

前导取 `prompt_range[0]` 向下对齐到 16 的倍数 —— 既不把最短的请求挤出规格区间，
又保证前导是整数个 block（否则最后一个不完整 block 无法被后续请求复用）。

### 结果一：prefix cache 占比只由 workload 决定，与并发度无关

| workload | 逻辑 block 获取 | 缓存命中 | 占比 |
|---|---|---|---|
| short-qa | 2,478 | 572 | **23.1%** |
| code-gen | 2,647 | 1,120 | **42.3%** |
| long-summarize | 8,851 | 4,992 | **56.4%** |

并发 1 / 4 / 8 逐位相同，三次重复也逐位相同。命中数还有个精确到个位的解析式：

```
命中数 = (请求数 − 1) × 前导block数
short-qa        4 × 143 =   572 ✓
code-gen       32 ×  35 = 1,120 ✓
long-summarize 128 × 39 = 4,992 ✓
```

这个等式能把"缓存命中率"这个通常含糊的指标**完全解释掉**：跨请求复用 100% 来自
那段共享前导，正文片段一次都没有被复用。它的另一面是下面「限制三」。

### 结果二：block 生命周期与访问次数，workload 间差异显著

| 指标 | short-qa | code-gen | long-summarize |
|---|---|---|---|
| 生命周期 P50 (C=1→C=8) | 1.354 → 1.437s | 5.055 → 5.386s | 5.575 → 6.735s |
| 生命周期 P90 | 1.879 → 1.933s | 7.182 → 7.616s | 7.210 → 8.142s |
| 存活 <1s 占比 | 22.8% → 21.9% | 5.3% → 5.0% | 1.8% → 1.5% |
| 单 block 访问 P50 | 86 | 288 | 321 |
| 单 block 访问 max | 14,123 | 13,508 | 14,812 |

生命周期 P50 跨 workload 差 **4.7 倍**，存活 <1s 占比差 **12 倍**。含义是
**换出收益随请求类型而变**：short-qa 里五分之一的换出是白费功夫，
long-summarize 里只有 1.5%。

`访问 max` 三者几乎相同不是巧合：第 1 个共享前导 block 被该类**所有**请求持有
整个解码过程，累计读数就等于该类输出 token 的总量（14,123 / 13,508 / 14,808）。
这是精确恒等式，可以当校验用。

### ⚠️ 限制一：本次实验没有触发任何逐出

| workload | 物理 block 数 | 池容量 | 池利用率 | 峰值同时存活 (C=1/4/8) |
|---|---|---|---|---|
| short-qa | 1,775 | 7,810 | 22.7% | 24 / 82 / 142 |
| code-gen | 1,496 | 7,810 | 19.2% | 88 / 301 / 567 |
| long-summarize | 3,824 | 7,810 | 49.0% | 272 / 958 / 1,760 |

long-summarize 的 **token 足迹**是池容量的 1.13 倍，事先预计会逐出，**实测没有**。
原因是足迹按"所有请求的 token 总量"算，而 prefix cache 命中 56.4% 加上物理 block
释放后回收，实际被触达的**不同**物理 block 只有 3,824 个。空闲链从未耗尽。

所以本次实验刻画的是 **block 的行为**（生命周期、访问热度、缓存复用来源），
**没有检验逐出** —— 而逐出是 swap 策略唯一的作用对象。

### ⚠️ 限制二：三次重复确认的是确定性，不是噪声

三次重复的墙钟差异只有 0.005–0.040s（0.01%），结构量逐位相同，标准差近乎 0。
这是 `temperature=0` + `ignore_eos` + 独占 GPU 的必然后果。

好处是：**任何观察到的 workload 间差异都是真实差异，不是噪声。**
代价是：**没有测到环境噪声**（共享 GPU、邻居负载、热漂移）。报告里不能把
"± 0.000" 当成"很稳定"来宣传 —— 它只说明这台机器上这几分钟里没有扰动。

### ⚠️ 限制三：可复用集合太小，策略对比会退化

跨请求可复用的只有共享前导，所以可复用 block 总数只有 4 / 32 / 128 个，
而垃圾 block 有 1,771 / 1,464 / 3,696 个：

| workload | 可复用 block | B=64 | B=128 | B=256 |
|---|---|---|---|---|
| short-qa | 4 | 全装得下 → 无区分 | 同 | 同 |
| code-gen | 32 | 全装得下 → 无区分 | 同 | 同 |
| long-summarize | 128 | **有压力** | 刚好装下 | 无区分 |

9 个格子里只有 1 个能让 LRU / LFU / Adaptive 分出高下。要让 64/128/256 这个
区间有意义，需要一份**倾斜复用**的 workload（K 个系统提示模板 + Zipf 分布），
把可复用集合放大到预算的数倍 —— 见 Step 4.7。

**这条限制同时也是一个方法论收获**：判断"预算档位选得对不对"不能靠感觉，
要先把「命中 50% / 80% 的请求各需要多少 block」算出来。热集合远小于预算时，
任何策略都能把该留的全留下，对比会退化成没有差别 —— 那时该改的是预算，
不是策略。这个量现在由 `workloads.describe()` 直接报出来。

### 踩过的坑

**两个必须分清的口径**，混用会得出错的热度结论：

- free 事件里的 `reads` 是「**这个请求**持有期间读了这个 block 多少次」，
  按 `(block_id, request_id)` 归因。一个 block 被 N 个请求持有过，就有 N 条
  带 reads 的 free 事件。
- **热度**是「这个 block 总共被读了多少次」，要把同一 block 的各条 free 事件的
  `reads` **相加**。`analyze.compute_access_counts` 就是这么算的。

  踩过的坑：这里写成覆盖而不是累加，会让 23% 的 block 读数失真、总量丢掉
  30%~43%，最热的 block 从 2,064 次被低估到 768 次。更阴的是"热 block 占比"
  这类粗指标几乎看不出变化（23.3% vs 20.5%）—— 只有 Max / P90 会暴露它。

**`reads` 的精确语义**（已逐请求验证）：一个 block 的 `reads` 等于「从它分配到
请求结束之间的 decode step 数」。`reads_max` 精确等于该请求的 `completion_tokens`，
`reads = completion` 的 block 数恰好是完整的 prompt 段（短 prompt 请求为
16 个 = 256/16）。所以 replay 展开访问流时，在 `(alloc_t, free_t]` 上均匀铺点
**与真实逐步时序等价**，不是近似。

**`/reset_prefix_cache` 无论成败都返回 HTTP 200**，底层只在「除 null block 外
全部空闲」时才真清，失败只打 warning。判定只能靠数日志里
`Successfully reset prefix cache` 的行数。

---

## Step 4.6：GPU↔CPU 迁移层的单价与干扰

### 先别自己写迁移层 —— vLLM 里已经有了

`vllm/v1/kv_offload/`（8,196 行）里的 `cpu/gpu_worker.py` 就是一个完整的异步迁移层：
每次传输一个独立 CUDA stream（带 stream/event 池）、GPU→CPU 前
`stream.wait_stream(current_stream())` 等计算写完、传输之间用 event 串链、
CPU→GPU 方向用 `CU_MEMCPY_SRC_ACCESS_ORDER_ANY` 让驱动做流水，并且用
`enable_timing=True` 的 CUDA event **记录每次传输耗时**。

配置：`--kv-offloading-size <GiB>`，策略由 `extra_config["eviction_policy"]` 选，
**自带 LRU 和 ARC**。策略基类是 `vllm/v1/kv_offload/cpu/policies/base.py` 的
`CachePolicy`（`get/insert/remove/touch/evict(n, protected)/clear`），注册表
`manager.py:30` 的 `_CACHE_POLICIES` 是个普通模块级 dict —— 我们的 plugin
可以像装 tracer hook 一样把它 patch 进去，不用 fork vLLM。

所以增量价值在 **LFU / Adaptive 两个策略类**，以及**迁移流量到底能不能被
decode 藏住**这件事 —— 后者原先只是一句"异步发出的，不阻塞"，从未被测量过；
本节的干扰曲线把它测了（红线 ~19 个 block/step，比当初假定的少 21 倍）。

### 重算 vs 换入：换入便宜约 31 倍

Qwen2.5-7B 每 token KV = 2×4×128×2×28 = 57,344 B，每 block（16 token）
= 917,504 B。于是：

```
换回: 917,504 B/block ÷ 17.5 GB/s(实测，见下) = 52.4 µs/block
重算: 101.27 µs/token × 16 token            = 1,620 µs/block   ← 实测
```

**换回比重算便宜约 31 倍**，2048-token 前缀是 6.7 ms vs 207 ms。
这为整个迁移层提供了前提，也意味着 vLLM v1 抢占默认选「重算」而非「换出」，
在这个硬件上对长前缀是次优的。

#### 重算单价换过三版，留个记录

这条结论的方向完全由重算的单价决定，而它错了两次：

| 版本 | 值 | 怎么来的 | 问题 |
|---|---|---|---|
| 1 | 38.5 µs/token | TTFT 中位数配 prompt 中位数，拟合 25,974 tok/s | 中位数不是同一个请求。反推 396 TFLOPS，是 4090 峰值的 2.4 倍 |
| 2 | 38.5 µs/token | 改从 decode 吞吐反推，当成"保守下界" | prefill 是 compute-bound 批量矩阵乘，decode 是访存受限，拿 decode 给 prefill 定价没有依据 |
| 3 | **101.27 µs/token** | 单请求微基准实测（`scripts/measure_prefill.py`） | — |

第 2 版尤其值得记：它被写成了"取对重算最有利的估计，所以结论更稳"，
听上去是保守处理，实际上是**把重算算便宜了 2.6 倍**。而且"保守"这个
说法自带光环，让人不再去查那个数本身。真正的保守是把它测出来。

实测方法（`measurements/prefill_marginal_cost.txt`）：唯一 prompt、C=1、
`max_tokens=1`、关掉 prefix cache，测 512/1024/2048/3072/4096 五档 TTFT
后线性拟合，得

```
TTFT ≈ 9.79 ms + 101.27 µs × prompt_tokens      →  9,874 tok/s
```

固定开销只有 9.79 ms，反推 150 TFLOPS = 4090 bf16 峰值的 91% —— 确实
测到了 compute-bound 的纯 prefill。用**斜率**而不是绝对值，是因为网络、
排队、调度与长度无关，会被截距吸收掉。

#### PCIe 单价：也测了，而且发现原来的形状假设是错的

重算那一侧测准之后，整条结论就完全压在这个未测的常数上了。原先写的是
「24 GB/s，**假定值**」，并且用一句"它的不确定性影响不了结论"带过 ——
这是同一个毛病：一个看起来保守的数字，因为看着可信，就再没人去查。

真去测（`scripts/measure_pcie.py` → `measurements/pcie_link_ceiling.txt`）
之后，冒出两件没预料到的事。

**第一件：链路空闲时确实是降速的，但负载一到就升。** 同样先是个闸门：

```
空闲               2.5 GT/s (Gen1)   ×16  → 约 3.2 GB/s
负载下             16.0 GT/s (Gen4)   ×16   100% 的采样点
升上去花的时间      4 ms
```

所以不用按 3.2 GB/s 重写（那会差 7 倍），但**每次测量都得预热**。

**第二件（更要紧）：一次 block 拷贝不是「1 次 875 KB」，是「28 次 32 KB」。**
原先的推理是「block 917,504 B 远大于 vLLM 的 28 KB Triton 阈值，所以走 C++
DMA 路径」—— 结论对，理由和余量都错：

- `_select_swap_blocks_fn`（`kv_offload/cpu/gpu_worker.py:49-58`）比较的是
  **逐 data_ref** 的 `page_size_bytes`，即**单层**大小 = **32,768 B**，
  不是整块的 917,504 B。阈值 `28,672 B`，**余量只有 14.3%，不是 14 倍**。
- 描述符按 `(group, data_ref)` 展开（`gpu_worker.py:283-357`），所以
  一次 `cuMemcpyBatchAsync` 里是 **28 个 32,768 B 的 op**。
- 顺带：路径选择是**看 size 的**，`--block-size 8` 会让 CPU→GPU 方向
  静默切到 Triton kernel。所以"两个方向都走 C++"是当前 flag 的性质，
  不是模型的属性。

32,768 B ÷ 24 GB/s = 1.37 µs，**低于 DMA 启动开销**。把总字节数固定、
只改描述符大小，就量到了这个开销：

| 描述符大小 | 描述符数 | per-op 实测 | 链路项 @23.4 GB/s | 残差 |
|---|---|---|---|---|
| 8,192 B | 14,336 | 0.79 µs | 0.35 | 0.44 |
| 16,384 B | 7,168 | 1.14 µs | 0.70 | 0.44 |
| 32,768 B | 3,584 | 1.88 µs | 1.40 | 0.48 |
| 65,536 B | 1,792 | 3.24 µs | 2.80 | 0.44 |
| 917,504 B | 128 | 38.25 µs | 39.2 | （被淹没）|

残差稳定在 **0.44 µs/描述符**，于是生产单价是

```
917,504 / 23.4 GB/s + 28 × 0.44 µs = 39.2 + 12.3 = 51.5 µs/块
实测                                              52.6 µs/块
```

**24 GB/s 这个旧值错得不大也不小**：边际带宽 23.4 几乎就是 24，漏掉的是那
28 个描述符的固定开销（+33%）。换回的优势因此从 42× 变成 **31×** —— 量级没变，
但这是第一次它站在测量上而不是假设上。

另两条一起测出来、后面要用的事实：

- **两个方向不可加。** d2h 单独 17.5、h2d 单独 17.2、两个同时跑聚合只有
  20.0 GB/s（若可加应接近 34.8）。所以强度曲线的横轴必须是**聚合**字节/s。
- 渐近上限 23.2–24.4 GB/s（64 MB 连续），生产形状 17.5 GB/s。差的就是描述符开销。

### 迁移流量的干扰曲线：每 step 能"免费"迁移多少 block

上一节测的是**链路自身**能跑多快（空载）。这一节测的是**跑迁移的同时 decode
被拖慢多少** —— 两者是不同的数，混用会高估迁移层的收益。

`scripts/measure_interference.py` 让一个常驻进程按**连续限速**灌 PCIe 流量
（token bucket，chunk 8 block ≈ 0.4 ms ≪ 一步；不是占空比 —— 占空比会把每步
延迟变成双峰，"平均 GB/s"就成了没有哪一步真正经历过的数），同时用 short-qa
在 C=1 下连续 decode，主信号取 `EngineCoreProc._process_engine_step` 的
**start-to-start** 间隔，逐 block 交错 baseline/load 配对。服务配置与其余
Step 4 的数据同口径（`--gpu-memory-utilization 0.45 --block-size 16
--enable-prefix-caching`）。

**噪声底 = 0.020 ms（0.13%）**，取全部 **90 个** baseline block 的步中位数跨度。
判据：|Δ中位步长| 超过它就记为"看得见"。这里刻意用**跨三个服务实例**的全局跨度
（单看每个实例只有 0.012 / 0.016 / 0.013 ms），偏保守；而配对始终在实例内部做，
不跨实例比。

三段合起来 180 个 block：主 run 打四个方向 × 5 个档位（5.5–22 GB/s），sham
单独补跑（主 run 的那一段是在修限速 bug 之前跑的，作废），低速率段再补
1.1 / 2.2 / 4.4 / 8.8 GB/s 把红线夹住。合并与判读在
`scripts/interference_report.py`，产物
`measurements/pcie_interference_redline.txt`（红线 + 机制分解）、
`pcie_interference.csv`（逐 block 明细）、`pcie_interference.txt`（配对表）：

| 速率 GB/s | block/s | d2h Δ | h2d Δ | both Δ |
|---|---|---|---|---|
| 1.10 | 1,199 | **+0.013**（藏得住）| +0.043 | — |
| 2.20 | 2,398 | +0.028 | +0.096 | — |
| 4.40 | 4,796 | +0.065 | +0.161 | — |
| 5.50 | 5,990 | +0.083 | +0.210 | +0.144 |
| 8.80 | 9,590 | +0.126 | +0.394 | — |
| 11.0 | 11,970 | +0.160 | +0.557 | +0.440 |
| 16.5 | 17,970 | +0.206 | +0.607 | +0.561 |
| 22.0 | 23,960 | +0.264 | +0.917 | +0.616 |

（Δ 单位 ms，相对配对的 baseline；22 GB/s 是实测链路饱和值，见上一节。
`both` 的横轴是**聚合**字节/s —— 两个方向不可加。）

**红线**：

```
d2h（换出）   < 1.1 GB/s ≈ 1,200 block/s ≈ 一步 19 个 block
h2d（换回）   < 1.1 GB/s（最低档就已经看得见；按 1.1–4.4 档线性外推 ≈ 0.5）
```

d2h 的红线是**夹住**的：1.10 GB/s 三组配对 Δ = +0.015/+0.015/+0.008 全部在
0.020 以内，2.20 GB/s 三组 +0.032/+0.023/+0.029 全部超出。h2d 在 1.10 GB/s
就已经 2 倍于噪声底，所以它的红线更低；再往下测需要把 chunk 缩到 1 block，
而那会掉进上一节记录过的**小批量提交开销污染区**（提交 > 传输），所以没有
继续压，标为外推。

**两条必须一起看的结果。**

1. **sham 对照说明效应来自 DMA，不是来自 CPU。** 零字节 DMA、完全相同的
   CPU 循环与描述符构造：12 个负载 block 全部落在噪声底内，最大 |Δ| =
   0.011 ms，各档均值 +0.000 / +0.005 / +0.006 / +0.000 ms（对应等价
   5.5 / 11 / 16.5 / 22 GB/s 的迭代次数）。既不超过噪声底，也没有随"负载"
   上升的趋势 —— 做 CPU 活最多的那一档，均值是 +0.000 ms。

   这一条必须做，不能靠推理：这台机器的 CPU 频率是 `schedutil` 调的，实测在
   **1.5–3.5 GHz**（下限到上限）之间大幅摆动，所以"测量期间 CPU 速度不变"
   这个前提是假的，它会把 CPU 侧的活算快或算慢。基线时负载进程是 idle 的、
   负载时它在跑，两者 CPU 状态本就不同。sham 是这个差异的**实测上界**：
   它一样会触发升频、一样抢 CPU，却测不出任何步长变化。如果 sham 也动，
   上面所有数字就都只能读成"CPU 争用"，而不能读成"PCIe 干扰"。
2. **拖慢几乎全部落在 GPU 执行段上。** 用 `--enable-logging-iteration-details`
   精确定位（它包的是 `future.result()`，`core.py:495-499`），配 plugin 记的
   `iter` 逐迭代 join：

   | 22 GB/s | 完整步 Δ | 执行段 Δ | 其余 Δ |
   |---|---|---|---|
   | d2h | +0.264 | +1.80 | −1.54 |
   | h2d | +0.917 | +2.48 | −1.55 |

   **实测到的**是：执行段涨了 1.8–2.5 ms，而完整步只涨 0.26–0.92 ms，
   差额被 schedule + `update_from_output` 那段**反向吃掉**（逐块看，
   `Δ执行段 + Δ其余 ≈ Δ完整步` 在 36 个负载 block 上一致成立）。

   **据此的推断**（是模型，不是直接测量）：C=1 的 decode 循环约 15.4 ms 的活
   里有 ~3.2 ms 是 CPU 侧、不占用 GPU 的（baseline 完整步 15.41 − 执行段
   12.17）；DMA 把 GPU 那段顶长以后，这 3.2 ms 被逐步吃满，所以步长到
   ~15.4 ms 之前几乎不动。若这个模型成立，**它是个台阶不是斜坡**：GPU 执行段
   一旦涨到 ~15.4 ms 撞上 CPU 侧的底，之后步长会开始 1:1 跟着涨。h2d 在
   22 GB/s 已经把执行段推到 14.7 ms，**离那个底只剩约 5%**。

   这条推断**本轮没有直接验证**（要验证得再往上加压，看是否出现拐点），
   所以别把 22 GB/s 那档的 +0.92 ms 线性外推到更高速率 —— 外推方向正好
   落在拐点之后。

对照旧值：这条路原本写的是 `W ≈ 24 GB/s（假定值）→ 408 个 block/step`。
实测红线是 **~19 个 block/step**，差 **21 倍**。而且旧写法默认"把链路跑满
也是免费的"，实测把链路跑满（22 GB/s = 374 block/step）代价是换出 +1.7%、
换回 +5.9%。**两个方向里换回贵得多**（同速率下约 2.7–3.5 倍）——注意本实验里
两个方向都是异步发出、探针从不同步等待，所以这个不对称是 DMA 路径本身的性质，
不是 vLLM 那条"h2d 压在关键路径上"的调度性质，两者不要混为一谈。

### 一个必然出现的约束

正在跑的请求持有的 block **不能被换出**（每步都要读），所以 GPU 池必须装得下
并发工作集。实测各组合的单请求最大与并发峰值：

| 组合 | 单请求最大 block | 并发峰值 |
|---|---|---|
| short-qa | 24 | 24 / 82 / 142 |
| code-gen | 88 | 88 / 301 / 567 |
| long-summarize | 272 | 272 / 958 / 1,760 |

所以「热池预算」在物理上指的是 **GPU 上可缓存空闲 block 的预算**（正在跑的
请求另占一块，不参与淘汰），这也正是 `CachePolicy.manager.py:205` 的口径
（"cache-policy removes only idle blocks"）。

---

## Step 4.7：倾斜复用 workload 与离线 replay

Step 4.5 那批数据刻画了 block 的**行为**，但没有检验逐出 —— 而逐出是 swap
策略唯一的作用对象。这一节补上那一步：再造一份可复用集合足够大的 workload，
然后在同一份 trace 上离线比较逐出策略。

### 为什么不能直接拿 Step 4.5 的 trace 做策略对比

可复用 block 只有 4 / 32 / 128 个（就是那段共享前导），而预算是 64/128/256。
9 个格子里 8 个是「所有策略都能把该留的全留下」，比不出任何东西。

### skewed 模式：K 个模板 + Zipf

`workload/run.py --prefix-mode skewed`。三个设计要点，缺一个就退化：

1. **长度各异**。所有模板一样长的话，"部分命中"（前 j 块在池里、第 j+1 块被
   逐出）根本不会出现 —— 而部分命中恰恰是策略差异最大的地方。
2. **长度与热度无关**。长度按 block 对齐后 shuffle 再按序号赋 Zipf 权重。
   不 shuffle 的话最短的模板天然对最多请求可用，同时又最热，两个变量绑死。
3. **逐块互不相同**。这一条是被实测打回来的：最初写成
   `[template-i] + 同一段 instruction + 随机填充`，以为序号标记能让第 0 块分叉，
   结果 instruction 有一百多个 token，第 1~7 块在所有模板间逐字节相同 ——
   `check_templates_disjoint()` 当场查出 code-gen 有 69 个重复块。模板互相命中
   会让 "请求命中自己的模板" 这个模型直接失效。现在标记之后只保留指令开头
   6 个词（塞得进第 0 块），其余全部用每模板独立的随机种子生成。

**prompt 长度必须先采、再挑模板**，不能反过来：先挑模板再定长度的话，prompt
分布会被模板长度分布带偏，元数据里那条 "uniform(lo,hi)" 就成了假话。

跑出来的规格（`--dry-run` 直接打印，不用起服务）：

| workload | K | 可复用 block | 独有 prompt block | 覆盖 50% 请求 | 覆盖 80% 请求 |
|---|---|---|---|---|---|
| short-qa | 64 | 259 | 672 | 59 | 152 |
| code-gen | 32 | 418 | 945 | 101 | 264 |
| long-summarize | 16 | 535 | 5780 | 104 | 255 |

最后一列是选预算的依据：三类的「覆盖 80%」都落在 150~265，横跨 64/128/256，
每一档都有真实取舍。（"独有 prompt block" 不含解码过程中新分配的块。）

```bash
REPS=1 bash workload/run_skewed.sh          # 先跑一轮看数据（约 25 分钟）
bash workload/run_skewed.sh                 # 完整 27 次（约 70 分钟）
```

产物写在 `traces-skew/ results-skew/ logs-skew/`，和 Step 4.5 那批完全分开 ——
那批的结论已经写进 README，不能被覆盖或混在一起统计。

### 离线 replay：三种结局，代价差约 30 倍

`scripts/replay.py` 在同一份 trace 上模拟 B 个**空闲** block 的热池。一次
block 获取有三种结局：

| 结局 | 条件 | 代价 |
|---|---|---|
| 命中 | 块还在热池里，或已被同批请求共享 | 0 |
| 换回 | 曾被逐出、现在在 CPU 上 | 52.4 µs/块（**实测** 17.5 GB/s 有效带宽） |
| 重算 | 从没见过 | 1,620 µs/块（**实测** 101.27 µs/token × 16 token） |

**这一行里两个数现在都是实测的了**，而且是被测了两次、两次都和原来的估计
不一样：

- 重算：旧值 616 µs/块 把差距压成 16×（低估了重算的代价）
- 换回：旧值 38.2 µs/块 把差距撑到 42×（高估了换回的速度，漏掉描述符开销）

两次修正方向相反，都来自同一类错误 —— 一个看着可信的数字再没人去查。
最终落在 **31×**。

换出**不阻塞**请求：迁移层是异步的（见 Step 4.6 的 CUDA stream/event 结构），
没有任何一步会去等它完成，所以 replay 的代价模型里它记 0 是对的。

但"不阻塞"**不等于"不计入延迟"** —— 这一条原先就是这么写的，Step 4.6 的干扰
曲线把它测掉了。换出一旦超过 ~1.1 GB/s 就会**把 decode 的每一步都拖慢一点**
（异步发出去的 DMA 在 GPU 侧和模型抢），到把链路跑满（22 GB/s）时是 +1.7%
（换回同期是 +5.9%）。所以 replay 报出的换出量要拿红线过一遍：**约 19 个
block/step 以内才是真的免费**，超出的部分不是 0 代价，只是代价记在
decode 吞吐上而不是记在这块 block 的获取延迟上。

**不按 block_id 记账。** 物理 block_id 会被池子反复复用，同一个 id 在不同时刻
是完全不同的内容；按它记账会把两个不同模板的块当成同一个。改用**逻辑块**，
这正是 prefix cache 哈希的东西：

    模板块   ("t", 模板号, 块内偏移)     ← 跨请求可复用
    独有块   ("j", 请求id, 块内偏移)     ← 只服务一个请求

映射不需要重新分词：元数据的 `schedule` 逐请求记了 template 和 template_blocks，
trace 逐块给了获取/释放时刻，请求的块按获取时刻排一下就是偏移序（prompt 块一定
先于解码块）。

### 先验证模型，再信数字

```bash
python scripts/replay.py verify --results-dir results-skew
```

它逐个请求核对「真实 trace 的 prefix cache 命中数」与「模型的预测」。skewed
workload 上实测 **93/93 请求、584/584 块完全一致，零个模型没预测到却发生了的
命中**。这条不过就不要看 replay 的数字 —— 重放跑得再顺，模型错了也是错的。

### 又抓到两个：只在 c8 显形，而且都会让结果"看起来正常"

这两个是把 27 条 trace 全部过一遍 `verify` 之后才暴露的 —— c1/c4 全绿，只有
code-gen c8 每轮报 1 个请求的模板块数对不上。两个都是**模型错了但输出毫无异常**
的那一类。

- **解码块被误判成模板块。** 物理块会被回收再分配：一个模板块被释放之后，它的
  id 可能落到某个请求的解码块上。最初的判据只有 `blk in idx`，于是那个解码块被
  当成模板块，**凭空多出一次"共享命中"**。修法是加一道"只有 prompt 里的块才可能
  是模板块"。c1/c4 池子宽裕、模板块从不释放，所以查不出来 —— 只有在 c8 才复发。

- **prompt 被切成多批 prefill。** 划 prompt/解码边界的判据原本是"最早那个时刻
  一次拿到的块"，依据是当时实测 132/132 个请求都满足"prompt 一次拿完"。**这个
  前提在 c8 垮了**：调度器会把 prompt 分块 prefill，第二批及以后的块落在更晚的
  时刻，被整批漏掉 —— 实测少 8 个模板块。现在改用元数据里的 `prompt_blocks` 划
  这条线（客户端按同一套 tokenizer 和 block size 算出来的），不依赖调度器行为。

教训是：**"实测 132/132 都满足"是样本结论，不是不变量。** 两个错误都寄生在一个
更极端的配置（更高的并发）里，而那个配置恰好是要对比的维度之一。顺带一提，
`verify` 报"1 个请求对不上"时如果不写明方向（多还是少），根本无从下手 —— 现在
它会打出差值，正负分别对应"误判"和"漏取"两种成因。

### 更早的两个仿真坑（都不会在输出里显形）

- **LFU 的频次存在了池子记录里**，块一被钉住（从 pool 里 pop 出来）频次就丢了，
  释放回池时从 0 重新开始。结果是 LFU 静悄悄地退化成"随便挑一个"，而报表上
  一切正常。现在 `pool / freq / last` 三份状态分开存。
- **预算为 0 时直接 `continue`**，块凭空消失而不是换出到 CPU。于是本该最贵的
  档位算出最便宜的结果（换回数恒为 0、全算成重算）。删掉那个特判，让通用路径
  处理：`len(pool) >= 0` 恒真，自然走到"淘汰的就是刚释放的这块"。

另外 `adaptive` 的衰减常数 tau 默认取**同一个块相邻两次被获取的间隔**中位数
—— 用 trace 自己的时间尺度，而不是拍一个常数。tau→∞ 时它**逐位退化成 LFU**，
这个性质在单测里钉住了，保证两种策略的对比是自洽的而不是在比一个随手写的启发式。

（这里也错了一版：最初取的是**持有**时长的中位数，注释还写着"用 trace 自己的
时间尺度"。短上下文 workload 里块只活 1.4s，而池子的换手比这快得多，衰减项把
一切历史频次都抹成 0，adaptive 逐档退化成 LRU —— 报表上没有任何异常，只是
"adaptive 和 LRU 居然一样"。持有 1.4s 的块完全可能在 10s 后才被复用，两者不是
一回事。正确的尺度是**复用间隔**，它们差了一个数量级。）

### 结果：三档预算 × 三种在线策略 × 三档并发

27 次运行全部通过 `verify`（零警告）。完整表格在 `measurements/replay_skew.md`，
原始数据 `results-skew/*.json`。**容量缺失率**（分母剔掉 compulsory，即
"工作集本身就比池子大"那部分不算策略的账）：

#### short-qa

| 策略 | 并发 | B=64 | B=128 | B=256 | 换回数 64/128/256 |
|---|---|---|---|---|---|
| LRU | 1 | 86.3% | 72.4% | 37.3% | 504 / 423 / 218 |
|  | 4 | 75.9% | 57.9% | 33.0% | 443 / 338 / 193 |
|  | 8 | 56.7% | 42.3% | 25.2% | 331 / 247 / 147 |
| adaptive | 1 | 55.0% | 47.8% | 26.0% | 321 / 279 / 152 |
|  | 4 | 50.2% | 40.6% | 22.9% | 293 / 237 / 134 |
|  | 8 | 38.9% | 27.7% | 16.8% | 227 / 162 / 98 |
| **LFU** | 1 | 41.8% | 25.7% | **16.3%** | 244 / 150 / 95 |
|  | 4 | 34.8% | 24.5% | **14.9%** | 203 / 143 / 87 |
|  | 8 | 28.9% | 18.7% | **12.7%** | 169 / 109 / 74 |
| belady (离线) | 1 | 11.5% | 0.0% | 0.0% | 67 / 0 / 0 |
|  | 4 | 8.9% | 0.0% | 0.0% | 52 / 0 / 0 |
|  | 8 | 6.3% | 0.0% | 0.0% | 37 / 0 / 0 |

#### code-gen

| 策略 | 并发 | B=64 | B=128 | B=256 | 换回数 64/128/256 |
|---|---|---|---|---|---|
| LRU | 1 | 81.7% | 62.5% | 59.0% | 349 / 267 / 252 |
|  | 4 | 46.1% | 45.7% | 41.9% | 197 / 195 / 179 |
|  | 8 | 28.8% | 24.8% | 19.4% | 123 / 106 / 83 |
| adaptive | 1 | 42.6% | 37.7% | 37.7% | 182 / 161 / 161 |
|  | 4 | 30.2% | 30.2% | 30.2% | 129 / 129 / 129 |
|  | 8 | 23.0% | 17.3% | 17.3% | 98 / 74 / 74 |
| **LFU** | 1 | 35.4% | 34.0% | **34.0%** | 151 / 145 / 145 |
|  | 4 | 30.2% | 30.2% | **30.2%** | 129 / 129 / 129 |
|  | 8 | 23.0% | 23.0% | **17.3%** | 98 / 98 / 74 |
| belady (离线) | 1 | 19.4% | 4.2% | 0.0% | 83 / 18 / 0 |
|  | 4 | 10.1% | 0.0% | 0.0% | 43 / 0 / 0 |
|  | 8 | 1.2% | 0.0% | 0.0% | 5 / 0 / 0 |

#### long-summarize

| 策略 | 并发 | B=64 | B=128 | B=256 | 换回数 64/128/256 |
|---|---|---|---|---|---|
| LRU | 1 | 87.3% | 87.3% | 73.0% | 1,269 / 1,269 / 1,061 |
|  | 4 | 52.6% | 52.6% | 42.4% | 765 / 765 / 616 |
|  | 8 | 24.5% | 20.6% | 18.6% | 356 / 300 / 270 |
| adaptive | 1 | 67.8% | 50.7% | 43.2% | 985 / 737 / 628 |
|  | 4 | 44.7% | 33.8% | 30.9% | 649 / 491 / 448 |
|  | 8 | 19.1% | 17.1% | 8.2% | 277 / 249 / 119 |
| **LFU** | 1 | 63.4% | 47.6% | **25.2%** | 921 / 691 / 366 |
|  | 4 | 35.2% | 17.8% | **9.7%** | 511 / 259 / 141 |
|  | 8 | 11.8% | 6.5% | **4.6%** | 171 / 95 / 67 |
| belady (离线) | 1 | 47.3% | 23.5% | 3.6% | 687 / 341 / 52 |
|  | 4 | 18.8% | 6.5% | 0.0% | 273 / 94 / 0 |
|  | 8 | 3.2% | 0.0% | 0.0% | 47 / 0 / 0 |

### 三条结论

**一、LFU 在全部 27 格里都赢，而且赢得不小。** B=256、c=8 时 LFU 相对 LRU：
short-qa 12.7% vs 25.2%（2.0×），code-gen 17.3% vs 19.4%，long-summarize
4.6% vs 18.6%（4.0×）。

**二、预算越大越划算，但在 code-gen 上有一大段是平的。** code-gen 从 64 加到
256（4 倍预算）几乎不改善。实测原因是**池子被根本不会再被复用的块占住了**：
在 B=256 下池内平均 **61.8%** 的条目是独有块（short-qa 是 54.3%）—— 也就是
`("j", 请求, 偏移)` 这类只服务过一个请求一次、之后再无引用的块。

把预算继续加到 512、1024 才见效（code-gen c4 换回数 129 → 74 → 20 → 0；
B=4000 时池子从不溢出，换回数归零）。

这条对迁移层设计是有直接含义的：**光加预算不够，得让策略认得出哪些块根本不会
被复用。** LFU 恰好部分做到了（垃圾块的频次恒为 1），这正是它领先的来源。也
说明"把池子做大"和"把池子用对"是两件独立的事，后者不需要额外显存。

**三、并发度越高，缺失率越低 —— 这一条反直觉，但它是真的。** 同一批请求在
c1/c4/c8 下 `n_acq` 与 compulsory **逐位相同**（实测 short-qa 2394/1810、
code-gen 2635/2208、long-summarize 8683/7230），所以**分母是常数，缺失率与
换回数严格成正比**。下降完全来自绝对量：short-qa B=64 的 LFU 换回数
244 → 203 → 169。

原因是并发制造了时间局部性：两个同时运行的请求若用同一模板，后到的那个是
`shared` 命中，根本不经过池子。所以**跨并发度比较缺失率是苹果比橘子** ——
它混进了"共享变多"这个效应。策略之间的比较必须在同一并发度内做。

### 还剩多少空间

belady（离线最优，知道未来，**不可实现**）在 B=128 时几乎全部归零：short-qa
和 code-gen 的九格里只有 code-gen c1 还有 4.2%，其余全是 0.0%。这说明**可复用
集合本身不大** —— 在 128 个块的预算下，只要选得对就没有容量缺失。

对照之下 LFU 在同样条件下是 14.9%~34.0%。这个差距就是"改进策略"这件事的
全部空间，也是这份 replay 对迁移层的实际价值：**不是"再调调启发式还能好多少"，
而是"最优也就这样了"。** 再往上走要靠把工作集搬出显存（也就是迁移层本身），
而不是靠逐出策略。

### 一个不能省的限制

**这里的 Zipf 热度是平稳的**，即模板的受欢迎程度不随时间漂移。这恰好是 LFU
的主场 —— 它比 LRU 好，很大程度上是在利用一个永不变化的分布。真实负载里
热门模板会轮换，那时 LFU 会因为积累的陈旧频次而迟钝，adaptive 才有机会反超。
**所以"LFU 最好"这个结论只在这份 workload 上成立**，换成热度漂移的负载要重测，
只需要加一个让模板热度随时间轮转的变体。

---

## Step 4.8：把逐出策略搬到真机上

Step 4.7 的结论是**离线模拟**出来的：LFU 在全部 27 格里都赢，B=256/c=8 时
short-qa 缺失率 12.7% vs LRU 的 25.2%。这一节把它放到 vLLM 自己的 CPU
offloading 层上真跑一遍，看那条结论转不转得过去。

增量代码是 `plugin/vllm_kvpolicy.py`：LFU / Adaptive 两个策略类补进
`_CACHE_POLICIES`（LRU / ARC 用 vLLM 自带），四个策略走**同一套** manager /
连接器 / 迁移层代码，policy 对象是唯一变量。跑法：

```bash
REPS=3 bash workload/run_policy_compare.sh     # 4 策略 × 3 次，约 30 分钟
python scripts/policy_report.py                # → measurements/policy_compare.md
```

### 策略只管得了三层缓存里的一层

Step 4.7 的 replay 里，策略管的是**热池**那一层。真机上不是：复用被两级吸收，
GPU prefix cache（vLLM 自己的 LRU，**所有策略共用**）在前，CPU offload 池在后，
后者只在 GPU 落空时才被问到。实测这个 workload（short-qa c=8，GPU 池 256 块、
CPU 池 292 块）：

| 层 | 每轮服务了多少 | 谁在管 |
|---|---|---|
| GPU prefix cache | 6,208 token = **388 块**（命中率 26.8%）| vLLM 内部的 LRU，四个策略完全一样 |
| CPU offload 池 | **150 ~ 170 块** | 被比较的那个策略 |

所以被比较的那一层只占缓存服务的 **28%~30%**（150/538 ~ 170/558），而策略对它
上下也就 13% 的差别（170 vs 150）—— 折算到全部缓存命中上是百分之几。
**TTFT 因此全程不动**
（12 次运行的 P50 落在 51.0~51.6 ms，组间差 0.1 ms、组内极差 0.6 ms —— 
比 lru 自己两次重复之间的差别还小）。端到端指标在这里没有分辨力，只能靠
`load_blocks`（从 CPU 池换回来的块数）读。

这不是实验没做好，是**离线 replay 和真机测的根本不是同一件事**：replay 里
B=256 换的是第一层的策略，真机上换的是第二层。把 4.7 的倍数直接搬过来会高估。

### 坑一：GPU 池装得下工作集时，CPU 池是个只写不读的坟场

这是本轮最容易出一份"干净但全错"的报告的地方。用默认口径起服务
（`--gpu-memory-utilization 0.45`）时，**GPU 池是 124,960 token（7,810 块），
而整个 workload 的足迹只有 37,382 token** —— GPU 池自己就装得下一切，
CPU 池于是永远不会被问到。实测：

| | 默认 GPU 池（124,960 token）| 本实验口径（4,096 token）|
|---|---|---|
| `store_evicted`（换出次数）| 556 | 556 |
| 存进 CPU 池的字节 | **778 MB** | — |
| 从 CPU 池读回的字节 | **0** | — |
| `lookup_hit` / `load_blocks` | **0 / 0** | 150~170 |

换出计数一切正常、指标齐全、报表干净 —— 而四个策略的"命中率"全是 0，
比不出任何东西。**判据是 `load_blocks`，不是 `store_evicted`**：
后者只说明池子满了，不说明池子有用。

要把 GPU 池压到工作集以下，`--gpu-memory-utilization` 只能粗调，得直接给
`--kv-cache-memory-bytes <块数 × 917,504>`。它的下限是 `max_model_len` 那么多
token（一个请求得装得下），所以 `--max-model-len` 要一并降到 2048。

`scripts/policy_report.py` 会在 `load_blocks == 0` 时显式报出来，不让这份报告
以"四个策略完全一样"的形式发出去。

### 坑二：观测器被两个进程交错写

第一版观测器在 `register()` 里开文件 —— 而 `register()` 在**每个** vLLM 进程里
都跑一次，包括从来不建 manager 的 frontend。结果是 frontend 和 EngineCore
两个进程往同一个 JSONL 里交错写，**frontend 那份计数器恒为 0**，而它写在后面。

分析脚本取"段内最后一条"，取到的就是那个 0 —— 四个策略的差值全是 0，
报告看着干净，结论是"四个策略完全一样"。**真正有数据的 EngineCore 那一行被
后写的 0 盖住了。**

改法有两处，都要：观测器改挂在 `CPUOffloadingManager.__init__` 上（谁有
manager 谁才写文件），分析脚本再按 pid 分组取最活跃的那个（`_last_active`），
并且把"窗口里出现了多个 pid"当成告警打出来。

### 坑三：tau 量错了尺度，adaptive 静悄悄退化成了 LRU

这一条和 Step 4.7 里离线 replay 踩过的是**同一个坑的第二次发作**（那一版把
tau 取成"持有时长"的中位数，这一版取成"相邻两次使用的间隔"中位数），换了个
入口又犯了一遍。

`adaptive` 的分数是 `freq × exp(-age/tau)`。原先 tau 按"同一个块相邻两次使用
的间隔"取中位数，实测 **tau = 63 ms**（n = 2,769）—— 而这个量根本不是复用间隔：
产生"使用"事件的有 `insert` / `mark_evictable` / `touch`，前两个之间只隔着一次
store 传输（几十毫秒），`touch` 又会在每次请求 lookup 后对整张 key 列表跑一遍，
并发请求共用热前缀时同一个块几十毫秒内被连着 touch 好几次。真正的时间尺度是
**秒级**。

后果是静默的：tau = 63 ms 时，任何歇了半秒以上的块 `exp(-age/tau)` 都下溢成 0，
逐出退化成"挑最旧的"—— **adaptive 就是 LRU**。当时四个策略的计数正好两两
相等（`adaptive ≡ lru`、`arc ≡ lfu`，逐位相同），差点被当成"策略之间没差别"。

改成**空闲时长**：块被放回池子（`mark_evictable`，ref_cnt 归零）到它下次被
需要之间的时间。逐出决策面对的正是"这个块现在是闲的，它还要闲多久"，这才是
tau 该量的东西。修完 tau 落在 **0.72 ~ 1.38 s**（随窗口移动），比 63 ms 大了
一个量级、也回到了秒级。

**这个数会在快照里带出来**（`tau_s` / `tau_n`），因为不把它打印出来，
"adaptive 和 LRU 打平"和"adaptive 根本就是 LRU"在指标上长得一模一样。

顺带一条从快照流里看得见的：tau 在**观测数够 32 个之前一直停在种子值 1.0 s**
（`_TAU_SEED_S`），实测到 n ≈ 261 才第一次更新。种子和真值同量级（1.0 vs 1.38），
所以这段预热没有把事情弄歪 —— 但这是个**巧合，不是保证**：换成复用尺度相差很远的
workload，第一阶段就是在用一个拍出来的常数做决策，快照里的 `tau_n` 是唯一能看出
这件事的地方。

### 坑四：`touch` 传的是整个请求的 key 列表

`CPUOffloadingManager.touch` 的调用者是 offloading connector 的 scheduler
（`kv_connector/v1/offloading/scheduler.py:691`，每次 lookup 之后；`:935`，store
路径）。它传的 `keys` 是**整个请求的 offload key 列表，不是命中的那一段**。

`LRUCachePolicy.touch` 里有 `if key in self.evictable_blocks` 这道守卫，不是可有
可无：少了它，从没进过池子的块也会被记进 `freq` / `last` / `_seen`。而 `_seen`
是**有上界的** LRU（`max_tracker_size`），被这些块灌满之后就会把真在池子里的
块的频次挤掉 —— 频次归零，LFU 退化成"随便挑一个"。这正是 Step 4.7 记过的那个
坑（"频次存在了池子记录里"），换个入口又出现了一次。

### 结果：4 策略 × 3 次重复

全部 12 次运行都通过了策略闸门（日志里那行 `实际逐出策略` 必须与请求的
策略对应，不对就中止）：

| 策略 | CPU 命中块（3 次）| 存入 | 换出 | 扫描落空 | TTFT P50 |
|---|---|---|---|---|---|
| lru | 150 / 150 / **156** | 848 | 556 | 144 | 51.3ms |
| arc | **170 / 170 / 170** | 828 | 536 | 144 | 51.2ms |
| lfu | **170 / 170 / 170** | 828 | 536 | 144 | 51.2ms |
| adaptive | 150 / 150 / 150 | 848 | 556 | 144 | 51.3ms |

三条读数：

1. **LFU / ARC 比 LRU / adaptive 多换回 13%**（170 vs 150 块），存入少 2.4%
   （命中多了就不必重存）。方向与离线结论一致，**幅度远小于离线** —— 原因就是
   上面那条：真机上被比较的只是第二层，而离线 replay 换的是第一层。
   组内极差 ≤ 6、组间差 20，这个分组站得住（下面那条注意 lru 第 3 次是 156）。
2. **`lookup_miss` 四个策略全是 144，等于请求数**：每个请求都至少缺一次
   （`_maximal_prefix_lookup` 扫到第一个 MISS 就停），平均只在 CPU 池里命中
   约 1.1 个块。CPU 池对这份 workload 的贡献本来就小。
3. **`store_evicted` 不是指标。** 恒有 `存入 − 换出 = 292` = CPU 池容量 ——
   逐出**多少次**是存多少块决定的，策略只能决定**逐出谁**。

### 稳定的是计数，不是决策 —— 所以"arc ≡ lfu"没被解释掉

上表里 `arc` 和 `lfu` 的数字逐位相同（828/536/170），两个很不一样的算法
（ARC 走 T1/T2 双队列 + 自适应 target，LFU 走频次）跑出同一个数。看着像是
它们做了同样的决策。

为了分辨"真的做了同样决策"和"计数太粗把不同决策压成同一个数"，把每次逐出的
key 折进一个顺序敏感的 crc32 指纹（`evict_fp`；用 crc32 而不是内置 `hash()`，
后者对 str 加了逐进程随机盐，跨进程不可比）。

结果**推翻了这个判据本身**：12 次运行给出 **12 个互不相同的指纹**，包括
**同一个策略的 3 次重复之间**。

| 策略 | 3 次重复的指纹 | 重复之间一致 |
|---|---|---|
| lru | 353101233 / 1255053408 / 1475834755 | 否 |
| arc | 784848652 / 1983733765 / 4246442895 | 否 |
| lfu | 2154221560 / 2303580321 / 2435119311 | 否 |
| adaptive | 8092843 / 2308389054 / 2890518117 | 否 |

所以"四个策略指纹不同"**不能**用来证明它们的决策不同 —— 同一个策略每次都不一样，
本来就会不同。这一版判据是无效的（第一版就是这么写的，被自己的重复实验打回来了）。

它证明的是另一件事，而且更有意思：**逐出序列逐次不可复现，而那六个计数逐次
几乎完全相同**（12 次里 11 次落在 150/170，1 次 156）。也就是说这些计数是这个
随机性很强的过程的一个**很粗但很稳的投影**。

于是 "arc 和 lfu 计数一样" 有两种可能 —— 两种算法的决策分布本来就会给出同一个
计数，或者纯属巧合 —— **本轮的数据分不出来**。要分清楚得看点级别的证据
（逐块比对谁在什么时刻被逐出，而不只是一个哈希），本轮没有做。这是一个明确的
开放项，不是已解决的事。

至少有一条是确定的：**一个看起来正常的聚合指标，不能用来证明策略按预期在工作。**
这条路上反复栽的都是这一类 —— 前面是 `store_evicted` 正常而 `load_blocks` 为 0，
这里是 adaptive 三次重复计数逐位相同、逐出序列三次互不相同。

---

## Step 5：连接 gpu-swap

最终产出不是 trace 文件，而是三个 insight：

1. 短命 block 不应换出（换出时它已释放，白花 PCIe 带宽）
2. 热 block 不应换出（频繁换回的代价 > 收益）
3. 请求类型差异是否需要分类策略（取决于 blocks-per-request 的方差）

这三个 insight 就是 gpu-swap 的 Adaptive Policy 的设计依据。
