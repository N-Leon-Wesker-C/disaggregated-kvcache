# kvcache-tracer

**KV Cache 块级生命周期追踪器**

通过 monkey-patch vLLM (v0.26, v1 engine) 的 `SingleTypeKVCacheManager`，记录每个 KV Cache block 的分配、访问、释放事件。产出的 trace 数据直接驱动 [gpu-swap](../gpu-swap/) 的换出策略设计。

## Hook 点（按 vLLM 0.25.1 实际签名校对）

| 类 | 方法 | 事件 | 在生命周期中的位置 |
|----|------|------|-------------------|
| `SingleTypeKVCacheManager` | `allocate_new_blocks` | alloc | 出生 |
| `SingleTypeKVCacheManager` | `cache_blocks` | access | block 写满 |
| `SingleTypeKVCacheManager` | `pop_blocks_for_free` | free | 死亡 |
| `KVCacheManager` | `allocate_slots` | reads（计数） | 每个 step 被读 |

## 为什么必须走 plugin 注入

vLLM v1 的 **EngineCore 跑在一个 spawn 出来的独立进程里**，而
KVCacheManager 只存在于那个进程。在启动脚本里 monkey-patch 然后指望
子进程继承，是行不通的 —— spawn 是全新解释器，patch 一律不存在。

（容易踩的坑：`get_mp_context()` 在 frontend 刚启动时返回 `fork`，
但 vLLM 真正创建 EngineCore 时已经因为 CUDA 初始化被强制改成 `spawn`。
早测会得到假阳性。）

vLLM 自带的 `vllm.general_plugins` 入口点会加载进所有进程
（frontend / EngineCore / worker），是官方为此准备的机制。

## 快速开始

```bash
pip install vllm

# 1. 安装追踪 plugin（一次性）
cd plugin && pip install -e . && cd ..

# 2. 正常启动 vLLM —— hook 自动生效，不需要改启动方式
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000

# 3. 另开终端发请求
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"Hello","max_tokens":64}'

# 4. 分析（事件边产生边落盘，不用等 server 退出）
python scripts/analyze.py kvcache_trace.*.jsonl
```

## 产出的数据

事件写在 `kvcache_trace.<pid>.jsonl`（一个 vLLM 进程一个文件，每行一条）：

```json
{"event":"alloc","timestamp":48.19,"block_id":1,"request_id":"cmpl-...","source":"new","pid":17915}
{"event":"access","timestamp":48.22,"block_id":1,"request_id":"cmpl-...","pid":17915}
{"event":"free","timestamp":48.35,"block_id":1,"request_id":"cmpl-...","reads":104,"pid":17915}
```

`alloc` 的 `source` 区分两种来源，**两个都必须记**：`new` 是新分配，
`cache` 是 prefix cache 命中后直接挂到请求上的。只记前者会让 free 数
远大于 alloc 数（实测 4 请求共享 3000 token 前缀时差 562 条）。

`free` 事件里的 `reads` 是 **「这个请求持有期间读了这个 block 多少次」** ——
由 `allocate_slots` 每个 decode step 累加、释放时一次写出，是 hot/cold skew
的唯一数据来源。（`access` 事件的计数恒为 1，别拿它算热度。）

它按 `(block_id, request_id)` 而不是只按 `block_id` 计数。一个 block 会被多个
请求持有（prefix cache 命中时共享、释放后又被下一个请求复用），只按 block_id
计数的话，先释放的那个请求会把所有持有者的读数一并取走，后面的取到 0 ——
实测 8 并发下 shared-prefix 类别出现 128 条 `reads=0`，而先释放的拿到被夸大的
值。要算某个 block 的总热度，把各次释放的 `reads` 相加即可。

调试：`KVTRACE_DEBUG=1` 会在每次 hook 触发时打印一行（带 pid），
用来确认 hook 到底有没有被调用、被哪个进程调用。
`KVTRACE_OUTPUT=<前缀>` 可以改输出路径，`KVTRACE_FLUSH_EVERY=<n>` 调刷盘批大小。

## 分析产出

- block 生命周期分布（短命 vs 长命）
- block 访问频率分布（hot/cold skew）
- 每请求 block 数与请求生命周期
- 针对 swap 策略设计的 insights（见 `analyze.py` 输出）

## 正式 workload 与并发度扫描

`workload/` 下的负载是确定性生成的，用来测**并发度 1 / 4 / 8 下的 block 行为差异**：

```bash
# 看请求集构成和足迹占比（不发请求）
python workload/workload.py --dry-run

# 一键跑完三个并发度：每个并发度重启服务、跑负载、分析
bash workload/run_sweep.sh

# 横向对比三次运行
python scripts/compare_runs.py c1=traces/c1.*.jsonl \
                                c4=traces/c4.*.jsonl \
                                c8=traces/c8.*.jsonl
```

四类请求（short-chat / long-prefill / long-decode / shared-prefix）覆盖的是
对 swap 策略要求**相反**的行为，详见 [workload/workload.py](workload/workload.py)
的模块文档。两个容易踩的点：

1. **池子必须能被压满。** 本机 4090-48G 跑 7B，`--gpu-memory-utilization 0.85`
   时池子有 465,424 tokens —— 任何合理负载都塞得下，永远不会逐出，
   于是 trace 里全是"从生到死没人打扰"的 block，关于 swap 的结论会是空的。
   扫描固定用 `0.45`（池子 110,416 tokens），负载足迹约 15 万，是池子的 1.37 倍。
   **这个值三次运行必须完全一致**，否则并发度就不是唯一变量了。

2. **每个并发度都重启服务。** 重启能同时清掉 prefix cache、空闲链、
   allocator 碎片状态；`/reset_prefix_cache` 只清第一样。缺了另外两样，
   三次运行的起点就不一样了。

`compare_runs.py` 里最该看的指标是 **物理 block 数 vs 池容量**：
明显小于池容量说明空闲链从未耗尽，逐出根本没发生过，那这次运行的
生命周期统计再漂亮也和 swap 无关。

## 学习目标

- vLLM PagedAttention 的 block 管理机制
- 无侵入式 instrumentation（monkey-patch）
- KV Cache 的 write-once-read-many 访问模式
- 为什么观测数据是策略设计的前提
