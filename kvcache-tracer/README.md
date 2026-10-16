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

## 正式实验：3 类 workload × 并发 1/4/8 × 3 次重复

规格由实验协议给定，逐请求在区间内均匀采样：

| workload | prompt tokens | output tokens | 请求数 | 共享前导 |
|---|---|---|---|---|
| short-qa | 64 – 256 | 64 – 128 | 144 | 64 tok（4 block）|
| code-gen | 512 – 1024 | 256 – 512 | 36 | 512 tok（32 block）|
| long-summarize | 2048 – 4096 | 256 – 512 | 40 | 2048 tok（128 block）|

`--prefix-mode` 有两种，默认 `single`：

- **`single`** —— 上面这张表，类内所有请求共享一段指令前导。已跑完的 27 次
  运行用的是这个模式，输出与当时的记录逐位一致（回归验证过）。
- **`skewed`** —— K 个长度各异的模板 + Zipf 分布。给逐出策略对比用，
  见下面「倾斜复用与离线 replay」一节。

```bash
# 看请求集的 token 分布（不发请求）
python workload/run.py --workload code-gen --concurrency 1 --dry-run --out /tmp/x.json

# 一键跑完 3 × 3 × 3 = 27 次运行（约 70 分钟）
bash workload/run_experiment.sh

# 聚合报告（均值 ± 标准差 + 可直接引用的结论句）
python scripts/report.py
```

### 实测结果

Qwen2.5-7B-Instruct / RTX 4090 48G / `--gpu-memory-utilization 0.45`
（池 124,960 tokens = 7,810 blocks）/ block size 16 / max-model-len 8192 /
prefix cache 开 / seed 20261015。

**逻辑 block 获取中来自 prefix cache 的比例**（并发 1/4/8 逐位相同）：

| workload | 逻辑获取 | 缓存命中 | 占比 |
|---|---|---|---|
| short-qa | 2,478 | 572 | **23.1%** |
| code-gen | 2,647 | 1,120 | **42.3%** |
| long-summarize | 8,851 | 4,992 | **56.4%** |

命中数有一个精确到个位的解析式：`命中数 = (请求数−1) × 前导block数`
—— short-qa 4×143 = 572 ✓，code-gen 32×35 = 1,120 ✓，long-summarize 128×39 = 4,992 ✓。
说明**跨请求复用 100% 来自那段共享前导**，正文片段没有产生任何复用（见下方限制）。

**block 生命周期与访问次数**（随并发度单调上升）：

| 指标 | short-qa | code-gen | long-summarize |
|---|---|---|---|
| 生命周期 P50 (C=1 → C=8) | 1.354 → 1.437s | 5.055 → 5.386s | 5.575 → 6.735s |
| 生命周期 P90 | 1.879 → 1.933s | 7.182 → 7.616s | 7.210 → 8.142s |
| 存活 <1s 占比 | 22.8% → 21.9% | 5.3% → 5.0% | 1.8% → 1.5% |
| 单 block 访问 P50 | 86 | 288 | 321 |
| 单 block 访问 max | 14,123 | 13,508 | 14,812 |

⚠️ **本次实验没有触发任何逐出，这是必须一并报告的负面结果。**

| workload | 物理 block 数 | 池容量 | 池利用率 | 峰值同时存活 (C=1/4/8) |
|---|---|---|---|---|
| short-qa | 1,775 | 7,810 | 22.7% | 24 / 82 / 142 |
| code-gen | 1,496 | 7,810 | 19.2% | 88 / 301 / 567 |
| long-summarize | 3,824 | 7,810 | 49.0% | 272 / 958 / 1,760 |

long-summarize 的 token 足迹是池容量的 1.13 倍，事先预计会逐出，**实测没有**：
足迹是按「所有请求的 token 总量」算的，而 prefix cache 命中 56.4% 加上物理
block 释放后回收，实际被触达的**不同**物理 block 只有 3,824 个。空闲链从未耗尽。

这条限制了结论的适用范围：**上面所有生命周期/访问统计都只刻画了 block 的行为，
没有检验逐出**——而逐出是 swap 策略唯一的作用对象。要做策略对比，必须先把热池
预算压到工作集以下（见 GUIDE 的 replay 一节）。

### 编排上的两个坑

1. **池容量必须逐轮相同。** vLLM 按启动时**当前显存实况**切池子，上一个服务刚被
   kill、驱动还没回收就启动，池子会偏小（实测同一套参数得到 110,416 vs 稳定的
   124,960）。脚本启动前等显存回落到 47,000 MiB 以上，启动后比对池容量，
   不一致立即中止整场。

2. **`/reset_prefix_cache` 无论成败都返回 HTTP 200**
   （`entrypoints/serve/dev/cache/api_router.py` 直接 `Response(status_code=200)`，
   body 里没有 success 字段），底层只在「除 null block 外全部空闲」时才真清，
   失败只打一条 warning。所以脚本**不信状态码**，改数日志里
   `Successfully reset prefix cache` 的行数，数不到就中止。

隔离策略分两层：换 (workload, 并发度) 时重启服务（9 次，清 prefix cache +
空闲链 + allocator 碎片），同一组合的 3 次重复之间只调 `/reset_prefix_cache`
（省下 18 次重启约 15 分钟）。冒烟验证过这个判断：两次重复的结构量完全相同
（102 alloc / 20 cached / 19.6%），若缓存残留第 2 次会白捡命中率而明显偏高。

### 重复的意义：确认确定性，不是测噪声

三次重复的墙钟差异只有 0.005–0.040s（0.01%），结构量逐位相同，标准差近乎 0。
这是 `temperature=0` + `ignore_eos` + 独占 GPU 的必然后果。所以三次重复**验证了
确定性**（任何观察到的 workload 间差异都是真实差异，不是噪声），但**没有测到
环境噪声**（共享 GPU、邻居负载、热漂移）。这一条要写在结论旁边，不能把
"± 0.000" 当成"很稳定"来宣传。

## 倾斜复用 workload 与离线 replay

上面那批数据刻画了 block 的**行为**，但没有检验逐出 —— 而逐出是 swap 策略
唯一的作用对象（见上面那条负面结果）。要让 64/128/256 三档热池预算有意义，
需要一份可复用集合足够大的 workload。

### `skewed` 模式：K 个模板 + Zipf

```bash
REPS=1 bash workload/run_skewed.sh     # 先跑一轮看数据（约 25 分钟）
bash workload/run_skewed.sh            # 完整 27 次（约 65 分钟）
RESUME=1 bash workload/run_skewed.sh   # 中断后接着跑，跳过已完成且 JSON 可解析的组合
python scripts/replay.py verify --results-dir results-skew    # 先验模型
python scripts/replay.py run    --results-dir results-skew    # 再比策略

python scripts/measure_prefill.py --base-url http://127.0.0.1:8001   # 重测重算单价
```

产物在 `traces-skew/ results-skew/ logs-skew/`，与正式实验完全分开。

| workload | 模板数 K | 可复用 block | 独有 prompt block | 覆盖 50% 请求 | 覆盖 80% 请求 |
|---|---|---|---|---|---|
| short-qa | 64 | 259 | 672 | 59 | 152 |
| code-gen | 32 | 418 | 945 | 101 | 264 |
| long-summarize | 16 | 535 | 5780 | 104 | 255 |

最后一列是**选预算的依据**：三类的「覆盖 80%」都落在 150~265，横跨
64/128/256，每一档都有真实取舍。热集合远小于预算时任何策略都能把该留的
全留下，对比会退化成没有差别 —— 那时该改的是预算，不是策略。

模板设计有三条硬要求，缺一条 replay 就失真：

1. **长度各异** —— 否则"部分命中"（前 j 块在池、第 j+1 块被逐出）根本不会
   出现，而那正是策略差异最大的地方。
2. **长度与热度无关** —— 长度对齐到 block 后 shuffle 再赋 Zipf 权重，否则
   最短的模板天然对最多请求可用又最热，两个变量绑死。
3. **逐块互不相同** —— 每条模板带独立随机种子。最初写成「序号标记 + 同一段
   指令 + 随机填充」，而指令有一百多 token，块 1~7 在所有模板间逐字节相同，
   被 `check_templates_disjoint()` 查出 code-gen 有 69 个重复块。模板互相命中
   会让整个模型失效。

### `scripts/replay.py`

在同一份 trace 上模拟 B 个**空闲** block 的热池（running 请求持有的块被钉住，
不参与逐出 —— 和 vLLM 自己 `kv_offload/cpu/manager.py:205` 的口径一致），比较
LRU / LFU / Adaptive，并用**离线最优 Belady 做参照**标出还剩多少空间。

一次 block 获取的三种结局，代价差约 40 倍 —— 这个差距决定了整个迁移层的
价值，所以两个数都得站得住：

| 结局 | 代价 | 来源 |
|---|---|---|
| 命中（在池里 / 被同批请求共享） | 0 | — |
| 换回（曾被逐出，现在在 CPU） | 52.4 µs/块 | **实测** 17.5 GB/s 有效带宽 |
| 重算（从没见过） | 1,620 µs/块 | **实测** 101.27 µs/token × 16 token |

换回单价用 `scripts/measure_pcie.py` 实测（`measurements/pcie_link_ceiling.txt`），
**换回比重算便宜 31×**。测量里有个关键订正：一次 block 拷贝在 vLLM 里不是
「1 次 875 KB」，而是**28 个 32,768 B 的描述符**（每层一个 data_ref），
阈值比较的也是**每层**的 32,768 B —— 余量只有 14.3%，不是 14 倍。32 KB 只有
1.37 µs 的链路时间，**低于 DMA 启动开销**，于是每块要多付 28 × 0.44 µs 的固定
开销：`39.2 + 12.3 = 51.5 µs`，实测 52.6 µs。

不过 52.4 µs/块是**块自己的**代价，不是全部代价：链路不是免费的公路，
灌满它会把 decode 拖慢。`scripts/measure_interference.py` 实测（交错
baseline/load 配对，噪声底 0.020 ms）**红线 < 1.1 GB/s ≈ 19 个 block/一步**；
再往上加就"看得见"了，一路加到把链路跑满（22 GB/s = 374 block/step）时，
换出 +1.7%、**换回 +5.9%**。两个对照定住了这条结论的归属：**sham 对照**
（同样的循环与描述符构造、零字节 DMA）12 个负载 block 全部落在噪声底内，
所以这是 DMA 的效应而不是负载进程抢 CPU；而 `--enable-logging-iteration-details`
的逐迭代 join 显示拖慢几乎全落在 GPU 执行段（+1.8~2.5 ms），完整步只担了
其中 1/7（换出）到 1/3（换回），其余被 CPU 侧那段反向吃掉。细节见 GUIDE
Step 4.6 与 `measurements/pcie_interference_redline.txt`。

重算单价用 `scripts/measure_prefill.py` 实测（唯一 prompt、C=1、关 prefix
cache，拟合 `TTFT ≈ 9.79 ms + 101.27 µs × tokens`，反推 150 TFLOPS =
4090 bf16 峰值的 91%）。原始输出存在 `measurements/prefill_marginal_cost.txt`。

这个数**改过三次，前两次都是错的**，而且第二次错得很有代表性：它从 decode
吞吐反推、自称"取对重算最有利的估计所以结论更保守"，实际把重算算便宜了
2.6 倍。教训是"保守估计"这个说法会让人不再去查那个数本身。

不按 `block_id` 记账：物理 block 会被池子反复复用，同一个 id 在不同时刻是
完全不同的内容。改用**逻辑块**（`("t", 模板号, 偏移)` / `("j", 请求id, 偏移)`），
这正是 prefix cache 哈希的东西。

先跑 `verify` 再信数字。它逐个请求核对真实 prefix cache 命中数与模型预测：

```
short-qa c1 rep1  逻辑块 4788  预测命中请求 93  未命中 0  多出来的命中 0
✓ 逐请求核对通过：真实 prefix cache 命中与逻辑块模型完全一致
```

27 次运行（3 workload × 并发 1/4/8 × 3 重复）全部通过，零警告。这一步不是走过场
—— 它后来抓出了两个只在 c8 显形的建模错误（见 GUIDE 4.7），两个都会让结果
**看起来正常但是错的**。

### 结果：容量缺失率（B=256，并发 8）

| workload | LRU | adaptive | **LFU** | belady (离线) |
|---|---|---|---|---|
| short-qa | 25.2% | 16.8% | **12.7%** | 0.0% |
| code-gen | 19.4% | 17.3% | **17.3%** | 0.0% |
| long-summarize | 18.6% | 8.2% | **4.6%** | 0.0% |

三档预算 × 三档并发的完整表格见 [GUIDE.md](GUIDE.md) 的 Step 4.7，原始输出在
`measurements/replay_skew.md`。三条值得单独记住的：

1. **LFU 在全部 27 格里都赢。** Zipf 热度下按频次逐出就是对的。
2. **加大预算有一大段是白加的。** code-gen 从 64 加到 256 几乎不改善，因为池子
   被**永不复用的独有块**占满了（B=256 下池内平均 61.8% 是这类块）。加预算和
   把预算用对是两件事。
3. **并发越高缺失率越低。** 反直觉，但机制清楚：并发制造共享，同时运行的请求
   命中同一模板时根本不经过池子。所以跨并发度比缺失率是苹果比橘子。

⚠️ **第三条结论的适用范围**：这里的 Zipf 热度是**平稳**的，正是 LFU 的主场。
真实负载里热门模板会轮换，那时 LFU 的陈旧频次会拖后腿、adaptive 才有机会反超。
"LFU 最好"只在这份 workload 上成立。

belady 全 0.0% 说明**可复用集合本身不大** —— 预算 128 时只要选得对就没有容量
缺失。这条界定了迁移层的价值边界：再往上要靠把工作集搬出显存，而不是靠继续
调逐出启发式。

### 搬到真机上：结论方向对上了，幅度没有

上面那张表是**离线模拟**的。把它放到 vLLM 自己的 CPU offloading 层上真跑
（`workload/run_policy_compare.sh`，4 策略 × 3 次，详见
[GUIDE.md](GUIDE.md) Step 4.8），得到的不是同一个数：

| 策略 | CPU 池换回块数（3 次）| 存入 | 换出 | TTFT P50 |
|---|---|---|---|---|
| lru | 150 / 150 / 156 | 848 | 556 | 51.3ms |
| arc | **170 / 170 / 170** | 828 | 536 | 51.2ms |
| lfu | **170 / 170 / 170** | 828 | 536 | 51.2ms |
| adaptive | 150 / 150 / 150 | 848 | 556 | 51.3ms |

**方向一致（频次类 > 纯最近性），幅度差一个量级**（13% vs 离线的 2~4 倍）。
原因是两者测的**不是同一层**：离线 replay 换的是热池（第一层）的策略，真机上
GPU prefix cache 由 vLLM 自己的 LRU 管着、四个策略共用，被比较的只是 CPU 池
（第二层），而它只占缓存服务的约 28~30%。**别把离线倍数直接搬到真机上。**

三个坑，都会产出"干净但是错的"报告，两个来自聚合指标太粗：

1. **GPU 池装得下工作集时，CPU 池只写不读。** 默认口径（GPU 池 124,960
   token > 足迹 37,382）下，换出计数 556 一切正常、存进 778 MB，**读回 0 字节**，
   四个策略命中率全是 0。判据是 `load_blocks`，不是 `store_evicted`。
2. **观测器被 frontend 和 EngineCore 两个进程交错写**，frontend 那份恒为 0 且
   写在后面 —— 分析脚本取到它，四个策略差值全是 0。
3. **tau 量错了尺度**：取"相邻两次使用的间隔"中位数得 63 ms，而这个量里混的是
   store 传输和并发 touch 的短间隔，真正的复用尺度是秒级。tau 小 50 倍 →
   `exp(-age/tau)` 下溢 → **adaptive 就是 LRU**（当时 `adaptive ≡ lru` 逐位相同）。
   这和 GUIDE 4.7 里离线 replay 踩的是同一个坑的第二次发作。

还有一条差点写错的，值得单独记：`arc` 和 `lfu` 的计数逐位相同（170/828/536），
而逐出**序列的指纹 12 次运行给出 12 个值 —— 包括同一个策略的 3 次重复之间**。
也就是说逐出序列逐次不可复现，计数却几乎逐次相同：**这些计数是一个随机过程的
很粗的稳定投影**。所以"arc 和 lfu 计数一样"到底是巧合还是真的一样，**本轮数据
分不出来**（要分清得看点级别的逐块比对）。教训是：**一个看起来正常的聚合指标，
不能用来证明策略按预期在工作。**

## 学习目标

- vLLM PagedAttention 的 block 管理机制
- 无侵入式 instrumentation（monkey-patch）
- KV Cache 的 write-once-read-many 访问模式
- 为什么观测数据是策略设计的前提
