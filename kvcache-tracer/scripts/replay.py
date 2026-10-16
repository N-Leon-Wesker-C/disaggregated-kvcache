"""
replay.py — 在同一份 trace 上离线重放，比较不同热池逐出策略

要回答的问题：GPU 上只留 B 个空闲 block 的话，LRU / LFU / Adaptive 谁更好，
好多少，以及**还剩多少空间**（用离线最优做参照）。

池的语义（与实验设计一致）
============================================================

    GPU 热池 = B 个**空闲** block 的预算。
    running 请求持有的 block 被钉住，不参与逐出 —— 这一点由 vLLM 自己的
    实现保证（vllm/v1/kv_offload/cpu/manager.py:205 只回收 idle block）。

    块从池里被逐出 = 换到 CPU（PCIe 上行，异步、不阻塞请求）。
    请求要用一个已被逐出的块 = 换回 GPU（PCIe 下行，在关键路径上）。
    从来没见过的块 = 必须重算（prefill）。

所以一次 block 获取有三种结局，代价差**三十倍**：

    在池里 / 已被同批请求共享   0            （已经在 GPU 上）
    被逐出过 → 换回             52.4 µs/块   （实测，见 PCIE_EFF_GBPS）
    从没见过 → 重算             1,620 µs/块  （实测 101.27 µs/token × 16 token）

最后一行是这份模拟里唯一**决定结论方向**的数，而且换过三版才测准 ——
详见 RECOMPUTE_US_PER_TOKEN 处的记录。早先低估到 616 µs/块（38× → 16×）。

换出**不阻塞**请求：迁移层是异步的（见 GUIDE 4.6 的 CUDA stream/event 结构），
没有任何一步等它完成，所以这份模拟按 0 计它的单块延迟是对的。带宽预算另算 ——
见输出里的 "换出流量"。

但**"不阻塞"不等于"不要钱"**。GUIDE 4.6 的干扰曲线实测：换出超过约
1.1 GB/s（≈1,200 block/s ≈ 一步 19 个 block）就开始把 decode 每一步拖慢，
跑满链路时 +1.7%。所以这份模拟报出的换出速率要拿那条红线过一遍 ——
红线以内才是真的免费，超出部分只是把代价从「这块 block 的延迟」挪到了
「decode 吞吐」。**本文件里的 0 是延迟口径的 0，不是总代价的 0。**

逻辑块怎么来
============================================================

**不按 block_id 记账。** 物理 block_id 会被池子反复复用，同一个 id 在不同
时刻是完全不同的内容；按它记账会把两个不同模板的块当成同一个。

按**逻辑块**记账，这正是 prefix cache 哈希的东西：

    模板块   ("t", 模板号, 块内偏移)     ← 跨请求可复用
    独有块   ("j", 请求id, 块内偏移)     ← 只服务一个请求

映射关系不需要重新分词：元数据里的 schedule 逐请求记了 template 和
template_blocks，trace 逐块给了获取/释放时刻。请求的块按获取时刻排序取
偏移即可（prompt 块一定先于解码块获取）。

模型验证
============================================================

`verify` 子命令会拿真实 trace 的 prefix cache 命中数对模型做逐个请求的核对。
skewed workload 上实测 93/93 请求、584/584 块完全一致，零个「模型没预测到
却发生了」的命中。

用法:
    # 先验证命中模型（强烈建议，几秒钟）
    python scripts/replay.py verify --results-dir results-skew

    # 三档预算 × 四种策略
    python scripts/replay.py run --results-dir results-skew \
        --budget 64,128,256 --policy lru,lfu,adaptive,belady
"""

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
from bisect import bisect_left
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP_RE = re.compile(r"-r(\d+)-")

#: Qwen2.5-7B-Instruct: 28 层 × 4 KV head × 128 head_dim × 2 (K,V) × 2 B (bf16)
BYTES_PER_TOKEN = 28 * 4 * 128 * 2 * 2

#: 有效带宽（GB/s），用来把「换回一个块」定价成 µs。**已实测**，
#: 但它是**等效值**，不是一个物理带宽 —— 见下面的分解。
#:
#: 测量：scripts/measure_pcie.py → measurements/pcie_link_ceiling.txt。
#: 用 vLLM 自己的 `cuda_mem_ops.copy_blocks`（同一个 cuMemcpyBatchAsync
#: v1 符号、同样的 layer-major 描述符），生产形状 = 1 block = 28 × 32,768 B。
#:
#:   23.4 GB/s   边际带宽（从粒度隔离那组拟合出来的斜率）
#:   0.44 µs     每个描述符的固定开销
#:   → 917,504 / 23.4e9 + 28 × 0.44µs = 39.2 + 12.3 = 51.5 µs/块
#:     实测 52.6 µs/块（批量 ≥64 时，此时提交开销可忽略）
#:
#: 所以 24.0 这个旧值**错得不大也不小**：边际带宽 23.4 几乎就是 24，
#: 漏掉的是那 28 个描述符的固定开销（+33%）。
#:
#: 为什么写 17.5 而不是 23.4：这个常数在代码里是乘进「每块」单价用的，
#: 拿 23.4 会漏掉描述符开销。17.5 = 917,504 B / 52.4 µs，是把开销折进去的
#: 等效带宽。
#:
#: 旧注释说「它的不确定性影响不了结论」，这句仍然成立，但**理由当时是错的**：
#: 它把一个未测的常数说成测不测都无所谓。实测下来换回是 52 µs/块对重算的
#: 1,620 µs/块，便宜 31×（原以为 42×）—— 量级没变，但数字变了。
#:
#: 已知的适用范围：批量很小时（1 块/次）实测只有约 40 µs/块，比大批量**更便宜**，
#: 机制未查清（已排除 block id 轮换造成的顺序访问，见 measure_pcie.py 的附注）。
#: replay 按块独立计价，这里取大批量值 —— 它更保守，且测量最干净。
PCIE_EFF_GBPS = 17.5

#: 重算一个 token 的 KV 的边际成本（µs）。**实测值**，不是反推的。
#:
#: 来源见 measurements/prefill_marginal_cost.txt（scripts/measure_prefill.py
#: 产出）：唯一 prompt、C=1、max_tokens=1，测 512/1024/2048/3072/4096 五档
#: 的 TTFT 后做线性拟合，得
#:
#:     TTFT ≈ 9.79 ms + 101.27 µs × prompt_tokens
#:
#: 拟合极干净（固定开销只有 9.79ms），反推 150 TFLOPS，是 4090 bf16 峰值
#: 165 TFLOPS 的 91% —— 说明确实测到了 compute-bound 的纯 prefill，
#: 没有混进排队和调度。
#:
#: 这里换过三版，留个记录免得再绕回去：
#:   1. 最早拿 TTFT **中位数**配 prompt **中位数**，拟合出 25,974 tok/s，
#:      反推 396 TFLOPS —— 是峰值的 2.4 倍。中位数不是同一个请求，整个口径
#:      不成立。
#:   2. 换成从 decode 吞吐反推的 38.5 µs/token，当成"保守下界"。但 prefill
#:      是 compute-bound 的批量矩阵乘，decode 是访存受限，两者每 token 的
#:      成本本来就不是一回事，拿 decode 给 prefill 定价没有依据。
#:   3. 实测 101.27 µs/token。比第 2 版高 2.6 倍 —— 也就是说之前**低估**了
#:      重算的代价，而这不是"更保守"，是直接把结论算反了一个量级。
RECOMPUTE_US_PER_TOKEN = 101.27


def swap_in_us(block_size):
    """换回一个 block 的 PCIe 时间。"""
    return block_size * BYTES_PER_TOKEN / (PCIE_EFF_GBPS * 1e9) * 1e6


def recompute_us(block_size):
    return block_size * RECOMPUTE_US_PER_TOKEN


# ================================================================
# 从 trace + 元数据重建逻辑访问流
# ================================================================

def rep_of(request_id):
    m = REP_RE.search(request_id or "")
    return int(m.group(1)) if m else None


def trace_glob_of(meta):
    """把元数据里的 trace_glob 解析成绝对路径。

    相对路径锚到 ROOT —— 不锚的话换个工作目录就静默 glob 不到，而静默是
    最糟的失败方式：重放照常跑完，只是所有指标都变成 0。
    """
    pattern = meta.get("trace_glob") or ""
    if pattern and not os.path.isabs(pattern):
        anchored = os.path.join(ROOT, pattern)
        if glob.glob(anchored):
            return anchored
    return pattern


def build_stream(meta, events, block_size):
    """→ accesses: [(t, kind, key, rid)]，kind 0=释放 1=获取，按 (t, kind) 升序。

    **同一时刻先释放后获取**，这不是随手定的：vLLM 一个 step 里就是先
    pop_blocks_for_free 回收跑完的请求，再 allocate_slots 给下一批分配。
    反过来排的话，同一个 step 内刚释放的块会被算成"来不及被复用"，
    凭空多出一批换回/重算，策略之间的差别会被这个假象淹没。
    """
    # 按请求分组
    per_req = defaultdict(lambda: {"alloc": [], "free": {}})
    for e in events:
        rid = e.get("request_id")
        if e["event"] == "alloc":
            per_req[rid]["alloc"].append((e["timestamp"], e["block_id"]))
        elif e["event"] == "free":
            per_req[rid]["free"][e["block_id"]] = e["timestamp"]

    sched = {s["id"]: s for s in meta.get("schedule", [])}

    # vLLM 把 request_id 包了一层（"short-qa-058" → "cmpl-short-qa-058-r1-0-…"）。
    # 取最长匹配，防止 short-qa-05 撞上 short-qa-058。
    def lookup(rid):
        best = None
        for sid, sv in sched.items():
            if sid in (rid or "") and (best is None or len(sid) > len(best)):
                best = sid
        return best

    rid_of = {rid: lookup(rid) for rid in per_req}
    rid_of = {k: v for k, v in rid_of.items() if v is not None}

    # 每个请求的 prompt 块集合。
    #
    # 老口径是"最早那个时刻一次拿到的块"，依据是当时实测 132/132 个请求都
    # 满足"prompt 一次拿完、之后每个 decode step 一块"。**这个前提在 c8 垮了**：
    # code-gen 每轮有 1 个请求的 prompt 被调度器切成多块 prefill，第二批及以后
    # 的块落在更晚的时刻，于是被漏掉 —— 实测少 8 个模板块。
    #
    # 所以改用元数据里的 prompt_blocks（客户端按同一套 tokenizer 和 block size
    # 算出来的）来划这条线：按获取时刻排序后，前 prompt_blocks 个就是 prompt。
    # 反正 prompt 块一定先于解码块，这个边界是稳定的。
    prompt_batch, prompt_t0 = {}, {}
    for rid, d in per_req.items():
        if not d["alloc"]:
            continue
        allocs = sorted(d["alloc"], key=lambda x: (x[0], x[1]))
        prompt_t0[rid] = allocs[0][0]
        n_prompt = sched[rid_of[rid]]["prompt_blocks"] if rid in rid_of else None
        if n_prompt and 0 < n_prompt <= len(allocs):
            prompt_batch[rid] = {b for _t, b in allocs[:n_prompt]}
        else:
            # 元数据缺这个字段时退回老口径（会低估 prompt 集，但不至于崩）
            t0 = allocs[0][0]
            prompt_batch[rid] = {b for t, b in allocs if t <= t0 + 1e-9}

    # 模板的物理块 = 所有使用该模板的请求的 prompt 批的**交集**。
    #
    # 这里绝不能改成"prompt 批按时间排序后的前 tpl_blocks 个"。prompt 块全部
    # 在同一时刻拿到，排序只能按 block_id 破平局 —— 而模板块来自 prefix cache
    # 的哈希表、正文块来自空闲链，两者的 id 毫无顺序关系。于是同一个模板块在
    # 不同请求里会落进不同的偏移，key 对不上，跨请求复用**静默消失**，最后
    # 看到的是"所有策略都一样差"，而那是模型坏了，不是策略没差别。
    # 交集没有这个问题：正文块各自独有，天然被排除掉，剩下的正好是模板块。
    by_tpl = defaultdict(list)
    for rid, sid in rid_of.items():
        t = sched[sid].get("template", -1)
        if t >= 0 and rid in prompt_batch:
            by_tpl[t].append(rid)

    tpl_phys, tpl_index, tpl_note = {}, {}, {}
    for t, rids in by_tpl.items():
        sets = [prompt_batch[r] for r in rids]
        shared = set.intersection(*sets) if len(sets) > 1 else set()
        want = sched[rid_of[rids[0]]]["template_blocks"]
        if len(sets) > 1:
            tpl_note[t] = (len(shared), want)
            tpl_phys[t] = sorted(shared)
        else:
            # 只有一次使用：没有别的请求可交叉印证，只能按 trace 里 prompt 批
            # 的获取顺序取前 want 个。反正它不会被任何别的请求引用，取错也
            # 不影响统计。
            order = [b for _t, b in sorted(per_req[rids[0]]["alloc"])]
            tpl_phys[t] = [b for b in order if b in sets[0]][:want]
        tpl_index[t] = {b: i for i, b in enumerate(tpl_phys[t])}

    accesses = []
    mismatches = []
    for rid, d in per_req.items():
        if rid not in rid_of:
            continue
        s = sched[rid_of[rid]]
        tpl = s.get("template", -1)
        tpl_blocks = s.get("template_blocks", 0)
        idx = tpl_index.get(tpl, {})
        want = tpl_blocks if tpl >= 0 else 0
        pset = prompt_batch.get(rid, set())

        blocks = sorted(d["alloc"])                # prompt 块一定先于解码块
        n_tpl = 0
        for off, (t_alloc, blk) in enumerate(blocks):
            # **只有 prompt 里的块才可能是模板块。**
            #
            # 这里踩过一个坑，而且只在 c8 显形：物理块会被回收再分配，一个模板
            # 块被释放之后，它的 id 可能落到某个请求的解码块上。如果这里只看
            # `blk in idx`，那个解码块就会被当成模板块，凭空多出一次"共享命中"。
            # c1/c4 的池子宽裕、模板块从不释放，所以查不出来。
            #
            # 判据必须用 pset 而不是"t_alloc == t0" —— prompt 可能被切成两批
            # prefill，第二批的块晚于 t0，按时刻判会把它错当成解码块。
            if blk in pset and blk in idx:
                key = ("t", tpl, idx[blk])
                n_tpl += 1
            else:
                key = ("j", s["id"], off)
            accesses.append((t_alloc, 1, key, s["id"]))
            t_free = d["free"].get(blk)
            if t_free is not None:
                accesses.append((t_free, 0, key, s["id"]))
        if n_tpl != want:
            # 记方向和差值：少了说明该请求没拿全模板，多了说明有块被错误地
            # 归进了模板 —— 两种情况的成因和严重性完全不同，只报一个计数
            # 等于把诊断信息扔掉。
            mismatches.append({"request": s["id"], "template": tpl,
                               "got": n_tpl, "want": want,
                               "delta": n_tpl - want})

    accesses.sort(key=lambda x: (x[0], x[1]))
    if mismatches:
        print(f"  警告: {len(mismatches)} 个请求识别出的模板块数与元数据不符；"
              f"差值 {[x['delta'] for x in mismatches]}"
              f"（负=没拿全模板，正=有块被误判为模板）", file=sys.stderr)
    build_stream.last_notes = {"tpl_note": tpl_note,
                               "tpl_mismatch": len(mismatches),
                               "mismatches": mismatches}
    return accesses


# ================================================================
# 逐出策略
# ================================================================

class Evictor:
    """从空闲池里挑一个块换出去。

    池里每条记录是 (key, last_access_t, freq, next_use_t)。
    next_use_t 只有 belady 用得上，其余策略拿到的是 inf。
    """

    def __init__(self, name, tau=None):
        self.name = name
        self.tau = tau

    def pick(self, entries, now):
        if self.name == "lru":
            # 最久没被访问的先走
            return min(entries, key=lambda e: (e[1], e[0]))
        if self.name == "lfu":
            # 访问次数最少的先走；次数相同则最久没用的先走
            return min(entries, key=lambda e: (e[2], e[1], e[0]))
        if self.name == "adaptive":
            # 频次 × 时间衰减。score 最小的先走。
            #
            # 这个形式把 LRU 和 LFU 当成两个极端包含进来了：
            #   tau → ∞ 时衰减项恒为 1，退化成纯 LFU
            #   tau → 0 时老块的 score 迅速归零，退化成近似 LRU
            # 所以"Adaptive 比两者都好/都差"是有信息量的，而不是在比一个
            # 随手拍出来的启发式。
            t = self.tau if self.tau else 1e18
            return min(entries,
                       key=lambda e: (e[2] * math.exp(-(now - e[1]) / t), e[1], e[0]))
        if self.name == "belady":
            # 离线最优的参照。**不是可达的**：它要知道未来。
            # 作用只是给出"还有多少空间"—— 在线策略再怎么调也到不了这条线。
            return max(entries, key=lambda e: (e[3], e[1], e[0]))
        raise ValueError(f"未知策略: {self.name}")


def next_use_map(accesses):
    """每个逻辑块的获取时刻序列（已按时间有序），供 belady 二分查询。

    早先用一个自增游标"边走边推进"，但游标只在当前事件的那个 key 上前进，
    别的 key 的游标是陈旧的 —— 池里的候选块很可能拿着一个已经过期的
    "下一次使用"，把 belady 这条参照线算错。二分查找没有这个状态，
    永远以当前时刻为基准。
    """
    seq = defaultdict(list)
    for t, kind, key, _rid in accesses:
        if kind == 1:
            seq[key].append(t)
    return seq


# ================================================================
# 重放
# ================================================================

def auto_tau(accesses):
    """adaptive 衰减常数的默认值：**同一个块相邻两次获取的间隔**中位数。

    这才是"热"的时间尺度：一个块隔多久会被再用一次。衰减比它快，历史频次
    就还没等到下次复用就被清零了。

    这里踩过一版：最初取的是**持有**时长的中位数。短上下文 workload 里块只
    活 1.4s，而池子的换手比这快得多，于是衰减项把一切历史频次都抹成 0，
    adaptive 逐档退化成 LRU —— 报表上看不出任何异常，只是"adaptive 和 LRU
    居然一样"。持有 1.4s 的块完全可能在 10s 之后才被复用，两者不是一回事。
    """
    last_seen, gaps = {}, []
    for t, kind, key, _rid in accesses:
        if kind != 1:
            continue
        if key in last_seen:
            gaps.append(t - last_seen[key])
        last_seen[key] = t
    return statistics.median(gaps) if gaps else 1.0


def replay(accesses, block_size, budget, policy, tau=None):
    """在一条逻辑访问流上模拟 B 个空闲块的热池。"""
    if policy == "adaptive" and not tau:
        tau = auto_tau(accesses)
    evictor = Evictor(policy, tau)

    # 三份状态分开存。早先把 freq 塞在 pool 的值里，块一被钉住（从 pool 里
    # pop 出来）频次就丢了，释放回池时又从 0 开始 —— LFU 会静悄悄地退化成
    # "随便挑一个"，而且在指标上还看不出异常。
    pool = set()                  # 空闲且仍在 GPU 上的键
    freq = defaultdict(int)       # 累计访问次数，贯穿整个模拟
    last = {}                     # 最近一次被访问的时刻
    cpu = set()                   # 已被换出、现在在 CPU 上的键
    holders = defaultdict(set)
    nxt = next_use_map(accesses) if policy == "belady" else {}
    inf = float("inf")

    def next_use(k, t):
        """k 在 t 之后（含 t）最早被再次获取的时刻；不会再有就是 inf。

        用 bisect_left 而不是「> t」：同一时刻先处理释放后处理获取，
        所以 t 这一刻的获取仍然在"未来"。"""
        seq = nxt.get(k)
        if not seq:
            return inf
        i = bisect_left(seq, t)
        return seq[i] if i < len(seq) else inf

    m = {"n_acq": 0, "hit_pool": 0, "shared": 0, "swap_in": 0,
         "compulsory": 0, "evictions": 0, "acq_tpl": 0, "hit_tpl": 0}
    lat = []
    #: 分请求累计的换回延迟。**这是唯一值得看分位数的量** ——
    #: 单块延迟只有三个取值（命中 0 / 换回 52.4µs / 重算 1620µs），
    #: 对它取 P50 只会落在 1620µs 这个常数上，每个策略每一档都一模一样，
    #: 看上去像一列有数字的指标，实际上不携带任何信息。按请求求和之后，
    #: 分布才随策略真正散开。
    req_swap = defaultdict(float)
    util_area = 0.0
    last_t = accesses[0][0] if accesses else 0.0

    for t, kind, key, rid in accesses:
        # 池占用按时间加权。只在事件时刻采样会低估满池的时长。
        util_area += len(pool) * (t - last_t)
        last_t = t

        if kind == 0:                                   # ---- 释放 ----
            holders[key].discard(rid)
            if holders[key]:
                continue                                # 还有别的请求在持有
            # 注意这里**不**给 budget<=0 开小灶：池子为 0 时 `len(pool) >= 0`
            # 恒真，会自然走到下面"淘汰的就是刚释放的这块"那一支，把它送去
            # CPU 并记一次换出。早先图省事直接 continue，块就凭空消失了 ——
            # 于是预算 0 的退化用例里所有获取都变成"重算"，换回数恒为 0，
            # 一个本该最贵的档位反而算出了最便宜的结果。
            if len(pool) >= budget:
                # 待插入的 key 也是候选 —— 池满时"干脆不留下它"是合法的。
                cands = list(pool) + [key]
                entries = [(k, last.get(k, t), freq[k], next_use(k, t))
                           for k in cands]
                victim = evictor.pick(entries, t)
                if victim[0] != key:
                    pool.discard(victim[0])
                    cpu.add(victim[0])
                    m["evictions"] += 1
                else:
                    # 淘汰的就是刚释放的这块：直接送 CPU，不占池位
                    cpu.add(key)
                    m["evictions"] += 1
                    continue
            pool.add(key)
            last[key] = t
            continue

        # ---- 获取 ----
        m["n_acq"] += 1
        freq[key] += 1
        is_tpl = key[0] == "t"
        if is_tpl:
            m["acq_tpl"] += 1
        if holders[key]:
            # 同批请求共享：块已经在 GPU 上，而且持有者还在用
            m["shared"] += 1
            lat.append(0.0)
        elif key in pool:
            m["hit_pool"] += 1
            if is_tpl:
                m["hit_tpl"] += 1
            lat.append(0.0)
            pool.discard(key)
        elif key in cpu:
            m["swap_in"] += 1
            lat.append(swap_in_us(block_size))
            req_swap[rid] += swap_in_us(block_size)
            cpu.discard(key)
        else:
            m["compulsory"] += 1
            lat.append(recompute_us(block_size))
        last[key] = t
        holders[key].add(rid)

    span = (accesses[-1][0] - accesses[0][0]) if accesses else 0.0
    span = max(span, 1e-9)
    n_evictable = m["n_acq"] - m["compulsory"]

    req_ids = {rid for _t, _k, _key, rid in accesses}

    def req_pct(q):
        """分请求的换回延迟分位数，含换回为 0 的请求（否则高分位会被放大）。"""
        if not req_ids:
            return 0.0
        v = sorted(req_swap.get(r, 0.0) for r in req_ids)
        return v[min(len(v) - 1, int(len(v) * q))]

    return {
        "policy": policy, "budget": budget, "tau": tau,
        "n_acq": m["n_acq"],
        "hit_pool": m["hit_pool"], "shared": m["shared"],
        "swap_in": m["swap_in"], "compulsory": m["compulsory"],
        "evictions": m["evictions"],
        "req_swap_p50_us": req_pct(.5),
        "req_swap_p99_us": req_pct(.99),
        "swap_in_total_ms": sum(req_swap.values()) / 1000.0,
        "swap_out_mb": m["evictions"] * block_size * BYTES_PER_TOKEN / 1e6,
        # 容量缺失率：分母剔掉 compulsory。全算进去的话，"工作集本身就比池子
        # 大"的那部分会被记成策略的锅，而那跟策略无关。
        "capacity_miss_rate": (m["swap_in"] / n_evictable) if n_evictable else 0.0,
        "pool_hit_rate": ((m["hit_pool"] + m["shared"]) / m["n_acq"])
                         if m["n_acq"] else 0.0,
        "pool_util": util_area / span / budget if budget else 0.0,
        # 可复用集合的保留率：模板块的获取里有多少直接命中热池。
        # 这是**唯一直接刻画策略有没有干对活**的指标 —— 总命中率会被大量
        # 独有块稀释，一个把独有块留得很多的策略也能刷出好看的总命中率。
        "tpl_hit_rate": (m["hit_tpl"] / m["acq_tpl"]) if m["acq_tpl"] else 0.0,
        # 这里**不**导出单块延迟的分位数。它的取值只有三个（命中 0 /
        # 换回 52.4µs / 重算 1620µs），P50 必然落在 compulsory 重算那一档上，
        # 每个策略每一档都给出同一个数。留着它迟早被当成一列有信息量的指标
        # 拿去引用 —— 上一版就是这么干的。只有总和是可加的、可信的。
        "lat_total_ms": sum(lat) / 1000.0,
        "span_s": span,
    }


# ================================================================
# 载入
# ================================================================

def load_groups(results_dir, workloads=None, concurrencies=None, reps=None):
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "*.c*.r*.json"))):
        meta = json.load(open(path))
        w, c = meta["run"]["workload"], meta["run"]["concurrency"]
        r = meta["run"]["rep"]
        if workloads and w not in workloads:
            continue
        if concurrencies and c not in concurrencies:
            continue
        if reps and r not in reps:
            continue
        events = []
        for p in sorted(glob.glob(trace_glob_of(meta))):
            for e in analyze.load(p).get("events", []):
                if rep_of(e.get("request_id")) == r:
                    events.append(e)
        if not events:
            print(f"  警告: {os.path.basename(path)} 切不出事件 "
                  f"(rep={r}, glob={meta.get('trace_glob')}) —— 跳过",
                  file=sys.stderr)
            continue
        groups[(w, c)].append((meta, events, os.path.basename(path)))
    return groups


# ================================================================
# 子命令
# ================================================================

def cmd_verify(args):
    """核对命中模型：真实 trace 的 prefix cache 命中数，是否等于模型预测。"""
    groups = load_groups(args.results_dir, args.workload, args.concurrency, args.rep)
    if not groups:
        print("没有可选的数据", file=sys.stderr)
        return 1
    total_bad = 0
    for (w, c), runs in sorted(groups.items()):
        for meta, events, name in runs:
            accesses = build_stream(meta, events, args.block_size)
            notes = getattr(build_stream, "last_notes", {})
            # 多请求共用的模板：交集大小必须正好等于元数据里的 template_blocks。
            # 不等就说明有模板块换了物理块（被逐出后重建），偏移映射不再可靠。
            bad_tpl = {t: v for t, v in notes.get("tpl_note", {}).items()
                       if v[0] != v[1]}
            if bad_tpl or notes.get("tpl_mismatch"):
                total_bad += 1
                print(f"  ✗ {w} c{c} rep{meta['run']['rep']}: "
                      f"{len(bad_tpl)} 个模板的交集大小与 template_blocks 不符，"
                      f"{notes.get('tpl_mismatch')} 个请求的模板块数对不上")
            pred_hit = set()
            seen_tpl = set()
            for s in meta["schedule"]:
                t = s["template"]
                if t >= 0 and t in seen_tpl:
                    pred_hit.add(s["id"])
                if t >= 0:
                    seen_tpl.add(t)
            actual = defaultdict(int)
            for e in events:
                if e["event"] == "alloc" and e.get("source") == "cache":
                    for sid in pred_hit:
                        if sid in (e.get("request_id") or ""):
                            actual[sid] += 1
                            break
            bad = [(k, actual.get(k, 0)) for k in pred_hit
                   if actual.get(k, 0) == 0]
            extra = sum(1 for k, v in actual.items() if v and k not in pred_hit)
            total_bad += len(bad) + extra
            print(f"  {w:<16} c{c} rep{meta['run']['rep']:<2} "
                  f"逻辑块 {len(accesses):>7}  "
                  f"预测命中请求 {len(pred_hit):>4}  未命中 {len(bad)}  "
                  f"多出来的命中 {extra}")
    print()
    if total_bad:
        print(f"✗ 有 {total_bad} 处不一致 —— 先别信 replay 的数字", file=sys.stderr)
        return 1
    print("✓ 逐请求核对通过：真实 prefix cache 命中与逻辑块模型完全一致")
    return 0


def cmd_run(args):
    groups = load_groups(args.results_dir, args.workload, args.concurrency, args.rep)
    if not groups:
        print("没有可选的数据", file=sys.stderr)
        return 1

    budgets = [int(x) for x in args.budget.split(",")]
    policies = args.policy.split(",")
    si, ri = swap_in_us(args.block_size), recompute_us(args.block_size)

    if args.format == "markdown":
        # 让粘进文档的片段自带条件。这些数字脱离条件就没有意义。
        print("### 离线 replay：热池逐出策略对比\n")
        print(f"- 块大小 {args.block_size} token"
              f" = {args.block_size * BYTES_PER_TOKEN / 2**20:.3f} MiB")
        print(f"- 换回一个块 **{si:.1f} µs**（有效带宽 {PCIE_EFF_GBPS} GB/s，"
              f"含 28 个描述符的固定开销，实测）")
        print(f"- 重算一个块 **{ri:.0f} µs**（实测 {RECOMPUTE_US_PER_TOKEN} µs/token）"
              f" → 换回比重算便宜 {ri / si:.0f}×")
        print(f"- 池语义：B 个**空闲** block；running 请求持有的块被钉住，不参与逐出")
        print(f"- adaptive 的 tau = "
              f"{args.tau if args.tau else '自动（同块相邻两次获取间隔的中位数）'}\n")
    else:
        print("=" * 100)
        print("离线重放：同一份 trace 上比较热池逐出策略")
        print("=" * 100)
        print(f"  块大小 {args.block_size} token = {args.block_size * BYTES_PER_TOKEN / 2**20:.3f} MiB")
        print(f"  换回一个块 {si:.1f} µs（有效带宽 {PCIE_EFF_GBPS} GB/s，"
              f"含 28 个描述符的固定开销，实测）")
        print(f"  重算一个块 {ri:.0f} µs（{RECOMPUTE_US_PER_TOKEN} µs/token）"
              f"  → 换回比重算便宜 {ri / si:.0f}×")
        print(f"  池语义：B 个**空闲** block；running 请求持有的块被钉住，不参与逐出")
        print(f"  adaptive 的 tau = "
              f"{args.tau if args.tau else '自动（取同块相邻两次获取间隔的中位数）'}")

    md = (args.format == "markdown")

    for (w, c), runs in sorted(groups.items()):
        if not md:
            print()
            print("#" * 100)
            print(f"# {w}  ×  并发度 {c}   （{len(runs)} 次重复）")
            print("#" * 100)
        for budget in budgets:
            per_policy = defaultdict(list)
            taus = []
            for meta, events, _name in runs:
                accesses = build_stream(meta, events, args.block_size)
                if not accesses:
                    continue
                if "adaptive" in policies:
                    taus.append(auto_tau(accesses))
                for pol in policies:
                    per_policy[pol].append(
                        replay(accesses, args.block_size, budget, pol, args.tau))

            rows = []
            for pol in policies:
                rs = per_policy.get(pol) or []
                if not rs:
                    continue

                def avg(k, _rs=rs):
                    return statistics.mean(x[k] for x in _rs)

                rows.append({"pol": pol, **{key: avg(key)
                                            for _l, key, sp in _COLS if sp}})

            label = (f"{w} × 并发 {c} × 热池 B={budget} blocks"
                     f"（{budget * args.block_size} tokens）, {len(runs)} 次重复")
            if md:
                note = None
                if taus:
                    note = (f"adaptive 的 tau = {statistics.mean(taus):.2f}s"
                            f"（取自该 trace 里同块相邻两次获取间隔的中位数）。"
                            f"belady 知道未来，**不可实现**，只用来标出还剩多少空间。")
                print(render_markdown(rows, label, note))
                continue

            print()
            print(f"--- 热池预算 B = {budget} blocks "
                  f"（{budget * args.block_size} tokens）---")
            widths = []
            for i, (_l, _k, sp) in enumerate(_COLS):
                cells = [("belady (离线)" if r["pol"] == "belady" else r["pol"])
                         if sp is None else _fmt(r[_k], sp) for r in rows]
                widths.append(max(len(_COLS[i][0]), *(len(x) for x in cells)) + 2)
            hdr = "".join(f"{c[0]:<{widths[i]}}" for i, c in enumerate(_COLS))
            print(hdr)
            print("-" * len(hdr))
            for r in rows:
                cells = [("belady (离线)" if r["pol"] == "belady" else r["pol"])
                         if sp is None else _fmt(r[_k], sp)
                         for _l, _k, sp in _COLS]
                print("".join(f"{x:<{widths[i]}}" for i, x in enumerate(cells)))
            if taus:
                print(f"      （adaptive 的 tau = "
                      f"{statistics.mean(taus):.2f}s，取自该 trace 里同块相邻两次"
                      f"获取间隔的中位数）")
            print("      （belady 是**离线**参照，知道未来，只在线的策略到不了；"
                  "它的作用是标出还剩多少空间）")
    return 0


#: 表格列定义。**渲染和 markdown 共用一份**，避免两边顺序或口径走岔 ——
#: 之前手抄表格进文档，抄错过一次。
_COLS = [
    ("策略", "pol", None),
    ("命中率", "pool_hit_rate", ("pct", 1)),
    ("模板命中率", "tpl_hit_rate", ("pct", 1)),
    ("容量缺失率", "capacity_miss_rate", ("pct", 1)),
    ("换出数", "evictions", ("int", 0)),
    ("换回数", "swap_in", ("int", 0)),
    ("池利用率", "pool_util", ("pct", 1)),
    ("换回P50", "req_swap_p50_us", ("us", 0)),
    ("换回P99", "req_swap_p99_us", ("us", 0)),
    ("换回合计", "swap_in_total_ms", ("ms", 0)),
    ("换出流量", "swap_out_mb", ("mb", 1)),
]


def _fmt(v, spec):
    kind, nd = spec
    if kind == "pct":
        return f"{v * 100:.{nd}f}%"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "us":
        return f"{v:.0f}µs"
    if kind == "ms":
        return f"{v:.0f}ms"
    if kind == "mb":
        return f"{v:.1f}MB"
    return f"{v}"


def render_markdown(rows, title, note=None):
    """输出可直接粘进文档的 markdown 表。

    存在的理由很实际：这些数字要写进 README / GUIDE，手抄 9 个组合 ×
    3 档预算 × 4 个策略一定会抄错，而且抄错了不会有人发现。
    """
    out = [f"**{title}**", ""]
    out.append("| " + " | ".join(c[0] for c in _COLS) + " |")
    out.append("|" + "|".join("---" for _ in _COLS) + "|")
    for r in rows:
        cells = []
        for _label, key, spec in _COLS:
            if spec is None:
                cells.append(str(r[key]))
            else:
                cells.append(_fmt(r[key], spec))
        # belady 是不可达的离线上界，加个标记免得被当成可选项引用
        if r["pol"] == "belady":
            cells[0] = "belady (离线)"
        out.append("| " + " | ".join(cells) + " |")
    if note:
        out.append("")
        out.append(note)
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="热池逐出策略离线重放")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--results-dir", default=os.path.join(ROOT, "results-skew"))
        p.add_argument("--workload", action="append", default=None)
        p.add_argument("--concurrency", action="append", type=int, default=None)
        p.add_argument("--rep", action="append", type=int, default=None)
        p.add_argument("--block-size", type=int, default=16)

    v = sub.add_parser("verify", help="核对逻辑块模型与真实 prefix cache 命中")
    common(v)

    r = sub.add_parser("run", help="三档预算 × 多策略")
    common(r)
    r.add_argument("--budget", default="64,128,256")
    r.add_argument("--policy", default="lru,lfu,adaptive,belady")
    r.add_argument("--tau", type=float, default=None,
                   help="adaptive 的时间衰减常数（秒）；不给则取自动值")
    r.add_argument("--format", choices=["table", "markdown"], default="table",
                   help="markdown 输出可直接粘进 README/GUIDE，省掉手抄这一步")

    args = ap.parse_args()
    return cmd_verify(args) if args.cmd == "verify" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
