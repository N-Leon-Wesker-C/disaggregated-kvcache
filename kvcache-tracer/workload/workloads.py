"""
workloads.py — 三类正式 workload 的规格与请求生成

规格（由实验协议给定）
============================================================

| workload         | prompt tokens | generation tokens |
|------------------|---------------|-------------------|
| short-qa         | 64 – 256      | 64 – 128          |
| code-gen         | 512 – 1024    | 256 – 512         |
| long-summarize   | 2048 – 4096   | 256 – 512         |

每个请求的 prompt 长度和输出长度都在区间内**均匀随机采样**，所以「prompt/output
token 分布」本身是可报告的实验量，而不是一个固定值。

共享指令前导
============================================================

每类 workload 内所有请求共享一段**完全相同的指令前导**（真实服务里 system
prompt 的形态），这是 prefix cache 命中的来源。前导长度取 `prompt_range[0]`
向下取到 16 的倍数：

    short-qa       64 tokens ( 4 blocks)
    code-gen      512 tokens (32 blocks)
    long-summarize 2048 tokens (128 blocks)

取 prompt 下限是为了不把最短的请求挤出规定区间（前导本身就占满了下限）。
这个取法让命中率天然随上下文长度递增 —— 长上下文 workload 有更长的可用共享段，
这是真实性质而非人为放大。每个请求的独有内容 = 目标 prompt 长度 − 前导长度。

请求数怎么定的
============================================================

C=1 时单流解码约 58 tok/s，所以每轮墙钟基本由「输出 token 总量」决定。
三类 workload 的请求数都按「C=1 每轮 ≲ 250s」倒推（输出总量约 14K tokens），
再让 prompt 部分尽量吃满池子：

    short-qa       144 请求 → prompt ≈ 23,000 tok
    code-gen        36 请求 → prompt ≈ 28,000 tok
    long-summarize  36 请求 → prompt ≈ 111,000 tok  ← 接近 KV 池容量

需要提前知道的后果：**只有 long-summarize 能把池子压到接近满**。另外两类是
解码主导的，在同样墙钟预算内 prompt 总量只有两三万 token，远低于 12.5 万的池子，
不会触发逐出。这不是缺陷，而是三类 workload 在同一个池预算下的真实差异 ——
报告里必须如实写出来，不能拿 long-summarize 的逐出行为去代表另外两类。

确定性
============================================================

所有随机采样走同一个 `random.Random(SEED)`，且生成顺序固定。同一个 seed 下
请求集逐字节可复现 —— 三次重复跑的是同一批请求，差异只来自调度。
"""

import random

#: 固定随机种子。改它 = 换一份 workload。
SEED = 20261015

#: block_size，用于把共享前导对齐到整块。必须和 vLLM 的 --block-size 一致。
BLOCK_SIZE = 16


# ================================================================
# 词表：只需要分词结果稳定、长度可控，不需要语义通顺
# ================================================================

VOCAB_QA = (
    "question answer explain describe define compare difference between what "
    "which when where why how does is are can could would should the a an of "
    "for with about into over under example reason cause effect result meaning "
    "purpose usage benefit drawback advantage limit constraint assumption "
    "context background overview summary detail step process method approach "
    "simple clear brief concise accurate correct common typical general"
).split()

VOCAB_CODE = (
    "def return import from class self async await yield lambda list dict set "
    "tuple str int float bool None True False try except finally raise with as "
    "for while if elif else in not and or is pass break continue global nonlocal "
    "assert del print len range enumerate zip map filter sorted sum min max abs "
    "open read write close append extend insert remove pop keys values items get "
    "update json parse encode decode request response client server handler route "
    "buffer stream chunk token parse render mount hook state props effect memo "
    "refactor test mock patch fixture assert equals raises parametrize benchmark"
).split()

VOCAB_DOC = (
    "the report describes analysis findings regarding performance characteristics "
    "of distributed storage systems under sustained load conditions observed during "
    "the evaluation period. measurements indicate throughput remained stable while "
    "latency increased gradually as concurrency rose across all tested configurations. "
    "further investigation revealed that cache behavior dominated the observed variance "
    "between runs and that memory pressure correlated strongly with request size rather "
    "than request rate alone. additional experiments are required to isolate the effect "
    "of background compaction on the tail latency distribution reported in section four. "
    "the authors conclude that adaptive policies outperform static thresholds when the "
    "workload composition shifts over time and recommend further study of migration cost."
).split()


WORKLOADS = {
    "short-qa": {
        "prompt_range": (64, 256),
        "gen_range": (64, 128),
        "n_requests": 144,
        "vocab": VOCAB_QA,
        # 固定指令前导，类内所有请求逐字节相同 → prefix cache 的复用来源
        "instruction": (
            "You are a concise technical assistant. Answer the following question "
            "accurately and briefly, using plain language and no unnecessary detail."
        ),
        "n_templates": 64,
        "template_range": (64, 160),
    },
    "code-gen": {
        "prompt_range": (512, 1024),
        "gen_range": (256, 512),
        "n_requests": 36,
        "vocab": VOCAB_CODE,
        "instruction": (
            "You are an expert software engineer. Write correct, idiomatic and "
            "well-tested code for the task below. Follow the existing style of the "
            "project, handle edge cases explicitly, and do not add commentary that "
            "was not requested. Prefer clarity over cleverness, keep functions small "
            "and single-purpose, and make sure the result is runnable as-is."
        ),
        "n_templates": 32,
        "template_range": (256, 496),
    },
    "long-summarize": {
        "prompt_range": (2048, 4096),
        "gen_range": (256, 512),
        # 40 请求 → 足迹约 1.11x 池容量。36 请求时正好卡在 1.00x，逐出变成
        # 刀锋上的行为，三次重复之间会不稳定；留出余量才能观察到稳定的逐出。
        "n_requests": 40,
        "vocab": VOCAB_DOC,
        "instruction": (
            "You are a careful research assistant. Read the document provided below "
            "and produce a faithful summary that preserves the authors' claims and "
            "the evidence given for them. Do not introduce information that is not "
            "present in the source, do not editorialize, and do not omit limitations "
            "the authors themselves state. Structure the summary so that a reader "
            "who has not seen the document can follow the argument, and keep the "
            "terminology consistent with the original wording throughout."
        ),
        "n_templates": 16,
        "template_range": (640, 960),
    },
}

#: Zipf 指数：第 i 个模板的权重 ∝ 1/(i+1)^alpha。
#: 1.0 是标准 Zipf，真实服务里系统提示的分布通常是这个量级 —— 少数几个模板
#: 吃掉大部分流量，长尾模板偶尔被用到。alpha 越大越集中。
SKEW_ALPHA = 1.0


def load_tokenizer(model_path):
    """本地分词器。必须用真实分词器数 token —— 估算会偏，偏了就不是确定性。"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def build_text(target_tokens, rng, tok, vocab, hi=None):
    """拼一段 target_tokens 个 token 左右的文本，可选硬上限 hi。

    增长是按批的（逐词 encode 太慢），所以会有一个 step 的溢出。给了 hi 就在
    超限时**从尾部删词**收回区间 —— 删词是精确的，不需要重新生成，也不会像
    按 token 截断那样产生半个字符。

    为什么需要 hi：prompt 长度必须落在规格区间内，溢出的样本会让「prompt
    token 分布」这个要写进元数据的量失真。实测不加约束时 short-qa 的 max
    冲到 279，超了规格上限 256。
    """
    step = max(1, min(32, target_tokens // 8))
    words = []
    while len(tok.encode(" ".join(words))) < target_tokens:
        words.extend(rng.choices(vocab, k=step))
    if hi is not None:
        while words and len(tok.encode(" ".join(words))) > hi:
            words.pop()
    return " ".join(words)


def _fit_to_tokens(text, target, rng, tok, vocab):
    """把 text 补/裁到恰好 target 个 token。

    为什么必须精确：prefix cache 以 block 为粒度复用，共享段若不是整数个
    block，最后一个不完整的 block 无法被后续请求复用，白白少掉一块命中。

    为什么要循环收敛而不是一次截断：按 token 截一段再 decode 回字符串时，
    BPE 在边界上偶尔会解出半个字符，重新分词就少一个 token（实测 short-qa
    卡在 63）。所以截断后还要收敛 —— 少了就补一个刚好只 +1 token 的词，
    多了就再截一刀。
    """
    if len(tok.encode(text)) < target:
        filler = build_text(target - len(tok.encode(text)), rng, tok, vocab)
        text = f"{text} {filler}"

    for _ in range(8):
        n = len(tok.encode(text))
        if n == target:
            break
        if n > target:
            text = tok.decode(tok.encode(text)[:target])
            continue
        for w in vocab:                     # 找一个刚好补 1 个 token 的词
            cand = f"{text} {w}"
            if len(tok.encode(cand)) == target:
                return cand
        text = f"{text} {rng.choice(vocab)}"
    return text


def build_preamble(spec, tok):
    """`single` 模式：一段类内逐字节相同的共享指令前导。

    长度取 `prompt_range[0]` 向下对齐到 block —— 取下限是为了不把最短的
    请求挤出规定区间。
    """
    target = (spec["prompt_range"][0] // BLOCK_SIZE) * BLOCK_SIZE
    return _fit_to_tokens(spec["instruction"], target,
                          random.Random(SEED), tok, spec["vocab"])


def build_templates(spec, tok):
    """`skewed` 模式：K 个彼此不同的系统提示模板，长度铺满 template_range。

    三个刻意的设计，缺一个 replay 就退化成没有信息量的场景：

    1. **长度各异**。所有模板一样长的话，"部分命中"（前 j 块在池里、第 j+1
       块被逐出）根本不会出现 —— 而部分命中恰恰是逐出策略差异最大的地方。
    2. **长度与热度无关**。长度按 block 对齐后会做一次 shuffle 再按序号赋
       Zipf 权重；不 shuffle 的话最短的模板天然对最多请求可用，同时又最热，
       两者叠加会把"热度"和"长度"这两个变量绑死，没法分开解释。
    3. **逐块互不相同**。这是被实测打回来的一条：最初每个模板写成
       `[template-i] + 同一段 instruction + 随机填充`，想着序号标记能让第 0
       块就分叉 —— 结果 instruction 有一百多个 token，第 1~7 块在所有模板间
       逐字节相同，`check_templates_disjoint()` 当场查出 code-gen 有 69 个
       重复块。模板互相命中会让 replay 的"请求命中自己的模板"模型直接失效。
       所以现在标记之后**只保留指令开头几个词**（塞得进第 0 块），其余全部
       用每模板独立的随机种子生成，从第 1 块起就分叉。
    """
    lo, hi = spec["template_range"]
    k = spec["n_templates"]
    rng = random.Random(SEED ^ 0x5EED)

    # 长度在 [lo, hi] 上均匀铺开，向下对齐到 block；对齐后可能重复，
    # 但重复的是长度不是内容，仍是彼此独立的块。
    if k == 1:
        blocks = [max(1, lo // BLOCK_SIZE)]
    else:
        blocks = [max(1, (lo + (hi - lo) * i // (k - 1)) // BLOCK_SIZE)
                  for i in range(k)]
    rng.shuffle(blocks)

    #: 指令里保留的词数。必须短到连同标记一起落在第 0 块内 —— 长了就会
    #: 跨块，跨出去的那部分在所有模板间相同，重复块又回来了。
    head_words = 6
    head = " ".join(spec["instruction"].split()[:head_words])

    templates = []
    for i, nb in enumerate(blocks):
        seed_text = f"[template-{i:03d}] {head}"
        text = _fit_to_tokens(seed_text, nb * BLOCK_SIZE,
                              random.Random(SEED + i), tok, spec["vocab"])
        templates.append({"id": i, "text": text,
                          "n_tokens": nb * BLOCK_SIZE, "n_blocks": nb})
    return templates


def zipf_weights(k, alpha=SKEW_ALPHA):
    """第 i 名的权重 ∝ 1/(i+1)^alpha。"""
    return [1.0 / ((i + 1) ** alpha) for i in range(k)]


def pick_template(prefixes, target_prompt, rng):
    """在「整段塞得下且还给正文留至少一块」的模板里按 Zipf 抽一个。

    没有可用模板时返回 None —— 请求退化成纯独有正文。这不是异常路径：
    最短的一小撮请求本来就装不下任何模板，真实服务里也一样。把它们硬塞进
    一个模板会同时破坏 prompt 长度分布和"命中"的含义。
    """
    elig = [j for j, p in enumerate(prefixes)
            if p["n_tokens"] + BLOCK_SIZE <= target_prompt]
    if not elig:
        return None
    w = zipf_weights(len(prefixes))
    return rng.choices(elig, weights=[w[j] for j in elig], k=1)[0]


def make_prefixes(name, tok, prefix_mode="single"):
    """按模式造共享前导。抽成独立函数是为了让 run.py 的元数据统计能拿到
    同一份模板，而不是各自再实现一遍 —— 两份实现迟早会漂移。"""
    spec = WORKLOADS[name]
    if prefix_mode == "single":
        p = build_preamble(spec, tok)
        return [{"id": 0, "text": p, "n_tokens": len(tok.encode(p)),
                 "n_blocks": len(tok.encode(p)) // BLOCK_SIZE}]
    if prefix_mode == "skewed":
        return build_templates(spec, tok)
    raise ValueError(f"未知 prefix_mode: {prefix_mode}")


def build_requests(name, tok, prefix_mode="single"):
    """生成某类 workload 的请求集。同一 (name, prefix_mode) 下逐字节可复现。

    prefix_mode:
      - "single": 类内所有请求共享同一段指令前导（原始实验用的模式）
      - "skewed": K 个长度各异的模板，按 Zipf 分布挑选

    两种模式的 prompt 长度分布**都是**规格区间上的均匀分布 —— 模板是先
    采长度、再挑模板，不是反过来。反过来的话（先挑模板再定长度）prompt
    分布会被模板长度分布带偏，元数据里那条"uniform(lo,hi)"就成了假话。
    """
    spec = WORKLOADS[name]
    rng = random.Random(SEED)          # 每类独立种子 → 换类型不影响其他类型
    lo, hi = spec["prompt_range"]
    prefixes = make_prefixes(name, tok, prefix_mode)

    requests = []
    for i in range(spec["n_requests"]):
        # 目标 prompt 长度在区间内均匀采样 —— 这一步必须在挑模板之前
        target_prompt = rng.randint(lo, hi)
        j = pick_template(prefixes, target_prompt, rng) \
            if prefix_mode == "skewed" else 0
        pre = prefixes[j] if j is not None else None
        n_pre = pre["n_tokens"] if pre else 0

        body = build_text(max(8, target_prompt - n_pre), rng, tok,
                          spec["vocab"], hi=hi - n_pre)
        requests.append({
            "id": f"{name}-{i:03d}",
            "workload": name,
            "prompt": f"{pre['text']} {body}" if pre else body,
            "max_tokens": rng.randint(*spec["gen_range"]),
            # 这两项进元数据，replay 靠它们重建"谁该命中谁"，
            # 不用再跑一遍分词器，也不会和生成逻辑漂移。
            "template": j if j is not None else -1,
            "template_blocks": pre["n_blocks"] if pre else 0,
        })

    # 打乱发送顺序：不打乱的话 prompt 长度是单调的，并发窗口里会全是
    # 极端长度的请求。固定种子 → 三次重复顺序一致。
    rng.shuffle(requests)
    return requests


def check_templates_disjoint(prefixes, tok):
    """验证 K 个模板的 block 两两不同 —— 用哈希数，不靠肉眼。

    如果两个模板共享了某个 block，"请求命中自己的模板" 这个 replay 模型就
    不成立了：命中数会凭空多出来，而且多出来的部分随模板长度对齐方式漂移。
    成本很低（全部模板合起来两万 token 级别），所以每次都真查一遍，不做假设。
    """
    if len(prefixes) < 2:
        return {"checked": False, "duplicate_block_hashes": 0}
    import hashlib
    seen = {}
    dup = 0
    for p in prefixes:
        ids = tok.encode(p["text"])
        for b in range(p["n_blocks"]):
            chunk = ids[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE]
            h = hashlib.blake2b(
                b"".join(int(t).to_bytes(4, "little") for t in chunk),
                digest_size=8).digest()
            if h in seen and seen[h] != p["id"]:
                dup += 1
            seen[h] = p["id"]
    return {"checked": True, "duplicate_block_hashes": dup,
            "n_block_hashes": len(seen)}


def describe(requests, tok, prefix_mode="single", prefixes=None):
    """统计请求集的 token 分布与复用结构 —— 要写进实验元数据的部分。"""
    prompts = sorted(len(tok.encode(r["prompt"])) for r in requests)
    gens = sorted(r["max_tokens"] for r in requests)
    n = len(prompts)

    def p(v, q):
        # 分位数按 v 自己的长度取。早先这里用的是请求数 n —— 对 prompts/gens
        # 恰好等价，对下面的模板列表就越界了。
        return v[min(len(v) - 1, int(len(v) * q))]

    desc = {
        "n_requests": n,
        "prefix_mode": prefix_mode,
        "prompt_tokens": {
            "min": prompts[0], "p50": p(prompts, .5), "p90": p(prompts, .9),
            "max": prompts[-1], "mean": round(sum(prompts) / n, 1),
            "total": sum(prompts),
        },
        "output_tokens": {
            "min": gens[0], "p50": p(gens, .5), "p90": p(gens, .9),
            "max": gens[-1], "mean": round(sum(gens) / n, 1),
            "total": sum(gens),
        },
    }

    # ---- 复用结构。replay 要模拟的"可复用集合"就是这个 reusable_blocks ----
    used = {}
    junk = 0
    for r in requests:
        j = r.get("template", 0)
        tb = r.get("template_blocks", 0)
        if j >= 0:
            used[j] = used.get(j, 0) + 1
        pb = -(-len(tok.encode(r["prompt"])) // BLOCK_SIZE)      # ceil
        junk += pb - tb

    # 按请求数从多到少累加，看覆盖掉 50% / 80% 的请求要用多少 block。
    #
    # 这是判断"热池预算 64/128/256 是不是有意义的档位"的唯一依据：热模板的
    # block 数远小于预算时，任何策略都能把热的全留住，LRU/LFU 的对比会退化
    # 成没有差别 —— 那是预算选错了，不是策略没差别。
    nb_of = {p["id"]: p["n_blocks"] for p in (prefixes or [])}
    cover, cum_req, cum_blk, ti = {}, 0, 0, 0
    targets = (0.5, 0.8)
    for tj, cnt in sorted(used.items(), key=lambda kv: -kv[1]):
        cum_req += cnt
        cum_blk += nb_of.get(tj, 0)
        while ti < len(targets) and cum_req >= targets[ti] * n:
            cover[f"blocks_for_{int(targets[ti] * 100)}pct_requests"] = cum_blk
            ti += 1

    counts = sorted(used.values())
    all_blocks = sum(p["n_blocks"] for p in (prefixes or []))
    desc["reuse"] = {
        "n_templates": len(prefixes) if prefixes else 1,
        "n_templates_used": len(used),
        "n_requests_without_template": sum(1 for r in requests
                                           if r.get("template", 0) < 0),
        # 只有被用过至少一次的模板才可能被命中，所以这个数才是 replay 里
        # 真正承压的集合大小
        "reusable_blocks_used": sum(
            p["n_blocks"] for p in (prefixes or []) if p["id"] in used),
        "reusable_blocks_all": all_blocks,
        "junk_blocks": junk,
        "requests_per_template": {
            "min": counts[0] if counts else 0,
            "p50": p(counts, .5) if counts else 0,
            "max": counts[-1] if counts else 0,
        },
        "top_template_share": round(counts[-1] / n, 3) if counts else 0.0,
        **cover,
    }
    if prefixes:
        desc["reuse"].update(check_templates_disjoint(prefixes, tok))
    return desc


def format_description(name, desc, pool_tokens=None):
    p, o = desc["prompt_tokens"], desc["output_tokens"]
    ru = desc.get("reuse") or {}
    lines = [
        f"[{name}] {desc['n_requests']} 个请求  "
        f"(prefix_mode={desc.get('prefix_mode', 'single')})",
        f"  prompt : min {p['min']:>5}  p50 {p['p50']:>5}  p90 {p['p90']:>5}"
        f"  max {p['max']:>5}  mean {p['mean']:>7}  合计 {p['total']:>8,}",
        f"  output : min {o['min']:>5}  p50 {o['p50']:>5}  p90 {o['p90']:>5}"
        f"  max {o['max']:>5}  mean {o['mean']:>7}  合计 {o['total']:>8,}",
    ]
    if ru:
        lines.append(
            f"  复用   : 模板 {ru['n_templates_used']}/{ru['n_templates']} 被用到，"
            f"可复用 block {ru['reusable_blocks_used']}（全部 "
            f"{ru['reusable_blocks_all']}），独有 prompt block "
            f"{ru['junk_blocks']}（不含解码过程中新分配的）")
        lines.append(
            f"           每模板请求数 min {ru['requests_per_template']['min']} / "
            f"p50 {ru['requests_per_template']['p50']} / "
            f"max {ru['requests_per_template']['max']}"
            f"（最热模板占 {ru['top_template_share'] * 100:.1f}% 请求）"
            + (f"  ⚠ 模板间有 {ru['duplicate_block_hashes']} 个重复 block"
               if ru.get("duplicate_block_hashes") else ""))
        lines.append(
            f"           命中 50% / 80% 请求分别需要 "
            f"{ru.get('blocks_for_50pct_requests', '?')} / "
            f"{ru.get('blocks_for_80pct_requests', '?')} 个 block —— "
            f"热池预算要选在这一带才有取舍")
    if pool_tokens:
        foot = p["total"] + o["total"]
        ratio = foot / pool_tokens
        lines.append(f"  足迹/池容量 = {foot:,}/{pool_tokens:,} = {ratio:.2f}x"
                     f"  → {'会逐出' if ratio > 1 else '不满池，不会逐出'}")
    return "\n".join(lines)
