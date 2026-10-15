"""
workload.py — 并发度扫描用的正式 workload

设计目标
============================================================

这份 workload 不是为了「跑满吞吐」，而是为了产出**能回答 swap 策略问题**的
KV cache trace。所以它必须满足四条：

1. **确定性**。固定 seed、`temperature=0`、`ignore_eos=True`（输出长度精确
   等于 max_tokens）。两次跑同一个并发度，请求集和 token 数完全一致。

2. **跨并发度复用同一请求集**。并发 1 / 4 / 8 三次运行发的是同一批请求、
   同样的顺序、同样的长度，**唯一的变量是同时在跑的请求数**。否则测出来的
   差异分不清是并发造成的还是请求集变了造成的。

3. **总足迹超过 KV cache 池**。这是关键。实测本机 4090-48G + Qwen2.5-7B
   在 `--gpu-memory-utilization 0.45` 下池子有 110,416 tokens，而满配置
   (0.85) 下有 465,424 —— 后者意味着任何合理的负载都塞得下，永远不会发生
   逐出，trace 里全是「从生到死没人打扰」的 block，关于 swap 的结论会是空的。
   本 workload 的独特 token 足迹约 15 万，是池子的 1.35 倍，逐出必然发生，
   且并发越高发生得越早越频繁。

4. **覆盖四类行为**，因为不同类别对 swap 策略的要求是相反的：

   | 类别         | 数量 | prompt | output | 考察点                                |
   |--------------|------|--------|--------|---------------------------------------|
   | short-chat   | 24   | 128    | 64     | 短命 block，swapping 纯属浪费          |
   | long-prefill | 22   | 5120   | 16     | 一次性申请大量 block，冲击池子         |
   | long-decode  | 10   | 384    | 768    | 长命 + 高频读，最该被保护的热 block    |
   | shared-prefix| 16   | 3072   | 128    | 跨请求复用的 block（前缀相同的 2048）  |

5. **可归因**。每个请求通过 `X-Request-Id` 头把自己的类别标签传下去，vLLM 拿它
   当 engine 层 request_id，所以 trace 里的 `request_id` 就是 `long-prefill-07`
   这样的标签，可以按类别直接聚合。不加这个头的话服务端会生成随机 uuid，
   四类请求在 trace 里就混成一团了。

   注意 shared-prefix：16 个请求共享**完全相同的 2048 token 前缀字符串**。
   因为是同一个字符串，服务端重新分词后 token id 必然逐位相同，prefix cache
   一定命中。这是故意造的「理想复用」场景 —— 用来测 block 被多个请求服务时
   的 reads 累加是否正确、生命周期统计是否会被跨请求复用搞乱。

用法
============================================================

    # 看请求集长什么样，不发请求
    python workload/workload.py --dry-run

    # 跑一个并发度
    python workload/workload.py --concurrency 4 --out results/c4.json

服务端需要已经在 8000 端口起来（见 run_sweep.sh）。
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict


# ================================================================
# 请求集定义
# ================================================================

#: 四类请求的 (类别, 数量, prompt tokens, 输出 tokens, 说明)
CATEGORIES = [
    ("short-chat",    24, 128,  64),
    ("long-prefill",  22, 5120, 16),
    ("long-decode",   10, 384,  768),
    ("shared-prefix", 16, 3072, 128),
]

#: shared-prefix 类共享的前缀长度（其余部分是每个请求独有的后缀）
SHARED_PREFIX_TOKENS = 2048

#: 唯一固定 seed。改它 = 换一份 workload，三次运行必须用同一个。
SEED = 20261015

#: 造文本用的词表。不需要通顺，只需要分词结果稳定、长度可控。
WORDS = (
    "system policy cache block token memory layer kernel tensor batch stream "
    "prefix suffix eviction promote demote latency throughput decode prefill "
    "queue scheduler worker shard replica fabric capacity pressure hotspot cold "
    "warm reuse migration bandwidth stall traversal attention residual norm "
    "projection embedding vocab context window horizon segment offset stride "
    "budget quota pool slab arena allocator reference lifetime interval counters "
    "signal probe trace sample region partition compaction fragmentation locality "
    "temporal spatial heuristic threshold margin headroom saturation contention "
    "backpressure admission reservation watermark deadline slack jitter variance "
    "median percentile tail mean drift plateau transient steady bursty adaptive"
).split()


def load_tokenizer(model_path):
    """本地分词器。用真实的 Qwen 分词器数 token —— 估算会偏，偏了就不是确定性。"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def build_text(target_tokens, rng, tok):
    """拼一段约 target_tokens 个 token 的文本。

    不做精确截断：分词边界上截断容易产生半个字，反而引入不确定性。
    按块累加、接近目标就停，允许 ±(target/20) 的偏差 —— 对 trace 分析
    无影响（block 数按 16 对齐，几十个 token 的差异落在噪声里）。
    """
    step = max(1, target_tokens // 20)
    words = []
    n = 0
    while n < target_tokens:
        words.extend(rng.choices(WORDS, k=step))
        n = len(tok.encode(" ".join(words)))
    return " ".join(words)


def build_workload(tok):
    """生成请求集。同一 seed 下每次调用结果完全一致。"""
    rng = random.Random(SEED)
    requests = []

    # shared-prefix 的公共前缀只生成一次，16 个请求引用同一个字符串
    shared_prefix = build_text(SHARED_PREFIX_TOKENS, rng, tok)

    req_id = 0
    for category, count, prompt_tokens, out_tokens in CATEGORIES:
        for i in range(count):
            if category == "shared-prefix":
                suffix = build_text(prompt_tokens - SHARED_PREFIX_TOKENS, rng, tok)
                prompt = shared_prefix + " " + suffix
            else:
                prompt = build_text(prompt_tokens, rng, tok)

            requests.append({
                "id": f"{category}-{i:02d}",
                "index": req_id,
                "category": category,
                "prompt": prompt,
                "max_tokens": out_tokens,
            })
            req_id += 1

    # 打乱顺序：不打乱的话四类各自成块发送，并发窗口里全是同一类请求，
    # 测不出类别之间的相互影响。seed 固定 → 三次运行的顺序一致。
    rng.shuffle(requests)
    return requests


def report_workload(requests, tok, kv_pool_tokens=None):
    """打印请求集概要 —— 这也是 workload 的一部分产出，写进实验记录用。"""
    per_cat = defaultdict(lambda: {"n": 0, "prompt": 0, "out": 0})
    for r in requests:
        c = per_cat[r["category"]]
        c["n"] += 1
        c["prompt"] += len(tok.encode(r["prompt"]))
        c["out"] += r["max_tokens"]

    print(f"{'类别':<16}{'数量':>6}{'总 prompt tok':>16}{'总 output tok':>16}")
    print("-" * 56)
    total_prompt = total_out = total_n = 0
    for cat, _, _, _ in CATEGORIES:
        c = per_cat[cat]
        print(f"{cat:<16}{c['n']:>6}{c['prompt']:>16,}{c['out']:>16,}")
        total_prompt += c["prompt"]
        total_out += c["out"]
        total_n += c["n"]
    print("-" * 56)
    print(f"{'合计':<16}{total_n:>6}{total_prompt:>16,}{total_out:>16,}")

    # 独特足迹要扣掉共享前缀的重复部分 —— prefix cache 只存一份
    n_shared = per_cat["shared-prefix"]["n"]
    effective = total_prompt + total_out - (n_shared - 1) * SHARED_PREFIX_TOKENS
    print(f"\n独特 token 足迹（扣掉共享前缀的重复）: {effective:,}")
    if kv_pool_tokens:
        print(f"KV cache 池容量:                      {kv_pool_tokens:,}")
        ratio = effective / kv_pool_tokens
        verdict = "会发生逐出" if ratio > 1 else "不会发生逐出"
        print(f"足迹 / 池容量 = {ratio:.2f}x  →  {verdict}")
        if ratio <= 1:
            print("  警告: 足迹没超过池子，trace 里不会有逐出，"
                  "并发度对比会失去意义。调大 workload 或调小 "
                  "--gpu-memory-utilization。")


# ================================================================
# 执行
# ================================================================

async def _one_request(session, sem, base_url, model, req):
    """发一个请求，返回客户端侧计时。

    信号量在**请求内部**获取（而不是在 gather 之前），这样 in-flight 窗口
    恰好等于并发度：前 C 个立刻发出，之后每完成一个补一个。
    """
    payload = {
        "model": model,
        "prompt": req["prompt"],
        "max_tokens": req["max_tokens"],
        "temperature": 0.0,
        "ignore_eos": True,          # 输出长度精确等于 max_tokens
        "stream": True,              # 只为拿 TTFT（prefill / decode 的分界）
        "stream_options": {"include_usage": True},
    }

    # X-Request-Id 会被 vLLM 取作 engine 层 request_id 的基名
    # （entrypoints/serve/engine/serving.py:_base_request_id），但它会在外面
    # 再包一层。实测（0.25.1）发 "long-prefill-99" 得到的是：
    #
    #     cmpl-long-prefill-99-0-957e927f
    #
    # 标签嵌在中间，按子串匹配就能归类 —— 够用了。不发这个头的话是纯随机
    # uuid，四类请求在 trace 里混成一团，而按类别拆解恰恰是设计四类的意义。
    headers = {"X-Request-Id": req["id"]}

    async with sem:
        t_submit = time.time()
        ttft = None
        usage = None
        try:
            async with session.post(f"{base_url}/v1/completions",
                                    json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return {**{k: req[k] for k in ("id", "category", "max_tokens")},
                            "ok": False, "error": f"HTTP {resp.status}: {body[:200]}"}

                async for raw in resp.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data: "):
                        continue
                    body = line[6:]
                    if body == "[DONE]":
                        break
                    chunk = json.loads(body)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                        continue
                    choices = chunk.get("choices") or []
                    if choices and choices[0].get("text") and ttft is None:
                        ttft = time.time() - t_submit
        except Exception as exc:                      # noqa: BLE001
            return {**{k: req[k] for k in ("id", "category", "max_tokens")},
                    "ok": False, "error": f"{type(exc).__name__}: {exc}"}

        done = time.time()
        return {
            "id": req["id"],
            "category": req["category"],
            "max_tokens": req["max_tokens"],
            "ok": True,
            "submit_t": round(t_submit, 6),
            "ttft_s": round(ttft, 6) if ttft is not None else None,
            "latency_s": round(done - t_submit, 6),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
        }


async def run(requests, concurrency, base_url, model):
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=1800, sock_read=600)
    wall_start = time.time()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *[_one_request(session, sem, base_url, model, r) for r in requests])

    wall = time.time() - wall_start
    return results, wall


def summarize(results, wall, concurrency):
    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]

    print(f"\n{'='*60}")
    print(f"并发度 {concurrency}  —  墙钟 {wall:.1f}s")
    print(f"{'='*60}")
    print(f"成功 {len(ok)} / {len(results)}")
    if bad:
        print(f"失败 {len(bad)}:")
        for r in bad[:5]:
            print(f"  {r['id']}: {r['error']}")
        return

    def pct(vals, q):
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(len(vals) * q))]

    lat = [r["latency_s"] for r in ok]
    print(f"端到端延迟  P50 {pct(lat, .5):.2f}s  P90 {pct(lat, .9):.2f}s  "
          f"max {max(lat):.2f}s")

    # 按类别拆开看 —— 四类之间的差异正是 workload 想暴露的东西
    per_cat = defaultdict(list)
    for r in ok:
        per_cat[r["category"]].append(r)

    print(f"\n{'类别':<16}{'数量':>5}{'TTFT中位':>11}{'延迟中位':>11}"
          f"{'prompt tok':>12}{'输出 tok':>10}")
    print("-" * 68)
    for cat, _, _, _ in CATEGORIES:
        rs = per_cat.get(cat, [])
        if not rs:
            continue
        ttfts = [r["ttft_s"] for r in rs if r["ttft_s"] is not None]
        print(f"{cat:<16}{len(rs):>5}"
              f"{(pct(ttfts, .5) if ttfts else 0):>10.3f}s"
              f"{pct([r['latency_s'] for r in rs], .5):>10.2f}s"
              f"{sum(r['prompt_tokens'] or 0 for r in rs):>12,}"
              f"{sum(r['completion_tokens'] or 0 for r in rs):>10,}")

    tot_in = sum(r["prompt_tokens"] or 0 for r in ok)
    tot_out = sum(r["completion_tokens"] or 0 for r in ok)
    print(f"\n吞吐: 输入 {tot_in/wall:,.0f} tok/s, "
          f"输出 {tot_out/wall:,.0f} tok/s, "
          f"合计 {(tot_in+tot_out)/wall:,.0f} tok/s")


def main():
    ap = argparse.ArgumentParser(description="KV cache 追踪用的正式 workload")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="同时在跑的请求数（信号量窗口）")
    ap.add_argument("--model-path", default="/workspace/models/Qwen2.5-7B-Instruct",
                    help="本地模型目录，只用来加载分词器")
    ap.add_argument("--served-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default=None, help="客户端侧计时结果写到这里")
    ap.add_argument("--kv-pool-tokens", type=int, default=None,
                    help="KV cache 池容量（来自服务端日志），用于打印足迹占比")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印请求集，不发请求")
    args = ap.parse_args()

    tok = load_tokenizer(args.model_path)
    requests = build_workload(tok)
    report_workload(requests, tok, args.kv_pool_tokens)

    if args.dry_run:
        print("\n--- 前 5 个请求（已打乱后的顺序） ---")
        for r in requests[:5]:
            n = len(tok.encode(r["prompt"]))
            print(f"  {r['id']:<18} prompt={n:>5} tok, out={r['max_tokens']:>4} tok")
        return

    print(f"\n开始发送: {len(requests)} 个请求, 并发度 {args.concurrency}")
    results, wall = asyncio.run(
        run(requests, args.concurrency, args.base_url, args.served_name))
    summarize(results, wall, args.concurrency)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "concurrency": args.concurrency,
                "wall_s": round(wall, 3),
                "seed": SEED,
                "n_requests": len(requests),
                "results": results,
            }, f, indent=2)
        print(f"\n客户端计时已写入 {args.out}")


if __name__ == "__main__":
    main()
