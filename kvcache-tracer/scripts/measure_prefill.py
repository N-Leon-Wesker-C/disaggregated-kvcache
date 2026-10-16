"""
measure_prefill.py — 实测 prefill 的边际成本，用来给「重算一个 block 多贵」定价

为什么必须实测
============================================================

replay 里「重算一个 block」的代价决定了整个换出策略的结论方向：

    重算贵  → 宁可花 PCIe 换回来（迁移层有价值）
    重算便宜 → 不如直接算，迁移层白折腾

这个数原本一直是拍的，而且是**决定结论方向**的那个数。

**实测结果（2026-10-15）**：90 秒就能测完，答案是

    TTFT ≈ 9.79 ms + 101.27 µs × prompt_tokens      → 9,874 tok/s

反推 150 TFLOPS = 4090 bf16 峰值的 91%，说明确实测到了 compute-bound 的纯
prefill。这个值（101.27）比原来用的 38.5 高 **2.6 倍**，重算一个 16-token 块
因此从 616 µs 修正到 1,620 µs。

（当时按假定的 24 GB/s 算出换回是 38.2 µs/块，优势 42×。后来 PCIe 那一侧
也测了，见 `scripts/measure_pcie.py`：生产形状下是 52.4 µs/块，所以最终是
**31×**。两次修正方向相反，都是把估计值换成实测值。）

注意方向：我原本预计 prefill 因为是批量矩阵乘会比 decode **便宜**，结论会
被削弱甚至反转。实测是反的。这也是为什么这个数必须测而不是推 —— 两种推理都
能讲得通，只有测出来是真的。

反向的估计同样不可信：早先用 TTFT 中位数配 prompt 中位数拟合出 25,974 tok/s，
反推 396 TFLOPS，超过这张卡 165 TFLOPS 的峰值 —— 说明那个口径里混进了
队列等待、采样等非 prefill 的开销（而且把中位延迟配中位长度，本来就不是
同一个请求）。

方法
============================================================

同一批请求只改 prompt 长度，测 TTFT，做线性拟合：

    TTFT ≈ a + L / R_prefill

斜率就是边际 prefill 成本。用**差分**而不是绝对值，是因为 a（网络、排队、
调度）与 L 无关，拟合时会被截距吸收掉。

几条必须守住的：
  * 每次只发一个请求（C=1），否则测到的是批处理吞吐不是边际成本
  * prompt 必须唯一，否则命中 prefix cache，prefill 根本不发生
  * max_tokens=1，把 decode 压到可忽略
  * 长度要够长（2048+）才进入 compute-bound 区间

用法:
    python scripts/measure_prefill.py --sizes 512,1024,2048,3072,4096 --reps 3
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time

sys.path.insert(0, sys.path[0] + "/../workload")
import workloads  # noqa: E402

#: Qwen2.5-7B 参数量，用来把测到的吞吐换算成 TFLOPS 做合理性检查
N_PARAMS = 7.6e9


async def one(session, base_url, model, prompt, max_tokens=1):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.time()
    ttft = None
    usage = None
    async with session.post(f"{base_url}/v1/completions", json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
        async for raw in resp.content:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            ch = json.loads(body)
            if ch.get("usage"):
                usage = ch["usage"]
                continue
            choices = ch.get("choices") or []
            if choices and choices[0].get("text") and ttft is None:
                ttft = time.time() - t0
    return ttft, (usage or {}).get("prompt_tokens")


async def run(args):
    import aiohttp
    tok = workloads.load_tokenizer(args.model_path)
    rng = random.Random(20261015)
    vocab = workloads.VOCAB_DOC

    # 每个长度单独构造，且逐次重新随机 —— 复用同一段文本会命中 prefix cache，
    # 那样测到的是缓存命中后的 TTFT，根本没有 prefill 发生。
    plan = [(L, r) for L in [int(x) for x in args.sizes.split(",")]
            for r in range(args.reps)]

    results = {}
    timeout = aiohttp.ClientTimeout(total=600, sock_read=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for L, r in plan:
            prompt = workloads.build_text(L, rng, tok, vocab)
            n = len(tok.encode(prompt))
            ttft, ptok = await one(session, args.base_url, args.served_name, prompt)
            results.setdefault(L, []).append(ttft)
            print(f"  L={L:>5} (实测 {n:>5} tok)  第{r+1}次  TTFT {ttft * 1000:>8.2f} ms"
                  f"  usage={ptok}")

    # 线性拟合 TTFT = a + L / R。斜率取倒数就是 µs/token。
    pts = [(L, statistics.median(ts)) for L, ts in sorted(results.items())]
    n = len(pts)
    mx = sum(L for L, _ in pts) / n
    my = sum(t for _, t in pts) / n
    denom = sum((L - mx) ** 2 for L, _ in pts)
    slope = sum((L - mx) * (t - my) for L, t in pts) / denom if denom else 0.0
    intercept = my - slope * mx

    print()
    print("=" * 70)
    print("拟合结果")
    print("=" * 70)
    print(f"  TTFT ≈ {intercept * 1000:.2f} ms + {slope * 1e6:.4f} µs × prompt_tokens")
    print(f"  边际 prefill 成本 = {slope * 1e6:.4f} µs/token")
    print(f"  → {1 / slope:,.0f} tok/s")
    print(f"  固定开销（网络+排队+调度）= {intercept * 1000:.2f} ms")
    if slope > 0:
        tflops = 2 * N_PARAMS / slope / 1e12
        print(f"  反推算力 = {tflops:,.0f} TFLOPS"
              f"（RTX 4090 bf16 峰值约 165 TFLOPS）")
        if tflops > 165:
            print("  ⚠ 超过峰值 —— 拟合不可信，多半混进了排队或批处理效应")
        elif tflops < 20:
            print("  ⚠ 远低于峰值 —— 可能没进入 compute-bound 区间，长度再加")
    print()
    print("  把这个数填回 replay：")
    print(f"    RECOMPUTE_US_PER_TOKEN = {slope * 1e6:.4f}")
    old = 38.5
    print(f"    （原来拍的是 {old}，相差 {old / (slope * 1e6):.1f}×）")
    return 0


def main():
    ap = argparse.ArgumentParser(description="实测 prefill 边际成本")
    ap.add_argument("--model-path", default="/workspace/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--served-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--sizes", default="512,1024,2048,3072,4096")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
