"""
run.py — 跑一类 workload 的一个 (并发度, 重复序号) 组合，并落一份完整元数据

用法:
    python workload/run.py --workload short-qa --concurrency 4 --rep 1 \
        --kv-pool-tokens 124960 --out results/short-qa.c4.r1.json

产出的 JSON 里既有客户端侧计时，也有本次运行的**全部实验条件**（模型、vLLM
版本、池预算、block size、max model length、并发度、token 分布、prefix cache
开关、随机种子）。这是实验协议要求的记录 —— 缺任何一项，结果都无法被复现或
和其他组对比。

服务端需要已经起来（见 run_experiment.sh）。
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workloads  # noqa: E402


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        name, mem = [x.strip() for x in out.split(",")[:2]]
        return {"name": name, "memory_total_mib": int(mem.split()[0])}
    except Exception:                                   # noqa: BLE001
        return {}


def vllm_version():
    try:
        import vllm
        return vllm.__version__
    except Exception:                                   # noqa: BLE001
        return "unknown"


async def _one(session, sem, base_url, model, req, max_retries=2):
    """发一个请求，返回客户端侧计时。

    信号量在请求内部获取，in-flight 窗口恰好等于并发度：前 C 个立刻发出，
    之后每完成一个补一个。

    重试是必要的：单次网络抖动或服务端瞬时压力会让一个请求失败，
    而失败了整个 (workload, 并发度, 重复) 组合就少了一个样本，三次重复的
    统计就不再可比。重试让失败变成可恢复的。
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
    # （entrypoints/serve/engine/serving.py:_base_request_id），实际得到的是
    # "cmpl-short-qa-007-0-957e927f" 这样包了一层的 id。标签嵌在中间，
    # 分析脚本按子串匹配就能归类。
    headers = {"X-Request-Id": req["run_id"]}

    async with sem:
        last_err = None
        for attempt in range(max_retries + 1):
            t_submit = time.time()
            ttft = None
            usage = None
            try:
                async with session.post(f"{base_url}/v1/completions",
                                        json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        last_err = f"HTTP {resp.status}: {body[:200]}"
                        continue
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
            except Exception as exc:                    # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                continue

            done = time.time()
            return {
                "id": req["id"], "run_id": req["run_id"],
                "workload": req["workload"],
                "max_tokens": req["max_tokens"], "ok": True,
                "attempt": attempt,
                "submit_t": round(t_submit, 6),
                "ttft_s": round(ttft, 6) if ttft is not None else None,
                "latency_s": round(done - t_submit, 6),
                "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "completion_tokens": (usage or {}).get("completion_tokens"),
            }

        return {
            "id": req["id"], "run_id": req["run_id"],
            "workload": req["workload"],
            "max_tokens": req["max_tokens"], "ok": False,
            "error": last_err, "attempt": max_retries,
        }


async def run_all(requests, concurrency, base_url, model):
    import aiohttp
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=1800, sock_read=600)
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *[_one(session, sem, base_url, model, r) for r in requests])
    return results, time.time() - t0


def pct(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(len(vals) * q))]


def summarize(results, wall):
    ok = [r for r in results if r["ok"]]
    out = {"n_ok": len(ok), "n_failed": len(results) - len(ok)}
    if not ok:
        return out
    lat = [r["latency_s"] for r in ok]
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    tin = sum(r["prompt_tokens"] or 0 for r in ok)
    tout = sum(r["completion_tokens"] or 0 for r in ok)
    out.update({
        "latency_s": {"p50": round(pct(lat, .5), 3),
                      "p90": round(pct(lat, .9), 3),
                      "max": round(max(lat), 3)},
        "ttft_s": {"p50": round(pct(ttfts, .5), 4),
                   "p90": round(pct(ttfts, .9), 4)} if ttfts else None,
        "input_tokens": tin, "output_tokens": tout,
        "input_tok_per_s": round(tin / wall, 1),
        "output_tok_per_s": round(tout / wall, 1),
        "total_tok_per_s": round((tin + tout) / wall, 1),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True,
                    choices=sorted(workloads.WORKLOADS))
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--rep", type=int, default=1, help="重复序号，从 1 开始")
    ap.add_argument("--model-path", default="/workspace/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--served-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--kv-pool-tokens", type=int, default=None,
                    help="服务端实测的 KV 池容量（token 数），来自启动日志")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=workloads.BLOCK_SIZE)
    ap.add_argument("--prefix-caching", default="on",
                    choices=["on", "off"])
    ap.add_argument("--prefix-mode", default="single",
                    choices=["single", "skewed"],
                    help="single: 类内共享一段前导；skewed: K 模板 + Zipf")
    ap.add_argument("--trace-glob", default=None,
                    help="本次运行的 trace 文件 glob，写进元数据便于回查")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="只发前 N 个请求。冒烟用 —— 正式运行不要给")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = workloads.load_tokenizer(args.model_path)
    prefixes = workloads.make_prefixes(args.workload, tok, args.prefix_mode)
    requests = workloads.build_requests(args.workload, tok, args.prefix_mode)
    # 请求 id 里带上重复序号：同一会话内三次重复共用一份 trace 文件，事件靠
    # request_id 归属到具体某次重复。重复之间只做 /reset_prefix_cache，不重启
    # 服务（重启要多花 27×50s），所以「哪条事件属于第几次」必须能从 id 读出来。
    for r in requests:
        r["run_id"] = f"{r['id']}-r{args.rep}"
    if args.limit:
        requests = requests[:args.limit]
    # describe 在截断**之后**算：元数据里的 token 分布必须描述真正发出去的那批，
    # 否则冒烟运行的元数据会写着一份它没跑过的分布。
    desc = workloads.describe(requests, tok, args.prefix_mode, prefixes)
    print(workloads.format_description(args.workload, desc,
                                       args.kv_pool_tokens))

    if args.dry_run:
        for r in requests[:5]:
            print(f"   {r['run_id']:<26} prompt={len(tok.encode(r['prompt'])):>5}"
                  f"  out={r['max_tokens']:>4}"
                  f"  template={r['template']:>3}({r['template_blocks']} blocks)")
        return

    print(f"\n开始: {desc['n_requests']} 请求, 并发度 {args.concurrency}, "
          f"第 {args.rep} 次")
    # run_all 是协程，必须 asyncio.run 驱动 —— 少这一层会拿到一个
    # "cannot unpack non-iterable coroutine object"，而且请求一个都不会发出去。
    results, wall = asyncio.run(run_all(requests, args.concurrency,
                                        args.base_url, args.served_name))
    summ = summarize(results, wall)
    print(f"完成: {summ['n_ok']}/{len(results)} 成功, 墙钟 {wall:.1f}s")
    if summ.get("total_tok_per_s"):
        print(f"  吞吐 {summ['total_tok_per_s']:,.0f} tok/s "
              f"(输入 {summ['input_tok_per_s']:,.0f} / "
              f"输出 {summ['output_tok_per_s']:,.0f})")
        print(f"  延迟 P50 {summ['latency_s']['p50']}s "
              f"P90 {summ['latency_s']['p90']}s")
    if summ["n_failed"]:
        print(f"  失败 {summ['n_failed']} 个:")
        for r in [x for x in results if not x["ok"]][:3]:
            print(f"    {r['id']}: {r.get('error')}")

    # ---- 完整实验条件。协议要求逐项记录，缺一项结果就不可比 ----
    meta = {
        "run": {
            "workload": args.workload,
            "concurrency": args.concurrency,
            "rep": args.rep,
            "wall_s": round(wall, 3),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "model": {"served_name": args.served_name,
                  "path": args.model_path},
        "vllm_version": vllm_version(),
        "hardware": gpu_info(),
        "server": {
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_pool_tokens": args.kv_pool_tokens,
            "kv_pool_blocks": (args.kv_pool_tokens // args.block_size
                               if args.kv_pool_tokens else None),
            "block_size": args.block_size,
            "max_model_len": args.max_model_len,
            "prefix_caching": args.prefix_caching == "on",
        },
        "workload_spec": {
            "seed": workloads.SEED,
            "prompt_range": list(workloads.WORKLOADS[args.workload]["prompt_range"]),
            "gen_range": list(workloads.WORKLOADS[args.workload]["gen_range"]),
            "skew_alpha": workloads.SKEW_ALPHA,
            **desc,
        },
        # 逐请求的复用结构。离线 replay 靠这一份重建"哪个请求期望命中哪些
        # block"，不需要分词器、也不会和生成逻辑漂移。template == -1 表示
        # 这个请求的 prompt 装不下任何模板，没有可复用前导。
        "schedule": [
            {"id": r["id"], "template": r["template"],
             "template_blocks": r["template_blocks"],
             "prompt_blocks": -(-len(tok.encode(r["prompt"]))
                                // workloads.BLOCK_SIZE),
             "max_tokens": r["max_tokens"]}
            for r in requests
        ],
        "client": summ,
        "trace_glob": args.trace_glob,
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"元数据 → {args.out}")


if __name__ == "__main__":
    main()
