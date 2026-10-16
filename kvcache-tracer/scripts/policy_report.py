#!/usr/bin/env python3
"""汇总真实系统上的逐出策略对比（results-policy/）。

数据是怎么来的
------------------------------------------------------------
`workload/run_policy_compare.sh` 每个策略起一个服务，每次重复前在计数流里插一条
`{"marker": "rep"}` 界标。观测器（plugin/vllm_kvpolicy.py）每 2s 追加一条累计快照。
本脚本按界标切段：段前最后一条 = 本轮起点，段后最后一条 = 本轮终点，差值就是
**这一轮**的计数。

为什么不能只在结束时读一次
------------------------------------------------------------
每个策略只起一个服务、跑多次重复，计数器是进程级的累计量。不切段的话第 2 轮会
把第 1 轮的数字一起算进去。

主指标
------------------------------------------------------------
`load_blocks` —— 从 CPU 池换回来的块数。**这个数才代表 CPU 池有用**。

坑：`store_evicted`（换出次数）不是指标。GPU 池装得下工作集时，offload 层根本
不会被查询（它只在 GPU prefix cache 落空时才被问到），于是换出计数一切正常、
`load_blocks` 恒为 0 —— 报表看着像成功，其实四个策略完全一样。本脚本会在
`load_blocks == 0` 时显式报出来。
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

ROOT = "/workspace/disaggregated-kvcache/kvcache-tracer"

COUNTERS = ("lookup_hit", "lookup_pending", "lookup_miss",
            "store_blocks", "store_evicted", "store_dropped", "load_blocks")


def _last_active(events):
    """窗口内按 pid 分组，取活跃度最高的那个 pid 的**最后一条**快照。

    坑（实测踩到）：一个 vLLM 服务有多个进程，而早期版本的观测器在每个进程里
    都开了同一个文件。frontend 从来不建 `CPUOffloadingManager`，它那份计数器
    恒为 0，但**物理上写在最后** —— 直接取"最后一条"取到的就是它，于是四个策略
    的差值全是 0，报告看着干净，结论是"四个策略完全一样"。真正有数据的
    EngineCore 那一行被后写的 0 盖住了。

    改成按 pid 分组、比活跃度，写文件的是谁就不再重要了。
    """
    per_pid = {}
    for kind, rec in events:
        if kind == "snap":
            per_pid[rec.get("pid")] = rec
    if not per_pid:
        return None, None
    pid, rec = max(per_pid.items(),
                   key=lambda kv: sum(kv[1].get(k, 0) for k in COUNTERS))
    return rec, (pid if len(per_pid) > 1 else None)


def _norm(name):
    """把策略标识归一到可比的形式：`LFUCachePolicy` / `lfu` → `lfu`。"""
    return (name or "").strip().lower().removesuffix("cachepolicy")


def load_segments(path):
    """按 marker 切段，返回 [(policy, rep, before, after, pids)]。

    `pids` 非空表示这一段的窗口里出现了**不止一个** pid 在写 —— 计数流被污染
    过或进程重启过，此时前后两条快照未必来自同一个进程，差值不可信，要在报告里
    显式说出来。
    """
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            events.append(
                ("marker" if rec.get("marker") == "rep" else "snap", rec))

    marks = [i for i, (k, _) in enumerate(events) if k == "marker"]
    segs = []
    for n, i in enumerate(marks):
        lo = marks[n - 1] if n else 0
        hi = marks[n + 1] if n + 1 < len(marks) else len(events)
        meta = events[i][1]
        before, pids_b = _last_active(events[lo:i])
        after, pids_a = _last_active(events[i + 1:hi])
        segs.append({"policy": meta["policy"], "rep": meta["rep"],
                     "before": before, "after": after,
                     "pids": pids_a or pids_b})
    return segs


def delta(after, before):
    if after is None:
        return None
    before = before or {}
    out = {}
    for k in COUNTERS:
        out[k] = after.get(k, 0) - before.get(k, 0)
    return out


def load_run(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def pct(vals, q):
    if not vals:
        return float("nan")
    vals = sorted(vals)
    i = min(int(q * len(vals)), len(vals) - 1)
    return vals[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results-policy")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="measurements/policy_compare.md")
    a = ap.parse_args()

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    rdir = resolve(a.results_dir)
    rows, notes = [], []
    for path in sorted(glob.glob(os.path.join(rdir, "*.counters.jsonl"))):
        policy = os.path.basename(path).split(".")[0]
        for seg in load_segments(path):
            if seg["pids"] is not None:
                notes.append(
                    f"! {policy} rep{seg['rep']}：这段窗口里有多个 pid 在写"
                    f"（{seg['pids']} 等）—— 进程可能重启过，前后快照未必同源")
            got = (seg["after"] or {}).get("policy")
            if got and _norm(got) != _norm(policy):
                notes.append(
                    f"! {policy} rep{seg['rep']}：计数文件自称策略是 {got}，"
                    f"与文件名不符 —— 配置可能没生效")
            d = delta(seg["after"], seg["before"])
            if d is None:
                print(f"  ! {policy} rep{seg['rep']}：没有终点快照，跳过")
                continue
            meta = load_run(os.path.join(
                rdir, f"{policy}.c{a.concurrency}.r{seg['rep']}.json"))
            ttfts, lats, ok = [], [], 0
            if meta:
                for r in meta.get("results", []):
                    if r.get("ok") and r.get("ttft_s") is not None:
                        ttfts.append(r["ttft_s"])
                        lats.append(r["latency_s"])
                        ok += 1
            rows.append({"policy": policy, "rep": seg["rep"], **d,
                         "evict_fp": (seg["after"] or {}).get("evict_fp"),
                         "tau_s": (seg["after"] or {}).get("tau_s"),
                         "n_ok": ok, "ttft": ttfts, "lat": lats, "meta": meta})

    if not rows:
        print("没有数据：results-policy/ 里没有带界标的计数文件。")
        return 1

    lines = ["# 真实系统上的逐出策略对比", ""]
    cfg = rows[0]["meta"]["server"] if rows[0]["meta"] else {}
    spec = rows[0]["meta"]["workload_spec"] if rows[0]["meta"] else {}
    lines += [
        f"workload={spec.get('name', '?')} 并发={a.concurrency} "
        f"GPU 池={cfg.get('kv_pool_blocks')} 块 block_size={cfg.get('block_size')} "
        f"max_model_len={cfg.get('max_model_len')}",
        "",
    ]
    if notes:
        lines += ["## 数据来源的告警", ""] + [f"- {n}" for n in notes] + [""]
    lines += [
        "## 每次重复的原始计数",
        "",
        "| 策略 | rep | CPU命中块 | 扫描命中 | 扫描落空 | 存入 | 换出 | 塞不下 | "
        "TTFT P50 | TTFT P90 | 逐出指纹 | tau |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fp = "—" if r["evict_fp"] is None else str(r["evict_fp"])
        tau = "—" if r["tau_s"] is None else f"{r['tau_s']}s"
        lines.append(
            f"| {r['policy']} | {r['rep']} | {r['load_blocks']} | "
            f"{r['lookup_hit']} | {r['lookup_miss']} | {r['store_blocks']} | "
            f"{r['store_evicted']} | {r['store_dropped']} | "
            f"{pct(r['ttft'], .5) * 1000:.1f}ms | {pct(r['ttft'], .9) * 1000:.1f}ms | "
            f"{fp} | {tau} |")

    # 按策略聚合
    pols = []
    for r in rows:
        if r["policy"] not in pols:
            pols.append(r["policy"])
    lines += ["", "## 按策略聚合（重复之间取中位数）", "",
              "| 策略 | CPU命中块 | 扫描落空 | 换出 | TTFT P50 | TTFT P90 |",
              "|---|---|---|---|---|---|"]
    agg = {}
    for p in pols:
        rs = [r for r in rows if r["policy"] == p]
        med = lambda k: st.median([r[k] for r in rs])  # noqa: E731
        tt = [v for r in rs for v in r["ttft"]]
        agg[p] = {"load_blocks": med("load_blocks"),
                  "lookup_miss": med("lookup_miss"),
                  "store_evicted": med("store_evicted"),
                  "ttft50": pct(tt, .5), "ttft90": pct(tt, .9)}
        lines.append(f"| {p} | {agg[p]['load_blocks']:.0f} | "
                     f"{agg[p]['lookup_miss']:.0f} | {agg[p]['store_evicted']:.0f} | "
                     f"{agg[p]['ttft50'] * 1000:.1f}ms | "
                     f"{agg[p]['ttft90'] * 1000:.1f}ms |")

    # 逐出指纹（顺序敏感的 crc32）：想回答"计数相同是不是等于决策相同"。
    #
    # 结论是**回答不了**：实测 12 次运行给出 12 个互不相同的指纹，包括同一个策略
    # 的 3 次重复之间 —— 逐出序列本来就逐次不可复现。所以"两个策略指纹不同"不能
    # 用来证明它们的决策不同（同一个策略也这样）。它证明的是另一个更有意思的事：
    # **逐出序列逐次不可复现，而这些计数逐次几乎完全相同** —— 计数是一个随机过程
    # 的很粗的稳定投影。
    #
    # 因此这里只如实报出"重复之间是否一致"，**不据此下"决策相同/不同"的结论**。
    fps = {}
    for r in rows:
        if r["evict_fp"] is not None:
            fps.setdefault(r["policy"], set()).add(r["evict_fp"])
    if fps:
        lines += ["", "## 逐出决策指纹（顺序敏感的 crc32）", "",
                  "| 策略 | 指纹 | 重复之间是否一致 |", "|---|---|---|"]
        for p in pols:
            if p not in fps:
                continue
            vals = sorted(fps[p])
            same = "是" if len(vals) == 1 else f"**否（{len(vals)} 个）**"
            lines.append(f"| {p} | {' / '.join(str(v) for v in vals)} | {same} |")
        multi = [p for p in fps if len(fps[p]) > 1]
        if multi:
            lines += ["",
                      f"{'、'.join(multi)} 的指纹在重复之间就不一致 —— 说明"
                      f"**逐出序列本身逐次不可复现**，而上面的计数却几乎逐次相同。"
                      f"所以这些计数是一个随机过程的粗投影；两个策略计数相同，"
                      f"**既不能证明也不能否证**它们做了同样的决策。"]

    print("\n".join(lines))

    # 必须显式报出来的失效模式
    dead = [p for p in pols if agg[p]["load_blocks"] == 0]
    if dead:
        print(f"\n⚠️  {', '.join(dead)} 的 CPU 命中块数为 0 —— CPU 池只写不读，"
              f"策略差异被掩盖了。把 GPU 池压到工作集以下再来一次。")

    tied = []
    for i, pa in enumerate(pols):
        for pb in pols[i + 1:]:
            if (agg[pa]["load_blocks"] == agg[pb]["load_blocks"]
                    and agg[pa]["store_evicted"] == agg[pb]["store_evicted"]):
                tied.append((pa, pb))
    for pa, pb in tied:
        fa, fb = fps.get(pa, set()), fps.get(pb, set())
        if fa and fb and fa != fb:
            print(f"\n⚠️  {pa} 与 {pb} 的计数逐位相同，指纹不同 —— 但**这个指纹"
                  f"证明不了什么**：同一个策略重复之间指纹也不同（见上表），"
                  f"逐出序列本来就逐次不可复现。所以'计数相同'到底是巧合还是"
                  f"两个策略真的一样，**本轮数据分不出来**。别把它读成"
                  f"'两个策略效果一样'，也别读成'决策不同'。")

    os.makedirs(os.path.dirname(resolve(a.out)), exist_ok=True)
    with open(resolve(a.out), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
