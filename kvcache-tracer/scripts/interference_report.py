#!/usr/bin/env python3
"""合并三段补测，重算「藏得住迁移的最大速率」这根红线。

三段是怎么来的
------------------------------------------------------------
* **main**  d2h/h2d/both/sham × 22% 档位一整套。但它的负载进程是在修 sham
  限速 bug **之前**启动的，所以它的 sham 段无效（budget=0 → 空转）。
  d2h/h2d/both 不受影响（实测达成速率与目标逐档吻合）。
* **sham** 用修好的 loadgen 单独补跑。
* **low**  主跑最低档是 5.5 GB/s，而那一档就已经「看得见」了，红线还没夹住。
  补跑 1.1 / 2.2 / 4.4 / 8.8 GB/s 把低端填上。

红线的定义
------------------------------------------------------------
每档与**紧邻前一个** baseline 配对（不跟全局 baseline 比，避免把漂移算成效应），
取 Δ中位步长。噪声底 = 所有 baseline 中位数的跨度（max−min）—— 这是**实测**的
分辨力，不是假设值。红线 = 最大的「Δ 仍在噪声底以内」的速率。

比"线性外推到 0"保守：不假定效应随速率线性，只看哪一档开始真的看得出来。
"""
import argparse
import bisect
import contextlib
import csv
import glob
import json
import os
import re
import statistics
import sys


class _Tee:
    """同时写终端和文件，省得把分析再跑一遍才能存下来。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()

ROOT = "/workspace/disaggregated-kvcache/kvcache-tracer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))

FLOAT_COLS = {"level", "step_med_ms", "step_p95_ms", "step_p99_ms",
              "step_period_ms", "bimodal_gap_ms", "itl_med_us", "itl_p95_us",
              "ach_gbps", "target_gbps"}
BLOCK_BYTES = 917504


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in FLOAT_COLS:
                if k in r:
                    try:
                        r[k] = float(r[k])
                    except (TypeError, ValueError):
                        r[k] = float("nan")
            for k in ("t0", "t1", "n_steps", "n_itl"):
                if k in r:
                    r[k] = int(float(r[k]))
            rows.append(r)
    return rows


def rehydrate(r):
    """CSV 行 → report() 期望的 dict 形状。"""
    return {
        "dir": r["dir"], "level": r["level"], "kind": r["kind"],
        "t0": r["t0"], "t1": r["t1"],
        "step": {"n": r["n_steps"], "med_ms": r["step_med_ms"],
                 "p95_ms": r["step_p95_ms"], "p99_ms": r["step_p99_ms"],
                 "med_period_ms": r["step_period_ms"],
                 "bimodal_gap_ms": r["bimodal_gap_ms"]},
        "itl_med": r["itl_med_us"], "itl_p95": r["itl_p95_us"],
        "ach_gbps": r["ach_gbps"],
        "lg": ({"achieved_gbps": r["ach_gbps"]}
               if r["kind"] == "load" and r["level"] > 0 else None),
    }


def pairs_for(blocks, direction):
    """按速率升序给出 (load, base) 配对。"""
    recs = [b for b in blocks if b["dir"] == direction]
    out = []
    for i, b in enumerate(recs):
        if b["kind"] != "load" or b["level"] == 0 or not b["step"]["n"]:
            continue
        base = next((recs[j] for j in range(i - 1, -1, -1)
                     if recs[j]["kind"] == "base" and recs[j]["step"]["n"]), None)
        if base:
            out.append((b, base))
    return out


def red_line(blocks, ceiling):
    bases = [b for b in blocks if b["kind"] == "base" and b["step"]["n"]]
    meds = [b["step"]["med_ms"] for b in bases]
    noise = max(meds) - min(meds)
    print("=" * 78)
    print("红线：藏得住迁移的最大速率")
    print("=" * 78)
    print(f"  baseline 步中位数 {min(meds):.3f} – {max(meds):.3f} ms"
          f"  → 噪声底 {noise:.3f} ms ({noise / min(meds) * 100:.2f}%)")
    print(f"  判据：|Δ中位步长| > 噪声底 = 看得见\n")
    print(f"  {'方向':<5}{'目标':>7}{'实际GB/s':>10}{'block/s':>9}"
          f"{'Δ中位ms':>10}{'Δ%':>7}{'ΔITL ms':>9}  判定")
    print("  " + "-" * 66)

    result = {}
    for direction in ("d2h", "h2d", "both", "sham"):
        hidden_max = 0.0
        for load, base in pairs_for(blocks, direction):
            d = load["step"]["med_ms"] - base["step"]["med_ms"]
            pct = d / base["step"]["med_ms"] * 100
            itl = (load["itl_med"] - base["itl_med"]) / 1000.0
            rate = load["ach_gbps"]
            vis = abs(d) > noise
            if not vis and rate > hidden_max:
                hidden_max = rate
            print(f"  {direction:<5}{load['level'] * ceiling:>7.1f}{rate:>10.2f}"
                  f"{rate * 1e9 / BLOCK_BYTES:>9.0f}{d:>+10.3f}{pct:>+7.2f}"
                  f"{itl:>+9.3f}  {'看得见' if vis else '藏得住'}")
        print()
        result[direction] = hidden_max

    print("  红线（该方向仍「藏得住」的最高实测速率）：")
    for d in ("d2h", "h2d", "both"):
        g = result[d]
        rates = [l["ach_gbps"] for l, _ in pairs_for(blocks, d)]
        if g <= 0:
            print(f"    {d:<5} 最低实测档（{min(rates):.2f} GB/s）就已经看得见"
                  f" —— 红线低于它")
        else:
            print(f"    {d:<5} < {g:5.2f} GB/s ≈ {g * 1e9 / BLOCK_BYTES:6.0f} block/s"
                  f"   一步(15.6ms)可藏 {g * 1e9 * 0.0156 / BLOCK_BYTES:5.1f} block")
    # sham 是零字节对照，没有「速率」可言
    sham_pairs = pairs_for(blocks, "sham")
    if sham_pairs:
        worst = max(abs(l["step"]["med_ms"] - b["step"]["med_ms"])
                    for l, b in sham_pairs)
        print(f"    sham  零字节对照（不做任何 DMA）—— 全程最大 |Δ| = "
              f"{worst:.3f} ms，"
              f"{'全部落在噪声底内' if worst <= noise else '**超出噪声底**'}"
              f" → 效应来自 DMA 本身，不是 CPU 侧的循环")
    return result, noise


def exec_decompose(blocks, keeps):
    """按 block 把 完整步 / 执行段 / 其余 拆开。

    执行段来自 `--enable-logging-iteration-details`，它包的是 `future.result()`
    （core.py:495-499）—— 即「等模型执行返回」那一段。用 plugin 记的 `iter`
    精确 join（两者共用 EngineCore 的 `_iteration_index`）。

    主跑和低速率补跑是两次独立的 vllm serve，各自的 iteration 日志与 trace 分开
    备份；按 block 的 t0 落在哪一次的时间范围里来选。
    """
    pat = re.compile(r"Iteration\((\d+)\).*iteration elapsed time: ([0-9.]+) ms")
    runs = []
    for keep in keeps:
        log = os.path.join(keep, "interference.log")
        if not os.path.exists(log):
            continue
        ex = {}
        with open(log, errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    ex[int(m.group(1))] = float(m.group(2))
        tr = []
        for p in glob.glob(os.path.join(keep, "traces-step", "interference*.jsonl")):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        if r.get("iter") is not None:
                            tr.append(r)
        if not tr:
            continue
        tr.sort(key=lambda r: r["t"])
        runs.append({"ex": ex, "tr": tr, "ts": [r["t"] for r in tr],
                     "lo": tr[0]["t"], "hi": tr[-1]["t"]})
    if not runs:
        return None

    def stats(t0, t1):
        for run in runs:
            if not (run["lo"] <= t0 <= run["hi"]):
                continue
            lo = bisect.bisect_left(run["ts"], t0)
            hi = bisect.bisect_right(run["ts"], t1)
            full = [r["dur_ns"] / 1e6 for r in run["tr"][lo:hi]]
            exe = [run["ex"][r["iter"]] for r in run["tr"][lo:hi]
                   if r["iter"] in run["ex"]]
            if not full or not exe:
                return None
            return statistics.median(full), statistics.median(exe)
        return None

    print("\n" + "=" * 78)
    print("机制分解：负载落在哪一段")
    print("=" * 78)
    print("  执行段 = future.result() 的等待（GPU 执行）；其余 = schedule + "
          "update_from_output")
    print(f"\n  {'方向':<5}{'实际GB/s':>10}{'完整步Δ':>10}{'执行段Δ':>10}"
          f"{'其余Δ':>9}{'执行/完整':>11}")
    print("  " + "-" * 62)
    for direction in ("d2h", "h2d", "both"):
        for load, base in pairs_for(blocks, direction):
            a, b = stats(int(load["t0"]), int(load["t1"])), \
                stats(int(base["t0"]), int(base["t1"]))
            if not a or not b:
                continue
            df, de = a[0] - b[0], a[1] - b[1]
            print(f"  {direction:<5}{load['ach_gbps']:>10.2f}{df:>+10.3f}"
                  f"{de:>+10.3f}{df - de:>+9.3f}{a[1] / a[0]:>11.3f}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="measurements/pcie_interference_main.csv",
                    help="主 run 的 CSV（**未合并过的**原始那份）")
    ap.add_argument("--sham", default="measurements/pcie_interference_sham.csv")
    ap.add_argument("--low", default="measurements/pcie_interference_low.csv")
    ap.add_argument("--out", default="measurements/pcie_interference.csv")
    ap.add_argument("--report", default="measurements/pcie_interference.txt")
    ap.add_argument("--keep", default="",
                    help="各 run 的 iteration 日志与 step trace 备份目录（逗号分隔）。"
                         "留空则跳过机制分解那节 —— 那节需要当时保下来的 "
                         "logs-step/interference.log 与 traces-step/，而它们会被"
                         "下一次 measure_interference.py 启动时覆盖。")
    ap.add_argument("--ceiling-gbps", type=float, default=22.0)
    ap.add_argument("--summary", default="measurements/pcie_interference_redline.txt")
    a = ap.parse_args()

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    rows = load_csv(resolve(a.main))
    # 主 run 的 sham 段是在修限速 bug 之前跑的（budget=0 → 空转），整段丢弃，
    # 用后面单独补跑的那份。d2h/h2d/both 不受影响，逐档达成速率与目标吻合。
    stale = [r for r in rows if r["dir"] == "sham"]
    rows = [r for r in rows if r["dir"] != "sham"]
    print(f"  + main  {len(rows):>4} 行  ({a.main})   [丢弃无效 sham {len(stale)} 行]")
    for label, path in (("sham", a.sham), ("low", a.low)):
        p = resolve(path)
        if os.path.exists(p):
            extra = load_csv(p)
            print(f"  + {label:<5} {len(extra):>4} 行  ({path})")
            rows += extra
        else:
            print(f"  ! 缺 {label}：{path}   （低速率补测可能还没跑完）")

    order = {"d2h": 0, "h2d": 1, "both": 2, "sham": 3}
    rows.sort(key=lambda r: (order.get(r["dir"], 9), r["level"], r["t0"]))
    with open(resolve(a.out), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写 {a.out}（{len(rows)} 行）\n")

    blocks = [rehydrate(r) for r in rows]
    # 红线与机制分解同时落盘一份 —— GUIDE 引用的就是这两个数，得能单独查。
    with open(resolve(a.summary), "w") as fh:
        with contextlib.redirect_stdout(_Tee(sys.stdout, fh)):
            red_line(blocks, a.ceiling_gbps)
            if a.keep:
                exec_decompose(blocks, [x for x in a.keep.split(",") if x])
            else:
                print("\n（跳过机制分解：未给 --keep）")
    print(f"已写 {a.summary}")

    import measure_interference as mi
    ns = argparse.Namespace(dirs="d2h,h2d,both,sham", chunk_blocks=8,
                            block_s=8.0, pairs=3, ceiling_gbps=a.ceiling_gbps)
    mi.report(resolve(a.report), blocks, ns)
    print(f"\n已写 {a.report}")


if __name__ == "__main__":
    main()
