#!/usr/bin/env python3
"""
measure_interference.py — Step 2：迁移流量对 decode 循环的干扰曲线

产出：`迁移强度 → decode 每步多花多少`，写进 GUIDE Step 4.6。

设计要点（每一条都是为了避免一个具体会给出「自信但错误」答案的坑）
============================================================

1. **主信号是完整步，不是 GPU 执行段。**
   `--enable-logging-iteration-details` 那个日志按构造只覆盖模型执行段，
   **排除了 schedule**。而 PCIe 迁移流量最可能的落点恰恰是 schedule / CPU
   那一侧（D2H 占主机 DRAM 带宽、IOMMU 与 DMA 映射吃 CPU）。只信那个日志
   会测出「几乎没影响」而用户可见的 ITL 明明在动。主信号因此改用
   `EngineCoreProc._process_engine_step`（plugin/vllm_steptrace.py），
   日志只做交叉验证。**两者之差本身就是一等结果。**

2. **强度是连续限速，不是占空比。**
   跑 X ms 停 Y ms 会让每步延迟变成双峰分布，「平均 GB/s」是没有任何单独
   一步真正经历过的数。所以用 token bucket（chunk 周期 ~0.4 ms ≪ 一步 12 ms）。
   分析时检测双峰（p95−p50）作为事后把关。

3. **交错 + 相邻配对。**
   baseline → load → baseline → load …，每个 load block 只跟**紧邻**的
   baseline 比。不做「全部 load 比全部 baseline」——那会把热漂移算成效应。
   baseline block 自身的离散度就是噪声底。

4. **sham 对照。** 同样的循环与 sleeps、同样的迭代次数，但不做拷贝。
   若 sham ≈ 真负载，效应来自 CPU 侧循环本身，结论必须照此改写。

5. **四个载入类型分开**：d2h / h2d / both / sham。不分开就无法归因
   （换出是发射后不管，换回在关键路径上且按步量化）。

用法:
    python scripts/measure_interference.py --pairs 3
    python scripts/measure_interference.py --dirs d2h --levels 0,0.5,1.0 --pairs 2   # 快速冒烟
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "workload"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/workspace/models/Qwen2.5-7B-Instruct"
SERVED = "Qwen2.5-7B-Instruct"
PORT = 8010
CORE_PAT = "VLLM::Engine[C]ore"

#: 与 run_experiment.sh / run_skewed.sh 逐项相同，保证与已有数据同口径
GPU_UTIL = 0.45
MAX_LEN = 8192
BLOCK_SIZE = 16


# ─────────────────────────── 负载进程控制 ───────────────────────────

class LoadGenCtl:
    """常驻负载进程的 stdin/stdout 控制端。

    常驻的理由：每次重启都要重新锁页 ~900 MB（好几秒），而一个测量 block
    只有 8 秒。分配开销只付一次，负载的进程身份整场不变。
    """

    def __init__(self, blocks: int, chunk_blocks: int):
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "pcie_loadgen.py"),
             "--blocks", str(blocks), "--chunk-blocks", str(chunk_blocks)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._done: list[dict] = []
        self._ready = False
        import threading
        self._th = threading.Thread(target=self._reader, daemon=True)
        self._th.start()
        for _ in range(600):                      # 最多等 120 秒（锁页很慢）
            if self._ready:
                break
            time.sleep(0.2)
        if not self._ready:
            raise RuntimeError("负载进程 120 秒内没就绪")

    def _reader(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith("DONE "):
                self._done.append(json.loads(line[5:]))
            elif line.startswith("{"):
                self._ready = True

    def start(self, rate_gbps: float, direction: str, duration_s: float):
        self._done.clear()
        # sham 也要**如实传速率**：它靠这个速率算出与真负载相同的迭代节奏，
        # 只是不做 DMA。传 0 会让它变成不限速空转，对照就失效了。
        payload = "max" if rate_gbps <= 0 else f"{rate_gbps:.4f}"
        self.proc.stdin.write(f"run {payload} {direction} {duration_s}\n")
        self.proc.stdin.flush()

    def collect(self, timeout_s: float) -> dict | None:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if self._done:
                return self._done.pop(0)
            time.sleep(0.05)
        return None

    def stop(self):
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:                                          # noqa: BLE001
            self.proc.kill()


# ─────────────────────────── 服务端 ───────────────────────────

def start_server(step_out: str) -> subprocess.Popen:
    log = os.path.join(ROOT, "logs-step/interference.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    env = dict(os.environ, KVSTEP_OUT=step_out, VLLM_SERVER_DEV_MODE="1",
               HF_HUB_OFFLINE="1")
    fh = open(log, "w")
    p = subprocess.Popen(
        ["vllm", "serve", MODEL, "--served-model-name", SERVED,
         "--gpu-memory-utilization", str(GPU_UTIL),
         "--max-model-len", str(MAX_LEN), "--block-size", str(BLOCK_SIZE),
         "--enable-prefix-caching", "--enable-logging-iteration-details",
         "--port", str(PORT)],
        stdout=fh, stderr=subprocess.STDOUT, env=env)
    return p


def wait_ready(timeout_s: int = 480) -> bool:
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3):
                return True
        except Exception:                                          # noqa: BLE001
            time.sleep(2)
    return False


def stop_server(p: subprocess.Popen):
    p.terminate()
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
    for _ in range(60):
        if subprocess.run(["pgrep", "-f", CORE_PAT],
                          capture_output=True).returncode != 0:
            break
        time.sleep(1)
    # 等显存归还，否则下一轮池容量会偏小
    for _ in range(60):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip()
        try:
            if int(out.split("\n")[0]) >= 47000:
                return
        except ValueError:
            pass
        time.sleep(2)


# ─────────────────────────── 客户端探针 ───────────────────────────

async def probe(session, prompts, deadline_ns: int, itl_sink: list):
    """C=1 连续 decode，逐 chunk 记录到达时刻。

    `ignore_eos=True` 是必须的：否则模型提前结束，请求长度参差，每个 block
    的 token 数都不一样，ITL 的样本量就不可比了。
    """
    tok_book = 0
    while time.time_ns() < deadline_ns:
        prompt = prompts[tok_book % len(prompts)]
        tok_book += 1
        payload = {
            "model": SERVED, "prompt": prompt, "max_tokens": 1200,
            "temperature": 0.0, "ignore_eos": True, "stream": True,
        }
        prev = None
        async with session.post(f"http://127.0.0.1:{PORT}/v1/completions",
                                json=payload) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                ch = json.loads(body)
                if not (ch.get("choices") or []):
                    continue
                now = time.time_ns()
                if prev is not None:
                    itl_sink.append((now - prev) / 1000.0)     # µs
                prev = now
                if now >= deadline_ns:
                    break


# ─────────────────────────── 步时序解析 ───────────────────────────

def load_steps(glob_dir: str, prefix: str) -> list[dict]:
    """读 plugin 写的步时序，只取行数最多的那个文件（EngineCore 进程）。"""
    cands = [os.path.join(glob_dir, f) for f in os.listdir(glob_dir)
             if f.startswith(prefix) and f.endswith(".jsonl")]
    best, best_n = [], 0
    for c in cands:
        rows = []
        try:
            with open(c) as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except OSError:
            continue
        if len(rows) > best_n:
            best, best_n = rows, len(rows)
    return best


def split_stats(steps: list[dict]) -> dict:
    if not steps:
        return {}
    durs = sorted(s["dur_ns"] / 1e6 for s in steps)          # ms

    def pct(p):
        if not durs:
            return float("nan")
        i = min(len(durs) - 1, int(round(p / 100 * (len(durs) - 1))))
        return durs[i]

    # start-to-start 周期：对边界画在哪不敏感，就是 decode 迭代速率本身
    ts = sorted(s["t"] for s in steps)
    periods = [(b - a) / 1e6 for a, b in zip(ts, ts[1:]) if 0 < (b - a) / 1e6 < 500]
    return {
        "n": len(durs),
        "med_ms": statistics.median(durs),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "mean_ms": statistics.fmean(durs),
        "med_period_ms": statistics.median(periods) if periods else float("nan"),
        # p95−p50 是双峰检测：占空比那种混合分布会把它顶起来
        "bimodal_gap_ms": pct(95) - statistics.median(durs),
    }


# ─────────────────────────── 主流程 ───────────────────────────

async def run(args) -> int:
    import aiohttp
    import workloads

    os.makedirs(os.path.join(ROOT, "traces-step"), exist_ok=True)
    step_prefix = "interference"
    for f in os.listdir(os.path.join(ROOT, "traces-step")):
        if f.startswith(step_prefix):
            os.remove(os.path.join(ROOT, "traces-step", f))
    step_out = os.path.join(ROOT, "traces-step", step_prefix)

    print("启动服务...", flush=True)
    srv = start_server(step_out)
    if not wait_ready():
        stop_server(srv)
        return 1
    print("服务就绪", flush=True)

    print("启动负载进程（锁页约 900 MB，要等一会儿）...", flush=True)
    lg = LoadGenCtl(args.blocks, args.chunk_blocks)

    tok = workloads.load_tokenizer(MODEL)
    reqs = workloads.build_requests("short-qa", tok, "single")
    prompts = [r["prompt"] for r in reqs[:16]]

    levels = [float(x) for x in args.levels.split(",")]
    dirs = args.dirs.split(",")
    blocks: list[dict] = []

    timeout = aiohttp.ClientTimeout(total=600, sock_read=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for direction in dirs:
            for level in levels:
                for pair in range(args.pairs):
                    # --- baseline ---
                    itl: list[float] = []
                    t0 = time.time_ns()
                    await probe(session, prompts, t0 + int(args.block_s * 1e9), itl)
                    t1 = time.time_ns()
                    blocks.append({"dir": direction, "level": level, "kind": "base",
                                   "t0": t0, "t1": t1, "itl": itl, "lg": None})
                    print(f"  {direction} L={level:<5} pair{pair} baseline  "
                          f"n_itl={len(itl):>5}", flush=True)

                    # --- load ---
                    rate = level * args.ceiling_gbps
                    itl = []
                    if level > 0:
                        lg.start(rate, direction, args.block_s)
                        time.sleep(0.4)          # 让负载先起来，别把爬坡算进去
                    t0 = time.time_ns()
                    await probe(session, prompts, t0 + int(args.block_s * 1e9), itl)
                    t1 = time.time_ns()
                    info = lg.collect(args.block_s + 30) if level > 0 else None
                    blocks.append({"dir": direction, "level": level, "kind": "load",
                                   "t0": t0, "t1": t1, "itl": itl, "lg": info})
                    ach = f"{info['achieved_gbps']:.2f} GB/s" if info else "—"
                    print(f"  {direction} L={level:<5} pair{pair} load      "
                          f"n_itl={len(itl):>5}  实际 {ach}", flush=True)

    lg.stop()
    stop_server(srv)

    # ---------------- 分析 ----------------
    steps = load_steps(os.path.join(ROOT, "traces-step"), step_prefix)
    print(f"\n步时序共 {len(steps)} 行", flush=True)
    steps.sort(key=lambda s: s["t"])

    for b in blocks:
        seg = [s for s in steps if b["t0"] <= s["t"] <= b["t1"]]
        b["step"] = split_stats(seg)
        b["itl_med"] = statistics.median(b["itl"]) if b["itl"] else float("nan")
        b["itl_p95"] = (sorted(b["itl"])[int(0.95 * (len(b["itl"]) - 1))]
                        if b["itl"] else float("nan"))

    write_csv(args.out, blocks, args.ceiling_gbps)
    report(args.report, blocks, args)
    return 0


def _pair_up(blocks: list[dict]) -> list[dict]:
    """把每个 load block 与**紧邻的前一个** baseline 配成对。"""
    out = []
    for i, b in enumerate(blocks):
        if b["kind"] != "load" or b["level"] == 0:
            continue
        base = None
        for j in range(i - 1, -1, -1):
            if blocks[j]["kind"] == "base" and blocks[j]["dir"] == b["dir"]:
                base = blocks[j]
                break
        if base:
            out.append((base, b))
    return out


def write_csv(path: str, blocks: list[dict], ceiling: float) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = ["dir", "level", "kind", "t0", "t1", "n_steps", "step_med_ms",
            "step_p95_ms", "step_p99_ms", "step_period_ms", "bimodal_gap_ms",
            "n_itl", "itl_med_us", "itl_p95_us", "ach_gbps", "target_gbps"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for b in blocks:
            s = b.get("step") or {}
            f.write(",".join(str(x) for x in [
                b["dir"], b["level"], b["kind"], b["t0"], b["t1"],
                s.get("n", 0), f"{s.get('med_ms', float('nan')):.4f}",
                f"{s.get('p95_ms', float('nan')):.4f}",
                f"{s.get('p99_ms', float('nan')):.4f}",
                f"{s.get('med_period_ms', float('nan')):.4f}",
                f"{s.get('bimodal_gap_ms', float('nan')):.4f}",
                len(b["itl"]), f"{b['itl_med']:.2f}", f"{b['itl_p95']:.2f}",
                f"{(b['lg'] or {}).get('achieved_gbps', 0):.3f}",
                f"{b['level'] * ceiling:.2f}",
            ]) + "\n")


def report(path: str, blocks: list[dict], args) -> None:
    out: list[str] = []
    say = out.append
    say("=" * 78)
    say("Step 2 — 迁移流量对 decode 的干扰曲线")
    say("=" * 78)
    say(f"  探针      short-qa prompt，C=1，ignore_eos，连续 decode，逐 chunk 记 ITL")
    say(f"  主信号    EngineCoreProc._process_engine_step（完整步，含 schedule）")
    say(f"  交叉验证  --enable-logging-iteration-details（仅执行段）+ 客户端 ITL")
    say(f"  强度      连续限速（chunk {args.chunk_blocks} block ≈ 0.4 ms ≪ 一步）")
    say(f"  每 block  {args.block_s:.0f} 秒，配对 {args.pairs} 组，交错 baseline/load")
    say("")

    # 噪声底：所有 baseline 的离散度
    bases = [b for b in blocks if b["kind"] == "base" and b.get("step")]
    if bases:
        meds = [b["step"]["med_ms"] for b in bases]
        noise = max(meds) - min(meds)
        say(f"  baseline 步中位数范围  {min(meds):.3f} – {max(meds):.3f} ms"
            f"  （跨度 {noise:.3f} ms = 噪声底）")
        say(f"  **小于这个跨度的效应不算显著**")
        say("")

    say(f"  {'方向':<6} {'档位':>6} {'目标':>7} {'实际':>8} "
        f"{'步中位':>9} {'Δ基':>8} {'Δ%':>7} {'ITL中位':>9} {'ΔITL':>8}")
    say("-" * 78)
    for direction in args.dirs.split(","):
        for base, load in _pair_up(blocks):
            if load["dir"] != direction:
                continue
            bs, ls = base.get("step") or {}, load.get("step") or {}
            if not bs or not ls:
                continue
            d_abs = ls["med_ms"] - bs["med_ms"]
            d_pct = d_abs / bs["med_ms"] * 100
            itl_b, itl_l = base["itl_med"], load["itl_med"]
            ach = (load["lg"] or {}).get("achieved_gbps", 0.0)
            say(f"  {direction:<6} {load['level']:>6.2f} "
                f"{load['level'] * args.ceiling_gbps:>7.1f} {ach:>8.2f} "
                f"{ls['med_ms']:>9.3f} {d_abs:>+8.3f} {d_pct:>+7.2f} "
                f"{itl_l:>9.1f} {itl_l - itl_b:>+8.1f}")
        say("")

    # 双峰检测
    say("双峰检测（p95 − p50）—— 占空比造成的混合分布会把这一列顶起来")
    say("-" * 78)
    for b in blocks:
        if b["kind"] == "load" and b.get("step"):
            say(f"  {b['dir']:<6} L={b['level']:<5} p95−p50 = "
                f"{b['step']['bimodal_gap_ms']:>7.3f} ms")

    text = "\n".join(out) + "\n"
    print(text)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        print(f"报告已写入 {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="PCIe 迁移流量对 decode 的干扰曲线")
    ap.add_argument("--dirs", default="d2h,h2d,both,sham")
    ap.add_argument("--levels", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--block-s", type=float, default=8.0)
    ap.add_argument("--ceiling-gbps", type=float, default=22.0,
                    help="1.0 档对应的目标速率；实测 chunk=8 时上限约 23 GB/s")
    ap.add_argument("--blocks", type=int, default=1024)
    ap.add_argument("--chunk-blocks", type=int, default=8)
    ap.add_argument("--out", default="measurements/pcie_interference.csv")
    ap.add_argument("--report", default="measurements/pcie_interference.txt")
    args = ap.parse_args()

    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
