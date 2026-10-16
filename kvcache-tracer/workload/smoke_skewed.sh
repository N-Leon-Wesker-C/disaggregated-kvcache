#!/usr/bin/env bash
# smoke_skewed.sh — 正式扫描前的冒烟：起一次服务，跑短的两组，验完再关
#
# 验的是**模型对不对**，不是"能不能跑起来"：
#   1. skew 模式下真实 prefix cache 命中数，是否等于「该请求之前已经出现过
#      的同模板 block 数」。这一条不成立，replay 里"命中自己的模板"就是错的。
#   2. long-summarize 的独有 body 涨到 5780 block 后，会不会把池子压垮。
set -euo pipefail

ROOT=/workspace/disaggregated-kvcache/kvcache-tracer
MODEL=/workspace/models/Qwen2.5-7B-Instruct
PORT=8000
PIDFILE=/tmp/kvtrace_smoke.pid
CORE_PAT='VLLM::Engine[C]ore'

cd "$ROOT"
mkdir -p traces-smoke logs-smoke results-smoke

stop_server() {
    [[ -f "$PIDFILE" ]] && { kill -TERM "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; }
    for _ in $(seq 1 60); do pgrep -f "$CORE_PAT" >/dev/null || break; sleep 1; done
    pgrep -f "$CORE_PAT" >/dev/null && { pkill -KILL -f "$CORE_PAT" || true; sleep 3; }
    for _ in $(seq 1 60); do
        local free
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        [[ "$free" -ge 47000 ]] && return 0
        sleep 2
    done
}
trap 'stop_server' EXIT

stop_server
rm -f traces-smoke/*.jsonl

LOG=logs-smoke/server.log
: > "$LOG"
VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1 \
KVTRACE_OUTPUT="${ROOT}/traces-smoke/skew" \
nohup vllm serve "$MODEL" --served-model-name Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.45 --max-model-len 8192 --block-size 16 \
    --enable-prefix-caching --port "$PORT" > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

for _ in $(seq 1 240); do
    curl -sf -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    sleep 2
done
POOL=$(grep -oP "GPU KV cache size: \K[0-9,]+" "$LOG" | tr -d ,)
echo "池容量: $POOL tokens"

# ---- 1. short-qa 全量跑一遍，用来验命中模型 ----
python workload/run.py --workload short-qa --concurrency 1 --rep 1 \
    --prefix-mode skewed --kv-pool-tokens "$POOL" \
    --gpu-memory-utilization 0.45 --max-model-len 8192 --block-size 16 \
    --trace-glob "traces-smoke/skew.*.jsonl" \
    --out results-smoke/short-qa.c1.r1.json

# ---- 2. 长上下文：只发 12 个，看足迹 ----
python workload/run.py --workload long-summarize --concurrency 1 --rep 1 \
    --prefix-mode skewed --limit 12 --kv-pool-tokens "$POOL" \
    --gpu-memory-utilization 0.45 --max-model-len 8192 --block-size 16 \
    --trace-glob "traces-smoke/skew.*.jsonl" \
    --out results-smoke/long-summarize.c1.r1.json

echo
echo "=== 关闭服务 ==="
stop_server
trap - EXIT
echo "done"
