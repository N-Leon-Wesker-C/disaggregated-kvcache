#!/usr/bin/env bash
# run_sweep.sh — 并发度扫描：同一份 workload 分别跑并发 1 / 4 / 8
#
# 为什么每个并发度都重启一次服务，而不是跑完用 /reset_prefix_cache 清缓存：
#
#   重启能同时清掉三样东西 —— prefix cache、block pool 的空闲链、以及
#   allocator 累积的碎片状态。reset_prefix_cache 只清第一样。既然并发度是
#   这次实验唯一的自变量，其余状态就必须每次都从同一个零点出发。
#
#   代价是每个并发度多花 ~50s 加载模型。这个代价换来的可比性值。
#
#   注意：--gpu-memory-utilization / --max-model-len / --block-size 必须三次
#   完全一致，否则池子大小不同，并发度就不是唯一变量了。
#
# 用法:
#   bash workload/run_sweep.sh              # 跑 1 4 8
#   CONCURRENCIES="8" bash workload/run_sweep.sh   # 只跑某个并发度

set -euo pipefail

ROOT=/workspace/disaggregated-kvcache/kvcache-tracer
MODEL=/workspace/models/Qwen2.5-7B-Instruct
SERVED_NAME=Qwen2.5-7B-Instruct
PORT=8000

# ---- 三次运行必须完全一致的固定配置 ----
GPU_UTIL=0.45            # 0.85 时池子有 46 万 token，负载根本压不满 → 不会逐出
MAX_LEN=8192             # 放得下最长的 5160-token prompt
BLOCK_SIZE=16            # vLLM v1 默认值
#
# 这三个值决定池容量，而池容量是「是否发生逐出」判定的分母，所以不能变。
# 脚本每轮都会打印实际测到的池容量，三次必须相同 —— 不同就说明中途换了
# 配置或有别的进程占显存，别拿这种数据做对比。
#
# 踩过的坑：vLLM 是启动时按当前显存实况切池子的，如果上一个服务刚用掉 40GB
# 就被 kill，驱动侧还没完全回收，池子会明显偏小（实测同一套参数先得到
# 110,416 tokens、之后稳定得到 124,960）。修法是给 vLLM 传
# --kv-cache-memory-bytes <字节数> 把池子钉死；本脚本靠 stop_server 里的
# 等待把它规避掉了。

CONCURRENCIES="${CONCURRENCIES:-1 4 8}"
PIDFILE=/tmp/kvtrace_server.pid

cd "$ROOT"
mkdir -p traces results logs

stop_server() {
    if [[ -f "$PIDFILE" ]]; then
        kill -TERM "$(cat "$PIDFILE")" 2>/dev/null || true
        rm -f "$PIDFILE"
    fi
    # 必须等显存真正释放，否则下一次启动会因显存不足直接失败
    for _ in $(seq 1 40); do
        pgrep -f "VLLM::EngineCore" >/dev/null || break
        sleep 1
    done
    pgrep -f "VLLM::EngineCore" >/dev/null && {
        echo "  警告: EngineCore 仍在，强制清理" >&2
        pkill -KILL -f "VLLM::EngineCore" || true
        sleep 3
    }
    return 0
}

wait_ready() {
    for _ in $(seq 1 240); do
        if curl -sf -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    echo "  错误: 服务 240s 内没起来，看 logs/ 下的日志" >&2
    return 1
}

trap 'stop_server' EXIT

stop_server

for C in $CONCURRENCIES; do
    echo
    echo "############################################################"
    echo "# 并发度 ${C}"
    echo "############################################################"

    LOG="logs/c${C}.log"
    : > "$LOG"
    stop_server

    # KVTRACE_OUTPUT 用绝对路径，每个并发度一个前缀 → trace 文件天然分开
    VLLM_SERVER_DEV_MODE=1 \
    HF_HUB_OFFLINE=1 \
    KVTRACE_OUTPUT="${ROOT}/traces/c${C}" \
    nohup vllm serve "$MODEL" \
        --served-model-name "$SERVED_NAME" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-model-len "$MAX_LEN" \
        --block-size "$BLOCK_SIZE" \
        --enable-prefix-caching \
        --port "$PORT" \
        > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"

    wait_ready

    POOL=$(grep -oP "GPU KV cache size: \K[0-9,]+" "$LOG" | tr -d ,)
    echo "  KV cache 池: ${POOL} tokens"

    python workload/workload.py \
        --concurrency "$C" \
        --model-path "$MODEL" \
        --served-name "$SERVED_NAME" \
        --base-url "http://127.0.0.1:${PORT}" \
        --kv-pool-tokens "$POOL" \
        --out "results/c${C}.json"

    # 等事件真正落盘再停服务 —— tracer 是每条 flush，但给一点余量
    sleep 3
    stop_server
    echo "  trace: $(ls traces/c${C}.*.jsonl 2>/dev/null | wc -l) 个文件"
done

echo
echo "扫描完成。逐个分析:"
for C in $CONCURRENCIES; do
    echo
    echo "################  并发度 ${C}  ################"
    python scripts/analyze.py "traces/c${C}.*.jsonl"
done
