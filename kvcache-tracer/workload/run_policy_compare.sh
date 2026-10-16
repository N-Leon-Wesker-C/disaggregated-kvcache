#!/usr/bin/env bash
# run_policy_compare.sh — 在**真实系统**上比 LRU / ARC / LFU / Adaptive 四个逐出
# 策略。这一层不是离线 replay，是 vLLM 自己的 CPU offloading 层。
#
# 要回答的问题
# ============================================================
#
#   scripts/replay.py 已经在同一份 trace 上离线比过这些策略，结论是 LFU 全赢。
#   但那是**模拟**：它自己实现了一套逐出逻辑、自己定义了代价模型。真实系统上
#   是不是也这样？离线模型的结论会不会崩？
#
#   增量代码是 plugin/vllm_kvpolicy.py —— 把 LFU / Adaptive 补进 vLLM 的
#   _CACHE_POLICIES，LRU / ARC 用 vLLM 自带的。四个策略跑的是**同一份代码路径**，
#   只有逐出决策不同。
#
# 为什么必须把 GPU 池压小
# ============================================================
#
#   这是本轮最容易踩的坑：
#
#   用默认的 --gpu-memory-utilization 0.45 起服务（GPU 池 7,810 块），再挂一个
#   0.25 GiB 的 CPU offload 池，会得到一个**看起来正常但是空的**结果 ——
#   store_evicted 有四百多次、指标齐全，但 load_blocks = 0、CPU_to_GPU 字节 = 0。
#
#   原因是 offload 层只在 **GPU prefix cache 没命中**时才被问到。GPU 池装得下
#   整个工作集时，所有复用都在 GPU 那层解决了，CPU 池是个只写不读的坟场。这时
#   任何策略的"命中率"都是 0，比不出任何东西 —— 而报表上看不出异常，因为换出
#   计数一切正常。
#
#   所以：GPU 池必须小于工作集。--gpu-memory-utilization 只能粗调，这里直接用
#   --kv-cache-memory-bytes 给字节数。它的下限是 max_model_len 那么多 token
#   （一个请求得装得下），所以 max-model-len 一并降到 2048。
#
# 用法
# ============================================================
#
#   bash workload/run_policy_compare.sh                 # 4 策略 × 1 次
#   REPS=3 bash workload/run_policy_compare.sh          # 4 策略 × 3 次
#   POLICIES="lru lfu" REPS=3 bash workload/run_policy_compare.sh
#
# 产物: results-policy/<policy>.counters.jsonl 与 .c<conc>.r<rep>.json
#       logs-policy/  traces-policy/
# 分析: python scripts/policy_report.py

set -euo pipefail

ROOT=/workspace/disaggregated-kvcache/kvcache-tracer
MODEL=/workspace/models/Qwen2.5-7B-Instruct
SERVED_NAME=Qwen2.5-7B-Instruct
PORT=8000

# ---- 本实验的固定口径 ----
# 改动会让跨策略对比失效；也意味着**不能**和 results-skew/ 那 27 次运行横向
# 比较 —— 那批没开 offload，GPU 池也大得多。
GPU_BLOCKS="${GPU_BLOCKS:-256}"     # GPU 上所有块的总预算（含运行中的）
BLOCK_BYTES=917504                  # 16 token × 2 × 4 × 128 × 2 B × 28 层
MAX_LEN="${MAX_LEN:-2048}"
BLOCK_SIZE=16
OFFLOAD_GIB="${OFFLOAD_GIB:-0.25}"  # → 292 个 CPU 块，与 replay 的 256 档同量级

POLICIES="${POLICIES:-lru arc lfu adaptive}"
REPS="${REPS:-1}"
WORKLOAD="${WORKLOAD:-short-qa}"
CONCUR="${CONCUR:-8}"
PREFIX_MODE="${PREFIX_MODE:-skewed}"

RESULTS=results-policy
LOGS=logs-policy
TRACES=traces-policy
PIDFILE=/tmp/kvpolicy_server.pid

CORE_PAT='VLLM::Engine[C]ore'

cd "$ROOT"
mkdir -p "$RESULTS" "$LOGS" "$TRACES"

stop_server() {
    if [[ -f "$PIDFILE" ]]; then
        kill -TERM "$(cat "$PIDFILE")" 2>/dev/null || true
        rm -f "$PIDFILE"
    fi
    for _ in $(seq 1 60); do
        pgrep -f "$CORE_PAT" >/dev/null || break
        sleep 1
    done
    pkill -KILL -f "$CORE_PAT" 2>/dev/null || true
    for _ in $(seq 1 60); do
        local free
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
               | head -1)
        [[ "$free" -ge 47000 ]] && return 0
        sleep 2
    done
}
trap 'stop_server' EXIT

wait_ready() {
    for _ in $(seq 1 240); do
        curl -sf -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
        sleep 2
    done
    echo "  错误: 服务 480s 内没起来，看 $LOG" >&2
    return 1
}

# 两级都要清：GPU prefix cache + connector 管的 CPU 池。
# 不带 reset_external 时只清前者，CPU 池会带着上一轮的块进入下一轮 ——
# 那样第 2、3 次重复是"热"的，和冷启动的第 1 次不可比。
reset_both() {
    local before after
    before=$(grep -c "Successfully reset prefix cache" "$LOG" || true)
    curl -sf -m 120 -X POST \
        "http://127.0.0.1:${PORT}/reset_prefix_cache?reset_external=true" \
        >/dev/null 2>&1 || true
    for _ in $(seq 1 240); do
        after=$(grep -c "Successfully reset prefix cache" "$LOG" || true)
        [[ "$after" -gt "$before" ]] && return 0
        sleep 0.5
    done
    echo "  错误: prefix cache 没重置成功（该接口无论成败都返回 200，只能看日志）" >&2
    return 1
}

echo "GPU 池 ${GPU_BLOCKS} 块 / CPU offload ${OFFLOAD_GIB} GiB / max-model-len ${MAX_LEN}"
echo "策略 [$POLICIES] × ${REPS} 次，workload=${WORKLOAD} c=${CONCUR} ${PREFIX_MODE}"
echo

stop_server

for POLICY in $POLICIES; do
    echo "############################################################"
    echo "# 策略 ${POLICY}  ×  ${REPS} 次"
    echo "############################################################"

    LOG="${LOGS}/${POLICY}.log"
    OUT="${RESULTS}/${POLICY}.counters.jsonl"
    : > "$LOG"; : > "$OUT"

    VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1 \
    KVTRACE_POLICY_OUT="$OUT" KVTRACE_POLICY_NAME="$POLICY" \
    KVTRACE_OUTPUT="${ROOT}/${TRACES}/${POLICY}" \
    nohup vllm serve "$MODEL" \
        --served-model-name "$SERVED_NAME" \
        --max-model-len "$MAX_LEN" \
        --block-size "$BLOCK_SIZE" \
        --enable-prefix-caching \
        --kv-cache-memory-bytes $(( GPU_BLOCKS * BLOCK_BYTES )) \
        --kv-offloading-size "$OFFLOAD_GIB" \
        --kv-transfer-config \
            "{\"kv_connector_extra_config\":{\"eviction_policy\":\"${POLICY}\"}}" \
        --port "$PORT" \
        > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    wait_ready

    # 策略有没有真的被用上，必须**看到**而不是推断：请求一个没注册的名字 vLLM
    # 会直接 ValueError 起不来，但请求一个已注册的名字而配置没生效是**完全静默**
    # 的 —— 那会得到一份"四个策略结果一样"的干净报告。
    #
    # 判据是 manager 打印的那行实际策略（四种策略都有），不是"我的类被实例化了"
    # （lru / arc 走 vLLM 自带的类，那行永远不会出现 —— 实测把 lru 误判成失败）。
    if ! grep -q "实际逐出策略" "$LOG"; then
        echo "  错误: 日志里没有'实际逐出策略'那一行 —— 插件没生效，本策略作废" >&2
        grep -E "KVPolicy|Unknown cache policy" "$LOG" | head -5 >&2
        exit 1
    fi
    grep -E "实际逐出策略" "$LOG" | tail -1 | sed 's/^/  /'

    # 光有那行还不够 —— 得是**要的那个**策略。选错策略时四种跑出来一模一样，
    # 而报表是干净的，没有任何东西会提示。
    ACTUAL=$(grep -oP "实际逐出策略 = \K\w+" "$LOG" | tail -1)
    case "$POLICY" in
        lru)      WANT=LRUCachePolicy ;;
        arc)      WANT=ARCCachePolicy ;;
        lfu)      WANT=LFUCachePolicy ;;
        adaptive) WANT=AdaptiveCachePolicy ;;
        *)        WANT="" ;;
    esac
    if [[ -n "$WANT" && "$ACTUAL" != "$WANT" ]]; then
        echo "  错误: 请求的 $POLICY 应对应 $WANT，实际是 $ACTUAL —— 中止" >&2
        exit 1
    fi

    POOL=$(grep -oP "GPU KV cache size: \K[0-9,]+" "$LOG" | tr -d ,)
    [[ -z "$POOL" ]] && { echo "  错误: 读不到池容量" >&2; exit 1; }
    echo "  GPU KV cache 池: ${POOL} tokens"

    for R in $(seq 1 "$REPS"); do
        if [[ "$R" -gt 1 ]]; then
            sleep 3
            echo "  --- 第 ${R} 次重复（清 GPU + CPU 两级缓存）---"
            reset_both
        else
            echo "  --- 第 1 次重复（全新服务，状态天然干净）---"
        fi

        # 在计数流里打个界标。观测器是 O_APPEND + 行缓冲，两边都追加小行，
        # 混写是安全的。分段与取数全交给 scripts/policy_report.py。
        python -c "
import json, sys, time
print(json.dumps({'marker': 'rep', 'policy': sys.argv[1],
                  'rep': int(sys.argv[2]), 't': time.time_ns()}))" \
            "$POLICY" "$R" >> "$OUT"

        python workload/run.py \
            --workload "$WORKLOAD" \
            --concurrency "$CONCUR" \
            --rep "$R" \
            --prefix-mode "$PREFIX_MODE" \
            --model-path "$MODEL" \
            --served-name "$SERVED_NAME" \
            --base-url "http://127.0.0.1:${PORT}" \
            --kv-pool-tokens "$POOL" \
            --max-model-len "$MAX_LEN" \
            --block-size "$BLOCK_SIZE" \
            --trace-glob "${TRACES}/${POLICY}.*.jsonl" \
            --out "${RESULTS}/${POLICY}.c${CONCUR}.r${R}.json"

        # 等观测器的定时刷盘（每 2s 一次）把最后几个计数落下来
        sleep 4
    done

    echo
    stop_server
done

echo "############################################################"
echo "# 完成。分析: python scripts/policy_report.py"
echo "############################################################"
