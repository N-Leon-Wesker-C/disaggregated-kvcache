#!/usr/bin/env bash
# run_skewed.sh — K 模板 + Zipf 复用模式的扫描，产出喂给离线 replay 的 trace
#
# 和 run_experiment.sh 的关系
# ============================================================
#
#   两者共用同一个客户端（run.py）和同一套编排（重启隔离、池容量校验、
#   逐日志确认 reset）。区别只在 --prefix-mode：
#
#     run_experiment.sh  single  类内所有请求共享一段指令前导
#     run_skewed.sh      skewed  K 个长度各异的模板，按 Zipf 挑选
#
#   产物刻意分开放（traces-skew/ results-skew/ logs-skew/）。正式实验那 27 次
#   运行的结论已经写进 README，不能被这一轮覆盖或者混在一起统计。
#
# 为什么需要 skewed 模式
# ============================================================
#
#   正式实验实测出的可复用集合太小：跨请求复用的 block 只有 4 / 32 / 128 个
#   （就是那段共享前导），而热池预算是 64/128/256。拿它做 LRU/LFU/Adaptive
#   的对比，9 个格子里有 8 个是「所有策略都能把该留的全留下」—— 对比会退化成
#   没有差别，而那是预算选错了，不是策略没差别。
#
#   skewed 模式把可复用集合放大到几百个 block，且**长度与热度无关**：
#
#     workload         模板数  可复用block  独有prompt block  命中80%请求需
#     short-qa           64       259            672              152
#     code-gen           32       418            945              264
#     long-summarize     16       535           5780              255
#
#   （"独有 prompt block" 不含解码过程中新分配的 block，那个数只有 trace 才有。）
#
#   三类的「命中 80% 请求所需 block」都落在 150~265，正好横跨 64/128/256
#   三档预算 —— 每一档都有真实的取舍，这才是有信息量的对比。
#
# 用法:
#   bash workload/run_skewed.sh                          # 全部 27 次
#   REPS=1 bash workload/run_skewed.sh                   # 先跑一轮看数据
#   WORKLOADS="long-summarize" bash workload/run_skewed.sh
#   RESUME=1 bash workload/run_skewed.sh                 # 跳过已有完整结果的组合
#
# RESUME 只跳过**已经跑完且 JSON 可解析**的那几次运行。被中断的组合会从
# 缺的那一次接着跑；它的第 1 次仍然是全新服务，状态天然干净。

set -euo pipefail

ROOT=/workspace/disaggregated-kvcache/kvcache-tracer
MODEL=/workspace/models/Qwen2.5-7B-Instruct
SERVED_NAME=Qwen2.5-7B-Instruct
PORT=8000

# ---- 必须和 run_experiment.sh 逐项相同，否则两批数据的池容量不可比 ----
GPU_UTIL=0.45
MAX_LEN=8192
BLOCK_SIZE=16

PREFIX_MODE=skewed
TRACES=traces-skew
RESULTS=results-skew
LOGS=logs-skew

WORKLOADS="${WORKLOADS:-short-qa code-gen long-summarize}"
CONCURRENCIES="${CONCURRENCIES:-1 4 8}"
REPS="${REPS:-3}"

PIDFILE=/tmp/kvtrace_server.pid
POOL_REF=""

cd "$ROOT"
mkdir -p "$TRACES" "$RESULTS" "$LOGS"

echo "实验矩阵: [$WORKLOADS] × [$CONCURRENCIES] × ${REPS} 次重复, prefix_mode=${PREFIX_MODE}"
echo

# 进程匹配一律写成 "VLLM::Engine[C]ore"：这是正则，能匹配到真正的 EngineCore，
# 但**匹配不到写它的那条命令行本身**（那里面是字面的 [C]）。直接写
# "VLLM::EngineCore" 时，pkill/pgrep -f 会连调用者的 shell 一起命中 —— 实测把
# 自己这条 shell 杀掉了（退出码 144），并且给出"EngineCore 还在"的假阳性。
CORE_PAT='VLLM::Engine[C]ore'

stop_server() {
    if [[ -f "$PIDFILE" ]]; then
        kill -TERM "$(cat "$PIDFILE")" 2>/dev/null || true
        rm -f "$PIDFILE"
    fi
    for _ in $(seq 1 60); do
        pgrep -f "$CORE_PAT" >/dev/null || break
        sleep 1
    done
    if pgrep -f "$CORE_PAT" >/dev/null; then
        echo "  警告: EngineCore 仍在，强制清理" >&2
        pkill -KILL -f "$CORE_PAT" || true
        sleep 3
    fi
    # 等显存真正归还。不等的话下一个服务会按偏小的可用显存切池子。
    for _ in $(seq 1 60); do
        local free
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
               | head -1)
        [[ "$free" -ge 47000 ]] && return 0
        sleep 2
    done
    echo "  警告: 显存 120s 内没回落到 47000MiB 以上（当前 ${free}MiB）" >&2
    return 0
}

wait_ready() {
    for _ in $(seq 1 240); do
        curl -sf -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
        sleep 2
    done
    echo "  错误: 服务 480s 内没起来，看 $LOG" >&2
    return 1
}

# 重置 prefix cache，并以**日志**为准判定成败（接口无论成败都返回 200）
reset_cache() {
    local before after
    before=$(grep -c "Successfully reset prefix cache" "$LOG" || true)
    curl -sf -m 30 -X POST "http://127.0.0.1:${PORT}/reset_prefix_cache" \
        >/dev/null 2>&1 || true
    for _ in $(seq 1 60); do
        after=$(grep -c "Successfully reset prefix cache" "$LOG" || true)
        if [[ "$after" -gt "$before" ]]; then
            echo "  prefix cache 已重置（第 ${after} 次，日志确认）"
            return 0
        fi
        sleep 0.5
    done
    echo "  错误: prefix cache 没重置成功。" >&2
    echo "  该接口无论成败都返回 200，判定依据是日志里没有 'Successfully reset prefix cache'；" >&2
    echo "  多半是上一次的请求还占着 block。重复之间不隔离，数据不可用。" >&2
    grep -o "Failed to reset prefix cache.*" "$LOG" | tail -1 >&2 || true
    return 1
}

# 结果文件是否已经存在且**完整**。只看文件非空是不够的 —— 扫描被中断时
# 可能留下半截 JSON，那种文件 `-s` 判定通过、json.load 才炸。所以要真的
# 解析一遍并检查必需字段。
have_result() {
    [[ -s "$1" ]] || return 1
    python -c "
import json, sys
try:
    m = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if m.get('results') and m.get('run') else 1)
" "$1" 2>/dev/null
}

trap 'stop_server' EXIT

echo "清理可能残留的服务..."
stop_server

for W in $WORKLOADS; do
  for C in $CONCURRENCIES; do
    # 整个组合都跑完就别启动服务了。否则 RESUME 会为已完成的组合白起一次
    # 服务（约 90s）再逐个 rep 跳过 —— 光启动就浪费六分钟。
    if [[ "${RESUME:-0}" == "1" ]]; then
        _all=1
        for R in $(seq 1 "$REPS"); do
            have_result "${RESULTS}/${W}.c${C}.r${R}.json" || { _all=0; break; }
        done
        if [[ "$_all" == "1" ]]; then
            echo "跳过 ${W} × 并发 ${C}：${REPS} 次结果均已完成"
            continue
        fi
    fi
    echo
    echo "############################################################"
    echo "# ${W}  ×  并发度 ${C}  ×  ${REPS} 次重复  [${PREFIX_MODE}]"
    echo "############################################################"

    LOG="${LOGS}/${W}.c${C}.log"
    : > "$LOG"
    stop_server

    VLLM_SERVER_DEV_MODE=1 \
    HF_HUB_OFFLINE=1 \
    KVTRACE_OUTPUT="${ROOT}/${TRACES}/${W}.c${C}" \
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
    if [[ -z "$POOL" ]]; then
        echo "  错误: 从启动日志里读不到池容量" >&2
        exit 1
    fi
    if [[ -z "$POOL_REF" ]]; then
        POOL_REF="$POOL"
        echo "  KV cache 池: ${POOL} tokens（作为后续所有轮的基准）"
    elif [[ "$POOL" != "$POOL_REF" ]]; then
        echo "  错误: 池容量从 ${POOL_REF} 变成 ${POOL} tokens —— 本轮不可比，中止。" >&2
        echo "  检查是否有别的进程占显存，或上一个服务是否没完全退出。" >&2
        exit 1
    else
        echo "  KV cache 池: ${POOL} tokens（与基准一致 ✓）"
    fi

    for R in $(seq 1 "$REPS"); do
        OUT="${RESULTS}/${W}.c${C}.r${R}.json"
        if [[ "${RESUME:-0}" == "1" ]] && have_result "$OUT"; then
            echo "  --- 第 ${R} 次重复（已有完整结果，跳过）---"
            continue
        fi
        if [[ "$R" == "1" ]]; then
            echo "  --- 第 1 次重复（全新服务，状态天然干净）---"
        else
            echo "  --- 第 ${R} 次重复（先清 prefix cache）---"
            sleep 5
            reset_cache
        fi

        python workload/run.py \
            --workload "$W" \
            --concurrency "$C" \
            --rep "$R" \
            --prefix-mode "$PREFIX_MODE" \
            --model-path "$MODEL" \
            --served-name "$SERVED_NAME" \
            --base-url "http://127.0.0.1:${PORT}" \
            --kv-pool-tokens "$POOL" \
            --gpu-memory-utilization "$GPU_UTIL" \
            --max-model-len "$MAX_LEN" \
            --block-size "$BLOCK_SIZE" \
            --trace-glob "${TRACES}/${W}.c${C}.*.jsonl" \
            --out "$OUT"
    done

    sleep 3
    stop_server
    echo "  trace: $(ls ${TRACES}/${W}.c${C}.*.jsonl 2>/dev/null | wc -l) 个文件"
  done
done

echo
echo "############################################################"
echo "# 扫描完成。报告:"
echo "#   python scripts/report.py --results-dir results-skew"
echo "############################################################"
