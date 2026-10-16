#!/usr/bin/env bash
# run_experiment.sh — 正式实验：3 类 workload × 并发度 1/4/8 × 每组合 3 次重复
#
# 实验设计
# ============================================================
#
# 自变量是 (workload, 并发度)，重复 3 次是为了给出方差 —— 只有一次运行的数字
# 无法区分「workload 的差异」和「这一次调度的运气」。
#
# 隔离策略分两层，代价和收益不同：
#
#   1. 换 (workload, 并发度) 时**重启服务**（9 次）。重启清掉 prefix cache、
#      block pool 空闲链、allocator 碎片状态 —— 自变量换挡时全部归零。
#   2. 同一组合的三次重复之间只调 `/reset_prefix_cache`，不重启。三次用的是
#      同一批请求（同一 seed），唯一该变的只有调度时序，而前缀缓存若残留，
#      第 2、3 次就会白捡第 1 次留下的命中率。
#
#   为什么敢只 reset 不重启：reset 清掉的正是唯一会跨重复泄漏的状态（缓存的
#   block hash 索引）。省下的是 27 次重启里多余的 18 次，约 15 分钟。
#
# /reset_prefix_cache 的坑
# ============================================================
#
#   这个接口**无论成败都返回 HTTP 200**（entrypoints/serve/dev/cache/
#   api_router.py 直接 Response(status_code=200)，body 里没有 success 字段）。
#   底层 block_pool.reset_prefix_cache 只在「除 null block 外全部空闲」时才真清，
#   否则打一条 warning 就返回 False。
#
#   所以脚本**不能信状态码**，只能数日志里 "Successfully reset prefix cache"
#   的行数是否增加。reset 失败 = 上一次的请求还占着 block = 重复之间不隔离，
#   这时必须停下来而不是继续跑。
#
# 池容量必须逐次相同
# ============================================================
#
#   池容量是判定「有没有发生逐出」的分母，也是 workload 足迹比的参照。它由
#   --gpu-memory-utilization / --max-model-len / --block-size 决定，但 vLLM 是
#   按**启动时当前显存实况**切池子的 —— 上一个服务刚被 kill、驱动还没回收完
#   就启动，池子会偏小（实测量过同一套参数得到 110,416 vs 稳定的 124,960）。
#   脚本因此做两件事：启动前等显存回落，启动后把池容量和第一次的值对比，
#   不一致就整个中止 —— 那种数据不能混进来。
#
# 用法:
#   bash workload/run_experiment.sh                       # 全部 27 次
#   WORKLOADS="short-qa" CONCURRENCIES="1" bash workload/run_experiment.sh
#   REPS=1 bash workload/run_experiment.sh                # 快速冒烟

set -euo pipefail

ROOT=/workspace/disaggregated-kvcache/kvcache-tracer
MODEL=/workspace/models/Qwen2.5-7B-Instruct
SERVED_NAME=Qwen2.5-7B-Instruct
PORT=8000

# ---- 全程固定。改任何一个都会改池容量，实验就不再可比 ----
GPU_UTIL=0.45            # 0.85 时池子 46 万 token，这三类负载都填不满 → 不逐出
MAX_LEN=8192             # 放得下最长的 4096-token prompt
BLOCK_SIZE=16            # vLLM v1 默认，也是 workloads.BLOCK_SIZE

WORKLOADS="${WORKLOADS:-short-qa code-gen long-summarize}"
CONCURRENCIES="${CONCURRENCIES:-1 4 8}"
REPS="${REPS:-3}"

PIDFILE=/tmp/kvtrace_server.pid
POOL_REF=""

cd "$ROOT"
mkdir -p traces results logs

echo "实验矩阵: [$WORKLOADS] × [$CONCURRENCIES] × ${REPS} 次重复 = $(echo $WORKLOADS | wc -w) × $(echo $CONCURRENCIES | wc -w) × ${REPS} 次运行"
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

trap 'stop_server' EXIT

echo "清理可能残留的服务..."
stop_server

for W in $WORKLOADS; do
  for C in $CONCURRENCIES; do
    echo
    echo "############################################################"
    echo "# ${W}  ×  并发度 ${C}  ×  ${REPS} 次重复"
    echo "############################################################"

    LOG="logs/${W}.c${C}.log"
    : > "$LOG"
    stop_server

    # KVTRACE_OUTPUT 是服务启动时固定的，所以整个会话（3 次重复）写进同一组
    # 文件；重复之间靠 request_id 里的 -rN 后缀区分（见 run.py）。
    VLLM_SERVER_DEV_MODE=1 \
    HF_HUB_OFFLINE=1 \
    KVTRACE_OUTPUT="${ROOT}/traces/${W}.c${C}" \
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
        if [[ "$R" == "1" ]]; then
            echo "  --- 第 1 次重复（全新服务，状态天然干净）---"
        else
            echo "  --- 第 ${R} 次重复（先清 prefix cache）---"
            # 给服务端一点时间把最后一次请求的 block 全部回收，否则 reset 会失败
            sleep 5
            reset_cache
        fi

        python workload/run.py \
            --workload "$W" \
            --concurrency "$C" \
            --rep "$R" \
            --model-path "$MODEL" \
            --served-name "$SERVED_NAME" \
            --base-url "http://127.0.0.1:${PORT}" \
            --kv-pool-tokens "$POOL" \
            --gpu-memory-utilization "$GPU_UTIL" \
            --max-model-len "$MAX_LEN" \
            --block-size "$BLOCK_SIZE" \
            --trace-glob "traces/${W}.c${C}.*.jsonl" \
            --out "results/${W}.c${C}.r${R}.json"
    done

    sleep 3
    stop_server
    echo "  trace: $(ls traces/${W}.c${C}.*.jsonl 2>/dev/null | wc -l) 个文件"
  done
done

echo
echo "############################################################"
echo "# 实验完成。生成报告:"
echo "#   python scripts/report.py"
echo "############################################################"
