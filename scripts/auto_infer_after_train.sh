#!/bin/bash
# ============================================================
#  等待训练结束后自动推理
#  
#  用法: 
#    nohup bash scripts/auto_infer_after_train.sh &
#    # 或在 screen 中运行:
#    bash scripts/auto_infer_after_train.sh
#
#  支持选项:
#    --ckpt <path>   指定 ckpt 目录 (默认自动选最佳)
#    --best          选择 NDCG 最高的 ckpt (默认)
#    --latest        选择最新的 ckpt
#    --now           不等待训练，直接用现有 ckpt 推理
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="${PROJECT_DIR}/ckpt"
DATA_DIR="${PROJECT_DIR}/data"
RESULT_DIR="${PROJECT_DIR}/results"
LOG_DIR="${PROJECT_DIR}/log"

# 环境变量
export TRAIN_DATA_PATH="${DATA_DIR}"
export EVAL_DATA_PATH="${DATA_DIR}"
export EVAL_RESULT_PATH="${RESULT_DIR}"
export USER_CACHE_PATH="${PROJECT_DIR}/cache"
export TRAIN_CKPT_PATH="${CKPT_DIR}"
export TRAIN_LOG_PATH="${LOG_DIR}"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

# 解析参数
SELECT_MODE="best"  # best | latest | specific
SPECIFIC_CKPT=""
WAIT_FOR_TRAIN=true

for arg in "$@"; do
    case "${arg}" in
        --best)    SELECT_MODE="best" ;;
        --latest)  SELECT_MODE="latest" ;;
        --now)     WAIT_FOR_TRAIN=false ;;
        --ckpt)    SELECT_MODE="specific" ;;
        *)
            if [ "${SELECT_MODE}" = "specific" ] && [ -z "${SPECIFIC_CKPT}" ]; then
                SPECIFIC_CKPT="${arg}"
            fi
            ;;
    esac
done

# 选择最佳 ckpt (NDCG 最高)
select_best_ckpt() {
    local best=""
    local best_ndcg=0
    for d in "${CKPT_DIR}"/global_step*; do
        [ -d "$d" ] || continue
        ndcg=$(echo "$d" | grep -oP 'NDCG=\K[0-9.]+' || echo "0")
        if [ "$(echo "$ndcg > $best_ndcg" | bc -l)" -eq 1 ]; then
            best_ndcg="$ndcg"
            best="$d"
        fi
    done
    echo "$best"
}

# 选择最新 ckpt
select_latest_ckpt() {
    ls -td "${CKPT_DIR}"/global_step* 2>/dev/null | head -1
}

# 等待训练进程结束
if [ "${WAIT_FOR_TRAIN}" = true ]; then
    TRAIN_PID=$(pgrep -f "python.*main.py.*--amp_bf16" | head -1 || true)
    if [ -n "${TRAIN_PID}" ]; then
        echo "⏳ 检测到训练进程 PID=${TRAIN_PID}，等待训练完成..."
        echo "   当前 ckpt 列表:"
        ls -1d "${CKPT_DIR}"/global_step* 2>/dev/null | while read d; do
            echo "     $(basename "$d")"
        done
        
        # 每 30 秒检查一次
        while kill -0 "${TRAIN_PID}" 2>/dev/null; do
            sleep 30
            # 打印最新 ckpt
            LATEST=$(select_latest_ckpt)
            if [ -n "${LATEST}" ]; then
                printf "\r   [$(date '+%H:%M:%S')] 训练中... 最新: $(basename "${LATEST}")"
            fi
        done
        echo ""
        echo "✅ 训练已结束！"
    else
        echo "ℹ️  未检测到训练进程，直接使用已有 ckpt"
    fi
fi

# 选择 ckpt
case "${SELECT_MODE}" in
    best)
        CKPT_PATH=$(select_best_ckpt)
        ;;
    latest)
        CKPT_PATH=$(select_latest_ckpt)
        ;;
    specific)
        CKPT_PATH="${SPECIFIC_CKPT}"
        ;;
esac

if [ -z "${CKPT_PATH}" ] || [ ! -d "${CKPT_PATH}" ]; then
    echo "❌ 未找到有效的 checkpoint"
    echo "   CKPT_DIR: ${CKPT_DIR}"
    ls -la "${CKPT_DIR}" 2>/dev/null
    exit 1
fi

export MODEL_OUTPUT_PATH="${CKPT_PATH}"

echo ""
echo "=========================================="
echo "  🔍 开始推理"
echo "=========================================="
echo "  Checkpoint: $(basename "${CKPT_PATH}")"
echo "  评测数据:   ${EVAL_DATA_PATH}"
echo "  结果输出:   ${EVAL_RESULT_PATH}"
echo ""

python -u "${PROJECT_DIR}/infer.py" \
    --mm_emb_id 82 \
    --ensemble_mode none \
    "$@"

echo ""
echo "=========================================="
echo "  ✅ 推理完成"
echo "=========================================="
echo "  结果: ${EVAL_RESULT_PATH}/"
ls -lh "${EVAL_RESULT_PATH}/" 2>/dev/null
