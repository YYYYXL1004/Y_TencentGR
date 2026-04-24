#!/bin/bash
# ============================================================
#  TAAC2025 本地复现脚本
#  
#  用法: bash run_local.sh [download|convert|preprocess|train|infer|all]
#  
#  完整流程:
#    1. pip install -r requirements.txt
#    2. bash run_local.sh download              # 下载HF数据集 (hf-mirror镜像)
#    3. bash run_local.sh convert               # Parquet → 比赛JSONL格式
#    4. bash run_local.sh preprocess            # 生成Alias负采样表
#    5. bash run_local.sh train                 # 训练
#    6. bash run_local.sh infer                 # 推理
#
#  或使用 demo 数据快速验证:
#    bash run_local.sh preprocess --demo
#    bash run_local.sh train --demo
# ============================================================

set -e

# ========================= 路径配置 =========================
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
CACHE_DIR="${PROJECT_DIR}/cache"
CKPT_DIR="${PROJECT_DIR}/ckpt"
LOG_DIR="${PROJECT_DIR}/log"

# Demo 数据目录
DEMO_DIR="${PROJECT_DIR}/data_demo"

# 检查 --demo 标志
USE_DEMO=false
for arg in "$@"; do
    if [ "$arg" == "--demo" ]; then
        USE_DEMO=true
    fi
done

if [ "${USE_DEMO}" = true ]; then
    DATA_DIR="${DEMO_DIR}"
    echo "📌 使用 Demo 数据: ${DATA_DIR}"
fi

# 创建必要目录
mkdir -p "${CACHE_DIR}" "${CKPT_DIR}" "${LOG_DIR}" "${LOG_DIR}/tf_events" "${PROJECT_DIR}/results"

# ========================= 设置环境变量 =========================
# 训练阶段环境变量
export TRAIN_DATA_PATH="${DATA_DIR}"
export USER_CACHE_PATH="${CACHE_DIR}"
export TRAIN_CKPT_PATH="${CKPT_DIR}"
export TRAIN_LOG_PATH="${LOG_DIR}"
export TRAIN_TF_EVENTS_PATH="${LOG_DIR}/tf_events"

# 推理阶段环境变量
export EVAL_DATA_PATH="${DATA_DIR}"
export EVAL_RESULT_PATH="${PROJECT_DIR}/results"
# MODEL_OUTPUT_PATH: 推理时自动选择最新checkpoint, 见下方infer函数

# ========================= 功能函数 =========================

do_download() {
    echo "📥 [Download] 从 HuggingFace 下载 TencentGR-1M 数据集..."
    python "${PROJECT_DIR}/scripts/download_hf_data.py" "$@"
    echo "✅ 下载完成"
}

do_convert() {
    echo "🔄 [Convert] Parquet → 比赛 JSONL 格式..."
    python "${PROJECT_DIR}/scripts/convert_hf_to_competition.py" \
        --data_dir "${PROJECT_DIR}/data" \
        "$@"
    echo "✅ 转换完成"
}

do_preprocess() {
    echo "🔧 [Step 1] 生成偏移文件 (如已存在则跳过)..."
    if [ ! -f "${DATA_DIR}/seq_offsets.pkl" ]; then
        python "${PROJECT_DIR}/scripts/generate_offsets.py" --data_dir "${DATA_DIR}"
    else
        echo "  ✅ 偏移文件已存在，跳过"
    fi

    echo "🔧 [Step 2] 生成 Alias Table 负采样表..."
    python "${PROJECT_DIR}/util/preprocess_alias.py"
    echo "✅ 预处理完成"
}

do_train() {
    echo "🚀 [Train] 开始训练..."
    echo "  数据路径: ${TRAIN_DATA_PATH}"
    echo "  缓存路径: ${USER_CACHE_PATH}"
    echo "  模型保存: ${TRAIN_CKPT_PATH}"
    
    python -u "${PROJECT_DIR}/main.py" \
        --use_gradient_checkpointing \
        --amp_bf16 \
        "$@"

    echo "✅ 训练完成"
}

do_infer() {
    # 自动选择 NDCG 最高的 checkpoint (而非最新)
    BEST_CKPT=$(ls -1d "${CKPT_DIR}"/global_step* 2>/dev/null | sort -t= -k2 -rn | head -1)
    if [ -z "${BEST_CKPT}" ]; then
        echo "❌ 未找到任何 checkpoint，请先运行训练"
        exit 1
    fi
    export MODEL_OUTPUT_PATH="${BEST_CKPT}"
    echo "🔍 [Infer] 使用最佳 checkpoint: $(basename ${MODEL_OUTPUT_PATH})"
    echo "  评测数据: ${EVAL_DATA_PATH}"
    echo "  结果输出: ${EVAL_RESULT_PATH}"

    python -u "${PROJECT_DIR}/infer.py" \
        --mm_emb_id 82 \
        --ensemble_mode none \
        "$@"

    echo "✅ 推理完成，结果保存在: ${EVAL_RESULT_PATH}"
}

# ========================= 主入口 =========================

MODE="${1:-help}"
shift 2>/dev/null || true  # 移除第一个参数，剩余传给子命令

# 过滤掉 --demo 标志，避免传递给子命令
FILTERED_ARGS=()
for arg in "$@"; do
    if [ "$arg" != "--demo" ]; then
        FILTERED_ARGS+=("$arg")
    fi
done

case "${MODE}" in
    download)
        do_download "${FILTERED_ARGS[@]}"
        ;;
    convert)
        do_convert "${FILTERED_ARGS[@]}"
        ;;
    preprocess)
        do_preprocess
        ;;
    train)
        do_train "${FILTERED_ARGS[@]}"
        ;;
    infer)
        do_infer "${FILTERED_ARGS[@]}"
        ;;
    all)
        do_preprocess
        do_train "${FILTERED_ARGS[@]}"
        do_infer
        ;;
    *)
        echo "=========================================="
        echo "  TAAC2025 本地复现工具"
        echo "=========================================="
        echo ""
        echo "用法: bash run_local.sh <命令> [选项]"
        echo ""
        echo "命令:"
        echo "  download    - 从 HuggingFace 下载 TencentGR-1M 数据集"
        echo "  convert     - 将 Parquet 转为比赛 JSONL 格式"
        echo "  preprocess  - 生成偏移文件和 Alias 采样表"
        echo "  train       - 训练模型"
        echo "  infer       - 推理"
        echo "  all         - 执行 preprocess + train + infer"
        echo ""
        echo "选项:"
        echo "  --demo      - 使用 data_demo/ 中的小规模数据快速验证"
        echo ""
        echo "完整 1M 数据集流程:"
        echo "  bash run_local.sh download"
        echo "  bash run_local.sh convert"
        echo "  bash run_local.sh preprocess"
        echo "  bash run_local.sh train"
        echo "  bash run_local.sh infer"
        echo ""
        echo "Demo 数据快速验证:"
        echo "  bash run_local.sh preprocess --demo"
        echo "  bash run_local.sh train --demo"
        exit 0
        ;;
esac
