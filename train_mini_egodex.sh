export OUTPUT_DIR="./checkpoints/mini-egodex"
export DATA_ROOT="/root/shared-nvme/egodex"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

EXTRA_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
    EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
if [ -n "${REPORT_TO:-}" ]; then
    EXTRA_ARGS+=(--report_to "$REPORT_TO")
fi

accelerate launch --num_processes "$NUM_PROCESSES" -m train.train_mini_vla \
    --config_path="configs/mini_vla_egodex.yaml" \
    --data_root="$DATA_ROOT" \
    --output_dir="$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
