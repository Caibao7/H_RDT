export OUTPUT_DIR="./checkpoints/mini-egodex"
export DATA_ROOT="/root/shared-nvme/egodex"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

accelerate launch --num_processes 4 -m train.train_mini_vla \
    --config_path="configs/mini_vla_egodex.yaml" \
    --data_root="$DATA_ROOT" \
    --output_dir="$OUTPUT_DIR"
