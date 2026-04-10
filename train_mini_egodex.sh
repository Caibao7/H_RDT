export OUTPUT_DIR="./checkpoints/mini-egodex"
export DATA_ROOT="/root/shared-nvme/egodex"

accelerate launch --num_processes 4 -m train.train_mini_vla \
    --config_path="configs/mini_vla_egodex.yaml" \
    --data_root="$DATA_ROOT" \
    --output_dir="$OUTPUT_DIR"
