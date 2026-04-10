export OUTPUT_DIR="./checkpoints/mini-egodex"
export DATA_ROOT="/path/to/egodex"

accelerate launch train/train_mini_vla.py \
    --config_path="configs/mini_vla_egodex.yaml" \
    --data_root="$DATA_ROOT" \
    --output_dir="$OUTPUT_DIR"
