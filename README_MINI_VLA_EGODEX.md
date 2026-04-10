# Mini VLA on EgoDex

This is a lightweight EgoDex-only training path for predicting future human hand actions with a diffusion head. It reuses H-RDT's EgoDex preprocessing (`actions_48d`, normalization stats, optional precomputed language embeddings) but does not use the large H-RDT backbone.

The current mini architecture is:

- Vision: small SigLIP vision tower
- Language: T5-small online encoder or matching precomputed embeddings
- Diffusion head: 1D conditional UNet with DDPM scheduler

## What It Supports

- EgoDex-only training and validation split
- 48D hand action state/action representation from `actions_48d`
- Single-frame or short-history RGB input
- Optional precomputed language embeddings from `datasets/pretrain/encode_lang_batch.py`
- Diffusion-style action chunk prediction
- Multi-GPU training through `accelerate`
- Optional `wandb` logging through `accelerate`
- Checkpoint save/resume through `accelerate.save_state` / `load_state`

## Main Files

- Dataset: [datasets/pretrain/mini_egodex_dataset.py](/home/caiyy/codefield/VLA/H_RDT/datasets/pretrain/mini_egodex_dataset.py)
- Model: [models/mini_vla_diffusion.py](/home/caiyy/codefield/VLA/H_RDT/models/mini_vla_diffusion.py)
- Config: [configs/mini_vla_egodex.yaml](/home/caiyy/codefield/VLA/H_RDT/configs/mini_vla_egodex.yaml)
- Train entry: [train/train_mini_vla.py](/home/caiyy/codefield/VLA/H_RDT/train/train_mini_vla.py)
- Launch script: [train_mini_egodex.sh](/home/caiyy/codefield/VLA/H_RDT/train_mini_egodex.sh)

## Data Requirements

Before training, EgoDex episodes need:

1. `actions_48d` written into each `.hdf5`
2. `egodex_stat.json`
3. Optional per-episode `.pt` language embedding files

Use the existing preprocessing scripts:

```bash
python -m datasets.pretrain.precompute_48d_actions --data_root /path/to/egodex
python -m datasets.pretrain.calc_stat --data_root /path/to/egodex
python -m datasets.pretrain.encode_lang_batch \
  --data_root /path/to/egodex \
  --model_path google-t5/t5-small \
  --config_path configs/mini_vla_egodex.yaml \
  --num_gpus 4 \
  --processes_per_gpu 1 \
  --output_root /path/to/egodex_t5_small_embeds
```

If you want to overwrite the existing per-episode `.pt` files in-place instead, omit `--output_root` and add `--force_overwrite`.

Important:

- If you train with precomputed language embeddings and later want online text evaluation, the safest setup is to precompute embeddings with the same text encoder family you plan to use online.
- The default mini config now assumes T5-small-compatible embeddings with dimension `512`.
- If you still use the old H-RDT T5-XXL embeddings, change `model.text.precomputed_dim` accordingly.

## Configure

Edit [configs/mini_vla_egodex.yaml](/home/caiyy/codefield/VLA/H_RDT/configs/mini_vla_egodex.yaml):

- `dataset.data_root`: EgoDex root
- `dataset.stats_path`: path to `egodex_stat.json`
- `dataset.lang_embed_root`: optional alternate root for precomputed language embeddings
- `train.output_dir`: checkpoint directory
- `train.report_to`: `none`, `wandb`, `tensorboard`, or `all`
- `model.text.use_online_text_encoder`: `false` for cached embeddings during training, `true` to encode raw text online

Recommended first-run settings:

- `model.obs_cond_dim: 384`
- `model.unet_down_dims: [256, 512, 1024]`
- `common.action_chunk_size: 8`
- `dataset.image_size: 224`
- `train.train_batch_size`: per-GPU batch size
- `global batch size = train_batch_size * num_gpus * gradient_accumulation_steps`

With the current default config:

- `train.train_batch_size: 32`
- `train.gradient_accumulation_steps: 4`
- `num_gpus: 4`
- Global batch size = `32 * 4 * 4 = 512`

## Launch

Single node, 4 GPUs:

```bash
accelerate launch --num_processes 4 -m train.train_mini_vla \
  --config_path configs/mini_vla_egodex.yaml \
  --data_root /root/shared-nvme/egodex \
  --output_dir ./checkpoints/mini-egodex
```

Or use the helper script after editing its env vars:

```bash
bash train_mini_egodex.sh
```

## Resume

Resume from a specific checkpoint:

```bash
accelerate launch --num_processes 4 -m train.train_mini_vla \
  --config_path configs/mini_vla_egodex.yaml \
  --data_root /root/shared-nvme/egodex \
  --output_dir ./checkpoints/mini-egodex \
  --resume_from_checkpoint ./checkpoints/mini-egodex/checkpoint-2000
```

Resume from the latest checkpoint in `output_dir`:

```bash
accelerate launch --num_processes 4 -m train.train_mini_vla \
  --config_path configs/mini_vla_egodex.yaml \
  --data_root /root/shared-nvme/egodex \
  --output_dir ./checkpoints/mini-egodex \
  --resume_from_checkpoint latest
```

## Multi-GPU Behavior

- Training uses `accelerate` data parallelism on a single node
- `train.train_batch_size` is per process, which in your setup means per GPU
- Each GPU keeps a full model replica
- Data is sharded across GPUs by `accelerate.prepare(...)`
- Gradients are synchronized automatically by DDP through `accelerate`
- This is not model parallelism, FSDP, or DeepSpeed ZeRO

## Save Behavior

- Periodic checkpoints are saved under `output_dir/checkpoint-<step>`
- Final weights are saved to `output_dir/mini_vla_final.pt`
- Resolved config is saved to `output_dir/resolved_config.yaml`

`mini_vla_final.pt` contains:

- `model_state_dict`
- `config`
- `global_step`

## Current Training Features Check

- `wandb`: supported if `train.report_to=wandb` and the environment is already configured for Weights & Biases
- Resume: supported for explicit checkpoint paths and `latest`
- Save: periodic `accelerate` checkpoints plus final `.pt` export
- Multi-GPU: supported through `accelerate launch --num_processes N`
- Mixed precision: controlled by `train.mixed_precision`
- Gradient accumulation: controlled by `train.gradient_accumulation_steps`

## Current Limitations

- The online and precomputed text paths are only distribution-matched if they come from the same text encoder family
- There is no standalone inference script yet for this mini model
- Validation currently reports loss and sampled MSE only
- The final export is a plain PyTorch checkpoint, not a Hugging Face `save_pretrained` package
