import argparse
import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from datasets.pretrain.mini_egodex_dataset import MiniEgoDexCollator, MiniEgoDexDataset
from models.mini_vla_diffusion import MiniVLADiffusionPolicy
from train.train_mini_vla import (
    build_image_transform,
    dataloader_worker_init_fn,
    evaluate,
    load_checkpoint_config,
    load_yaml_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a mini VLA checkpoint on the EgoDex test split.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint-* dir or mini_vla_final.pt.")
    parser.add_argument("--config_path", type=str, default="configs/mini_vla_egodex.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--vision_model_path", type=str, default=None)
    parser.add_argument("--text_model_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None, help="Use <=0 to evaluate the full test split.")
    parser.add_argument("--sample_actions", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--no_use_ema", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def load_eval_config(checkpoint: str, config_path: str) -> dict:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_file() and checkpoint_path.suffix == ".pt":
        payload = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(payload, dict) and "config" in payload:
            return payload["config"]

    if checkpoint_path.is_dir():
        checkpoint_config, _ = load_checkpoint_config(str(checkpoint_path))
        if checkpoint_config is not None:
            return checkpoint_config

    return load_yaml_config(config_path)


def build_model(config: dict) -> MiniVLADiffusionPolicy:
    return MiniVLADiffusionPolicy(
        state_dim=config["common"]["state_dim"],
        action_dim=config["common"]["action_dim"],
        pred_horizon=config["common"]["action_chunk_size"],
        image_history_size=config["common"]["img_history_size"],
        obs_cond_dim=config["model"]["obs_cond_dim"],
        vision_model_name_or_path=config["model"]["vision"]["model_name_or_path"],
        vision_use_pretrained=config["model"]["vision"]["pretrained"],
        vision_freeze=config["model"]["vision"]["freeze"],
        vision_num_tokens=config["model"]["vision"]["num_tokens"],
        text_model_name_or_path=config["model"]["text"]["model_name_or_path"],
        text_use_pretrained=config["model"]["text"]["pretrained"],
        text_freeze=config["model"]["text"]["freeze"],
        text_max_length=config["model"]["text"]["max_length"],
        precomputed_text_dim=config["model"]["text"]["precomputed_dim"],
        use_online_text_encoder=config["model"]["text"]["use_online_text_encoder"],
        unet_global_cond_dim=config["model"].get("unet_global_cond_dim"),
        diffusion_step_embed_dim=config["model"]["diffusion_step_embed_dim"],
        unet_down_dims=tuple(config["model"]["unet_down_dims"]),
        kernel_size=config["model"]["kernel_size"],
        n_groups=config["model"]["n_groups"],
        num_train_timesteps=config["model"]["num_train_timesteps"],
        num_inference_steps=config["model"]["num_inference_steps"],
    )


def find_checkpoint_weight_file(checkpoint_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = checkpoint_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}")


def strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_model_weights(model: MiniVLADiffusionPolicy, checkpoint: str, *, use_ema: bool) -> None:
    checkpoint_path = Path(checkpoint)
    ema_state = None

    if checkpoint_path.is_file() and checkpoint_path.suffix == ".pt":
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
        ema_state = payload.get("ema_state") if isinstance(payload, dict) else None
    elif checkpoint_path.is_dir():
        weight_file = find_checkpoint_weight_file(checkpoint_path)
        if weight_file.suffix == ".safetensors":
            import safetensors.torch

            state_dict = safetensors.torch.load_file(str(weight_file), device="cpu")
        else:
            state_dict = torch.load(weight_file, map_location="cpu")
        ema_path = checkpoint_path / "ema_state.pt"
        if ema_path.exists():
            ema_state = torch.load(ema_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model.load_state_dict(strip_module_prefix(state_dict), strict=True)

    if not use_ema or ema_state is None:
        return

    shadow_params = ema_state.get("shadow_params", [])
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    all_params = list(model.parameters())
    if len(shadow_params) == len(trainable_params):
        target_params = trainable_params
    elif len(shadow_params) == len(all_params):
        target_params = all_params
    else:
        print(
            "Skipping EMA weights because parameter count does not match "
            f"(ema={len(shadow_params)}, trainable={len(trainable_params)}, total={len(all_params)}).",
            flush=True,
        )
        return

    for shadow, param in zip(shadow_params, target_params):
        param.data.copy_(shadow.to(device=param.device, dtype=param.dtype))


def main():
    args = parse_args()
    config = load_eval_config(args.checkpoint, args.config_path)
    if args.data_root is not None:
        config["dataset"]["data_root"] = args.data_root
    if args.vision_model_path is not None:
        config["model"]["vision"]["model_name_or_path"] = args.vision_model_path
    if args.text_model_path is not None:
        config["model"]["text"]["model_name_or_path"] = args.text_model_path
    if args.batch_size is not None:
        config["train"]["eval_batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["train"]["val_num_workers"] = args.num_workers
    if args.max_eval_batches is not None:
        config["train"]["max_eval_batches"] = None if args.max_eval_batches <= 0 else args.max_eval_batches
    if args.num_inference_steps is not None:
        config["model"]["num_inference_steps"] = args.num_inference_steps

    mixed_precision = args.mixed_precision or config["train"].get("mixed_precision", "no")
    accelerator = Accelerator(mixed_precision=mixed_precision)

    image_transform = build_image_transform(
        config["dataset"]["image_size"],
        config["model"]["vision"]["model_name_or_path"],
    )
    val_dataset = MiniEgoDexDataset(
        data_root=config["dataset"]["data_root"],
        config=config,
        image_transform=image_transform,
        val=True,
        upsample_rate=config["dataset"]["upsample_rate"],
        use_precomp_lang_embed=config["dataset"]["use_precomp_lang_embed"],
        stats_path=config["dataset"]["stats_path"],
        lang_embed_root=config["dataset"].get("lang_embed_root"),
    )

    val_num_workers = int(config["train"].get("val_num_workers", 0))
    loader_kwargs = {
        "batch_size": config["train"]["eval_batch_size"],
        "shuffle": False,
        "num_workers": val_num_workers,
        "pin_memory": True,
        "collate_fn": MiniEgoDexCollator(),
    }
    if val_num_workers > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": int(config["train"].get("prefetch_factor", 2)),
                "worker_init_fn": dataloader_worker_init_fn,
            }
        )
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    model = build_model(config)
    load_model_weights(model, args.checkpoint, use_ema=not args.no_use_ema)
    model, val_loader = accelerator.prepare(model, val_loader)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    metrics = evaluate(
        accelerator,
        model,
        val_loader,
        weight_dtype,
        max_batches=config["train"].get("max_eval_batches"),
        sample_actions=args.sample_actions,
    )
    metrics = {key: float(value) for key, value in metrics.items()}
    if accelerator.is_main_process:
        print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
        if args.output_json is not None:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "checkpoint": args.checkpoint,
                        "use_ema": not args.no_use_ema,
                        "sample_actions": args.sample_actions,
                        "metrics": metrics,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )


if __name__ == "__main__":
    main()
