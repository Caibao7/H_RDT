import argparse
import os
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import SiglipImageProcessor

from datasets.pretrain.mini_egodex_dataset import MiniEgoDexCollator, MiniEgoDexDataset
from models.mini_vla_diffusion import MiniVLADiffusionPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight EgoDex-only VLA diffusion policy.")
    parser.add_argument("--config_path", type=str, default="configs/mini_vla_egodex.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--report_to", type=str, default=None)
    return parser.parse_args()


def build_image_transform(image_size: int, vision_model_name_or_path: str):
    processor = SiglipImageProcessor.from_pretrained(vision_model_name_or_path)
    mean = tuple(processor.image_mean)
    std = tuple(processor.image_std)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


@torch.no_grad()
def evaluate(accelerator, model, dataloader, weight_dtype):
    model.eval()
    loss_sum = 0.0
    sample_mse_sum = 0.0
    count = 0
    for batch in dataloader:
        states = batch["states"].to(accelerator.device, dtype=weight_dtype)
        actions = batch["actions"].to(accelerator.device, dtype=weight_dtype)
        images = batch["images"].to(accelerator.device, dtype=weight_dtype)
        lang_embeds = batch.get("lang_embeds")
        lang_attn_mask = batch.get("lang_attn_mask")
        if lang_embeds is not None:
            lang_embeds = lang_embeds.to(accelerator.device, dtype=weight_dtype)
        if lang_attn_mask is not None:
            lang_attn_mask = lang_attn_mask.to(accelerator.device)

        output = model(
            states=states,
            actions=actions,
            images=images,
            lang_tokens=lang_embeds,
            lang_attn_mask=lang_attn_mask,
            instructions=batch["instructions"],
        )
        pred_actions = model.sample_actions(
            states=states,
            images=images,
            lang_tokens=lang_embeds,
            lang_attn_mask=lang_attn_mask,
            instructions=batch["instructions"],
        )
        mse = torch.mean((pred_actions - actions) ** 2)

        gathered_loss = accelerator.gather_for_metrics(output.loss.detach().unsqueeze(0))
        gathered_mse = accelerator.gather_for_metrics(mse.detach().unsqueeze(0))
        loss_sum += gathered_loss.mean().item()
        sample_mse_sum += gathered_mse.mean().item()
        count += 1
    model.train()
    if count == 0:
        return {"eval/loss": 0.0, "eval/sample_mse": 0.0}
    return {"eval/loss": loss_sum / count, "eval/sample_mse": sample_mse_sum / count}


def main():
    args = parse_args()
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_root = args.data_root or config["dataset"]["data_root"]
    output_dir = args.output_dir or config["train"]["output_dir"]
    report_to = args.report_to or config["train"].get("report_to", "none")
    if report_to == "none":
        report_to = None
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging_dir = os.path.join(output_dir, config["train"].get("logging_dir", "logs"))
    project_config = ProjectConfiguration(
        project_dir=logging_dir,
        total_limit=config["train"].get("checkpoints_total_limit"),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config["train"]["gradient_accumulation_steps"],
        mixed_precision=config["train"]["mixed_precision"],
        log_with=report_to,
        project_config=project_config,
    )
    set_seed(config["train"]["seed"])
    if accelerator.is_main_process:
        with open(os.path.join(output_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    accelerator.init_trackers("mini-vla-egodex", config=config)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    image_transform = build_image_transform(
        config["dataset"]["image_size"],
        config["model"]["vision"]["model_name_or_path"],
    )
    train_dataset = MiniEgoDexDataset(
        data_root=data_root,
        config=config,
        image_transform=image_transform,
        val=False,
        upsample_rate=config["dataset"]["upsample_rate"],
        use_precomp_lang_embed=config["dataset"]["use_precomp_lang_embed"],
        stats_path=config["dataset"]["stats_path"],
    )
    val_dataset = MiniEgoDexDataset(
        data_root=data_root,
        config=config,
        image_transform=image_transform,
        val=True,
        upsample_rate=config["dataset"]["upsample_rate"],
        use_precomp_lang_embed=config["dataset"]["use_precomp_lang_embed"],
        stats_path=config["dataset"]["stats_path"],
    )
    collator = MiniEgoDexCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["train_batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
        persistent_workers=config["train"]["num_workers"] > 0,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["eval_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )

    model = MiniVLADiffusionPolicy(
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
        diffusion_step_embed_dim=config["model"]["diffusion_step_embed_dim"],
        unet_down_dims=tuple(config["model"]["unet_down_dims"]),
        kernel_size=config["model"]["kernel_size"],
        n_groups=config["model"]["n_groups"],
        num_train_timesteps=config["model"]["num_train_timesteps"],
        num_inference_steps=config["model"]["num_inference_steps"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=max(config["train"]["max_train_steps"] // 100, 100),
        num_training_steps=config["train"]["max_train_steps"],
    )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    global_step = 0
    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
        if resume_path == "latest":
            checkpoint_dirs = [
                os.path.join(output_dir, name)
                for name in os.listdir(output_dir)
                if name.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, name))
            ]
            if checkpoint_dirs:
                checkpoint_dirs.sort(key=lambda path: int(os.path.basename(path).split("-")[1]))
                resume_path = checkpoint_dirs[-1]
            else:
                resume_path = None
        if resume_path is not None:
            accelerator.print(f"Resuming from checkpoint: {resume_path}")
            accelerator.load_state(resume_path)
            try:
                global_step = int(os.path.basename(resume_path).split("-")[1])
            except (IndexError, ValueError):
                global_step = 0

    progress_bar = tqdm(
        initial=global_step,
        total=config["train"]["max_train_steps"],
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )
    while global_step < config["train"]["max_train_steps"]:
        for batch in train_loader:
            with accelerator.accumulate(model):
                states = batch["states"].to(accelerator.device, dtype=weight_dtype)
                actions = batch["actions"].to(accelerator.device, dtype=weight_dtype)
                images = batch["images"].to(accelerator.device, dtype=weight_dtype)
                lang_embeds = batch.get("lang_embeds")
                lang_attn_mask = batch.get("lang_attn_mask")
                if lang_embeds is not None:
                    lang_embeds = lang_embeds.to(accelerator.device, dtype=weight_dtype)
                if lang_attn_mask is not None:
                    lang_attn_mask = lang_attn_mask.to(accelerator.device)

                output = model(
                    states=states,
                    actions=actions,
                    images=images,
                    lang_tokens=lang_embeds,
                    lang_attn_mask=lang_attn_mask,
                    instructions=batch["instructions"],
                )
                accelerator.backward(output.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config["train"]["max_grad_norm"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=output.loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])
                accelerator.log(
                    {
                        "train/loss": output.loss.detach().item(),
                        "train/diff_loss": output.diff_loss.detach().item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

                if global_step % config["train"]["eval_period"] == 0:
                    metrics = evaluate(accelerator, model, val_loader, weight_dtype)
                    accelerator.log(metrics, step=global_step)

                if global_step % config["train"]["checkpointing_period"] == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(checkpoint_dir)

                if global_step >= config["train"]["max_train_steps"]:
                    break

    progress_bar.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(output_dir, "mini_vla_final.pt")
        torch.save(
            {
                "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                "config": config,
                "global_step": global_step,
            },
            final_path,
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
