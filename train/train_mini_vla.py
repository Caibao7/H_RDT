import argparse
import copy
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


class EMAModel:
    """Lightweight EMA state tracker for periodic eval/save."""

    def __init__(
        self,
        parameters,
        *,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ):
        params = list(parameters)
        self.shadow_params = [p.detach().clone() for p in params]
        self.update_after_step = int(update_after_step)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.optimization_step = 0
        self.collected_params = None

    def get_decay(self) -> float:
        step = max(0, self.optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** (-self.power)
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, parameters):
        params = list(parameters)
        decay = self.get_decay()
        one_minus_decay = 1.0 - decay
        for shadow, param in zip(self.shadow_params, params):
            if not param.requires_grad:
                shadow.copy_(param.detach())
            else:
                shadow.lerp_(param.detach(), one_minus_decay)
        self.optimization_step += 1

    @torch.no_grad()
    def store(self, parameters):
        self.collected_params = [param.detach().clone() for param in parameters]

    @torch.no_grad()
    def copy_to(self, parameters):
        for shadow, param in zip(self.shadow_params, parameters):
            param.data.copy_(shadow.data)

    @torch.no_grad()
    def restore(self, parameters):
        if self.collected_params is None:
            return
        for stored, param in zip(self.collected_params, parameters):
            param.data.copy_(stored.data)
        self.collected_params = None

    def state_dict(self):
        return {
            "shadow_params": [tensor.detach().cpu().clone() for tensor in self.shadow_params],
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }

    def load_state_dict(self, state_dict):
        self.optimization_step = int(state_dict["optimization_step"])
        self.update_after_step = int(state_dict["update_after_step"])
        self.inv_gamma = float(state_dict["inv_gamma"])
        self.power = float(state_dict["power"])
        self.min_value = float(state_dict["min_value"])
        self.max_value = float(state_dict["max_value"])
        self.shadow_params = [tensor.detach().clone() for tensor in state_dict["shadow_params"]]


def log_stage(accelerator: Accelerator, message: str) -> None:
    if accelerator.is_main_process:
        print(f"[mini-vla] {message}", flush=True)


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
        states = batch["states"].to(accelerator.device)
        actions = batch["actions"].to(accelerator.device)
        images = batch["images"].to(accelerator.device)
        lang_embeds = batch.get("lang_embeds")
        lang_attn_mask = batch.get("lang_attn_mask")
        if lang_embeds is not None:
            lang_embeds = lang_embeds.to(accelerator.device)
        if lang_attn_mask is not None:
            lang_attn_mask = lang_attn_mask.to(accelerator.device)

        with accelerator.autocast():
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


def save_training_state(accelerator, output_dir, global_step, epoch, ema=None):
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(checkpoint_dir)
    if accelerator.is_main_process:
        torch.save(
            {
                "global_step": global_step,
                "epoch": epoch,
            },
            os.path.join(checkpoint_dir, "training_meta.pt"),
        )
        if ema is not None:
            torch.save(ema.state_dict(), os.path.join(checkpoint_dir, "ema_state.pt"))
    return checkpoint_dir


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
    log_stage(
        accelerator,
        "accelerator initialized "
        f"(processes={accelerator.num_processes}, mixed_precision={accelerator.mixed_precision})",
    )
    set_seed(config["train"]["seed"])
    if accelerator.is_main_process:
        with open(os.path.join(output_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    accelerator.init_trackers("mini-vla-egodex", config=config)
    logging_steps = max(int(config["train"].get("logging_steps", 1)), 1)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    image_transform = build_image_transform(
        config["dataset"]["image_size"],
        config["model"]["vision"]["model_name_or_path"],
    )
    log_stage(accelerator, "building EgoDex datasets")
    train_dataset = MiniEgoDexDataset(
        data_root=data_root,
        config=config,
        image_transform=image_transform,
        val=False,
        upsample_rate=config["dataset"]["upsample_rate"],
        use_precomp_lang_embed=config["dataset"]["use_precomp_lang_embed"],
        stats_path=config["dataset"]["stats_path"],
        lang_embed_root=config["dataset"].get("lang_embed_root"),
    )
    val_dataset = MiniEgoDexDataset(
        data_root=data_root,
        config=config,
        image_transform=image_transform,
        val=True,
        upsample_rate=config["dataset"]["upsample_rate"],
        use_precomp_lang_embed=config["dataset"]["use_precomp_lang_embed"],
        stats_path=config["dataset"]["stats_path"],
        lang_embed_root=config["dataset"].get("lang_embed_root"),
    )
    log_stage(
        accelerator,
        f"datasets ready (train={len(train_dataset)}, val={len(val_dataset)})",
    )
    collator = MiniEgoDexCollator()

    log_stage(
        accelerator,
        "building dataloaders "
        f"(per_gpu_batch={config['train']['train_batch_size']}, "
        f"workers_per_process={config['train']['num_workers']})",
    )
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

    log_stage(accelerator, "building model")
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
    log_stage(accelerator, "model ready")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        betas=(config["train"]["adam_beta1"], config["train"]["adam_beta2"]),
        weight_decay=config["train"]["weight_decay"],
    )
    steps_per_epoch = max(len(train_loader) // config["train"]["gradient_accumulation_steps"], 1)
    if config["train"]["use_epoch_training"]:
        total_train_steps = config["train"]["total_epochs"] * steps_per_epoch
    else:
        total_train_steps = config["train"]["max_train_steps"]
    lr_scheduler = get_scheduler(
        config["train"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["train"]["lr_warmup_steps"],
        num_training_steps=max(total_train_steps, 1),
        num_cycles=config["train"].get("lr_num_cycles", 1),
        power=config["train"].get("lr_power", 1.0),
    )

    log_stage(accelerator, "preparing model, optimizer, dataloaders, and scheduler with accelerate")
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    log_stage(accelerator, "accelerate.prepare complete")

    unwrapped_model = accelerator.unwrap_model(model)
    ema = None
    ema_config = config["train"].get("ema", {})
    if ema_config.get("enabled", False):
        ema = EMAModel(
            unwrapped_model.parameters(),
            update_after_step=ema_config.get("update_after_step", 0),
            inv_gamma=ema_config.get("inv_gamma", 1.0),
            power=ema_config.get("power", 0.75),
            min_value=ema_config.get("min_value", 0.0),
            max_value=ema_config.get("max_value", 0.9999),
        )

    global_step = 0
    start_epoch = 0
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
            meta_path = os.path.join(resume_path, "training_meta.pt")
            if os.path.exists(meta_path):
                meta = torch.load(meta_path, map_location="cpu")
                global_step = int(meta.get("global_step", 0))
                start_epoch = int(meta.get("epoch", 0))
            else:
                try:
                    global_step = int(os.path.basename(resume_path).split("-")[1])
                except (IndexError, ValueError):
                    global_step = 0
            if ema is not None:
                ema_path = os.path.join(resume_path, "ema_state.pt")
                if os.path.exists(ema_path):
                    ema.load_state_dict(torch.load(ema_path, map_location="cpu"))

    progress_bar = tqdm(
        initial=global_step,
        total=total_train_steps,
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )
    epoch = start_epoch
    max_epochs = config["train"]["total_epochs"] if config["train"]["use_epoch_training"] else 10**9
    stop_training = False
    log_stage(accelerator, "starting training loop")
    while not stop_training and epoch < max_epochs:
        for batch in train_loader:
            with accelerator.accumulate(model):
                states = batch["states"].to(accelerator.device)
                actions = batch["actions"].to(accelerator.device)
                images = batch["images"].to(accelerator.device)
                lang_embeds = batch.get("lang_embeds")
                lang_attn_mask = batch.get("lang_attn_mask")
                if lang_embeds is not None:
                    lang_embeds = lang_embeds.to(accelerator.device)
                if lang_attn_mask is not None:
                    lang_attn_mask = lang_attn_mask.to(accelerator.device)

                with accelerator.autocast():
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
                if ema is not None:
                    ema.step(accelerator.unwrap_model(model).parameters())
                progress_bar.update(1)
                progress_bar.set_postfix(
                    epoch=epoch + 1,
                    loss=output.loss.detach().item(),
                    lr=lr_scheduler.get_last_lr()[0],
                )
                if global_step == 1 or global_step % logging_steps == 0:
                    accelerator.log(
                        {
                            "train/loss": output.loss.detach().item(),
                            "train/diff_loss": output.diff_loss.detach().item(),
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

                do_step_eval = (not config["train"]["use_epoch_training"]) and global_step % config["train"]["eval_period"] == 0
                if do_step_eval:
                    if ema is not None:
                        ema.store(accelerator.unwrap_model(model).parameters())
                        ema.copy_to(accelerator.unwrap_model(model).parameters())
                    metrics = evaluate(accelerator, model, val_loader, weight_dtype)
                    if ema is not None:
                        ema.restore(accelerator.unwrap_model(model).parameters())
                    accelerator.log(metrics, step=global_step)

                do_step_save = (not config["train"]["use_epoch_training"]) and global_step % config["train"]["checkpointing_period"] == 0
                if do_step_save:
                    save_training_state(accelerator, output_dir, global_step, epoch, ema=ema)

                if global_step >= total_train_steps:
                    stop_training = True
                    break
        epoch += 1
        if config["train"]["use_epoch_training"]:
            if config["train"].get("epoch_eval_freq", 0) > 0 and epoch % config["train"]["epoch_eval_freq"] == 0:
                if ema is not None:
                    ema.store(accelerator.unwrap_model(model).parameters())
                    ema.copy_to(accelerator.unwrap_model(model).parameters())
                metrics = evaluate(accelerator, model, val_loader, weight_dtype)
                if ema is not None:
                    ema.restore(accelerator.unwrap_model(model).parameters())
                accelerator.log(metrics, step=global_step)
            if config["train"].get("epoch_save_freq", 0) > 0 and (
                epoch % config["train"]["epoch_save_freq"] == 0 or epoch == config["train"]["total_epochs"]
            ):
                save_training_state(accelerator, output_dir, global_step, epoch, ema=ema)
            if global_step >= total_train_steps:
                stop_training = True

    progress_bar.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_state_dict = accelerator.unwrap_model(model).state_dict()
        if ema is not None:
            ema.store(accelerator.unwrap_model(model).parameters())
            ema.copy_to(accelerator.unwrap_model(model).parameters())
            save_state_dict = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
            ema.restore(accelerator.unwrap_model(model).parameters())
        final_path = os.path.join(output_dir, "mini_vla_final.pt")
        torch.save(
            {
                "model_state_dict": save_state_dict,
                "config": config,
                "global_step": global_step,
                "epoch": epoch,
                "ema_state": ema.state_dict() if ema is not None else None,
            },
            final_path,
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
