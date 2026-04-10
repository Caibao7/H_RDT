from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from transformers import SiglipVisionModel, T5EncoderModel, T5Tokenizer


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=5, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels * 2))
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        scale, bias = torch.chunk(embed, 2, dim=1)
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims=(128, 256, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size=kernel_size, n_groups=n_groups),
            ]
        )

        self.down_modules = nn.ModuleList([])
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx == len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                        ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList([])
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = idx >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        sample = sample.moveaxis(-1, -2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timestep)
        global_feature = torch.cat([global_feature, global_cond], dim=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


class SiglipVisionEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        pooled_tokens: int,
        freeze: bool,
        use_pretrained: bool,
    ) -> None:
        super().__init__()
        self.model = SiglipVisionModel.from_pretrained(model_name_or_path)
        if not use_pretrained:
            self.model.init_weights()
        self.hidden_size = self.model.config.hidden_size
        self.freeze = freeze
        self.pooled_tokens = pooled_tokens
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
        else:
            outputs = self.model(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        if tokens.shape[1] > self.pooled_tokens:
            tokens = F.adaptive_avg_pool1d(tokens.transpose(1, 2), self.pooled_tokens).transpose(1, 2)
        return tokens


class T5TextConditioner(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        freeze: bool,
        max_length: int,
        use_pretrained: bool,
    ) -> None:
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        self.encoder = T5EncoderModel.from_pretrained(model_name_or_path)
        if not use_pretrained:
            self.encoder.init_weights()
        self.hidden_size = self.encoder.config.d_model
        self.max_length = max_length
        self.freeze = freeze
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def encode_texts(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        if self.freeze:
            with torch.no_grad():
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, attention_mask.bool()


@dataclass
class MiniVLAOutput:
    loss: torch.Tensor
    diff_loss: torch.Tensor


class MiniVLADiffusionPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        pred_horizon: int,
        image_history_size: int,
        obs_cond_dim: int = 256,
        vision_model_name_or_path: str = "google/siglip-base-patch16-224",
        vision_use_pretrained: bool = True,
        vision_freeze: bool = True,
        vision_num_tokens: int = 16,
        text_model_name_or_path: str = "google-t5/t5-small",
        text_use_pretrained: bool = True,
        text_freeze: bool = True,
        text_max_length: int = 128,
        precomputed_text_dim: int = 4096,
        use_online_text_encoder: bool = True,
        diffusion_step_embed_dim: int = 128,
        unet_down_dims=(128, 256, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 20,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.image_history_size = image_history_size
        self.use_online_text_encoder = use_online_text_encoder
        self.num_inference_steps = num_inference_steps

        self.vision_encoder = SiglipVisionEncoder(
            model_name_or_path=vision_model_name_or_path,
            pooled_tokens=vision_num_tokens,
            freeze=vision_freeze,
            use_pretrained=vision_use_pretrained,
        )
        self.vision_proj = nn.Linear(self.vision_encoder.hidden_size, obs_cond_dim)

        self.text_encoder = None
        if self.use_online_text_encoder:
            self.text_encoder = T5TextConditioner(
                model_name_or_path=text_model_name_or_path,
                freeze=text_freeze,
                max_length=text_max_length,
                use_pretrained=text_use_pretrained,
            )
            text_input_dim = self.text_encoder.hidden_size
        else:
            text_input_dim = precomputed_text_dim
        self.text_proj = nn.Linear(text_input_dim, obs_cond_dim)
        self.state_proj = nn.Linear(state_dim, obs_cond_dim)

        global_cond_dim = image_history_size * vision_num_tokens * obs_cond_dim + obs_cond_dim + obs_cond_dim
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def _encode_obs(
        self,
        states: torch.Tensor,
        images: torch.Tensor,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_attn_mask: Optional[torch.Tensor] = None,
        instructions: Optional[list[str]] = None,
    ) -> torch.Tensor:
        batch_size, history, channels, height, width = images.shape
        flat_images = images.view(batch_size * history, channels, height, width)
        vision_tokens = self.vision_encoder(flat_images)
        vision_tokens = self.vision_proj(vision_tokens)
        vision_tokens = vision_tokens.view(batch_size, -1)

        state_token = self.state_proj(states[:, 0])

        if self.use_online_text_encoder:
            if instructions is None:
                raise ValueError("instructions are required when use_online_text_encoder=True")
            text_hidden, text_mask = self.text_encoder.encode_texts(instructions, device=states.device)
            masked = text_hidden * text_mask.unsqueeze(-1)
            pooled = masked.sum(dim=1) / text_mask.sum(dim=1, keepdim=True).clamp_min(1)
        else:
            if lang_tokens is None:
                raise ValueError("lang_tokens are required when use_online_text_encoder=False")
            if lang_attn_mask is None:
                lang_attn_mask = torch.ones(lang_tokens.shape[:2], dtype=torch.bool, device=lang_tokens.device)
            masked = lang_tokens * lang_attn_mask.unsqueeze(-1)
            pooled = masked.sum(dim=1) / lang_attn_mask.sum(dim=1, keepdim=True).clamp_min(1)

        text_token = self.text_proj(pooled)
        return torch.cat([vision_tokens, state_token, text_token], dim=-1)

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        images: torch.Tensor,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_attn_mask: Optional[torch.Tensor] = None,
        instructions: Optional[list[str]] = None,
    ) -> MiniVLAOutput:
        batch_size = actions.shape[0]
        obs_cond = self._encode_obs(states, images, lang_tokens, lang_attn_mask, instructions)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=actions.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
        diff_loss = F.mse_loss(noise_pred.float(), noise.float())
        return MiniVLAOutput(loss=diff_loss, diff_loss=diff_loss)

    @torch.no_grad()
    def sample_actions(
        self,
        states: torch.Tensor,
        images: torch.Tensor,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_attn_mask: Optional[torch.Tensor] = None,
        instructions: Optional[list[str]] = None,
    ) -> torch.Tensor:
        obs_cond = self._encode_obs(states, images, lang_tokens, lang_attn_mask, instructions)
        batch_size = states.shape[0]
        action = torch.randn(
            (batch_size, self.pred_horizon, self.action_dim),
            device=states.device,
            dtype=states.dtype,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=states.device)
        for timestep in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(action, timestep, global_cond=obs_cond)
            action = self.noise_scheduler.step(noise_pred, timestep, action).prev_sample
        return action

    def forward(self, *args, **kwargs) -> MiniVLAOutput:
        return self.compute_loss(*args, **kwargs)
