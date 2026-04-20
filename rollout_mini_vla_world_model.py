#!/usr/bin/env python3
"""Smoke-test rollout for Mini VLA actions inside the EgoDex hand world model.

This script intentionally lives in H_RDT so the hand-wm checkout can remain
unchanged. It supports two useful first-pass checks:

1. Drive the world model with ground-truth EgoDex actions_48d.
2. Drive the world model with actions sampled from a trained Mini VLA checkpoint.

The main contract to verify is the 48D action convention. Mini VLA uses the
actions_48d order produced by datasets/pretrain/precompute_48d_actions.py:
left hand 24D, then right hand 24D; each hand is wrist xyz, wrist rotation 6D,
then five fingertip xyz positions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import h5py
import numpy as np
import torch
import yaml
from PIL import Image


HAND_WM_ROOT = Path("/home/caiyy/codefield/VLA/hand-wm")


@dataclass
class HandWorldModelConfig:
    device: str
    n_frames: int = 20
    action_dim: int = 48
    image_size: int = 256
    patch_size: int = 2
    model_dim: int = 1024
    layers: int = 16
    heads: int = 16
    timesteps: int = 1000
    sampling_timesteps: int = 10
    cfg: float = 1.0
    chunk_size: int = 1
    action_norm_stats_path: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out Mini VLA actions in an EgoDex-trained hand world model."
    )
    parser.add_argument("--hand_wm_root", type=Path, default=HAND_WM_ROOT)
    parser.add_argument("--wm_checkpoint", type=Path, default=HAND_WM_ROOT / "checkpoints/step_000160000")
    parser.add_argument("--mini_checkpoint", type=Path, default=Path("checkpoints/mini-egodex-rate1-chunk8/checkpoint-102063"))
    parser.add_argument("--mini_config", type=Path, default=Path("configs/mini_vla_egodex.yaml"))
    parser.add_argument("--no_mini_ema", action="store_true", help="Do not apply EMA weights when loading Mini VLA.")

    parser.add_argument("--data_root", type=Path, default=Path("/root/shared-nvme/egodex"))
    parser.add_argument("--egodex_hdf5", type=Path, default=None)
    parser.add_argument("--egodex_mp4", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)

    parser.add_argument("--action_source", choices=["gt", "mini"], default="gt")
    parser.add_argument(
        "--wm_action_space",
        choices=["raw", "minmax"],
        default="raw",
        help=(
            "Action values expected by the world model. Use raw if the WM was trained "
            "on raw actions_48d or uses its own action_norm_stats_path; use minmax if "
            "the WM was trained on Mini VLA style [-1, 1] actions."
        ),
    )
    parser.add_argument(
        "--wm_action_norm_stats_path",
        type=Path,
        default=None,
        help="Optional z-score stats file consumed by hand-wm WorldModel before action padding.",
    )
    parser.add_argument("--rollout_steps", type=int, default=8)
    parser.add_argument("--policy_replan_every", type=int, default=1)
    parser.add_argument("--fps", type=int, default=8)

    parser.add_argument("--wm_action_dim", type=int, default=48)
    parser.add_argument("--wm_n_frames", type=int, default=20)
    parser.add_argument("--wm_image_size", type=int, default=256)
    parser.add_argument("--wm_model_dim", type=int, default=1024)
    parser.add_argument("--wm_layers", type=int, default=16)
    parser.add_argument("--wm_heads", type=int, default=16)
    parser.add_argument("--wm_patch_size", type=int, default=2)
    parser.add_argument("--wm_sampling_timesteps", type=int, default=10)
    parser.add_argument("--wm_cfg", type=float, default=1.0)
    parser.add_argument("--wm_chunk_size", type=int, default=1)
    parser.add_argument("--disable_kv_cache", action="store_true")

    parser.add_argument("--output_dir", type=Path, default=Path("rollouts/mini_vla_world_model_smoke"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Build inputs and metadata without loading models.")
    return parser.parse_args()


def import_hand_world_model(hand_wm_root: Path):
    eval_dir = hand_wm_root / "src/world_model_eval"
    if not eval_dir.exists():
        raise FileNotFoundError(f"hand-wm world_model_eval directory not found: {eval_dir}")
    sys.path.insert(0, str(eval_dir))
    from world_model import WorldModel  # type: ignore

    return WorldModel


def collect_egodex_episodes(data_root: Path, split: str) -> list[dict[str, Path | str]]:
    parts = ["test"] if split == "test" else ["part1", "part2", "part3", "part4", "part5", "extra"]
    episodes: list[dict[str, Path | str]] = []
    for part in parts:
        part_dir = data_root / part
        if not part_dir.exists():
            continue
        for task_dir in sorted(part_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for hdf5_path in sorted(task_dir.glob("*.hdf5")):
                mp4_path = task_dir / f"{hdf5_path.stem}.mp4"
                if mp4_path.exists():
                    episodes.append({"hdf5": hdf5_path, "mp4": mp4_path, "task": task_dir.name})
    return episodes


def resolve_episode(args: argparse.Namespace) -> tuple[Path, Path, str]:
    if args.egodex_hdf5 is not None:
        hdf5_path = args.egodex_hdf5
        mp4_path = args.egodex_mp4 or hdf5_path.with_suffix(".mp4")
        if not hdf5_path.exists():
            raise FileNotFoundError(f"EgoDex HDF5 not found: {hdf5_path}")
        if not mp4_path.exists():
            raise FileNotFoundError(f"EgoDex MP4 not found: {mp4_path}")
        return hdf5_path, mp4_path, hdf5_path.parent.name

    episodes = collect_egodex_episodes(args.data_root, args.split)
    if not episodes:
        raise FileNotFoundError(f"No EgoDex episodes found under {args.data_root} split={args.split}")
    if args.episode_index < 0 or args.episode_index >= len(episodes):
        raise IndexError(f"episode_index={args.episode_index} out of range for {len(episodes)} episodes")
    episode = episodes[args.episode_index]
    return Path(episode["hdf5"]), Path(episode["mp4"]), str(episode["task"])


def load_egodex_stats(stats_path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    stat = stats["egodex"] if "egodex" in stats else stats
    action_min = np.asarray(stat["min"], dtype=np.float32)
    action_max = np.asarray(stat["max"], dtype=np.float32)
    if action_min.shape != (48,) or action_max.shape != (48,):
        raise ValueError(f"Expected 48D EgoDex min/max stats, got {action_min.shape}, {action_max.shape}")
    return action_min, action_max


def load_yaml_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_mini_config(checkpoint: Path, config_path: Path) -> dict[str, Any]:
    if checkpoint.is_file() and checkpoint.suffix == ".pt":
        payload = torch.load(checkpoint, map_location="cpu")
        if isinstance(payload, dict) and "config" in payload:
            return payload["config"]

    if checkpoint.is_dir():
        candidates = [
            checkpoint / "resolved_config.yaml",
            checkpoint / "training_config.yaml",
            checkpoint.parent / "resolved_config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return load_yaml_config(candidate)

    return load_yaml_config(config_path)


def minmax_normalize(action: np.ndarray | torch.Tensor, action_min: np.ndarray, action_max: np.ndarray):
    is_tensor = torch.is_tensor(action)
    if is_tensor:
        amin = torch.as_tensor(action_min, device=action.device, dtype=action.dtype)
        amax = torch.as_tensor(action_max, device=action.device, dtype=action.dtype)
        denom = torch.clamp(amax - amin, min=1e-6)
        return torch.clamp(((action - amin) / denom) * 2.0 - 1.0, -1.0, 1.0)
    denom = np.clip(action_max - action_min, a_min=1e-6, a_max=None)
    return np.clip(((action - action_min) / denom) * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)


def minmax_denormalize(action_norm: torch.Tensor, action_min: np.ndarray, action_max: np.ndarray) -> torch.Tensor:
    amin = torch.as_tensor(action_min, device=action_norm.device, dtype=action_norm.dtype)
    amax = torch.as_tensor(action_max, device=action_norm.device, dtype=action_norm.dtype)
    return ((action_norm + 1.0) * 0.5) * (amax - amin) + amin


def load_actions_and_instruction(hdf5_path: Path) -> tuple[np.ndarray, str]:
    with h5py.File(hdf5_path, "r") as f:
        if "actions_48d" not in f:
            raise KeyError(f"{hdf5_path} is missing actions_48d")
        actions = f["actions_48d"][:].astype(np.float32)
        attrs = f.attrs
        llm_type = attrs.get("llm_type", "")
        if isinstance(llm_type, bytes):
            llm_type = llm_type.decode("utf-8")
        if llm_type == "reversible":
            which = attrs.get("which_llm_description", "1")
            if isinstance(which, bytes):
                which = which.decode("utf-8")
            key = "llm_description2" if str(which) == "2" else "llm_description"
            instruction = attrs.get(key, "")
        else:
            instruction = attrs.get("llm_description", "")
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
    return actions, str(instruction)


def read_rgb_frame(mp4_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {mp4_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Video has no decodable frames: {mp4_path}")
        frame_idx = min(max(frame_idx, 0), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to decode frame {frame_idx} from {mp4_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def center_square_resize(frame_rgb: np.ndarray, image_size: int) -> np.ndarray:
    image = Image.fromarray(frame_rgb)
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def frame_to_world_tensor(frame_rgb: np.ndarray, image_size: int, device: str) -> torch.Tensor:
    frame = center_square_resize(frame_rgb, image_size)
    tensor = torch.from_numpy(frame).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def frame_to_mini_tensor(frame_rgb_float: torch.Tensor, image_transform, device: str) -> torch.Tensor:
    frame_np = (frame_rgb_float.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    image = Image.fromarray(frame_np)
    tensor = image_transform(image)
    return tensor.unsqueeze(0).unsqueeze(0).to(device)


def save_mp4(frames_rgb: Iterable[np.ndarray], path: Path, fps: int) -> None:
    frames = list(frames_rgb)
    if not frames:
        raise ValueError("No frames to save")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def action_dim_names() -> list[str]:
    names: list[str] = []
    fingertips = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]
    for hand in ["left", "right"]:
        names.extend([f"{hand}.wrist_pos.{axis}" for axis in ["x", "y", "z"]])
        names.extend([f"{hand}.wrist_rot6d.c0.{axis}" for axis in ["x", "y", "z"]])
        names.extend([f"{hand}.wrist_rot6d.c1.{axis}" for axis in ["x", "y", "z"]])
        for finger in fingertips:
            names.extend([f"{hand}.{finger}.{axis}" for axis in ["x", "y", "z"]])
    return names


def load_mini_policy(args: argparse.Namespace, mini_config: dict[str, Any]):
    from train.eval_mini_vla import build_model as build_mini_vla_model
    from train.eval_mini_vla import load_model_weights

    model = build_mini_vla_model(mini_config)
    load_model_weights(model, str(args.mini_checkpoint), use_ema=not args.no_mini_ema)
    model.to(args.device)
    model.eval()
    return model


def prepare_world_model(args: argparse.Namespace):
    WorldModel = import_hand_world_model(args.hand_wm_root)
    config = HandWorldModelConfig(
        device=args.device,
        n_frames=args.wm_n_frames,
        action_dim=args.wm_action_dim,
        image_size=args.wm_image_size,
        patch_size=args.wm_patch_size,
        model_dim=args.wm_model_dim,
        layers=args.wm_layers,
        heads=args.wm_heads,
        sampling_timesteps=args.wm_sampling_timesteps,
        cfg=args.wm_cfg,
        chunk_size=args.wm_chunk_size,
        action_norm_stats_path=str(args.wm_action_norm_stats_path) if args.wm_action_norm_stats_path else None,
    )
    return WorldModel(
        checkpoint_path=str(args.wm_checkpoint),
        config=config,
        use_kv_cache=not args.disable_kv_cache,
    ), config


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.policy_replan_every <= 0:
        raise ValueError("--policy_replan_every must be positive")
    if args.wm_chunk_size != 1:
        raise ValueError("This smoke-test script currently expects --wm_chunk_size 1")

    hdf5_path, mp4_path, task = resolve_episode(args)
    actions_raw, instruction = load_actions_and_instruction(hdf5_path)
    if actions_raw.shape[1] != 48:
        raise ValueError(f"Expected actions_48d shape (*, 48), got {actions_raw.shape}")
    if args.start_idx < 0 or args.start_idx >= actions_raw.shape[0] - 1:
        raise IndexError(f"start_idx={args.start_idx} invalid for {actions_raw.shape[0]} action frames")

    mini_config = load_mini_config(args.mini_checkpoint, args.mini_config)
    stats_path = mini_config["dataset"]["stats_path"]
    action_min, action_max = load_egodex_stats(stats_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "hdf5_path": str(hdf5_path),
        "mp4_path": str(mp4_path),
        "task": task,
        "instruction": instruction,
        "action_source": args.action_source,
        "wm_action_space": args.wm_action_space,
        "start_idx": args.start_idx,
        "rollout_steps": args.rollout_steps,
        "mini_checkpoint": str(args.mini_checkpoint),
        "wm_checkpoint": str(args.wm_checkpoint),
        "wm_config": asdict(
            HandWorldModelConfig(
                device=args.device,
                n_frames=args.wm_n_frames,
                action_dim=args.wm_action_dim,
                image_size=args.wm_image_size,
                patch_size=args.wm_patch_size,
                model_dim=args.wm_model_dim,
                layers=args.wm_layers,
                heads=args.wm_heads,
                sampling_timesteps=args.wm_sampling_timesteps,
                cfg=args.wm_cfg,
                chunk_size=args.wm_chunk_size,
                action_norm_stats_path=str(args.wm_action_norm_stats_path) if args.wm_action_norm_stats_path else None,
            )
        ),
        "action_dim_names": action_dim_names(),
    }

    if args.dry_run:
        with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Dry run complete. Metadata written to {args.output_dir / 'metadata.json'}")
        return

    mini_policy = None
    mini_image_transform = None
    if args.action_source == "mini":
        from train.train_mini_vla import build_image_transform

        mini_policy = load_mini_policy(args, mini_config)
        mini_image_transform = build_image_transform(
            mini_config["dataset"]["image_size"],
            mini_config["model"]["vision"]["model_name_or_path"],
        )

    world_model, wm_config = prepare_world_model(args)
    world_model.model.eval()

    init_frame_rgb = read_rgb_frame(mp4_path, args.start_idx)
    current_frame_world = frame_to_world_tensor(init_frame_rgb, wm_config.image_size, args.device)
    world_model.reset(current_frame_world)

    output_frames: list[np.ndarray] = [center_square_resize(init_frame_rgb, wm_config.image_size)]
    rollout_actions: list[dict[str, Any]] = []

    current_state_norm = torch.from_numpy(
        minmax_normalize(actions_raw[args.start_idx], action_min, action_max)
    ).float().view(1, 1, 48).to(args.device)
    current_frame_for_policy = current_frame_world[0]
    planned_actions_norm: torch.Tensor | None = None
    plan_offset = 0

    with torch.no_grad():
        for step in range(args.rollout_steps):
            src_frame_idx = args.start_idx + step + 1
            if src_frame_idx >= actions_raw.shape[0]:
                break

            if args.action_source == "gt":
                action_raw = torch.from_numpy(actions_raw[src_frame_idx]).float().view(1, 48).to(args.device)
                action_norm = minmax_normalize(action_raw, action_min, action_max)
            else:
                assert mini_policy is not None
                assert mini_image_transform is not None
                if planned_actions_norm is None or plan_offset >= min(args.policy_replan_every, planned_actions_norm.shape[1]):
                    image_tensor = frame_to_mini_tensor(current_frame_for_policy, mini_image_transform, args.device)
                    planned_actions_norm = mini_policy.sample_actions(
                        states=current_state_norm,
                        images=image_tensor,
                        instructions=[instruction],
                    )
                    plan_offset = 0
                action_norm = planned_actions_norm[:, plan_offset]
                action_raw = minmax_denormalize(action_norm, action_min, action_max)
                plan_offset += 1

            wm_action = action_raw if args.wm_action_space == "raw" else action_norm
            generated_any = False
            for frame_idx, decoded in world_model.generate_chunk(wm_action):
                generated_any = True
                frame = decoded[0, 0].detach().float().clamp(0, 1).cpu()
                frame_np = (frame.numpy() * 255.0).astype(np.uint8)
                output_frames.append(frame_np)
                current_frame_for_policy = decoded[0, 0].detach()

            if not generated_any:
                raise RuntimeError(f"World model did not yield a decoded frame at rollout step {step}")

            current_state_norm = action_norm.view(1, 1, 48)
            rollout_actions.append(
                {
                    "step": step,
                    "source_frame_idx": src_frame_idx,
                    "wm_frame_idx": int(frame_idx),
                    "action_raw_first8": action_raw[0, :8].detach().cpu().tolist(),
                    "action_norm_first8": action_norm[0, :8].detach().cpu().tolist(),
                }
            )

    metadata["actual_rollout_steps"] = len(rollout_actions)
    metadata["rollout_actions"] = rollout_actions
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    save_mp4(output_frames, args.output_dir / "rollout.mp4", fps=args.fps)
    print(f"Saved rollout video: {args.output_dir / 'rollout.mp4'}")
    print(f"Saved metadata: {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
