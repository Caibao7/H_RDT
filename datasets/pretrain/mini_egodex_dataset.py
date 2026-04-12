import os
import random
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass


class MiniEgoDexDataset(Dataset):
    """EgoDex dataset tailored for lightweight hand-action prediction."""

    def __init__(
        self,
        data_root: str,
        config: Dict,
        image_transform=None,
        val: bool = False,
        upsample_rate: int = 3,
        use_precomp_lang_embed: bool = True,
        stats_path: str | None = None,
        lang_embed_root: str | None = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.config = config
        self.val = val
        self.upsample_rate = upsample_rate
        self.use_precomp_lang_embed = use_precomp_lang_embed
        self.image_transform = image_transform or transforms.ToTensor()
        self.lang_embed_root = Path(lang_embed_root) if lang_embed_root else None

        self.action_chunk_size = config["common"]["action_chunk_size"]
        self.img_history_size = config["common"]["img_history_size"]
        dataset_config = config.get("dataset", {})
        self.max_lang_cache_items = int(dataset_config.get("max_lang_cache_items", 512))
        self.max_video_cache_items = int(dataset_config.get("max_video_cache_items", 4))
        self._lang_cache: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()
        self._video_cache: OrderedDict[str, cv2.VideoCapture] = OrderedDict()

        if stats_path is None:
            stats_path = os.path.join(os.path.dirname(__file__), "egodex_stat.json")
        self.stats_path = stats_path

        self.action_min = None
        self.action_max = None
        if os.path.exists(self.stats_path):
            if self.stats_path.endswith(".pt"):
                stat = torch.load(self.stats_path, map_location="cpu")
            else:
                with open(self.stats_path, "r", encoding="utf-8") as f:
                    stat = json.load(f)
            if "egodex" in stat:
                self.action_min = np.asarray(stat["egodex"]["min"], dtype=np.float32)
                self.action_max = np.asarray(stat["egodex"]["max"], dtype=np.float32)

        self.episodes = self._collect_episodes()
        split_name = "test" if val else "train"
        print(f"MiniEgoDexDataset loaded {len(self.episodes)} {split_name} episodes")

    def _collect_episodes(self) -> List[Dict[str, Path]]:
        parts = ["test"] if self.val else ["part1", "part2", "part3", "part4", "part5", "extra"]
        episodes: List[Dict[str, Path]] = []
        for part in parts:
            part_dir = self.data_root / part
            if not part_dir.exists():
                continue
            for task_dir in sorted(part_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                for hdf5_path in sorted(task_dir.glob("*.hdf5")):
                    stem = hdf5_path.stem
                    mp4_path = task_dir / f"{stem}.mp4"
                    if self.lang_embed_root is None:
                        pt_path = task_dir / f"{stem}.pt"
                    else:
                        relative_task_dir = task_dir.relative_to(self.data_root)
                        pt_path = self.lang_embed_root / relative_task_dir / f"{stem}.pt"
                    if not mp4_path.exists():
                        continue
                    episodes.append(
                        {
                            "hdf5": hdf5_path,
                            "mp4": mp4_path,
                            "pt": pt_path,
                            "task": task_dir.name,
                        }
                    )
        if not episodes:
            raise ValueError(f"No valid EgoDex episodes found under {self.data_root}")
        return episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_min is None or self.action_max is None:
            return action.astype(np.float32)
        denom = np.clip(self.action_max - self.action_min, a_min=1e-6, a_max=None)
        normalized = (action - self.action_min) / denom
        normalized = normalized * 2.0 - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _remember_lang_embedding(self, key: tuple[str, str], embedding: torch.Tensor) -> torch.Tensor:
        if self.max_lang_cache_items <= 0:
            return embedding
        self._lang_cache[key] = embedding
        self._lang_cache.move_to_end(key)
        while len(self._lang_cache) > self.max_lang_cache_items:
            self._lang_cache.popitem(last=False)
        return embedding

    def _load_language_embedding(self, pt_path: Path, hdf5_attrs) -> torch.Tensor:
        which = hdf5_attrs.get("which_llm_description", "1")
        if isinstance(which, bytes):
            which = which.decode("utf-8")
        which = str(which)
        cache_key = (str(pt_path), which)
        cached = self._lang_cache.get(cache_key)
        if cached is not None:
            self._lang_cache.move_to_end(cache_key)
            return cached

        if not pt_path.exists():
            raise FileNotFoundError(f"Missing precomputed language embedding: {pt_path}")

        embed_data = torch.load(pt_path, map_location="cpu")

        if which == "2" and "embeddings2" in embed_data and embed_data["embeddings2"] is not None:
            embeddings = embed_data["embeddings2"]
        else:
            embeddings = embed_data["embeddings"]

        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
        return self._remember_lang_embedding(cache_key, embeddings.float())

    def _load_instruction_text(self, hdf5_attrs) -> str:
        llm_type = hdf5_attrs.get("llm_type", "")
        if isinstance(llm_type, bytes):
            llm_type = llm_type.decode("utf-8")

        if llm_type == "reversible":
            which = hdf5_attrs.get("which_llm_description", "1")
            if isinstance(which, bytes):
                which = which.decode("utf-8")
            key = "llm_description2" if str(which) == "2" else "llm_description"
            instruction = hdf5_attrs.get(key, "")
        else:
            instruction = hdf5_attrs.get("llm_description", "")

        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
        return str(instruction)

    def _get_video_capture(self, mp4_path: Path) -> tuple[cv2.VideoCapture, bool]:
        path_key = str(mp4_path)
        if self.max_video_cache_items <= 0:
            return cv2.VideoCapture(path_key), True

        cap = self._video_cache.get(path_key)
        if cap is not None and cap.isOpened():
            self._video_cache.move_to_end(path_key)
            return cap, False

        if cap is not None:
            cap.release()
            del self._video_cache[path_key]

        cap = cv2.VideoCapture(path_key)
        self._video_cache[path_key] = cap
        self._video_cache.move_to_end(path_key)
        while len(self._video_cache) > self.max_video_cache_items:
            _, old_cap = self._video_cache.popitem(last=False)
            old_cap.release()
        return cap, False

    def _sample_video_frames(self, mp4_path: Path, frame_idx: int) -> torch.Tensor:
        cap, should_release = self._get_video_capture(mp4_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_idx = max(frame_idx - (self.img_history_size - 1) * self.upsample_rate, 0)

        frames: List[torch.Tensor] = []
        for idx in range(start_idx, frame_idx + 1, self.upsample_rate):
            idx = min(idx, max(total_frames - 1, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(self.image_transform(frame))
        if should_release:
            cap.release()

        if not frames:
            raise RuntimeError(f"Failed to decode frames from {mp4_path}")

        while len(frames) < self.img_history_size:
            frames.insert(0, frames[0].clone())

        return torch.stack(frames[-self.img_history_size :], dim=0)

    def __del__(self):
        for cap in getattr(self, "_video_cache", {}).values():
            cap.release()

    def __getitem__(self, index: int) -> Dict:
        # A tiny fraction of EgoDex files may be corrupt or missing the derived
        # actions_48d dataset after preprocessing. Replace those samples lazily
        # instead of failing a long distributed training job.
        max_attempts = 10
        last_missing_path = None
        for attempt in range(max_attempts):
            sample_index = index if attempt == 0 else random.randrange(len(self.episodes))
            episode = self.episodes[sample_index]

            with h5py.File(episode["hdf5"], "r") as f:
                if "actions_48d" not in f:
                    last_missing_path = episode["hdf5"]
                    continue

                actions_ds = f["actions_48d"]
                total_frames = actions_ds.shape[0]
                max_current = max(total_frames - 2, 0)
                current_idx = random.randint(0, max_current)

                current_state = actions_ds[current_idx : current_idx + 1].astype(np.float32)
                future_indices = list(
                    range(
                        current_idx + 1,
                        min(total_frames, current_idx + 1 + self.action_chunk_size * self.upsample_rate),
                        self.upsample_rate,
                    )
                )
                if not future_indices:
                    future_indices = [min(current_idx + 1, total_frames - 1)]
                while len(future_indices) < self.action_chunk_size:
                    future_indices.append(future_indices[-1])

                future_actions = np.stack(
                    [actions_ds[future_idx].astype(np.float32) for future_idx in future_indices[: self.action_chunk_size]],
                    axis=0,
                )

                instruction = self._load_instruction_text(f.attrs)
                lang_embeds = (
                    self._load_language_embedding(episode["pt"], f.attrs) if self.use_precomp_lang_embed else None
                )

            return {
                "states": torch.from_numpy(self._normalize_action(current_state)),
                "actions": torch.from_numpy(self._normalize_action(future_actions)),
                "action_norm": torch.ones(self.action_chunk_size, future_actions.shape[-1], dtype=torch.float32),
                "images": self._sample_video_frames(episode["mp4"], current_idx),
                "lang_embeds": lang_embeds,
                "instruction": instruction,
                "episode_path": str(episode["hdf5"]),
                "task": episode["task"],
            }

        raise ValueError(
            f"Failed to sample a valid EgoDex episode after {max_attempts} attempts. "
            f"Last missing actions_48d file: {last_missing_path}"
        )


class MiniEgoDexCollator:
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            "states": torch.stack([instance["states"] for instance in instances], dim=0),
            "actions": torch.stack([instance["actions"] for instance in instances], dim=0),
            "action_norm": torch.stack([instance["action_norm"] for instance in instances], dim=0),
            "images": torch.stack([instance["images"] for instance in instances], dim=0),
            "instructions": [instance["instruction"] for instance in instances],
            "episode_paths": [instance["episode_path"] for instance in instances],
            "tasks": [instance["task"] for instance in instances],
        }

        lang_embeds = [instance["lang_embeds"] for instance in instances if instance["lang_embeds"] is not None]
        if len(lang_embeds) == len(instances):
            lang_lens = [embed.shape[0] for embed in lang_embeds]
            batch["lang_embeds"] = torch.nn.utils.rnn.pad_sequence(lang_embeds, batch_first=True, padding_value=0.0)
            lang_attn_mask = torch.zeros(
                batch["lang_embeds"].shape[0], batch["lang_embeds"].shape[1], dtype=torch.bool
            )
            for idx, length in enumerate(lang_lens):
                lang_attn_mask[idx, :length] = True
            batch["lang_attn_mask"] = lang_attn_mask

        return batch
