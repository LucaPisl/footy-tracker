from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class JerseySample:
    image_path: Path
    label_index: int
    label_name: str
    track_id: str


def load_track_labels(split_dir: Path) -> dict[str, int]:
    gt_path = split_dir / f"{split_dir.name}_gt.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground-truth file: {gt_path}")
    with gt_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {gt_path}")
    return {str(key): int(value) for key, value in data.items()}


def collect_track_images(split_dir: Path) -> dict[str, list[Path]]:
    images_root = split_dir / "images"
    if not images_root.exists():
        raise FileNotFoundError(f"Missing images directory: {images_root}")
    tracks: dict[str, list[Path]] = {}
    for track_dir in sorted(images_root.iterdir()):
        if not track_dir.is_dir():
            continue
        image_paths = sorted(
            [path for path in track_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if image_paths:
            tracks[track_dir.name] = image_paths
    if not tracks:
        raise ValueError(f"No track image folders found under {images_root}")
    return tracks


def build_label_mapping(
    labels: Iterable[int],
    include_unknown: bool,
    unknown_label: str,
) -> tuple[dict[str, int], dict[int, str]]:
    unique_labels = sorted({label for label in labels if label >= 0})
    label_to_index: dict[str, int] = {}
    index_to_label: dict[int, str] = {}
    for idx, label in enumerate(unique_labels):
        name = str(label)
        label_to_index[name] = idx
        index_to_label[idx] = name
    if include_unknown:
        unknown_index = len(label_to_index)
        label_to_index[unknown_label] = unknown_index
        index_to_label[unknown_index] = unknown_label
    if not label_to_index:
        raise ValueError("No valid labels found to build the label mapping.")
    return label_to_index, index_to_label


def split_track_ids(track_ids: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1).")
    rng = random.Random(seed)
    shuffled = track_ids[:]
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    val_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])
    return train_ids, val_ids


def subsample_images(
    image_paths: list[Path],
    max_frames: int,
    seed: int,
) -> list[Path]:
    if max_frames <= 0 or len(image_paths) <= max_frames:
        return image_paths
    rng = random.Random(seed)
    return sorted(rng.sample(image_paths, max_frames))


def build_samples(
    track_ids: Iterable[str],
    tracks: dict[str, list[Path]],
    track_labels: dict[str, int],
    label_to_index: dict[str, int],
    include_unknown: bool,
    unknown_label: str,
    max_frames_per_track: int,
    seed: int,
) -> list[JerseySample]:
    samples: list[JerseySample] = []
    for track_id in track_ids:
        if track_id not in tracks:
            continue
        raw_label = track_labels.get(track_id, -1)
        if raw_label < 0 and not include_unknown:
            continue
        if raw_label < 0:
            label_name = unknown_label
        else:
            label_name = str(raw_label)
        if label_name not in label_to_index:
            continue
        image_paths = subsample_images(tracks[track_id], max_frames_per_track, seed)
        label_index = label_to_index[label_name]
        for path in image_paths:
            samples.append(
                JerseySample(
                    image_path=path,
                    label_index=label_index,
                    label_name=label_name,
                    track_id=track_id,
                )
            )
    if not samples:
        raise ValueError("No jersey samples were generated. Check dataset paths and label settings.")
    return samples


class JerseyFrameDataset(Dataset[JerseySample]):
    def __init__(self, samples: list[JerseySample], transform=None):
        self._samples = samples
        self._transform = transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        sample = self._samples[index]
        from PIL import Image

        try:
            image = Image.open(sample.image_path).convert("RGB")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Failed to read image: {sample.image_path}") from exc
        if self._transform is not None:
            image = self._transform(image)
        return image, sample.label_index


def compute_class_weights(samples: Iterable[JerseySample], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float)
    for sample in samples:
        counts[sample.label_index] += 1
    weights = torch.zeros_like(counts)
    non_zero = counts > 0
    weights[non_zero] = counts[non_zero].sum() / (counts[non_zero] * non_zero.sum())
    weights[~non_zero] = 0.0
    return weights
