import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from footy_tracker.jersey.dataset import (  # noqa: E402
    JerseyFrameDataset,
    build_label_mapping,
    build_samples,
    collect_track_images,
    compute_class_weights,
    load_track_labels,
    split_track_ids,
)


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return config


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("jersey_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def ensure_split_dir(root: Path, split: str, auto_extract: bool) -> Path:
    split_dir = root / split
    if split_dir.exists():
        return split_dir
    zip_path = root / f"{split}.zip"
    if auto_extract and zip_path.exists():
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(root)
        if split_dir.exists():
            return split_dir
    raise FileNotFoundError(
        f"Missing split directory {split_dir}. Extract {zip_path} or set data.auto_extract=true."
    )


def build_transforms(image_size: int, augment: bool) -> transforms.Compose:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if augment:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=6, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model architecture: {arch}")


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.numel())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy(logits, labels) * batch_size
        total += batch_size

    return running_loss / max(1, total), running_acc / max(1, total)


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    best_acc: float,
    history: list[dict[str, float]],
) -> Path:
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_acc": best_acc,
        "history": history,
    }
    path = output_dir / "checkpoint_last.pt"
    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
) -> tuple[int, float, list[dict[str, float]]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(data["model_state"])
    optimizer.load_state_dict(data["optimizer_state"])
    if scheduler is not None and data.get("scheduler_state") is not None:
        scheduler.load_state_dict(data["scheduler_state"])
    epoch = int(data.get("epoch", 0))
    best_acc = float(data.get("best_acc", 0.0))
    history = data.get("history", [])
    return epoch, best_acc, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train jersey number classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/jersey_train.yaml"),
        help="Path to jersey training config YAML.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})

    output_dir = Path(train_cfg.get("output_dir", "outputs/train/jersey"))
    logger = configure_logging(output_dir)

    seed = int(data_cfg.get("seed", 1337))
    random.seed(seed)
    torch.manual_seed(seed)

    root = Path(data_cfg.get("root", "data/sn_jersey/jersey-2023"))
    auto_extract = bool(data_cfg.get("auto_extract", False))
    train_split = str(data_cfg.get("train_split", "train"))
    train_dir = ensure_split_dir(root, train_split, auto_extract)

    track_labels = load_track_labels(train_dir)
    tracks = collect_track_images(train_dir)
    include_unknown = bool(data_cfg.get("include_unknown", False))
    unknown_label = str(data_cfg.get("unknown_label", "unknown"))

    label_to_index, index_to_label = build_label_mapping(
        track_labels.values(), include_unknown, unknown_label
    )

    if include_unknown:
        track_ids = sorted(tracks.keys())
    else:
        track_ids = sorted(
            [track_id for track_id in tracks.keys() if track_labels.get(track_id, -1) >= 0]
        )
    if not track_ids:
        raise ValueError("No labeled tracks available for training. Check include_unknown or dataset labels.")
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    train_ids, val_ids = split_track_ids(track_ids, val_ratio, seed)

    max_frames = int(data_cfg.get("max_frames_per_track", 0))
    train_samples = build_samples(
        train_ids,
        tracks,
        track_labels,
        label_to_index,
        include_unknown,
        unknown_label,
        max_frames,
        seed,
    )
    if val_ids:
        val_samples = build_samples(
            val_ids,
            tracks,
            track_labels,
            label_to_index,
            include_unknown,
            unknown_label,
            max_frames,
            seed,
        )
    else:
        logger.warning("val_ratio produced no validation tracks; reusing train samples for validation.")
        val_samples = train_samples

    image_size = int(model_cfg.get("image_size", 128))
    train_transform = build_transforms(image_size, augment=True)
    val_transform = build_transforms(image_size, augment=False)

    train_dataset = JerseyFrameDataset(train_samples, transform=train_transform)
    val_dataset = JerseyFrameDataset(val_samples, transform=val_transform)

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    device_name = str(train_cfg.get("device", "auto"))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model = build_model(
        arch=str(model_cfg.get("arch", "resnet18")),
        num_classes=len(label_to_index),
        pretrained=bool(model_cfg.get("pretrained", False)),
    ).to(device)

    use_class_weights = bool(train_cfg.get("use_class_weights", True))
    if use_class_weights:
        weights = compute_class_weights(train_samples, len(label_to_index)).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    scheduler_cfg = train_cfg.get("scheduler", {}) or {}
    scheduler_name = str(scheduler_cfg.get("name", "none")).lower()
    scheduler = None
    if scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 5)),
            gamma=float(scheduler_cfg.get("gamma", 0.1)),
        )
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(train_cfg.get("epochs", 20))
        )

    epochs = int(train_cfg.get("epochs", 20))
    best_acc = 0.0
    history: list[dict[str, float]] = []
    start_epoch = 1

    resume_cfg = train_cfg.get("resume", {}) or {}
    resume_enabled = bool(resume_cfg.get("enabled", False))
    resume_path = Path(resume_cfg.get("checkpoint_path", output_dir / "checkpoint_last.pt"))
    if resume_enabled:
        start_epoch, best_acc, history = load_checkpoint(
            resume_path, model, optimizer, scheduler
        )
        start_epoch += 1
        logger.info(
            "resume=enabled checkpoint=%s start_epoch=%d best_acc=%.4f",
            resume_path,
            start_epoch,
            best_acc,
        )

    logger.info("train_samples=%d val_samples=%d classes=%d", len(train_samples), len(val_samples), len(label_to_index))
    logger.info("device=%s model=%s", device, model_cfg.get("arch", "resnet18"))

    checkpoint_every = int(train_cfg.get("checkpoint_every", 1))
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        if scheduler is not None:
            scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        logger.info(
            "epoch=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best.pt")

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(
                output_dir,
                epoch,
                model,
                optimizer,
                scheduler,
                best_acc,
                history,
            )

    torch.save(model.state_dict(), output_dir / "last.pt")
    save_checkpoint(output_dir, epochs, model, optimizer, scheduler, best_acc, history)
    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "label_to_index": label_to_index,
                "index_to_label": {str(k): v for k, v in index_to_label.items()},
            },
            handle,
            indent=2,
        )
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_val_acc": best_acc,
                "history": history,
                "classes": index_to_label,
            },
            handle,
            indent=2,
        )
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    logger.info("training_complete best_val_acc=%.4f output_dir=%s", best_acc, output_dir)


if __name__ == "__main__":
    main()
