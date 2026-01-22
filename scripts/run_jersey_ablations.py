import copy
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def run_variant(name: str, base_cfg: dict, overrides: dict, output_root: Path) -> None:
    cfg = copy.deepcopy(base_cfg)
    for section, values in overrides.items():
        if section not in cfg or not isinstance(values, dict):
            cfg[section] = values
        else:
            cfg[section].update(values)

    cfg.setdefault("train", {})
    cfg["train"]["output_dir"] = str(output_root / name)
    cfg["train"]["resume"] = {"enabled": False, "checkpoint_path": str(output_root / name / "checkpoint_last.pt")}

    config_path = output_root / f"{name}.yaml"
    write_config(config_path, cfg)
    subprocess.run(
        ["python", "scripts/train_jersey.py", "--config", str(config_path)],
        check=True,
    )


def main() -> None:
    base_path = Path("configs/jersey_train.yaml")
    base_cfg = load_config(base_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path("outputs/train/jersey/ablations") / timestamp

    variants = {
        "A_baseline_tuned": {
            "model": {"arch": "resnet34", "pretrained": True, "image_size": 192},
            "augment": {"color_jitter": 0.2, "random_erasing_prob": 0.05},
            "train": {
                "epochs": 80,
                "batch_size": 128,
                "lr": 0.0003,
                "label_smoothing": 0.02,
                "use_class_weights": True,
                "loss": {"type": "cross_entropy"},
                "mixup_cutmix": {"mixup_prob": 0.0, "mixup_alpha": 0.0, "cutmix_prob": 0.0, "cutmix_alpha": 0.0},
            },
        },
        "B_mixup_only": {
            "train": {
                "mixup_cutmix": {"mixup_prob": 0.1, "mixup_alpha": 0.2, "cutmix_prob": 0.0, "cutmix_alpha": 0.0},
            }
        },
        "C_higher_res": {"model": {"image_size": 224}},
        "D_focal_loss": {"train": {"loss": {"type": "focal", "focal_gamma": 2.0}}},
        "A_long": {"train": {"epochs": 120}},
        "C_long": {"model": {"image_size": 224}, "train": {"epochs": 120}},
        "E_resnet50": {"model": {"arch": "resnet50", "pretrained": True, "image_size": 192}},
        "A_no_class_weights": {"train": {"use_class_weights": False}},
        "A_stronger_aug": {"augment": {"color_jitter": 0.3, "random_erasing_prob": 0.1}},
        "A_low_lr": {"train": {"lr": 0.0001}},
        "F_efficientnet_b0": {"model": {"arch": "efficientnet_b0", "pretrained": True, "image_size": 224}},
    }

    for name, overrides in variants.items():
        run_variant(name, base_cfg, overrides, output_root)


if __name__ == "__main__":
    main()
