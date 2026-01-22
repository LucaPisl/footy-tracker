import json
from pathlib import Path
from typing import Any

import cv2
import torch
import yaml
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _choose_video_path(default_path: Path) -> Path:
    print(f"Default input video: {default_path}")
    use_default = input("Use this video? [Y/n]: ").strip().lower()
    if use_default in {"", "y", "yes"}:
        return default_path
    while True:
        manual = input("Enter video path: ").strip()
        if not manual:
            continue
        path = Path(manual)
        if path.exists():
            return path
        print(f"Path not found: {path}")


def build_model(num_classes: int) -> torch.nn.Module:
    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_jersey_model(weights_path: Path, mapping_path: Path, device: torch.device):
    mapping = json.loads(mapping_path.read_text())
    index_to_label = {int(k): v for k, v in mapping["index_to_label"].items()}
    model = build_model(len(index_to_label)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, index_to_label


def _load_image_size(config_path: Path, fallback: int = 224) -> int:
    if not config_path.exists():
        return fallback
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return fallback
    model_cfg = data.get("model", {})
    if isinstance(model_cfg, dict) and model_cfg.get("image_size"):
        return int(model_cfg["image_size"])
    return fallback


def _crop_jersey_region(
        frame: Any,
        box: tuple[int, int, int, int],
        crop_cfg: dict[str, Any],
) -> Any | None:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None
    y_start_frac = float(crop_cfg.get("y_start_frac", 0.2))
    y_end_frac = float(crop_cfg.get("y_end_frac", 0.7))
    x_pad_frac = float(crop_cfg.get("x_pad_frac", 0.05))
    min_w = int(crop_cfg.get("min_width_px", 20))
    min_h = int(crop_cfg.get("min_height_px", 20))
    cx1 = max(0, int(x1 + width * x_pad_frac))
    cx2 = min(frame.shape[1], int(x2 - width * x_pad_frac))
    cy1 = max(0, int(y1 + height * y_start_frac))
    cy2 = min(frame.shape[0], int(y1 + height * y_end_frac))
    if cx2 - cx1 < min_w or cy2 - cy1 < min_h:
        return None
    return frame[cy1:cy2, cx1:cx2]


def predict_jersey(
        model: torch.nn.Module,
        index_to_label: dict[int, str],
        crop_bgr: Any,
        device: torch.device,
        transform: transforms.Compose,
) -> tuple[str, float]:
    image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)
    label = index_to_label[int(pred)]
    return label, float(conf.item())


def _draw_label(frame, x1, y1, x2, y2, text, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        text,
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths_local = _safe_load_yaml(root / "configs" / "paths_local.yaml")
    default_video = Path(
        paths_local.get("paths", {}).get("input_video", "data/videos/sample.mp4")
    )
    video_path = _choose_video_path(default_video)

    yolo_cfg = _safe_load_yaml(root / "configs" / "yolo.yaml").get("yolo", {})
    jersey_cfg = _safe_load_yaml(root / "configs" / "jersey.yaml").get("jersey", {})
    jersey_crop_cfg = jersey_cfg.get("crop", {}) if isinstance(jersey_cfg.get("crop"), dict) else {}

    yolo_weights = root / yolo_cfg.get("model_path", "weights/yolo/player.pt")
    jersey_weights = root / jersey_cfg.get("model_path", "outputs/train/jersey/best.pt")
    mapping_path = root / "outputs" / "train" / "jersey" / "label_mapping.json"
    jersey_train_cfg = root / "outputs" / "train" / "jersey" / "config.yaml"

    if not yolo_weights.exists():
        raise FileNotFoundError(f"Missing YOLO weights: {yolo_weights}")
    if not jersey_weights.exists():
        raise FileNotFoundError(f"Missing jersey weights: {jersey_weights}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing label mapping: {mapping_path}")
    if not video_path.exists():
        # Fallback to absolute if it was intended to be relative to root
        if not video_path.is_absolute():
            video_path = root / video_path
        if not video_path.exists():
            raise FileNotFoundError(f"Missing input video: {video_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jersey_model, index_to_label = load_jersey_model(jersey_weights, mapping_path, device)
    image_size = _load_image_size(jersey_train_cfg, fallback=224)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    yolo = YOLO(str(yolo_weights))
    conf = float(yolo_cfg.get("conf_threshold", 0.25))
    iou = float(yolo_cfg.get("iou_threshold", 0.45))
    player_class_id = int(yolo_cfg.get("player_class_id", 0))
    ref_class_id = int(yolo_cfg.get("referee_class_id", 1))
    ball_class_id = int(yolo_cfg.get("ball_class_id", 2))
    jersey_conf_thresh = float(jersey_cfg.get("conf_threshold", 0.6))
    unknown_label = str(jersey_cfg.get("unknown_label", "unknown"))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_out = root / "outputs" / "jersey_overlay_temp.mp4"
    temp_out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(temp_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        results = yolo.predict(frame, conf=conf, iou=iou, verbose=False)
        boxes = results[0].boxes
        for det in boxes:
            xyxy = det.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cls_id = int(det.cls[0].item())
            if cls_id == player_class_id:
                crop = _crop_jersey_region(frame, (x1, y1, x2, y2), jersey_crop_cfg)
                if crop is not None:
                    label, j_conf = predict_jersey(
                        jersey_model, index_to_label, crop, device, transform
                    )
                else:
                    label, j_conf = unknown_label, 0.0
                if j_conf < jersey_conf_thresh:
                    label = unknown_label
                text = f"player {label} ({j_conf:.2f})"
                color = (0, 220, 0)
            elif cls_id == ref_class_id:
                text = "referee"
                color = (0, 180, 255)
            elif cls_id == ball_class_id:
                text = "ball"
                color = (255, 180, 0)
            else:
                text = f"class {cls_id}"
                color = (200, 200, 200)
            _draw_label(frame, x1, y1, x2, y2, text, color)

        writer.write(frame)
        frame_index += 1
        if frame_index % 60 == 0:
            print(f"Processed {frame_index} frames")

    capture.release()
    writer.release()

    print(f"Temporary overlay video saved to: {temp_out}")
    show = input("Play preview now? [Y/n]: ").strip().lower()
    if show in {"", "y", "yes"}:
        preview = cv2.VideoCapture(str(temp_out))
        delay = max(1, int(1000 / fps))
        while True:
            ok, frame = preview.read()
            if not ok:
                break
            cv2.imshow("Jersey Overlay Preview", frame)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
        preview.release()
        cv2.destroyAllWindows()

    save = input("Save output video? [Y/n]: ").strip().lower()
    if save in {"", "y", "yes"}:
        dest = input(
            f"Enter save path (default: outputs/demos/jersey_overlay.mp4): "
        ).strip()
        if not dest:
            dest_path = root / "outputs" / "demos" / "jersey_overlay.mp4"
        else:
            dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(temp_out.read_bytes())
        print(f"Saved output video to: {dest_path}")


if __name__ == "__main__":
    main()
