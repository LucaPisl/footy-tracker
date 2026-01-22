# Footy Tracker

An end-to-end football player tracking demo that takes an HD match video and produces an overlay video with:

- player bounding boxes
- persistent track IDs
- jersey number + confidence (when readable)

This repo is demo-focused and **config-driven**. Large artifacts (videos, datasets, weights, runs, outputs) are not
tracked in git.

---

## How It Works

Pipeline stages:

1. **Detection**: YOLO model detects players, referees, and ball.
2. **Tracking**: assigns stable IDs across frames (adapter planned).
3. **Jersey recognition**: classifier predicts jersey numbers for player crops.
4. **Overlay**: draws boxes + labels on frames and writes output video.

---

## Repo Layout (Important)

Tracked source:

- `src/footy_tracker/` — Python package
- `configs/` — YAML configs
- `scripts/` — training + demo scripts
- `test-stuff/` — local experiments (ignored in git)

Not tracked (you must supply locally):

- `data/` — datasets + sample videos
- `weights/` — trained models
- `outputs/` — generated demos/logs
- `runs/` — Ultralytics training outputs

---

## Setup

Create a local paths override:

```bash
cp configs/paths_local_example.yaml configs/paths_local.yaml
```

Edit `configs/paths_local.yaml` to point at your local videos and weights.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Demo (Frame IO + Stitching)

The core demo pipeline handles **frame extraction** and **video stitching**. An ML step is expected to write annotated
frames into the ML output directory before stitching.

Run:
```bash
python -m footy_tracker --config configs/pipeline.yaml
```

If you want only extraction or only stitching, toggle `steps` in `configs/pipeline.yaml`.

---

## Quick Video Overlay (YOLO + Jersey)

For a simple, interactive end-to-end pass on a single video:

```bash
python scripts/quick_video_jersey_overlay.py
```

What it does:

- Prompts for input video (default from `configs/paths_local.yaml`)
- Runs YOLO detection
- Crops jersey regions for **players only**
- Runs jersey classifier on those crops
- Draws boxes + labels, writes a temporary video
- Plays a preview and asks where to save

Expected local files:

- `weights/yolo/player.pt`
- `weights/jersey/jersey.pt` (symlinked to `outputs/train/jersey/best.pt` if trained locally)
- `outputs/train/jersey/label_mapping.json`

---

## Jersey Classifier Training (sn-jersey)

1. Ensure dataset is extracted:
    - `data/sn_jersey/jersey-2023/train/`
2. Train:
```bash
python scripts/train_jersey.py --config configs/jersey_train.yaml
```

Outputs:

- `outputs/train/jersey/best.pt`
- `outputs/train/jersey/label_mapping.json`

You can symlink to inference path:

```bash
ln -sf ../../outputs/train/jersey/best.pt weights/jersey/jersey.pt
```

---

## YOLO Training (SoccerNet-Tracking)

Train the detector (optional if you already have weights):

```bash
python scripts/train_yolo.py
```

Ensure `configs/yolo.yaml` points to `weights/yolo/player.pt`.

---

## Configs to Know

- `configs/pipeline.yaml` — end-to-end pipeline wiring
- `configs/yolo.yaml` — detection thresholds + weights
- `configs/jersey.yaml` — jersey model path + crop strategy
- `configs/tracker.yaml` — tracking params
- `configs/paths_local.yaml` — local paths (git-ignored)

---

## Windows Notes

Commands are designed to work on Windows/PowerShell:

```powershell
python -m footy_tracker --config configs\pipeline.yaml
```

---

## Status

This repo focuses on a **demo-ready pipeline** with:

- YOLO player detection trained on SoccerNet-Tracking
- Jersey classifier trained on sn-jersey
- Interactive video overlay script

Tracking integration and full end-to-end orchestration are still in progress.

