````md
# AGENTS.md — Footy Tracker

This file defines how humans and coding agents should work in this repo: goals, boundaries, conventions, and “definition of done”. Follow it strictly.

---

## Mission

Build a **10-day demo** that takes an **HD football match video** and outputs an **overlay video** with:

- player **bounding boxes**
- **persistent track IDs** (stable across frames)
- **jersey number + confidence** (when visible)

Target: **60–70%+ jersey accuracy** on clearly visible numbers.

---

## Team roles (operational constraint)

- **Luca (GPU machine):** trains/exports YOLO player detector + jersey number model weights.
- **Abdurrahman (no training GPU):** implements full pipeline, tracking integration, demo tooling.

### Practical implications
- The repo must run end-to-end **without requiring a training GPU**.
- Training code/configs may exist, but the **demo pipeline must work on CPU** (slower is OK).
- We intentionally **do not pin `torch`** in `requirements.txt` to avoid breaking CUDA installs.

---

## Non-goals (avoid scope creep)

- No full broadcast-grade analytics (events, tactics, xG, passes).
- No perfect re-identification in extreme occlusions; demo-level stability is fine.
- No fully automated dataset curation; only documented download + scripts.

---

## Repo layout (source of truth)

- `src/footy_tracker/` — all Python code (importable package)
- `configs/` — YAML configs (pipeline wiring, thresholds, paths)
- `data/` — datasets (NOT tracked; keep only `data/README.md`)
- `weights/` — model weights (NOT tracked; keep only `weights/README.md`)
- `outputs/` — generated demos/logs (NOT tracked)
- `runs/` — Ultralytics training outputs (NOT tracked)

Agents must **not** commit large binaries (videos, datasets, weights, runs).

---

## Primary entrypoints (must remain working)

### End-to-end demo
PowerShell:
```powershell
python -m footy_tracker --config configs\pipeline.yaml
````

Equivalent:

```powershell
python -m footy_tracker.pipeline.process_video --config configs\pipeline.yaml
```

### Pipeline steps (conceptual contract)

1. YOLO detection → per-frame detections
2. Tracking → stable track IDs
3. Crop jersey region per track
4. Jersey inference → number + confidence
5. Overlay → output video in `outputs/demos/`

If you change step contracts, you must update:

* configs schema (YAML keys)
* CLI help (if applicable)
* README (if pipeline usage changes)

---

## Config-driven rule

All user-tunable values belong in YAML (not hardcoded):

* model paths
* thresholds (conf/iou)
* class mappings
* tracker params
* output settings
* cropping strategy (jersey ROI)

Code should accept a config object and run deterministically from it.

### Local paths

* Users copy `configs/paths_local_example.yaml` → `configs/paths_local.yaml`.
* `configs/paths_local.yaml` must remain **git-ignored**.

---

## Data contracts (keep modules decoupled)

Agents should preserve a clean boundary between these modules:

### Detector output

Per frame:

* bounding boxes (xyxy or xywh, but choose one and standardize)
* confidence
* class id (player class)
* frame index

### Tracker output

Per frame:

* track id for each box
* smoothed box (optional)
* track state (active/lost) (optional)

### Jersey recognizer output

Per track (when crop is valid):

* predicted jersey number (string or int)
* confidence
* “not visible / unreadable” state

**Rule:** If jersey is not confidently readable, output `None` (or a designated “unknown”) + confidence.

---

## Engineering standards

### Python / style

* Keep code importable and module-based (no notebook-only logic).
* Prefer explicit types for core data structures (dataclasses/TypedDict acceptable).
* Functions should be small and single-purpose (detector, tracker, overlay separated).

### Logging

* Use structured logging (at minimum: frame index, #detections, #tracks).
* Do not spam per-pixel logs; per-frame summary is enough.
* Save a run summary to `outputs/` (git-ignored).

### Error handling

* Fail fast on missing config keys or missing files.
* Provide actionable error messages (what path/key is missing).

---

## Performance and correctness priorities

1. **Correct pipeline output** (overlay video generated reliably)
2. **Stable tracking IDs** (no frequent ID switches)
3. **Jersey inference correctness** when visible
4. Speed optimizations after correctness

Agents should prefer changes that improve reliability over clever optimizations.

---

## Tracking policy (demo-friendly)

* The tracker must support:

  * multiple players
  * short occlusions
  * camera motion (broadcast panning)

If implementing a simple tracker (baseline), ensure:

* association by IoU / distance
* max age (lost frames) configurable
* minimum hits configurable
* deterministic assignment

If using an existing tracker implementation, keep it behind a clean adapter so it can be swapped.

---

## Video I/O policy

* Always read frames in a streaming way (no loading full video into RAM).
* Output video must preserve:

  * resolution
  * fps (unless explicitly configured otherwise)
* Overlay must include:

  * box
  * track ID
  * jersey number + confidence (or “unknown”)

---

## What agents should do when asked to implement a feature

1. Identify which pipeline step it touches (detection/tracking/jersey/overlay/config/cli).
2. Implement in the correct module with minimal diff.
3. Add/update YAML keys (with defaults) if behavior is tunable.
4. Add a small smoke test or a runnable script path when feasible.
5. Ensure end-to-end CLI still runs with `configs/pipeline.yaml`.

---

## Definition of done (for any change)

* `python -m footy_tracker --config configs\pipeline.yaml` runs without crashing.
* Output video is produced under `outputs/`.
* No new large files committed.
* Config keys are documented (README/config comments).
* Changes do not assume GPU availability (unless explicitly “Luca-only training”).

---

## Guardrails (hard rules)

* Do **not** commit:

  * videos
  * datasets
  * weights
  * `runs/` outputs
  * large logs
* Do **not** pin CUDA-specific torch builds in `requirements.txt`.
* Do **not** break Windows/PowerShell commands; this repo must remain usable on Windows.

---

## Notes for collaboration

* Keep interfaces stable: downstream steps should not be forced to change unless necessary.
* Prefer adapters over deep refactors during the 10-day demo window.
* If you introduce a new dependency, justify it by:

  * clear benefit to tracking stability or jersey accuracy
  * minimal install friction on Windows

```

::contentReference[oaicite:0]{index=0}
```
