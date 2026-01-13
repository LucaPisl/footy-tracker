# Footy Tracker

An end-to-end football player tracking system using Python, YOLO, and SoccerNet datasets. The goal is to process match
videos and output tracked players with persistent IDs and jersey numbers.

### Project Overview

The system follows a tracking-by-detection pipeline:

1. **Detection**: YOLOv8/v11 model trained on SoccerNet-Tracking to detect players, referees, and the ball.
2. **Tracking**: DeepSORT or ByteTrack to maintain consistent IDs across frames.
3. **Jersey Recognition**: A dedicated model trained on the `sn-jersey` dataset to identify player numbers.
4. **Overlay**: Visualizing bounding boxes, IDs, and jersey numbers on the final video output.

### Current Status

**Phase 1: Environment & Data Preparation (Completed)**

The environment is fully configured, and all necessary datasets have been acquired.

**Phase 2: Player Detection Training (Completed)**

A YOLOv8 model has been successfully fine-tuned on the SoccerNet-Tracking dataset for detecting players, referees, and
the ball.

#### What's Working So Far

* **Hardware Acceleration**: PyTorch is correctly configured with CUDA, using the **NVIDIA GeForce RTX 5090 Laptop GPU
  **.
* **Data Preparation**: Automatic conversion of SoccerNet MOT format to YOLO format via
  `scripts/prepare_soccernet_yolo.py`.
* **Trained Detector**: A fine-tuned YOLOv8n model (`weights/yolo/player.pt`) capable of detecting:
    * **Players** (mAP50: 0.942)
    * **Referees** (mAP50: 0.726)
    * **Ball** (mAP50: 0.198)
* **Environment**: Core deep learning and tracking libraries (`torch`, `ultralytics`, `deep-sort-realtime`,
  `supervision`) are installed and verified.
* **Data Acquisition**: SoccerNet-Tracking and sn-jersey datasets are fully extracted and ready.

#### Current Implementation

* **Detection Pipeline**:
    * `scripts/prepare_soccernet_yolo.py`: Handles dataset conversion and organization.
    * `scripts/train_yolo.py`: Training script using `ultralytics` for YOLO fine-tuning.
    * `configs/soccernet_data.yaml`: YOLO-specific dataset configuration.
* **Configuration**: `configs/yolo.yaml` is updated to point to the trained `player.pt` weights and includes classes for
  player, referee, and ball.
* **Codebase**: The `src/footy_tracker/` package is initialized.

### Demo Pipeline (Frame IO + Stitching)

The demo pipeline currently handles **frame extraction** and **video stitching**. The ML tracking/overlay step is
handled by the GPU contributor and should write annotated PNGs that this pipeline stitches back into an MP4.

1. Set paths in `configs/pipeline.yaml`:
   * `paths.input_video` -> input MP4
   * `paths.frames_dir` -> extracted PNGs (input for ML step)
   * `paths.ml_output_dir` -> annotated PNGs produced by the ML step
2. Run the pipeline:

```bash
python -m footy_tracker --config configs/pipeline.yaml
```

If you need to run only extraction or only stitching, update the `steps` section in `configs/pipeline.yaml`.

### Remaining Tasks

*   [x] **Player Detection Training**: Fine-tune YOLO on SoccerNet-Tracking.
*   [ ] **Tracking Integration**: Implement the tracker logic (BoT-SORT or ByteTrack) to link detections across frames.
*   [ ] **Ball Detection Enhancement**: Improve ball detection accuracy (currently 0.198 mAP50) through higher
    resolution training or additional epochs.
*   [ ] **Jersey Number Recognition**: Develop and train the model for jersey number classification using the
    `sn-jersey` dataset.
*   [ ] **Team Identification**: Integrate jersey color/team classification.
*   [ ] **Pipeline Development**: Connect all components in `src/footy_tracker/` to process videos from end to end.
*   [ ] **Visualization**: Finalize the overlay logic for bounding boxes, IDs, and labels.
