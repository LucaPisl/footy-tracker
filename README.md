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

The project is currently in the initial setup phase. The environment is fully configured, and all necessary datasets
have been acquired and prepared for training.

#### What's Working So Far

* **Hardware Acceleration**: PyTorch is correctly configured with CUDA, successfully detecting the **NVIDIA GeForce RTX
  5090 Laptop GPU**.
* **Environment**: Core deep learning and tracking libraries (`torch`, `ultralytics`, `deep-sort-realtime`,
  `supervision`) are installed.
* **Data Acquisition**:
    * **SoccerNet-Tracking**: Dataset downloaded and extracted into `data/soccernet_tracking/`.
    * **sn-jersey**: Dataset (train/test/challenge) downloaded and extracted into `data/sn_jersey/`.
* **Project Skeleton**: Directory structure and configuration YAMLs are in place.

#### Current Implementation

* **Configuration**: Initial templates for YOLO detection (`yolo.yaml`), tracking parameters (`tracker.yaml`), and the
  end-to-end pipeline (`pipeline.yaml`) are defined in the `configs/` directory.
* **Scripts**: Utility scripts for dataset management and environment verification are available in `scripts/`.
* **Codebase**: The `src/footy_tracker/` package is initialized, ready for the implementation of the core logic.

### Remaining Tasks

*   [ ] **Player Detection Training**: Fine-tune YOLO on SoccerNet-Tracking.
*   [ ] **Tracking Integration**: Implement the tracker logic to link detections across frames.
*   [ ] **Jersey Number Recognition**: Develop and train the model for jersey number classification.
*   [ ] **Pipeline Development**: Connect all components in `src/footy_tracker/` to process videos from end to end.
*   [ ] **Visualization**: Finalize the overlay logic for bounding boxes and labels.