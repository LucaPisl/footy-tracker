# Jersey Training Report (sn-jersey)

## Purpose

Document the jersey number recognition experiments, results, and takeaways so the work can be presented clearly.

---

## Quick Summary (Talk Track)

- Goal: Train a jersey number classifier for the demo pipeline (target 60–70%+ accuracy on clearly visible numbers).
- Best observed result: **~0.78 val accuracy** on the sn-jersey validation split using **ResNet34** with pretrained
  weights.
- Main challenge: **Overfitting**. Training accuracy climbs to ~1.0 while validation plateaus ~0.75–0.79.
- Aggressive augmentation (random crop + erasing + mixup/cutmix) reduced generalization.

---

## Dataset

- Dataset: SoccerNet **sn-jersey** (`data/sn_jersey/jersey-2023`)
- Labels: `train_gt.json` maps track IDs to jersey numbers, `-1` for unreadable.
- Split used: `train` split with a **track-level 90/10 train/val** split for evaluation.

---

## Training Pipeline (What We Implemented)

Files added/updated to support training:

- `scripts/train_jersey.py`: configurable training script (ResNet/MobileNet/EfficientNet), logging, checkpointing,
  resume.
- `configs/jersey_train.yaml`: all hyperparameters/config in YAML.
- `src/footy_tracker/jersey/dataset.py`: dataset loading, label mapping, class weights.
- `scripts/run_jersey_ablations.py`: runs a set of ablations with separate output dirs.

Key features in the training script:

- Config-driven settings (paths, model, augment, loss, early stopping).
- Class weighting support (to address class imbalance).
- Optional mixup/cutmix and label smoothing (used in ablations).
- Optional focal loss.
- Checkpoint save/resume.
- Early stopping (added later for generalization control).

---

## Best Configuration (What We Kept)

**Model**: ResNet34 (pretrained)  
**Image size**: 192  
**Augmentation**: resize only (no random resized crop) + light color jitter  
**Loss**: Cross entropy  
**Regularization**: no mixup/cutmix, no label smoothing  
**Optimizer**: AdamW  
**Batch**: 128

Result: **best val acc ~0.7781** (epoch ~63)

Output artifacts:

- Weights: `outputs/train/jersey/best.pt`
- Mapping: `outputs/train/jersey/label_mapping.json`
- Log file: `outputs/train/jersey/train_20260119_005333.log`

---

## Additional Experiments (Ablations)

We ran a full ablation pack for 8 hours using `scripts/run_jersey_ablations.py`.
All outputs saved under:  
`outputs/train/jersey/ablations/20260118_165048/`

**Summary of best validation accuracies:**

- `E_resnet50`: **0.5778**
- `A_no_class_weights`: **0.5693**
- `A_stronger_aug`: **0.5634**
- `A_baseline_tuned`: **0.5582**
- `B_mixup_only`: **0.5448**
- `C_higher_res` (224px): **0.5477**
- `A_long` (120 epochs): **0.5418**
- `C_long` (224px, 120 epochs): **0.5422**
- `A_low_lr`: **0.5363**
- `D_focal_loss`: **0.5294**
- `F_efficientnet_b0`: **incomplete** (timed out)

**Key observation:** These were **much worse** than the best run (~0.78). The likely cause
was the switch to more aggressive augmentations and random resized crop.

---

## Overfitting Findings

Evidence from logs:

- Training accuracy consistently rises above **0.99** while validation remains around **0.75–0.78**.
- Validation loss increases even as training loss approaches zero.

Interpretation:

- The model memorizes training samples; the dataset’s limited size and label noise reduce generalization.

---

## Follow-Up Runs

We retrained using a **224px input size** with early stopping and lower weight decay:

- Best val acc: **0.7647**
- Early stopping triggered around epoch 24

Conclusion: the 224px run did not beat the 192px best run.

---

## Test Script (Single Image Demo)

We created a demo script to visualize predictions using YOLO + jersey classifier:

**Script**

- `test-stuff/quick_jersey_test.py`

**What it does**

- Runs YOLO player detection on `test-stuff/test-image-jersey-1.png`
- Crops each player box
- Runs jersey classification for each crop
- Draws boxes + jersey label + confidence on the image
- Saves output to `outputs/jersey_test_overlay.png`

**Command**

```bash
python test-stuff/quick_jersey_test.py
```

---

## Recommendations (If We Continue)

Short, practical steps to increase accuracy:

1. **Expand dataset** with more clean, readable jersey crops (most likely to boost generalization).
2. **Avoid aggressive cropping** for small jersey crops (random resized crop can drop accuracy).
3. **Early stopping + mild augmentation** seems best for this dataset.

---

## Status

- Current best model is **good enough for demo** (77–78% val accuracy).
- Target of 80%+ is possible but likely requires cleaner data or more diverse samples.

