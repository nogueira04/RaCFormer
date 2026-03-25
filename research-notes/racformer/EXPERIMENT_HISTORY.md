# RaCFormer Experiment History

**Project**: Radar-Camera Fusion for 3D Object Detection (nuScenes)
**Research Goal**: Close the 40% night-time performance gap
**Period**: December 2025 - March 2026

---

## Timeline

```
2025-12        2026-01        2026-02        2026-03
|              |              |              |
|--3cam/3rad-->|              |              |
               |--baseline--->|              |
               |--dropout---->|              |
               |---nightaug (run1,2)-->|     |
               |   depth oracle        |     |
               |   path B experiments  |     |
                              |--nightaug (run3)--------->|
                              |  night evals (dropout,    |
                              |  baseline)                |
                                             |--nightaug night eval
                                             |  ALL EXPERIMENTS COMPLETE
```

---

## Reference Metrics

### Pretrained Checkpoint (Best Available Model)

| Metric | Value |
|--------|-------|
| Overall mAP | 54.18% |
| Overall NDS | 61.44% |
| Day mAP | 53.88% |
| Night mAP | 32.28% |
| Day-Night Gap | 21.6% absolute (40.1% relative) |

**Checkpoint**: `checkpoints/racformer_r50_f8.pth`
**Config**: `configs/racformer_r50_nuimg_704x256_f8.py`

### Dataset Composition

| Condition | Val Samples | % of Total |
|-----------|-------------|------------|
| Day | 4,449 | 73.9% |
| Rain | 968 | 16.1% |
| Night | 602 | 10.0% |
| **Total** | **6,019** | **100%** |

Night classes with 0% AP (not present in night scenes): bus, trailer, construction_vehicle.

---

## Experiment 1: 3-Camera 3-Radar (Front Sensors Only)

**Dates**: Dec 13-22, 2025
**Duration**: ~9.5 days (36 epochs)
**Config**: Front-facing cameras and radars only (3 of each)
**Goal**: Evaluate impact of sensor coverage reduction

### Results

| Metric | Value | vs Pretrained |
|--------|-------|---------------|
| mAP | 47.87% | -6.31% |
| NDS | 56.47% | -4.97% |

**Conclusion**: Reducing from 6 to 3 cameras costs ~6.3% mAP (88% of full model). Not directly relevant to night investigation but establishes a sensor coverage baseline.

---

## Experiment 2: Baseline Retrain

**Dates**: Jan 25 - Feb 4, 2026
**Duration**: ~9.5 days (36 epochs, 2 GPUs)
**Config**: `configs/racformer_r50_nuimg_704x256_f8.py`
**Goal**: Establish a fair comparison baseline with identical training setup
**Output**: `outputs/racformer_r50_nuimg_704x256_f8/2026-01-25/20-49-49/`

### Results

| Condition | mAP | NDS |
|-----------|-----|-----|
| All | 51.46% | 59.35% |
| Day | 51.19% | 59.34% |
| Night | 31.03% | 38.45% |
| Rain | 55.00% | 62.46% |
| **Day-Night Gap** | **20.16% (39.4% rel)** | |

### Key Observations

- **2.72% below pretrained checkpoint** (51.46% vs 54.18%). Possible causes: different hyperparameters, random seed, or the pretrained being best-of-N.
- Night performance proportionally similar to pretrained (39.4% vs 40.1% relative gap).
- All subsequent experiments compared against this retrained baseline for fairness.

---

## Experiment 3: Depth Oracle

**Dates**: January 2026
**Goal**: Test if depth estimation is the bottleneck for night performance
**Method**: Replace predicted depth with ground-truth LiDAR depth during inference

### Results

| Metric | Predicted Depth | GT Depth | Delta |
|--------|----------------|----------|-------|
| Night mAP | 32.28% | 32.59% | +0.31% |

**Conclusion**: **REJECTED.** Depth is NOT the bottleneck. Perfect depth only improves night mAP by 0.31%. See `DEPTH_ORACLE_EXPERIMENT.md`.

---

## Experiment 4: Inference-Time Fusion Modifications (Path B)

**Dates**: January 2026
**Goal**: Modify fusion weights at inference time to reduce camera reliance at night
**Methods tested**:
- Radar weight boosting (multiple scales)
- Camera weight reduction
- Brightness-based gating

### Results

| Approach | Night mAP Delta |
|----------|----------------|
| Boost radar 1.5x | -2.4% |
| Boost radar 2.0x | -3.1% |
| Boost radar 3.0x | -4.4% |
| Zero camera features | -4.56% (catastrophic: 1 TP, 1987 FPs) |
| Brightness gating | Negative |

**Conclusion**: **ALL REJECTED.** Any post-hoc modification to fusion weights produces OOD inputs to downstream layers. The model expects the original feature distribution. See `PATH_B_EXPERIMENT_REPORT.md`.

---

## Experiment 5: Hard Camera Dropout (p=0.2)

**Dates**: Jan 25 - Feb 3, 2026
**Duration**: ~9 days (36 epochs, 2 GPUs)
**Config**: `configs/racformer_r50_nuimg_704x256_f8_dropout.py`
**Method**: Zero camera features (`query_feat`) with 20% probability per decoder layer forward pass
**Implementation**: `models/racformer_transformer.py:269-329`
**Output**: `outputs/racformer_r50_nuimg_704x256_f8_dropout/2026-01-25/22-45-16/`

### Results

| Condition | mAP | NDS | vs Baseline |
|-----------|-----|-----|-------------|
| All | 51.49% | 59.26% | +0.03% / -0.09% |
| Day | 51.38% | 59.33% | +0.19% / -0.01% |
| Night | 28.27% | 35.96% | **-2.76% / -2.49%** |
| Rain | 54.54% | 62.66% | -0.46% / +0.20% |
| **Day-Night Gap** | **23.11% (45.0% rel)** | | **Worsened** |

### Per-Class Night Impact

| Class | Baseline Night | Dropout Night | Delta | Type |
|-------|---------------|---------------|-------|------|
| car | 78.5% | 78.1% | -0.4% | Radar-detectable |
| truck | 52.3% | 52.1% | -0.2% | Radar-detectable |
| **pedestrian** | **63.7%** | **53.6%** | **-10.1%** | Camera-dependent |
| **bicycle** | **30.0%** | **20.0%** | **-10.0%** | Camera-dependent |
| **traffic_cone** | **14.7%** | **7.4%** | **-7.3%** | Camera-dependent |
| barrier | 36.4% | 38.0% | +1.6% | Mixed |

### Training Dynamics

- Convergence slower but fully recovers: 3.6% behind at epoch 12, within noise by epoch 36.
- Higher final loss (9.24 vs 8.49) expected due to 20% forward passes without camera.
- Overall mAP masks night regression because night is only 10% of val set.

### Key Insight

Hard zeroing trains for binary (camera present/absent), but night is continuous degradation. Camera features at night carry partial useful signal. Reducing camera reliance loses this signal, especially for camera-dependent classes.

**Conclusion**: **REJECTED.** Worsens night by -2.76%, day-night gap from 39.4% to 45.0%.

---

## Experiment 6: Night Data Augmentation (SimulateNight, p=0.3)

**Dates**: Jan 25 - Mar 12, 2026 (3 runs)
**Duration**: ~45 days total (including restarts and pipeline optimization)
**Config**: `configs/racformer_r50_nuimg_704x256_f8_nightaug.py`
**Method**: Apply synthetic night transformation to 30% of training images
**Implementation**: `loaders/pipelines/night_augmentation.py`
**Output**: `outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-02-01/01-41-59/`

### Augmentation Parameters

| Parameter | Value |
|-----------|-------|
| Probability | 30% |
| Brightness | 0.42-0.52 |
| Gamma | 1.5-1.9 |
| Contrast | 0.68-0.78 |
| Noise std | 6-12 |
| Vignette | strength 0.3-0.45 |
| Headlight gradient | strength 0.3-0.45 |
| Bright spots | 4-6 spots, brightness 150-230 |
| Color shift | strength 0.1-0.18 |
| Preserve bright | threshold 200, factor 0.55 |

### Run History

| Run | Dates | Epochs | Status | Notes |
|-----|-------|--------|--------|-------|
| 1 | Jan 25-28 | 3 | Stopped | Configuration tuning |
| 2 | Jan 28-31 | 6 | Stopped | Parameter adjustments |
| 3 | Feb 1 - Mar 12 | 36 | **Complete** | Pipeline optimized at epoch 9 |

**Pipeline optimization**: At epoch 9, SimulateNight was moved from before image resize (900x1600) to after (256x704), reducing data loading overhead by ~8x. Spot sizes adjusted from (12, 40) to (4, 14). This cut epoch time from ~15h to ~8h.

### Results

| Condition | mAP | NDS | vs Baseline |
|-----------|-----|-----|-------------|
| All | 49.38% | 58.12% | **-2.08% / -1.23%** |
| Day | 49.38% | 58.08% | -1.81% / -1.26% |
| Night | 31.21% | 38.71% | **+0.18% / +0.26%** |
| Rain | 52.36% | 61.21% | -2.64% / -1.25% |
| **Day-Night Gap** | **18.17% (36.8% rel)** | | **Only better because day dropped** |

### Per-Class Night Impact

| Class | Baseline Night | NightAug Night | Delta |
|-------|---------------|----------------|-------|
| car | 78.5% | 77.0% | -1.5% |
| truck | 52.3% | 51.6% | -0.7% |
| pedestrian | 63.7% | 56.1% | -7.6% |
| motorcycle | 34.7% | 33.4% | -1.3% |
| bicycle | 30.0% | 35.6% | +5.6% |
| traffic_cone | 14.7% | 6.1% | -8.6% |
| barrier | 36.4% | 52.4% | +16.0% |

### Key Insight

Hand-crafted night augmentation doesn't capture real night degradation. The synthetic-to-real domain gap means the backbone doesn't learn transferable features. The augmentation diluted clean training data (-2.08% overall) without compensating improvement at night (+0.18%).

**Conclusion**: **REJECTED.** Negligible night improvement, significant overall regression.

---

## Analysis: Feature Statistics Investigation

**Dates**: January 2026
**Goal**: Understand why camera features fail at night
**Method**: Compare feature norms, standard deviations, and activation patterns between day and night

### Results

- Feature norms: **Identical** between day and night
- Feature std: **Identical** between day and night
- Activation patterns: Visually similar distributions

**Conclusion**: Night camera features are "confidently wrong" — they have correct magnitude/statistics but degraded semantic content. This makes the problem fundamentally harder because the model cannot distinguish reliable from unreliable features using simple statistics.

---

## Summary: All Experiments

| # | Experiment | Night mAP | Overall mAP | Night Delta | Overall Delta | Status |
|---|-----------|-----------|-------------|-------------|---------------|--------|
| - | Pretrained checkpoint | 32.28% | 54.18% | reference | reference | - |
| - | Baseline retrain | 31.03% | 51.46% | -1.25% | -2.72% | Fair baseline |
| 1 | 3-cam 3-rad | -- | 47.87% | -- | -6.31% | Sensor study |
| 2 | Depth oracle | +0.31% | -- | +0.31% | -- | **REJECTED** |
| 3 | Inference-time fusion | -2.4 to -4.6% | -- | negative | -- | **REJECTED** |
| 4 | Hard camera dropout | 28.27% | 51.49% | **-2.76%** | +0.03% | **REJECTED** |
| 5 | Night augmentation | 31.21% | 49.38% | **+0.18%** | **-2.08%** | **REJECTED** |

---

## What We've Learned

### The Core Problem Is Hard

Night image features are "confidently wrong" — identical statistics but corrupted semantics. This means:

1. **The model can't detect degradation** (features look normal)
2. **Simple interventions don't work** (dropout, augmentation, weight scaling)
3. **The problem is in the backbone's learned representations**, not in the fusion mechanism

### What Doesn't Work

| Category | Approaches Tried | Why They Failed |
|----------|-----------------|-----------------|
| **Depth-focused** | GT depth oracle | Not the bottleneck (+0.31%) |
| **Inference-time** | Radar boosting, camera reduction, brightness gating | OOD inputs break trained layers |
| **Training-time dropout** | Hard camera dropout (p=0.2) | Binary present/absent doesn't match continuous degradation |
| **Training-time augmentation** | SimulateNight (p=0.3) | Synthetic night doesn't match real night conditions |
| **Feature analysis** | Norm-based quality detection | Features are identical day/night |

### Key Constraints

1. **Night is only 10% of val set** — overall mAP hides night regressions. A 5% night loss shows as ~0.5% overall.
2. **LiDAR ceiling for night is 35.4%** — even the best sensor achieves limited night performance.
3. **Baseline retrain gap of 2.72%** — confounds all comparisons with pretrained checkpoint.
4. **Camera contributes 52.7%** of detection signal — it can't be zeroed without catastrophic loss.

---

## Remaining Directions

With training-time interventions exhausted, remaining options require architectural changes:

| Approach | Description | Difficulty |
|----------|-------------|------------|
| **Radar-Guided Query Init** | Initialize queries at radar detection locations instead of learned anchors | Medium |
| **Learned Condition Fusion** | Predict fusion weights from raw image appearance during training | High |
| **CycleGAN Day-to-Night** | Use learned style transfer (not hand-crafted) for augmentation | High |
| **Night-specific head** | Train a separate detection head for night conditions | Medium |

---

## File Reference

| Item | Path |
|------|------|
| Pretrained checkpoint | `checkpoints/racformer_r50_f8.pth` |
| Baseline retrain checkpoint | `outputs/racformer_r50_nuimg_704x256_f8/2026-01-25/20-49-49/latest.pth` |
| Dropout checkpoint | `outputs/racformer_r50_nuimg_704x256_f8_dropout/2026-01-25/22-45-16/latest.pth` |
| Night aug checkpoint | `outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-02-01/01-41-59/epoch_36.pth` |
| Dropout implementation | `models/racformer_transformer.py:269-329` |
| Night aug implementation | `loaders/pipelines/night_augmentation.py` |
| Night eval script | `tools/eval_night_gap.py` |
| Depth oracle report | `research-notes/racformer/DEPTH_ORACLE_EXPERIMENT.md` |
| Path B report | `research-notes/racformer/PATH_B_EXPERIMENT_REPORT.md` |
| Error analysis | `research-notes/racformer/WHY_ANALYSIS_ERROR_TYPES.md` |
| Training analysis | `research-notes/racformer/TRAINING_EXPERIMENT_ANALYSIS.md` |

---

*Last updated: 2026-03-12*

---

## Phase 6: Autonomous Inference-Time Screening (2026-03-25)

**Goal**: Find inference-time-only improvements to mAP/NDS without retraining.
**Method**: 300-sample mini-set screening (~3 min/experiment), promote >0.3% mAP to full 6019-sample validation.
**Duration**: ~3 hours, 8 experiments (12 configurations tested).

### Results Table

| # | Hypothesis | Mini mAP | Δ mAP | Full mAP | Full NDS | Status |
|---|-----------|----------|-------|----------|----------|--------|
| 0 | Baseline | 0.4830 | — | 0.5418 | 0.6144 | — |
| 1 | BEV NMS (IoU=0.5) | 0.4834 | +0.04% | — | — | Discard |
| 1b | BEV NMS (IoU=0.2) | 0.4794 | -0.36% | — | — | Discard |
| 2 | topk 300→500 | 0.4841 | +0.11% | — | — | Discard |
| 3 | Query init α=1.25 (H2.2) | — | — | — | — | Skipped |
| 4 | Radar-CLOCs (boost=0.2, r=3m) | 0.4892 | +0.63% | 0.5416 | 0.5995 | Discard |
| 5a | Decoder logit ensemble (last 3) | 0.4812 | -0.18% | — | — | Discard |
| 5b | Decoder logit ensemble (all 6) | 0.4796 | -0.34% | — | — | Discard |
| 6a | Box averaging (last 3 layers) | 0.4834 | +0.04% | — | — | Discard |
| 6b | Box averaging (last 2 layers) | 0.4827 | -0.03% | — | — | Discard |
| 7 | Zero velocity static classes | 0.4829 | -0.01% | — | — | Discard |
| 8 | CLAHE image enhancement | 0.4829 | -0.01% | — | — | Discard |

### Detailed Findings

**1. NMS is harmful for set-based DETR (Exp 1, 1b)**
- IoU=0.5: neutral (+0.04%). IoU=0.2: -0.36%.
- DETR's set-based training with Hungarian matching already prevents duplicates. NMS removes correct predictions, not duplicates. More aggressive NMS = more damage.
- **Conclusion**: NMS/Soft-NMS/WBF are counterproductive for NMS-free DETR architectures.

**2. Top-K bottleneck doesn't exist (Exp 2)**
- Increasing from 300 to 500 predictions: +0.11% (noise).
- The model concentrates high scores on true objects; rare classes are not crowded out.

**3. Query initialization is a training-time parameter (Exp 3 — skipped)**
- The paper's α=1.25 non-uniform distribution is overwritten by `load_checkpoint`. The learned `init_query_bbox.weight` from training replaces any initialization.
- **Flagged as training candidate**: Retrain with α=1.25 to test if non-uniform distance distribution helps far-range detection.

**4. Radar-CLOCs: screening artifact (Exp 4)**
- Most interesting result: mini-set showed +0.63% mAP with radar proximity boost.
- Full validation: mAP -0.02%, NDS -1.49%. The gain was specific to the 300-sample subset.
- **Root cause**: The model already fuses radar internally (23.1% contribution). Post-hoc radar verification is redundant — it re-applies information the model already incorporated.
- **Lesson**: Always validate screening gains on full dataset. Mini-set can produce false positives.

**5. Decoder layer ensembling doesn't help (Exp 5, 6)**
- Logit averaging across layers: -0.18% (3 layers) to -0.34% (6 layers). Earlier layers have less refined classification and dilute the final layer's accuracy.
- Box averaging across layers: ±0.04% (noise). Decoder layers converge rapidly; box predictions are nearly identical in the last 2-3 layers.
- **Conclusion**: DETR decoder iterative refinement works as designed — the last layer is definitively the best.

**6. Model already handles static classes (Exp 7)**
- Zeroing predicted velocities for barrier/traffic_cone: -0.01% mAP, +0.00% NDS.
- The model already predicts near-zero velocities for these classes (mAVE improvement: 0.0003).

**7. Input preprocessing has zero effect (Exp 8)**
- CLAHE (clip_limit=2.0) on all camera images: -0.01% mAP, +0.00% NDS.
- ResNet-50 features are robust to mild contrast changes. The night-time problem is semantic feature degradation, not pixel brightness.

### Meta-Analysis: Why Inference-Time Improvements Fail

1. **nuScenes mAP is rank-based**: Computed as area under the precision-recall curve at fixed recall thresholds. Any monotonic score transformation (temperature, calibration, boosting) preserves ranking and thus preserves AP. Only changes to box positions or adding/removing detections can change mAP.

2. **The model's output is near-optimal for its architecture**: The published checkpoint represents a well-trained model with carefully tuned hyperparameters. Post-processing cannot compensate for architectural limitations.

3. **Information redundancy**: The model already uses radar (23.1% feature contribution), temporal data (8-frame ConvGRU), and multi-camera fusion. Post-hoc use of any of these signals is redundant.

4. **TTA is blocked by architecture**: `MultiScaleFlipAug3D` with `flip=True` requires an `aug_test` method, which RaCFormer does not implement. Proper multi-camera TTA needs camera pair swapping and extrinsic updates — substantial engineering.

### Eliminated Inference-Time Approaches (Comprehensive)

The following can be definitively added to the "do not try" list for this model:

| Approach | Why It Fails |
|----------|-------------|
| Any form of NMS (hard, soft, WBF) | Set-based DETR has no duplicates |
| Score threshold tuning | Rank-based mAP is threshold-invariant |
| Top-K increase | No bottleneck at K=300 |
| Post-hoc radar score boosting | Model already uses radar internally |
| Decoder layer ensembling (scores) | Earlier layers dilute accuracy |
| Decoder layer ensembling (boxes) | Layers already converged |
| Static class velocity correction | Model already handles correctly |
| CLAHE / image contrast enhancement | ResNet features robust, night issue is semantic |

