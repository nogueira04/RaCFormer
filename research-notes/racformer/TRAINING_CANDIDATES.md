# RaCFormer Training Candidates — Ranked by Expected Impact

**Date**: 2026-03-25
**Context**: After exhaustive inference-time screening (12 configs, 0 keepers), the following training-time experiments are the most promising paths to publishable improvements.

---

## Tier 1: High Confidence, Publishable Impact

### T1. Query Initialization with α=1.25 (Paper's Intended Design)

**What**: The paper claims queries are distributed in concentric circles with α=1.25 growth factor (80→100→125→156→195→244 queries per circle). The released code uses uniform distribution (150 per circle). This is likely a code/paper discrepancy.

**Why it matters**: Non-uniform distribution places more queries at far range where BEV area is larger, better matching object density distribution. Could improve far-range detection (+mAP) without any architectural changes.

**Implementation**:
```python
def generate_points(self):
    alpha = 1.25
    n_inner = 80
    num_per_circle = [int(n_inner * alpha**i) for i in range(self.num_clusters)]
    # Distribute angles proportionally per circle
    # distances same as current (uniform in [0,1])
```

**Training cost**: ~24h (36 epochs on livenode03)
**Expected gain**: +0.3-1.0% overall mAP, possibly larger at far range
**Publishability**: Correcting a paper/code discrepancy is a valid finding
**Risk**: Low — worst case we confirm uniform is equivalent

---

### T2. Modality Dropout Training (Learned Robustness)

**What**: During training, randomly zero out one modality branch (image, radar, or LSS) with probability p=0.1. Forces the model to develop fallback pathways.

**Why it matters**: The core night-time problem is that image features are "confidently wrong" but the model doesn't know to ignore them. Training with random modality dropout teaches the model to function with degraded/missing inputs.

**Implementation** (partially exists in codebase):
```python
# In racformer_transformer.py forward():
if self.training:
    if random.random() < 0.1:
        img_feat = torch.zeros_like(img_feat)
    if random.random() < 0.05:
        radar_feat = torch.zeros_like(radar_feat)
```

**Training cost**: ~24h
**Expected gain**: +1-3% night mAP, +0.3-0.5% overall mAP
**Publishability**: High — "Modality-robust radar-camera fusion via training-time dropout" is a clear narrative
**Risk**: Medium — hard dropout (p=0.2) already tested and hurt night by -2.76%. The key difference: use lower p (0.05-0.1) and drop modalities independently, not together. Previous experiment used p=0.2 which was too aggressive.

---

### T3. Condition-Adaptive Training with Night Augmentation (Refined)

**What**: Apply targeted night simulation during training with improved augmentation (gamma darkening + noise + reduced contrast), but with a curriculum: start with mild augmentation, increase over epochs.

**Why it matters**: Previous night augmentation (SimulateNight p=0.3) gave night +0.18% but overall -2.08%. The augmentation was too aggressive and applied too frequently. A curriculum approach with lower probability and milder transforms could help night without hurting day.

**Implementation**:
```python
# Curriculum: epoch 1-12 no aug, 13-24 mild aug (p=0.1), 25-36 stronger (p=0.2)
SimulateNight(p=schedule(epoch), gamma_range=(0.3, 0.6), noise_std=0.01)
```

**Training cost**: ~24h
**Expected gain**: +1-2% night mAP, -0.5% to +0.2% overall
**Publishability**: Medium — augmentation is incremental, but curriculum is a novel twist
**Risk**: Medium — previous night augmentation experiments showed the approach is fragile

---

## Tier 2: Medium Confidence, Worth Investigating

### T4. Radar-Guided Query Initialization (Novel Architecture)

**What**: Instead of fixed circular query distribution, initialize some queries at radar detection locations. Radar provides reliable geometry even at night.

**Implementation**:
```python
def generate_queries(self, radar_points, max_queries=900):
    # Cluster radar points into N clusters
    # Place N queries at cluster centers (theta, distance in polar)
    # Fill remaining (900-N) with standard circular distribution
```

**Training cost**: ~24h + 1-2 weeks implementation
**Expected gain**: +0.5-1.5% overall mAP, larger gains at night
**Publishability**: High — novel contribution if it works
**Risk**: High — radar is sparse (50-100 points), may not provide enough query seeds. Requires careful handling of variable-count radar points.

### T5. Stronger Backbone (ResNet-101 or ConvNeXt)

**What**: Replace ResNet-50 with ResNet-101 or ConvNeXt-Base for better image features.

**Training cost**: ~36-48h (larger model, slower)
**Expected gain**: +1-2% overall mAP (standard scaling law)
**Publishability**: Low — backbone upgrade is not novel
**Risk**: Low but high cost. Memory may require reduced batch size on 24GB RTX 4090.

### T6. TTA Implementation (aug_test Method)

**What**: Implement proper `aug_test` with horizontal flip for RaCFormer. Requires camera pair swapping, extrinsic updates, prediction merging.

**Implementation cost**: 1-2 days engineering, no training needed
**Expected gain**: +0.3-0.8% mAP (based on literature for similar models)
**Publishability**: Low — TTA is a standard trick, not novel
**Risk**: Low, but engineering-heavy for multi-camera BEV models

---

## Tier 3: Speculative / Long-Term

### T7. Knowledge Distillation from LiDAR Teacher

**What**: Train a LiDAR-only teacher model, distill its BEV features into RaCFormer's radar+camera BEV. The teacher provides "correct" BEV features that the student learns to approximate.

**Expected gain**: +2-4% overall, larger night gains
**Publishability**: High — KD for radar-camera is novel
**Risk**: High — requires training two models, complex distillation pipeline

### T8. Domain Adaptation (Day→Night)

**What**: Use adversarial domain adaptation to align day and night feature distributions in the BEV space.

**Expected gain**: +2-5% night mAP
**Publishability**: High
**Risk**: High — adversarial training is unstable, domain adaptation for 3D detection is underexplored

---

## Recommended Priority Order

For a **single-GPU, ~1 week budget**:
1. **T1** (Query init α=1.25) — lowest risk, 24h, tests paper's claim
2. **T2** (Modality dropout, p=0.05-0.1) — 24h, targets the core problem
3. **T6** (TTA implementation) — no training, pure engineering

For a **publication-oriented campaign** (2-4 weeks):
1. T1 + T2 simultaneously (different seeds, same GPU alternating)
2. If either improves: combine T1+T2 into a single model
3. Add T6 (TTA) on top for the best model → this becomes the "bag of tricks" paper
4. If neither improves: pivot to T4 (radar-guided queries, novel architecture contribution)

---

*Generated by autoresearch session 2026-03-25. Raw data: `research/results.tsv`*
