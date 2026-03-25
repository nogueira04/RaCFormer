"""
RaCFormer with Camera Feature Noise Injection

Adds per-element Gaussian noise to camera features during training to
simulate night-time degradation where features have correct statistics
but corrupted semantic content ("confidently wrong").

Key insight: Night features have identical norms to day features but
degraded semantics. Uniform scaling (soft_camera) doesn't simulate this
and is partially neutralized by LayerNorm. Noise injection corrupts the
feature direction (semantics) while preserving magnitude, which is exactly
what night does.

Noise std is sampled from [0.1, 0.5] * feature_std per forward pass,
giving a range from mild to significant corruption.
"""

_base_ = ['./racformer_r50_nuimg_704x256_f8.py']

# Noise injection on camera features
modality_dropout_prob = 0.3  # 30% of forward passes get noise
modality_dropout_mode = 'noise_camera'

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            modality_dropout_prob=modality_dropout_prob,
            modality_dropout_mode=modality_dropout_mode,
        ),
    ),
)

checkpoint_config = dict(interval=6, max_keep_ckpts=7)
work_dir = './work_dirs/racformer_r50_noise_camera30'
