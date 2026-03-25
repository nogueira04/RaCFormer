"""
RaCFormer with Modality Dropout Training

This config enables sensor/modality dropout during training to improve robustness,
particularly for night-time performance where camera features may be degraded.

Based on research findings:
- Night mAP is 32.3% vs Day mAP 53.9% (40% relative gap)
- Inference-time fusion modifications don't work (model expects original distribution)
- Training with dropout forces the model to learn fallback pathways

Dropout modes:
- 'camera': Drop camera features (recommended for night improvement)
- 'radar': Drop radar features
- 'lss': Drop LSS BEV features
- 'mutual': 50/50 drop camera or radar
- 'any': Randomly drop one of three modalities

Usage:
    python tools/train.py configs/racformer_r50_nuimg_704x256_f8_dropout.py
"""

_base_ = ['./racformer_r50_nuimg_704x256_f8.py']

# Modality dropout configuration
# Start with 20% camera dropout (recommended by research notes)
modality_dropout_prob = 0.2
modality_dropout_mode = 'camera'  # Options: 'camera', 'radar', 'lss', 'mutual', 'any'

# Override the transformer config to include dropout parameters
model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            modality_dropout_prob=modality_dropout_prob,
            modality_dropout_mode=modality_dropout_mode,
        ),
    ),
)

# Optionally save more checkpoints for analysis
checkpoint_config = dict(interval=6, max_keep_ckpts=7)  # Save every 6 epochs

# Work directory for this experiment
work_dir = './work_dirs/racformer_r50_dropout_cam20'
