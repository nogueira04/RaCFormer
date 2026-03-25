"""
RaCFormer with 10% Camera Dropout Training

Gentle camera dropout for conservative baseline.
Use this if 20% dropout causes instability.
"""

_base_ = ['./racformer_r50_nuimg_704x256_f8.py']

# Gentle 10% camera dropout
modality_dropout_prob = 0.1
modality_dropout_mode = 'camera'

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            modality_dropout_prob=modality_dropout_prob,
            modality_dropout_mode=modality_dropout_mode,
        ),
    ),
)

checkpoint_config = dict(interval=6, max_keep_ckpts=7)
work_dir = './work_dirs/racformer_r50_dropout_cam10'
