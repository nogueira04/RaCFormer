"""
RaCFormer with Mutual Modality Dropout Training

Randomly drops either camera or radar features with 50/50 probability.
This helps the model learn to work with either modality alone.

Expected gain: +1-2% night mAP
"""

_base_ = ['./racformer_r50_nuimg_704x256_f8.py']

# Mutual dropout: randomly drop camera or radar
modality_dropout_prob = 0.2
modality_dropout_mode = 'mutual'

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            modality_dropout_prob=modality_dropout_prob,
            modality_dropout_mode=modality_dropout_mode,
        ),
    ),
)

checkpoint_config = dict(interval=6, max_keep_ckpts=7)
work_dir = './work_dirs/racformer_r50_dropout_mutual20'
