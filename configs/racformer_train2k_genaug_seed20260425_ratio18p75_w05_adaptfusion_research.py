"""Adaptive-fusion branch on top of S3 seed20260425 ratio18p75 w05.

This enables an identity-initialized decoder fusion gate after image/radar/LSS
query features are aligned. The gate is disabled by default in the code patch,
so this config is the only experiment surface that activates it.
"""

_base_ = ["./racformer_train2k_genaug_seed20260425_ratio18p75_w05_research.py"]

model = dict(
    pts_bbox_head=dict(
        transformer=dict(adaptive_fusion_gate=True),
    ),
)
