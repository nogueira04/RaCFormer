"""Adaptive-fusion replication on top of S3 seed20260502 ratio18p75 w05.

This keeps the same generated-sample 0.5x loss weighting as the seed20260425
adaptive pass, but swaps to the held-out generated manifest seed.
"""

_base_ = ["./racformer_train2k_genaug_seed20260502_ratio18p75_w05_research.py"]

model = dict(
    pts_bbox_head=dict(
        transformer=dict(adaptive_fusion_gate=True),
    ),
)
