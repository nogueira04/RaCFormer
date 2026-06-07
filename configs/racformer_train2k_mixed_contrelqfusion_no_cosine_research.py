"""Ablation for Branch A: remove pairwise branch-agreement cosine cues."""

_base_ = ["./racformer_train2k_mixed_contrelqfusion_research.py"]

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            reliability_use_pairwise_cosine=False,
            reliability_use_query_geometry=True,
        ),
    ),
)
