"""Ablation for Branch A: remove query range/speed geometry cues."""

_base_ = ["./racformer_train2k_mixed_contrelqfusion_research.py"]

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            reliability_use_pairwise_cosine=True,
            reliability_use_query_geometry=False,
        ),
    ),
)
