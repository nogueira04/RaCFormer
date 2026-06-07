"""Ablation for Branch A: continuous reliability gate without cosine/geometry cues."""

_base_ = ["./racformer_train2k_mixed_contrelqfusion_research.py"]

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            reliability_use_pairwise_cosine=False,
            reliability_use_query_geometry=False,
        ),
    ),
)
