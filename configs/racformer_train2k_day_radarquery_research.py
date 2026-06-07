"""Branch D radar-guided query initialization screen.

Train on the S0 2K day subset, but reserve a small slice of decoder queries for
object-centric radar point anchors from the current frame. This tests whether
radar-derived query proposals are a better story than dense radar BEV expansion
or global weather fusion.
"""

_base_ = ["./racformer_train2k_day_research.py"]

model = dict(
    pts_bbox_head=dict(
        radar_query_init=True,
        radar_query_topk=180,
        radar_query_use_velocity=False,
        radar_query_min_range=1.0,
        radar_query_score="rcs_speed",
    ),
)
