"""Branch D radar-guided query initialization top-k ablation.

Matches the S6 radar-query screen, but reserves only the first 90 decoder
queries for top-scored current-frame radar points. This tests whether the full
180-query override is helping or crowding out useful learned anchor coverage.
"""

_base_ = ["./racformer_train2k_day_research.py"]

model = dict(
    pts_bbox_head=dict(
        radar_query_init=True,
        radar_query_topk=90,
        radar_query_use_velocity=False,
        radar_query_min_range=1.0,
        radar_query_score="rcs_speed",
    ),
)
