"""S0 day-only plus zero-init radar RCS + velocity + time BEV residual.

Hypothesis: the RCS+velocity residual was the only radar-stat branch with a
positive night mAP delta; adding sweep-relative time may separate stale and
recent radar support without muting the useful RCS cue.
"""

_base_ = ["./racformer_train2k_day_research.py"]

model = dict(
    radar_rcs_bev_residual=dict(
        output_shape=(128, 128),
        rcs_index=3,
        rcs_scale=32.0,
        extra_indices=(4, 5, 6),
        extra_scales=(20.0, 20.0, 1.0),
    ),
)
