"""S0 occupancy + sweep-time BEV residual ablation.

This removes compensated velocity from the passing occupancy+velocity+time
branch while keeping RCS muted. It tests whether the night gain needs velocity
or is mostly from occupancy plus recency.
"""

_base_ = ["./racformer_train2k_day_research.py"]

model = dict(
    radar_rcs_bev_residual=dict(
        output_shape=(128, 128),
        rcs_index=3,
        rcs_scale=1000000.0,
        extra_indices=(6,),
        extra_scales=(1.0,),
    ),
)
