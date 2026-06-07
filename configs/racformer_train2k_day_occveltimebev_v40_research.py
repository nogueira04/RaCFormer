"""Velocity-scale ablation for S0 occupancy + velocity + time BEV residual.

This keeps the seed-0 S0_occveltimebev structure but weakens compensated
velocity normalization by increasing vx/vy scales from 20.0 to 40.0. The
residual code uses tanh(stat / scale), so a larger scale reduces saturation.
"""

_base_ = ["./racformer_train2k_day_research.py"]

model = dict(
    radar_rcs_bev_residual=dict(
        output_shape=(128, 128),
        rcs_index=3,
        rcs_scale=1000000.0,
        extra_indices=(4, 5, 6),
        extra_scales=(40.0, 40.0, 1.0),
    ),
)
