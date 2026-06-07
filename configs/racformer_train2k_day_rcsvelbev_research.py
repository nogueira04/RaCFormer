"""S0 day-only plus zero-init radar kinematic BEV residual.

Hypothesis: occupancy, RCS, and compensated velocity statistics in radar BEV
give the model a lightweight motion/reflectivity prior for night robustness
without changing the detector head or transformer contract.
"""

_base_ = ["./racformer_train2k_day_research.py"]

model = dict(
    radar_rcs_bev_residual=dict(
        output_shape=(128, 128),
        rcs_index=3,
        rcs_scale=32.0,
        extra_indices=(4, 5),
        extra_scales=(20.0, 20.0),
    ),
)
