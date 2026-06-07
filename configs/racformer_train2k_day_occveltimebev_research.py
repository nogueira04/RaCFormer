"""S0 day-only plus zero-init radar occupancy + velocity + time BEV residual.

Hypothesis: occupancy, compensated velocity, and sweep-relative time statistics
may preserve motion context while still muting the noisy RCS channel. This is a
config-only fallback; do not submit it while the current velocity branches are
running.
"""

_base_ = ["./racformer_train2k_day_research.py"]

model = dict(
    radar_rcs_bev_residual=dict(
        output_shape=(128, 128),
        rcs_index=3,
        rcs_scale=1000000.0,
        extra_indices=(4, 5, 6),
        extra_scales=(20.0, 20.0, 1.0),
    ),
)
