"""S0 day-only plus zero-init radar occupancy + velocity BEV residual.

Hypothesis: occupancy plus compensated velocity BEV statistics may provide the
motion cue wanted for night robustness while avoiding the noisy RCS channel that
hurt the earlier RCS-only and occupancy-dominant branches.
"""

_base_ = ["./racformer_train2k_day_research.py"]

model = dict(
    radar_rcs_bev_residual=dict(
        output_shape=(128, 128),
        rcs_index=3,
        rcs_scale=1000000.0,
        extra_indices=(4, 5),
        extra_scales=(20.0, 20.0),
    ),
)
