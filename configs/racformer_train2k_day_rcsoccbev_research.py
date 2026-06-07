"""S0 day-only plus zero-init radar occupancy-dominant BEV residual.

This is a config-only ablation of the RCS BEV residual branch. It keeps radar
occupancy as the main BEV cue and effectively mutes RCS by using a very large
normalization scale. The goal is to test whether raw RCS is noisy for this S0
night-transfer setting while preserving a radar hit prior.
"""

_base_ = ["./racformer_train2k_day_research.py"]

model = dict(
    radar_rcs_bev_residual=dict(
        output_shape=(128, 128),
        rcs_index=3,
        rcs_scale=1000000.0,
    ),
)
