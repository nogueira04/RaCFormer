"""S0 day-only plus zero-init radar RCS BEV residual.

Hypothesis: radar occupancy and RCS statistics in BEV provide a less destructive
radar-side cue than broad Gaussian feature expansion. The residual branch is
zero-initialized, preserving the S0 forward output at initialization while
allowing the model to learn an RCS-aware correction during training.
"""

_base_ = ["./racformer_train2k_day_research.py"]

model = dict(
    radar_rcs_bev_residual=dict(
        output_shape=(128, 128),
        rcs_index=3,
        rcs_scale=32.0,
    ),
)
