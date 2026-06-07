"""S0 day-only plus lightweight radar BEV expansion.

Hypothesis: a RobuRCDet-style local expansion of encoded radar BEV features can
improve night/rain robustness without changing the S0 day-only data distribution.
The residual projection is zero-initialized, so the baseline forward output is
preserved at initialization and the model must learn any expansion benefit during
training. The extra projection parameters mean this config is not strict-checkpoint
compatible with pre-expansion RaCFormer checkpoints.
"""

_base_ = ["./racformer_train2k_day_research.py"]

model = dict(
    radar_bev_expansion=dict(
        kernel_sizes=(3, 5, 7),
    ),
)
