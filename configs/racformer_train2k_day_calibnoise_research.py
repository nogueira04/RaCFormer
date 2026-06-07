"""Train2k day baseline with controlled train-time calibration noise.

Branch: calibration-aware robustness. The transform perturbs only lidar2img after
depth/radar-depth targets are built, so labels, images, radar BEV, and stored datasets
remain untouched. This trains the decoder image-query sampler to tolerate small
extrinsic projection errors.
"""

_base_ = ["./racformer_train2k_day_research.py"]

from mmcv import Config

_base_cfg = Config.fromfile("configs/racformer_train2k_day_research.py")
train_pipeline = [step.copy() for step in _base_cfg.train_pipeline]
train_pipeline.insert(
    -2,
    dict(
        type="CalibrationPerturbLidar2Img",
        prob=1.0,
        yaw_range_deg=(-2.0, 2.0),
        trans_range_m=((-0.20, 0.20), (-0.20, 0.20), (-0.05, 0.05)),
    ),
)

data = dict(
    train=dict(
        pipeline=train_pipeline,
    ),
)
