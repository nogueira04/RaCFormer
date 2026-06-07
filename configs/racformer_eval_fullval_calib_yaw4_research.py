"""Full-val evaluation with fixed +4 degree yaw calibration perturbation.

This is a stronger stress point for the calibration robustness branch and should only be
used after the +2 degree smoke/eval path is verified.
"""

_base_ = ["./racformer_eval_fullval_research.py"]

from mmcv import Config

_base_cfg = Config.fromfile("configs/racformer_eval_fullval_research.py")
test_pipeline = [step.copy() for step in _base_cfg.test_pipeline]
test_pipeline.insert(
    -1,
    dict(
        type="CalibrationPerturbLidar2Img",
        prob=1.0,
        fixed_yaw_deg=4.0,
        fixed_trans_m=(0.0, 0.0, 0.0),
    ),
)

data = dict(
    val=dict(
        pipeline=test_pipeline,
    ),
)
