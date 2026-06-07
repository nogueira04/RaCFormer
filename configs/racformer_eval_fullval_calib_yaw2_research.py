"""Full-val evaluation with fixed +2 degree yaw calibration perturbation.

Use this config for both S0 and calibration-noise-trained checkpoints so the robustness
comparison uses the same controlled projection error.
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
        fixed_yaw_deg=2.0,
        fixed_trans_m=(0.0, 0.0, 0.0),
    ),
)

data = dict(
    val=dict(
        pipeline=test_pipeline,
    ),
)
