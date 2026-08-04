# (d2) extrinsic miscalibration, medium severity, seed 0
# Cell fragment for the robustness study. Inherits the frozen full-val eval config and swaps
# ONLY the two radar loader entries of the test pipeline (lines :227 and :228 of
# configs/racformer_r50_nuimg_704x256_f8.py, reached through
# configs/racformer_eval_fullval_research.py). New file; nothing tracked outside
# research/robust_study/ is touched.
#
# Run with the frozen driver, e.g.
#   conda run -n racformerfix python -u research/night_gen_phase1/eval_by_condition.py \
#     --config research/robust_study/configs/d2_extrinsic_medium_seed0.py \
#     --weights checkpoints/racformer_r50_f8.pth --full-val --out-dir <run dir>

from research.robust_study.corruptions.cell_config import BASE_CONFIG, build_pipeline

_base_ = [BASE_CONFIG]

# (d2) extrinsic miscalibration, level medium (sigma_rot=0.06, sigma_trans=0.006),
# corruption-realisation seed 0. Draws are keyed by (seed, scene_token, radar channel)
# and severity scales the same draws, so this cell shares common random numbers with
# d2_extrinsic_severe_seed0.py.
_pipeline = build_pipeline(
    frame_t_type="D2MiscalibLoadnuradarpoints",
    sweeps_type="D2MiscalibLoadradarpointsFromMultiSweeps",
    extra=dict(level="medium", seed=0),
)

data = dict(val=dict(pipeline=_pipeline), test=dict(pipeline=_pipeline))
