# (d1) radar-camera async, offset 1 sweep step(s) ~= 77 ms
# Cell fragment for the robustness study. Inherits the frozen full-val eval config and swaps
# ONLY the two radar loader entries of the test pipeline (lines :227 and :228 of
# configs/racformer_r50_nuimg_704x256_f8.py, reached through
# configs/racformer_eval_fullval_research.py). New file; nothing tracked outside
# research/robust_study/ is touched.
#
# AMENDED 2026-08-03. The ladder is re-based onto the PHYSICAL sweep grid: `offset` counts
# steps along the real sample_data prev chain (nuScenes radar sweep period ~77 ms), not
# positions in results["sweeps"]["prev"]. The superseded prev-index mechanism was measured to
# be a no-op (fault-families.md (d1), "AMENDED 2026-08-03"). Every element of radar_points,
# the frame-t element included, is aggregated from 1 sweep step(s) earlier; the camera path
# and the ego-motion reference frame (keyframe t) are untouched, which is what makes the
# devkit compensate stale sweeps against the wrong clock -- the async artifact itself.
#
# Run with the frozen driver, e.g.
#   conda run -n racformerfix python -u research/night_gen_phase1/eval_by_condition.py \
#     --config research/robust_study/configs/d1_async_offset1.py \
#     --weights checkpoints/racformer_r50_f8.pth --full-val --out-dir <run dir>

from research.robust_study.corruptions.cell_config import BASE_CONFIG, build_pipeline

_base_ = [BASE_CONFIG]

_pipeline = build_pipeline(
    frame_t_type="D1AsyncLoadnuradarpoints",
    sweeps_type="D1AsyncLoadradarpointsFromMultiSweeps",
    extra=dict(offset=1),
)

data = dict(val=dict(pipeline=_pipeline), test=dict(pipeline=_pipeline))
