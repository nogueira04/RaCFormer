# Radar BEV Expansion Plan

Status: local staging only. Do not submit until the user approves the next experiment.

## Motivation

S5 condition-aware fusion preserved the S5 night gain but failed to recover day, rain, and overall mAP. The next branch should not be another scalar/context gate. A bounded alternative is a RobuRCDet-inspired local radar BEV expansion that improves radar feature density without changing the S0 day-only data distribution.

## Hypothesis

Adding a zero-initialized residual projection over fixed Gaussian-smoothed radar BEV features can improve night/rain robustness while preserving day performance. Because the residual projection starts at zero, the baseline forward output is preserved at initialization and the expansion path must learn a useful residual during training.

This config is not strict-checkpoint compatible with pre-expansion RaCFormer checkpoints because it adds trainable projection parameters. It should train and evaluate its own checkpoint; do not use it to strictly evaluate old S0 weights.

## Staged Files

- `remote_patch_work/models/racformer.py`
  - Adds `RadarBEVExpansion`.
  - Adds optional `radar_bev_expansion` model config key.
  - Applies expansion after `radar_bev_conv` for normal and empty-radar branches.
- `remote_patch_work/configs/racformer_train2k_day_radarbevexp_research.py`
  - Inherits S0 day-only training.
  - Enables `radar_bev_expansion=dict(kernel_sizes=(3, 5, 7))`.

## Proposed Remote Files If Approved

- `models/racformer.py`
- `configs/racformer_train2k_day_radarbevexp_research.py`
- `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp.sbatch`
- `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp_eval.sbatch`
- `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp_summary.sbatch`
- `research/night_gen_phase1/staged_radar_bev_expansion/smoke_s0_radarbevexp_model.sbatch`
- `research/night_gen_phase1/staged_radar_bev_expansion/summarize_s0_radarbevexp.py`

## Validation So Far

- Local Python syntax check passed:
  `python -m py_compile remote_patch_work/models/racformer.py remote_patch_work/configs/racformer_train2k_day_radarbevexp_research.py remote_patch_work/staged_radar_bev_expansion/summarize_s0_radarbevexp.py`
- Local shell syntax check passed:
  `bash -n remote_patch_work/staged_radar_bev_expansion/smoke_s0_radarbevexp_model.sbatch remote_patch_work/staged_radar_bev_expansion/run_s0_radarbevexp.sbatch remote_patch_work/staged_radar_bev_expansion/run_s0_radarbevexp_eval.sbatch remote_patch_work/staged_radar_bev_expansion/run_s0_radarbevexp_summary.sbatch`

## Remote Validation Sequence

1. Back up current remote `models/racformer.py`.
2. Upload `models/racformer.py` and `configs/racformer_train2k_day_radarbevexp_research.py`.
3. Upload the staged scripts under `research/night_gen_phase1/staged_radar_bev_expansion/`.
4. Run `smoke_s0_radarbevexp_model.sbatch` and require model-build plus zero-init checks to pass.
5. Submit train, eval, and summary as a dependency chain only after the smoke succeeds.

Node constraint: use only `livenode02` or `livenode03`. Do not submit to `livenode01` because it has a known NVIDIA driver problem.

## Gate

Use the same S0 publication gate:

- night mAP >= +1.0 pp vs S0
- day mAP >= -1.0 pp vs S0
- overall mAP >= -1.5 pp vs S0
- night NDS >= -0.5 pp vs S0

If it passes, replicate or scale before claiming paper-worthiness.
