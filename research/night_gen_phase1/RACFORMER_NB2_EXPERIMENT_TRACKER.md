# RaCFormer NB2 Experiment Tracker

Last updated: 2026-05-16 22:57 UTC

Purpose: keep a durable record of what has already been tried, what is running, what is staged, and what should not be repeated while pursuing a paper-worthy RaCFormer/NB2 result without Vertex AI.

## Success Gate

Use S0 train-2k day baseline as the comparison point.

- S0 day mAP/NDS: `0.3152649818 / 0.3745762709`
- S0 night mAP/NDS: `0.1487749875 / 0.2150977574`
- S0 rain mAP/NDS: `0.2743174671 / 0.3713314930`
- S0 overall mAP/NDS: `0.3039905911 / 0.3697754272`

Current promotion gate for any candidate branch:

- Night mAP at least `+1.0 pp` versus S0.
- Day mAP no worse than `-1.0 pp` versus S0.
- Overall mAP no worse than `-1.5 pp` versus S0.
- Night NDS no worse than `-0.5 pp` versus S0.
- If a branch passes on one seed only, replicate before making any paper-worthy claim.

## Stop-Go Dashboard

This section is authoritative for current decisions. The long chronological notes below are audit history, not a queue of work to continue automatically.

## Running Now

- Branch D endpoint recorded `2026-05-16 22:57 UTC`: FAIL.
  - Primary gate failed for both active variants:
    - `S6_radarquery` (`N=180`): day `0.2554`, night `0.0958`, overall `0.2477`, night NDS `0.1675`.
    - `S6_radarquery_topk90` (`N=90`): day `0.2977`, night `0.1230`, overall `0.2863`, night NDS `0.1901`.
  - Secondary long-range/radar-rich path is closed because the required day mAP delta >= `-1.0 pp` fails
    for both variants (`N=180`: `-5.99 pp`; `N=90`: `-1.76 pp`).
  - Required endpoint artifacts:
    - `research/night_gen_phase1/reports/D_failure_mode_20260516T225745Z.md`
    - `research/night_gen_phase1/reports/BRANCH_CHOICE_20260516T225745Z.md`
    - `research/paper_goal_20260515/CLAIM_INVENTORY.md`
    - `research/paper_goal_20260515/FINAL_REPORT.md`
  - No `random_query_init`, seed-`20260502` replication, or additional top-k run is legal for this D result.
  - Next legal action: halt for user review of the failure-mode and branch-choice memos.
- Branch D radar-guided query initialization submitted `2026-05-15 19:47 UTC`; the
  `topk=180` screen completed and failed the S0 publication gate.
- Branch D top-k ablation submitted `2026-05-15 20:31 UTC` after the idle-node check and smoke pass;
  the `topk=90` ablation also completed and failed the S0 publication gate.
- GPU allocation decision:
  - `train.py` has a DDP path when `WORLD_SIZE > 1`, but the active SLURM wrappers are single-process,
    single-GPU jobs (`--nodes=1`, `--ntasks-per-node=1`) pinned to separate nodes.
  - Do not restart these jobs as multi-node DDP mid-run: it would change effective global batch/training dynamics
    unless separately retuned, and both GPU nodes are already occupied by comparable Branch D evidence.
- Active jobs:
  - No active research jobs remain. `squeue -u gnmp` only shows stale dependency jobs `1320` and `1321`.
  - `1397` `s6_radarquery`: completed train2k day radar-query screen; final checkpoint `epoch_12.pth`.
  - `1398` `s6_radarquery_eval`: completed full-val day/night/rain/overall eval.
  - `1399` `s6_radarquery_summary`: completed; S0 publication-gate verdict FAIL.
  - `1404` `s6_radarquery_subset`: completed; wrote CPU-only radar-rich/long-range subset diagnostics.
  - `1401` `s6_radarq90`: completed; final checkpoint `epoch_12.pth`.
  - `1402` `s6_radarq90_eval`: completed full-val day/night/rain/overall eval.
  - `1403` `s6_radarq90_summary`: completed; S0 publication-gate verdict FAIL.
  - `1405` `s6_radarq90_subset`: completed; wrote CPU-only radar-rich/long-range subset diagnostics.
- Active training health:
  - `1397` reached epoch 1 and saved
    `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_1.pth`
    at `2026-05-15 20:13 UTC`.
  - Checkpoint size: `764836753` bytes.
  - `1397` reached epoch 2 and saved
    `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_2.pth`
    at `2026-05-15 20:40 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1397` reached epoch 3 and saved
    `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_3.pth`
    at `2026-05-15 21:06 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1397` reached epoch 4 and saved
    `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_4.pth`
    at `2026-05-15 21:32 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1397` reached epoch 5 and saved
    `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_5.pth`
    at `2026-05-15 21:58 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1397` reached epoch 6 and saved
    `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_6.pth`
    at `2026-05-15 22:24 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1397` reached epoch 7 and saved
    `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_7.pth`
    at `2026-05-15 22:50 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1397` reached epoch 8 and saved
    `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_8.pth`
    at `2026-05-15 23:16 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1397` reached epoch 9 and saved
    `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_9.pth`
    at `2026-05-15 23:42 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1397` reached epoch 10 and saved
    `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_10.pth`
    at `2026-05-16 00:08 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1397` reached epoch 11 and saved
    `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_11.pth`
    at `2026-05-16 00:34 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1397` reached final epoch 12 and saved
    `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_12.pth`
    at `2026-05-16 01:00 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `slurm_1397.err` ended at `1746` bytes; inspected as tqdm/progress-bar output only.
  - Eval job `1398` completed full-val inference/evaluation on `6019` val samples.
  - Summary job `1399` wrote the S0 gate result:
    - day `0.2554 / 0.3312`, delta `-5.99 / -4.34 pp`.
    - night `0.0958 / 0.1675`, delta `-5.30 / -4.76 pp`.
    - rain `0.2343 / 0.3416`, delta `-4.00 / -2.98 pp`.
    - overall `0.2477 / 0.3285`, delta `-5.63 / -4.13 pp`.
    - Gate verdict: FAIL.
  - Decision: do not submit the staged seed-20260502 replication for `topk=180`.
    The result is far below gate, not a near-miss.
  - Eval logs:
    - `research/night_gen_phase1/results/S6_radarquery/eval_slurm_1398.out`
    - `research/night_gen_phase1/results/S6_radarquery/eval_slurm_1398.err`
  - Summary artifacts:
    - `research/night_gen_phase1/results/S6_radarquery/summary_metrics.md`
    - `research/night_gen_phase1/results/S6_radarquery/summary_metrics.json`
  - Full condition metrics:
    - `research/night_gen_phase1/results/S6_radarquery/eval/eval_by_condition.json`
  - Subset diagnostics job `1404` completed:
    - `research/night_gen_phase1/results/S6_radarquery/subset_eval/subset_metrics.md`
    - all samples `0.2477 / 0.3285`;
    - radar-supported samples `0.2481 / 0.3288`;
    - radar-rich top quartile samples `0.2745 / 0.3614`;
    - object far >=30m `0.0615 / 0.2052`;
    - object far >=40m `0.0275 / 0.1128`.
  - `1401` started on `livenode03` at `2026-05-15 20:31 UTC`; initial GPU log shows one RTX 4090.
  - `1401` reached epoch 1 iter 200 by `2026-05-15 20:37 UTC`; `slurm_1401.err` remained `0` bytes at the last check.
  - `1401` reached epoch 1 and saved
    `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_1.pth`
    at `2026-05-15 20:58 UTC`.
  - Checkpoint size: `764836753` bytes.
  - `1401` reached epoch 2 and saved
    `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_2.pth`
    at `2026-05-15 21:24 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1401` reached epoch 3 and saved
    `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_3.pth`
    at `2026-05-15 21:50 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1401` reached epoch 4 and saved
    `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_4.pth`
    at `2026-05-15 22:16 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1401` reached epoch 5 and saved
    `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_5.pth`
    at `2026-05-15 22:42 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1401` reached epoch 6 and saved
    `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_6.pth`
    at `2026-05-15 23:08 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1401` reached epoch 7 and saved
    `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_7.pth`
    at `2026-05-15 23:34 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1401` reached epoch 8 and saved
    `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_8.pth`
    at `2026-05-16 00:00 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1401` reached epoch 9 and saved
    `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_9.pth`
    at `2026-05-16 00:26 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1401` reached epoch 10 and saved
    `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_10.pth`
    at `2026-05-16 00:52 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1401` reached epoch 11 and saved
    `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_11.pth`
    at `2026-05-16 01:18 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `1401` reached final epoch 12 and saved
    `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_12.pth`
    at `2026-05-16 01:44 UTC`.
  - Checkpoint size: `764836945` bytes.
  - `slurm_1401.err` remained `0` bytes at train completion.
  - Eval job `1402` completed full-val inference/evaluation on `6019` val samples.
  - Summary job `1403` wrote the S0 gate result:
    - day `0.2977 / 0.3586`, delta `-1.76 / -1.60 pp`.
    - night `0.1230 / 0.1901`, delta `-2.58 / -2.50 pp`.
    - rain `0.2554 / 0.3550`, delta `-1.89 / -1.63 pp`.
    - overall `0.2863 / 0.3549`, delta `-1.77 / -1.49 pp`.
    - Gate verdict: FAIL.
  - Decision: do not submit the staged seed-20260502 replication for `topk=90`.
    The result misses the night mAP/NDS and day/overall preservation gates.
  - Eval logs:
    - `research/night_gen_phase1/results/S6_radarquery_topk90/eval_slurm_1402.out`
    - `research/night_gen_phase1/results/S6_radarquery_topk90/eval_slurm_1402.err`
  - Summary artifacts:
    - `research/night_gen_phase1/results/S6_radarquery_topk90/summary_metrics.md`
    - `research/night_gen_phase1/results/S6_radarquery_topk90/summary_metrics.json`
  - Full condition metrics:
    - `research/night_gen_phase1/results/S6_radarquery_topk90/eval/eval_by_condition.json`
  - Subset diagnostics job `1405` completed:
    - `research/night_gen_phase1/results/S6_radarquery_topk90/subset_eval/subset_metrics.md`
    - all samples `0.2863 / 0.3549`;
    - radar-supported samples `0.2867 / 0.3551`;
    - radar-rich top quartile samples `0.3119 / 0.3797`;
    - object far >=30m `0.0788 / 0.2138`;
    - object far >=40m `0.0365 / 0.1207`.
- Completed train jobs:
  - `1393` `s5_contrel`: completed train2k mixed continuous reliability query fusion screening train.
    - final checkpoint reached at `2026-05-15 15:38 UTC`.
    - checkpoint: `outputs/racformer_train2k_mixed_contrelqfusion_research/2026-05-15/10-30-04/epoch_12.pth`
    - checkpoint size: `764865729` bytes.
    - train log: `research/night_gen_phase1/results/S5_contrelqfusion/slurm_1393.out`
    - train stderr: `1622` bytes, inspected as tqdm/progress-bar output only.
- Completed eval/summary jobs:
  - `1394` `s5_contrel_eval`: completed full-val condition eval.
    - eval JSON: `research/night_gen_phase1/results/S5_contrelqfusion/eval/eval_by_condition.json`
    - eval stderr: `4397` bytes, inspected as eval progress/info output only.
  - `1395` `s5_contrel_summary`: completed summary vs S0 and S5 baselines.
    - summary: `research/night_gen_phase1/results/S5_contrelqfusion/summary_metrics.md`
    - summary JSON: `research/night_gen_phase1/results/S5_contrelqfusion/summary_metrics.json`
    - summary stderr: `0` bytes.
- Completed smoke jobs:
  - `1396` `s6_radarquery_smoke`: PASS on `livenode02`.
    - GPU: RTX 4090 visible, 24564 MiB, 0% utilization at start.
    - dataset/batch: `len=2000`, batch image shape `(2, 48, 3, 256, 704)`, `8` radar frames.
    - current-frame radar tensors: `(976, 7)` and `(995, 7)` with `434` and `474` in-range finite points.
    - mechanism: `_radar_points_to_query_bbox` changed exactly `180` queries per sample; query tail and velocity fields unchanged.
    - log: `research/night_gen_phase1/results/S6_radarquery/smoke_slurm_1396.out`
  - `1400` `s6_radarq90_smoke`: PASS on `livenode03`.
    - GPU: RTX 4090 visible, 24564 MiB, 0% utilization at start.
    - dataset/batch: `len=2000`, batch image shape `(2, 48, 3, 256, 704)`, `8` radar frames.
    - current-frame radar tensors: `(976, 7)` and `(995, 7)` with `428` and `438` in-range finite points.
    - mechanism: `_radar_points_to_query_bbox` changed exactly `90` queries per sample; query tail and velocity fields unchanged.
    - log: `research/night_gen_phase1/results/S6_radarquery_topk90/smoke_slurm_1400.out`
  - `1391` `s5_contrel_smoke`: failed before model/data smoke because the wrapper missed `PYTHONPATH=$PWD`.
  - `1392` `s5_contrel_smoke`: PASS on `livenode02`.
    - GPU: RTX 4090 visible, 24564 MiB, 0% utilization at start.
    - dataset: `len=2000`, first sample `scene_condition=day`, image tensor shape `(48, 3, 256, 704)`.
    - model: `ContinuousReliabilityQueryFusion` enabled, zero-initialized, identity gate verified.
    - log: `research/night_gen_phase1/results/S5_contrelqfusion/smoke_slurm_1392.out`
- Completed calibration robustness branch:
  - `1385` `s0_calibnoise`: completed train2k day calibration-noise screening train.
    - checkpoint: `outputs/racformer_train2k_day_calibnoise_research/2026-05-14/23-27-48/epoch_12.pth`
  - `1386` `s0_calibnoise_mini`: completed mini eval.
    - report: `research/night_gen_phase1/results/S0_calibnoise/mini_eval_1386/evaluation_report.txt`
  - `1387` / `1388` `s0_calibnoise_nom_eval` / `s0_calibnoise_nom_sum`: completed nominal full-val eval and summary.
    - summary: `research/night_gen_phase1/results/S0_calibnoise_nominal/summary_metrics.md`
  - `1389` / `1390` `s0_calibnoise_yaw2_eval` / `s0_calibnoise_yaw2_sum`: completed fixed `+2 deg` yaw eval and summary.
    - summary: `research/night_gen_phase1/results/S0_calibnoise_yaw2/summary_metrics.md`
- Completed diagnostic jobs:
  - `1383` `s0_calib_yaw2_eval`: completed on `livenode02`; S0 full-val eval with fixed `+2 deg` yaw perturbation.
  - `1384` `s0_calib_yaw2_summary`: completed; wrote `research/night_gen_phase1/results/S0_calib_yaw2/summary_metrics.md`.
- Stale dependency jobs remain:
  - `1320` `s0_rcsbev_eval`: `PD`, `DependencyNeverSatisfied`.
  - `1321` `s0_rcsbev_summary`: `PD`, dependency on the stale eval job.
- These stale jobs are not active experiments and should not drive decisions. Do not cancel them unless explicitly requested.
- Read-only fallback audit completed while S6 was running:
  - Branch C motion/temporal audit completed after S6 failed:
    - RaCFormer already computes `time_diff` and warps image/BEV sampling by query velocity in
      `models/racformer_transformer.py`.
    - CPU-only subset jobs `1406`-`1411` wrote moving/range subset metrics under
      `research/paper_goal_20260515/branch_c_motion_subset_audit/`.
    - `S0_occveltimebev` seed0 moving-sample mAP/NDS `0.3016 / 0.3690` did not survive seed1
      `0.2942 / 0.3579` or seed2 `0.2894 / 0.3554`; stable `S0_rcsvelbev` was stronger on
      moving-sample mAP at `0.3067`.
    - Branch C audit artifacts: `research/paper_goal_20260515/BRANCH_C_AUDIT.md` and
      `research/paper_goal_20260515/BRANCH_C_AUDIT.json`.
  - LiDAR-teacher distillation is blocked for the active loop because no local teacher/distillation/CenterPoint/LiDAR-teacher code or checkpoint names were found under the audited source/config/checkpoint paths.
  - Do not spend the next GPU slot on distillation unless a concrete teacher checkpoint/features path is supplied or found.
  - Expert-routed sensor degradation is blocked for the active loop: prior CamDrop20 full-checkpoint side eval hurt night by `-2.78 pp` mAP, 3Cam3Rad regressed overall by `-15.61 pp` mAP, and no expert/router/MoME implementation or synthetic-corruption benchmark is staged.
  - Branch F audit artifacts: `research/paper_goal_20260515/BRANCH_F_AUDIT.md` and `research/paper_goal_20260515/BRANCH_F_AUDIT.json`.
  - Foundation visual features are blocked/low priority for the active loop: existing DINOv3 inference-time FP filtering/score fusion did not produce a clean mAP/NDS gain, and RCDINO-style feature fusion is a heavier pretrained camera-backbone branch.
  - Low-light enhancement is blocked as a main path: DriveGEN visual QC failed geometry preservation, and LightDiff checkpoint access is dead/no Wayback recovery.
  - Branch G/H audit artifacts: `research/paper_goal_20260515/BRANCH_G_H_AUDIT.md` and `research/paper_goal_20260515/BRANCH_G_H_AUDIT.json`.
- Staged post-S6 helper:
  - `research/paper_goal_20260515/eval_radarquery_subsets.py` was uploaded and `py_compile`-checked on the remote.
  - `research/night_gen_phase1/staged_radarquery/run_s6_radarquery_subset_eval_livenode02.sbatch` was uploaded and `bash -n`-checked on the remote.
  - `research/night_gen_phase1/staged_radarquery_topk90/run_s6_radarquery_topk90_subset_eval_livenode03.sbatch` was uploaded and `bash -n`-checked on the remote.
  - Submitted as dependency-gated CPU jobs:
    - `1404` after eval job `1398`, expecting `research/night_gen_phase1/results/S6_radarquery/eval/submission_overall/pts_bbox/results_nusc.json`.
    - `1405` after eval job `1402`, expecting `research/night_gen_phase1/results/S6_radarquery_topk90/eval/submission_overall/pts_bbox/results_nusc.json`.
  - It produces sample-level radar-rich/long-range metrics and object-distance metrics from existing prediction JSONs; it does not run inference.
- Staged-only seed replication helpers, not submitted:
  - Configs:
    - `configs/racformer_train2k_day_radarquery_seed20260502_research.py`
    - `configs/racformer_train2k_day_radarquery_topk90_seed20260502_research.py`
  - Seeded entrypoint:
    - `research/night_gen_phase1/staged_radarquery_replication/train_seeded.py`
  - Train/eval/summary wrappers:
    - `research/night_gen_phase1/staged_radarquery_replication/run_s6_radarquery_seed20260502_livenode02.sbatch`
    - `research/night_gen_phase1/staged_radarquery_replication/run_s6_radarquery_seed20260502_eval_livenode02.sbatch`
    - `research/night_gen_phase1/staged_radarquery_replication/run_s6_radarquery_seed20260502_summary_livenode02.sbatch`
    - `research/night_gen_phase1/staged_radarquery_replication/run_s6_radarquery_topk90_seed20260502_livenode03.sbatch`
    - `research/night_gen_phase1/staged_radarquery_replication/run_s6_radarquery_topk90_seed20260502_eval_livenode03.sbatch`
    - `research/night_gen_phase1/staged_radarquery_replication/run_s6_radarquery_topk90_seed20260502_summary_livenode03.sbatch`
  - Local and remote validation passed:
    - `python -m py_compile` on both seed configs and `train_seeded.py`.
    - `bash -n` on all staged replication wrappers.
  - Use only if seed-0 `S6_radarquery` or `S6_radarquery_topk90` passes or lands near the gate.

## Kill Conditions

Calibration-noise training is complete and rejected. Its predeclared nominal method
summary rejected the branch if any condition was true:

- Day mAP is worse than S0 `-1.0 pp`.
- Overall mAP is worse than S0 `-1.0 pp`.
- Night NDS is worse than S0 `-0.5 pp`.

Its fixed-yaw robustness summary required a method gain of at least `+1.0 pp`
in overall, night, or rain mAP versus `S0_calib_yaw2`, and no overall NDS drop worse
than `-0.5 pp` versus `S0_calib_yaw2`.

For any future S0-targeted branch using the general promotion gate, reject at summary
if any condition is true:

- Night mAP is less than S0 `+1.0 pp`.
- Day mAP is worse than S0 `-1.0 pp`.
- Overall mAP is worse than S0 `-1.5 pp`.
- Night NDS is worse than S0 `-0.5 pp`.
- A single seed passes: replicate before any claim.

## Completed Branch Decisions

| Branch | Seed | Day mAP/NDS | Night mAP/NDS | Rain mAP/NDS | Overall mAP/NDS | Gate | Decision |
|---|---:|---:|---:|---:|---:|---|---|
| `S0_calib_yaw2` | default | `0.2623 / 0.3285` | `0.1201 / 0.1900` | `0.2187 / 0.3253` | `0.2526 / 0.3247` | N/A | diagnostic baseline: fixed `+2 deg` yaw damages S0 by `-5.14 pp` overall mAP; use as robustness comparator |
| `S0_calibnoise_nominal` | default | `0.2901 / 0.3567` | `0.1339 / 0.2027` | `0.2498 / 0.3527` | `0.2803 / 0.3523` | FAIL | reject: train-time calibration noise collapses nominal day/overall and night NDS |
| `S0_calibnoise_yaw2` | default | `0.2502 / 0.3229` | `0.1197 / 0.1844` | `0.2070 / 0.3169` | `0.2412 / 0.3191` | FAIL | reject: worse than fixed-yaw S0 robustness baseline and no robustness gain |
| `S3_seed20260425_ratio18p75` | `20260425` | `0.3083 / 0.3718` | `0.1681 / 0.2314` | `0.2643 / 0.3620` | `0.2990 / 0.3681` | PASS | replicate only; not paper-worthy |
| `S3_seed20260502_ratio18p75` | `20260502` | `0.3050 / 0.3662` | `0.1494 / 0.2130` | `0.2644 / 0.3634` | `0.2959 / 0.3633` | FAIL | reject |
| `S3_seed20260425_ratio18p75_w05` | `20260425` | `0.3083 / 0.3680` | `0.1515 / 0.2092` | `0.2681 / 0.3645` | `0.2993 / 0.3637` | FAIL | reject |
| `S3_seed20260425_ratio18p75_w025` | `20260425` | `0.2930 / 0.3620` | `0.1348 / 0.2001` | `0.2560 / 0.3604` | `0.2840 / 0.3582` | FAIL | reject |
| `S3_seed20260425_ratio18p75_w05_adaptfusion` | `20260425` | `0.3077 / 0.3715` | `0.1598 / 0.2211` | `0.2717 / 0.3708` | `0.2990 / 0.3679` | PASS | replicate only |
| `S3_seed20260502_ratio18p75_w05_adaptfusion` | `20260502` | `0.3070 / 0.3660` | `0.1341 / 0.2037` | `0.2807 / 0.3728` | `0.2974 / 0.3621` | FAIL | reject |
| `S3_seed20260425` | `20260425` | `0.2969 / 0.3605` | `0.1640 / 0.2225` | `0.2628 / 0.3590` | `0.2889 / 0.3569` | FAIL | reject |
| `S3_seed20260425_ratio12p5` | `20260425` | `0.3100 / 0.3696` | `0.1483 / 0.2058` | `0.2622 / 0.3534` | `0.2997 / 0.3642` | FAIL | reject |
| `S3_seed20260425_ratio21p25` | `20260425` | `0.3091 / 0.3706` | `0.1590 / 0.2274` | `0.2678 / 0.3669` | `0.3009 / 0.3680` | PASS | archive: single-seed generated-night; not paper-worthy |
| `S1` | default | `0.2824 / 0.3527` | `0.1586 / 0.2242` | `0.2385 / 0.3471` | `0.2742 / 0.3494` | FAIL | reject |
| `S0_rcsbev` | default | `0.3152 / 0.3753` | `0.1408 / 0.1996` | `0.2755 / 0.3726` | `0.3043 / 0.3697` | FAIL | reject |
| `S0_rcsoccbev` | default | `0.3126 / 0.3761` | `0.1381 / 0.2140` | `0.2725 / 0.3760` | `0.3030 / 0.3719` | FAIL | reject |
| `S0_rcsvelbev` | default | `0.3159 / 0.3749` | `0.1506 / 0.2150` | `0.2806 / 0.3682` | `0.3062 / 0.3697` | FAIL | archive: stable but insufficient |
| `S0_occvelbev` | default | `0.2990 / 0.3645` | `0.1348 / 0.1977` | `0.2665 / 0.3647` | `0.2896 / 0.3603` | FAIL | reject |
| `S0_occveltimebev` | seed0 | `0.3093 / 0.3733` | `0.1637 / 0.2228` | `0.2780 / 0.3695` | `0.3011 / 0.3693` | PASS | replicate only |
| `S0_occveltimebev_seed1` | seed1 | `0.3041 / 0.3638` | `0.1594 / 0.2217` | `0.2608 / 0.3546` | `0.2937 / 0.3581` | FAIL | archive |
| `S0_occveltimebev_seed2` | seed2 | `0.2992 / 0.3602` | `0.1543 / 0.2058` | `0.2546 / 0.3542` | `0.2888 / 0.3553` | FAIL | reject |
| `S0_occtimebev` | default | `0.3175 / 0.3754` | `0.1436 / 0.2022` | `0.2764 / 0.3694` | `0.3049 / 0.3691` | FAIL | reject |
| `S0_rcsveltimebev` | default | `0.3106 / 0.3746` | `0.1477 / 0.2107` | `0.2730 / 0.3719` | `0.3009 / 0.3703` | FAIL | reject |
| `S0_occveltimebev_v10` | default | `0.3153 / 0.3753` | `0.1325 / 0.1939` | `0.2768 / 0.3730` | `0.3037 / 0.3700` | FAIL | reject |
| `S0_occveltimebev_v40` | default | `0.3121 / 0.3711` | `0.1410 / 0.2006` | `0.2676 / 0.3681` | `0.3026 / 0.3669` | FAIL | reject |
| `S5` | default | `0.2600 / 0.3320` | `0.1741 / 0.2085` | `0.2189 / 0.3240` | `0.2536 / 0.3278` | FAIL | reject: night gain collapses day/overall |
| `S5_conditionfusion` | default | `0.2583 / 0.3374` | `0.1741 / 0.2204` | `0.2256 / 0.3262` | `0.2530 / 0.3332` | FAIL | reject: night gain collapses day/overall |
| `S5_contrelqfusion` | default | `0.2598 / 0.3305` | `0.1673 / 0.2094` | `0.2308 / 0.3251` | `0.2539 / 0.3269` | FAIL | reject: continuous reliability does not recover S5 day/overall collapse; night NDS also misses S0 tolerance |
| `S6_radarquery` | default | `0.2554 / 0.3312` | `0.0958 / 0.1675` | `0.2343 / 0.3416` | `0.2477 / 0.3285` | FAIL | reject: radar-guided initialization of 180 queries sharply regresses every split; do not run seed replication |
| `S6_radarquery_topk90` | default | `0.2977 / 0.3586` | `0.1230 / 0.1901` | `0.2554 / 0.3550` | `0.2863 / 0.3549` | FAIL | reject: smaller radar-query allocation reduces damage but still misses night and preservation gates; do not run seed replication |

## Running Branches Under Evaluation

| Branch | Jobs | Implementation | Smoke Evidence | Decision Pending |
|---|---|---|---|---|
| None | N/A | N/A | N/A | Branch C feasibility/blocker audit before any new GPU submission |

Staged-only ablation configs, not submitted:

- `configs/racformer_train2k_mixed_contrelqfusion_stats_only_research.py`
- `configs/racformer_train2k_mixed_contrelqfusion_no_cosine_research.py`
- `configs/racformer_train2k_mixed_contrelqfusion_no_geometry_research.py`
- `configs/racformer_train2k_day_radarquery_seed20260502_research.py`
- `configs/racformer_train2k_day_radarquery_topk90_seed20260502_research.py`

## Hard Rejections

- Do not repeat generated-night seed variants unless there is a new mechanism.
- Do not scale DriveGEN; visual QC failed and geometry/labels are likely invalid.
- Do not claim any single-seed pass as paper-worthy.
- Do not do more broad paper searches unless all currently running branches have failed and the failure mode is summarized.
- Do not run more vx/vy scale-only ablations.

## Submission Guard

Do not submit further GPU jobs unless all of the following are true:

- Both `livenode02` and `livenode03` current jobs are finished or failed.
- Their `summary_metrics.md/json` files have been read.
- This tracker has a PASS/FAIL verdict for each completed branch.
- The next job is justified by the previous failure mode and is not a repeat of a hard-rejected path.

Current status: Branches A-D have been attempted or audited to screening depth and are
negative. Branches E-H are blocked/low priority without new artifacts, implementation
substrate, or QC evidence. No new GPU branch is justified under the current evidence.
The final report records a BLOCKED/no-positive-claim outcome.

## Meaningful Checkpoints Only

Replace progress polling with these tracker events only:

- job started
- epoch checkpoint saved
- job failed
- eval started
- summary written
- gate decision made

## Single Next Action

No new GPU job should be submitted now. The remaining action is to preserve the final
blocked report and wait for a new unlock: a LiDAR teacher checkpoint/features, a real
expert-router/failure benchmark, a baseline-preserving temporal router design, or a
geometry-preserving visual/foundation branch with QC evidence.

## Current Branch Checkpoint Log

### Branch C motion subset audit and final blocked status - 2026-05-16 02:53 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - CPU subset audit jobs `1406`-`1411` completed.
  - `squeue -u gnmp` shows only stale dependency jobs `1320`/`1321`.
- Code evidence:
  - `models/racformer_transformer.py` already computes `time_diff` and warps image/BEV
    sampling points by query velocity times `time_diff`.
- Subset audit:
  - Outputs: `research/paper_goal_20260515/branch_c_motion_subset_audit/*/subset_metrics.md`.
  - `S0_occveltimebev` seed0 moving-sample `0.3016 / 0.3690`.
  - `S0_occveltimebev_seed1` moving-sample `0.2942 / 0.3579`.
  - `S0_occveltimebev_seed2` moving-sample `0.2894 / 0.3554`.
  - `S0_rcsvelbev` moving-sample `0.3067 / 0.3696`.
- Decision:
  - Branch C is rejected/blocked for this loop.
  - Final outcome is BLOCKED/no positive paper-worthy claim.

### S6 topk90 full-val gate failure and subset completion - 2026-05-16 02:28 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Topk90 eval `1402`, summary `1403`, and subset job `1405` completed.
  - `squeue -u gnmp` shows only stale dependency jobs `1320`/`1321`.
- Metrics versus S0:
  - day `0.2977 / 0.3586`, mAP/NDS delta `-1.76 / -1.60 pp`.
  - night `0.1230 / 0.1901`, mAP/NDS delta `-2.58 / -2.50 pp`.
  - rain `0.2554 / 0.3550`, mAP/NDS delta `-1.89 / -1.63 pp`.
  - overall `0.2863 / 0.3549`, mAP/NDS delta `-1.77 / -1.49 pp`.
- Topk90 subset diagnostic:
  - `research/night_gen_phase1/results/S6_radarquery_topk90/subset_eval/subset_metrics.md`
  - all samples `0.2863 / 0.3549`;
  - radar-supported samples `0.2867 / 0.3551`;
  - radar-rich top quartile samples `0.3119 / 0.3797`;
  - object far >=30m `0.0788 / 0.2138`;
  - object far >=40m `0.0365 / 0.1207`.
- Decision:
  - Gate verdict: FAIL.
  - Do not run seed replication for topk90.
  - Branch D is rejected in its current radar-query initialization form.

### S6 topk180 subset completion and topk90 eval handoff - 2026-05-16 01:48 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Topk180 subset job `1404` completed.
  - Topk90 train job `1401` completed with `epoch_12.pth`.
  - Topk90 eval job `1402` is RUNNING on `livenode03`.
- Topk90 evidence:
  - Final checkpoint:
    `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_12.pth`
  - Checkpoint size: `764836945` bytes.
  - `research/night_gen_phase1/results/S6_radarquery_topk90/slurm_1401.err`: `0` bytes.
  - Eval logs:
    - `research/night_gen_phase1/results/S6_radarquery_topk90/eval_slurm_1402.out`
    - `research/night_gen_phase1/results/S6_radarquery_topk90/eval_slurm_1402.err`
- Topk180 subset diagnostic:
  - `research/night_gen_phase1/results/S6_radarquery/subset_eval/subset_metrics.md`
  - all samples `0.2477 / 0.3285`;
  - radar-supported samples `0.2481 / 0.3288`;
  - radar-rich top quartile samples `0.2745 / 0.3614`;
  - object far >=30m `0.0615 / 0.2052`;
  - object far >=40m `0.0275 / 0.1128`.

### S6_radarquery full-val gate failure - 2026-05-16 01:35 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Eval job `1398` completed and summary job `1399` wrote gate artifacts.
  - Subset diagnostics job `1404` started on `livenode02`.
  - Topk90 train job `1401` remains RUNNING on `livenode03`.
- Metrics versus S0:
  - day `0.2554 / 0.3312`, mAP/NDS delta `-5.99 / -4.34 pp`.
  - night `0.0958 / 0.1675`, mAP/NDS delta `-5.30 / -4.76 pp`.
  - rain `0.2343 / 0.3416`, mAP/NDS delta `-4.00 / -2.98 pp`.
  - overall `0.2477 / 0.3285`, mAP/NDS delta `-5.63 / -4.13 pp`.
- Artifacts:
  - `research/night_gen_phase1/results/S6_radarquery/eval/eval_by_condition.json`
  - `research/night_gen_phase1/results/S6_radarquery/summary_metrics.md`
  - `research/night_gen_phase1/results/S6_radarquery/summary_metrics.json`
- Decision:
  - Gate verdict: FAIL.
  - Do not run seed replication for topk180.
  - Continue only the already-running topk90 ablation before deciding the next branch.

### S6_radarquery eval progress and topk90 epoch-11 checkpoint - 2026-05-16 01:22 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Eval job `1398` remains RUNNING on `livenode02`.
  - Topk90 train job `1401` remains RUNNING on `livenode03`.
- Main eval evidence:
  - `research/night_gen_phase1/results/S6_radarquery/eval_slurm_1398.out` reached about `4863/6019` inference samples.
  - `research/night_gen_phase1/results/S6_radarquery/eval_slurm_1398.err` remains at the initial info lines.
  - No `eval_by_condition.json`, `results_nusc.json`, summary, or subset output was present at this check.
- Topk90 epoch-11 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_11.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [11/12][1000/1000]` and `Saving checkpoint at 11 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery_topk90/slurm_1401.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The main top-k 180 eval is near inference completion but has not yet produced metric artifacts.
  - The top-k 90 ablation remains healthy through eleven epochs and is in final epoch 12.

### S6_radarquery final train and eval handoff - 2026-05-16 01:05 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1397` completed and left the scheduler queue.
  - Eval job `1398` is RUNNING on `livenode02`.
  - Summary `1399` and subset diagnostics `1404` remain dependency-pending.
- Final train evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_12.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [12/12][1000/1000]` and `Saving checkpoint at 12 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery/slurm_1397.err`: `1746` bytes, inspected as progress-bar output only.
- Eval handoff evidence:
  - `research/night_gen_phase1/results/S6_radarquery/eval_slurm_1398.err` records config, final weights, model build, and inference on `6019` samples.
  - `research/night_gen_phase1/results/S6_radarquery/eval_slurm_1398.out` shows the evaluator loaded `epoch_12.pth` and is producing inference progress.
- Interpretation:
  - The main top-k 180 radar-query train completed successfully and is now in full-val evaluation.
  - The branch decision remains pending until eval, summary, and subset diagnostics complete.

### S6_radarquery_topk90 epoch-10 checkpoint - 2026-05-16 00:52 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1401` remains RUNNING on `livenode03`.
  - Eval `1402`, summary `1403`, and subset diagnostics `1405` remain dependency-pending.
- Epoch-10 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_10.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [10/12][1000/1000]` and `Saving checkpoint at 10 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery_topk90/slurm_1401.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The top-k 90 ablation remains healthy through ten epochs.
  - This is still not a metric decision; final train/eval/summary must complete before comparing to `S6_radarquery` and the S0 gate.

### S6_radarquery epoch-11 checkpoint - 2026-05-16 00:34 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1397` remains RUNNING on `livenode02`.
  - Eval `1398`, summary `1399`, and subset diagnostics `1404` remain dependency-pending.
- Epoch-11 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_11.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [11/12][1000/1000]` and `Saving checkpoint at 11 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery/slurm_1397.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The top-k 180 radar-query branch remains healthy through eleven epochs.
  - This is still not a metric decision; final train/eval/summary must complete before comparing to S0.

### S6_radarquery_topk90 epoch-9 checkpoint - 2026-05-16 00:26 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1401` remains RUNNING on `livenode03`.
  - Eval `1402`, summary `1403`, and subset diagnostics `1405` remain dependency-pending.
- Epoch-9 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_9.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [9/12][1000/1000]` and `Saving checkpoint at 9 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery_topk90/slurm_1401.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The top-k 90 ablation remains healthy through nine epochs.
  - This is still not a metric decision; final train/eval/summary must complete before comparing to `S6_radarquery` and the S0 gate.

### S6_radarquery epoch-10 checkpoint - 2026-05-16 00:08 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1397` remains RUNNING on `livenode02`.
  - Eval `1398`, summary `1399`, and subset diagnostics `1404` remain dependency-pending.
- Epoch-10 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_10.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [10/12][1000/1000]` and `Saving checkpoint at 10 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery/slurm_1397.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The top-k 180 radar-query branch remains healthy through ten epochs.
  - This is still not a metric decision; final train/eval/summary must complete before comparing to S0.

### S6_radarquery_topk90 epoch-8 checkpoint - 2026-05-16 00:00 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1401` remains RUNNING on `livenode03`.
  - Eval `1402`, summary `1403`, and subset diagnostics `1405` remain dependency-pending.
- Epoch-8 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_8.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [8/12][1000/1000]` and `Saving checkpoint at 8 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery_topk90/slurm_1401.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The top-k 90 ablation remains healthy through eight epochs.
  - This is still not a metric decision; final train/eval/summary must complete before comparing to `S6_radarquery` and the S0 gate.

### S6_radarquery epoch-9 checkpoint - 2026-05-15 23:42 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1397` remains RUNNING on `livenode02`.
  - Eval `1398`, summary `1399`, and subset diagnostics `1404` remain dependency-pending.
- Epoch-9 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_9.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [9/12][1000/1000]` and `Saving checkpoint at 9 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery/slurm_1397.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The top-k 180 radar-query branch remains healthy through nine epochs.
  - This is still not a metric decision; final train/eval/summary must complete before comparing to S0.

### S6_radarquery_topk90 epoch-7 checkpoint - 2026-05-15 23:34 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1401` remains RUNNING on `livenode03`.
  - Eval `1402`, summary `1403`, and subset diagnostics `1405` remain dependency-pending.
- Epoch-7 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_7.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [7/12][1000/1000]` and `Saving checkpoint at 7 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery_topk90/slurm_1401.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The top-k 90 ablation remains healthy through seven epochs.
  - This is still not a metric decision; final train/eval/summary must complete before comparing to `S6_radarquery` and the S0 gate.

### S6_radarquery epoch-8 checkpoint - 2026-05-15 23:16 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1397` remains RUNNING on `livenode02`.
  - Eval `1398`, summary `1399`, and subset diagnostics `1404` remain dependency-pending.
- Epoch-8 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_8.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [8/12][1000/1000]` and `Saving checkpoint at 8 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery/slurm_1397.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The top-k 180 radar-query branch remains healthy through eight epochs.
  - This is still not a metric decision; final train/eval/summary must complete before comparing to S0.

### S6_radarquery_topk90 epoch-6 checkpoint - 2026-05-15 23:08 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1401` remains RUNNING on `livenode03`.
  - Eval `1402`, summary `1403`, and subset diagnostics `1405` remain dependency-pending.
- Epoch-6 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_6.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [6/12][1000/1000]` and `Saving checkpoint at 6 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery_topk90/slurm_1401.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The top-k 90 ablation remains healthy through six epochs.
  - This is not a metric decision; final train/eval/summary must complete before comparing to `S6_radarquery` and the S0 gate.

### S6_radarquery epoch-7 checkpoint - 2026-05-15 22:50 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1397` remains RUNNING on `livenode02`.
  - Eval `1398`, summary `1399`, and subset diagnostics `1404` remain dependency-pending.
- Epoch-7 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_7.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [7/12][1000/1000]` and `Saving checkpoint at 7 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery/slurm_1397.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The top-k 180 radar-query branch remains healthy through seven epochs.
  - This is not a metric decision; final train/eval/summary must complete before comparing to S0.

### S6_radarquery_topk90 epoch-5 checkpoint - 2026-05-15 22:42 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1401` remains RUNNING on `livenode03`.
  - Eval `1402`, summary `1403`, and subset diagnostics `1405` remain dependency-pending.
- Epoch-5 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_5.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [5/12][1000/1000]` and `Saving checkpoint at 5 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery_topk90/slurm_1401.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The top-k ablation remains healthy through five epochs.
  - This is not a metric decision; final train/eval/summary must complete before comparing to `S6_radarquery` and the S0 gate.

### S6_radarquery epoch-6 checkpoint - 2026-05-15 22:25 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1397` remains RUNNING on `livenode02`.
  - Eval `1398`, summary `1399`, and subset diagnostics `1404` remain dependency-pending.
- Epoch-6 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_6.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [6/12][1000/1000]` and `Saving checkpoint at 6 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery/slurm_1397.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The main radar-query branch remains healthy through half of the screening train.
  - This is not a metric decision; final train/eval/summary must complete before comparing to the S0 gate.

### S6_radarquery_topk90 epoch-4 checkpoint - 2026-05-15 22:17 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1401` remains RUNNING on `livenode03`.
  - Eval `1402`, summary `1403`, and subset diagnostics `1405` remain dependency-pending.
- Epoch-4 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_4.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [4/12][1000/1000]` and `Saving checkpoint at 4 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery_topk90/slurm_1401.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The top-k ablation remains healthy through one third of the screening train.
  - This is not a metric decision; final train/eval/summary must complete before comparing to `S6_radarquery` and the S0 gate.

### S6_radarquery epoch-5 checkpoint - 2026-05-15 21:58 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1397` remains RUNNING on `livenode02`.
  - Eval `1398`, summary `1399`, and subset diagnostics `1404` remain dependency-pending.
- Epoch-5 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_5.pth`
    - size: `764836945` bytes after the write completed.
  - Train log reached `Epoch [5/12][1000/1000]` and `Saving checkpoint at 5 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery/slurm_1397.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The main radar-query branch remains healthy through five epochs.
  - This is not a metric decision; final train/eval/summary must complete before comparing to the S0 gate.

### S6_radarquery_topk90 epoch-3 checkpoint - 2026-05-15 21:50 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1401` remains RUNNING on `livenode03`.
  - Eval `1402`, summary `1403`, and subset diagnostics `1405` remain dependency-pending.
- Epoch-3 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_3.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [3/12][1000/1000]` and `Saving checkpoint at 3 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery_topk90/slurm_1401.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The top-k ablation remains healthy through 25% of the screening train.
  - This is not a metric decision; final train/eval/summary must complete before comparing to `S6_radarquery` and the S0 gate.

### S6_radarquery epoch-4 checkpoint - 2026-05-15 21:33 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1397` remains RUNNING on `livenode02`.
  - Eval `1398`, summary `1399`, and subset diagnostics `1404` remain dependency-pending.
- Epoch-4 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_4.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [4/12][1000/1000]` and `Saving checkpoint at 4 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery/slurm_1397.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The main radar-query branch remains healthy through one third of the screening train.
  - This is not a metric decision; final train/eval/summary must complete before comparing to the S0 gate.

### S6_radarquery_topk90 epoch-2 checkpoint - 2026-05-15 21:24 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1401` remains RUNNING on `livenode03`, elapsed about `52:32`.
  - Eval `1402`, summary `1403`, and subset diagnostics `1405` remain dependency-pending.
- Epoch-2 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_2.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [2/12][1000/1000]` and `Saving checkpoint at 2 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery_topk90/slurm_1401.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The top-k ablation remains healthy through two epochs.
  - This is not a metric decision; final train/eval/summary must complete before comparing to `S6_radarquery` and the S0 gate.

### S6_radarquery epoch-3 checkpoint - 2026-05-15 21:06 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1397` remains RUNNING on `livenode02`, elapsed about `1:18`.
  - Eval `1398`, summary `1399`, and subset diagnostics `1404` remain dependency-pending.
- Epoch-3 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_3.pth`
    - size: `764836945` bytes.
  - Train log reached `Epoch [3/12][1000/1000]` and `Saving checkpoint at 3 epochs`.
  - `research/night_gen_phase1/results/S6_radarquery/slurm_1397.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The main radar-query branch remains healthy through 25% of the screening train.
  - This is not a metric decision; final train/eval/summary must complete before comparing to the S0 gate.

### S6_radarquery epoch-1 checkpoint - 2026-05-15 20:14 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1397` remains RUNNING on `livenode02`, elapsed `26:23`.
  - Eval `1398` and summary `1399` remain dependency-pending.
- Epoch-1 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_1.pth`
    - size: `764836753` bytes.
  - `research/night_gen_phase1/results/S6_radarquery/slurm_1397.err`: `0` bytes at the checkpoint check.
- Interpretation:
  - The radar-query branch has passed the first epoch/checkpoint health marker.
  - This is not a metric decision; final train/eval/summary must complete before comparing to the S0 gate.

### S5_contrelqfusion epoch-1 checkpoint - 2026-05-15 13:59 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1393` remains RUNNING on `livenode02`, elapsed `29:09` of `8:00:00`.
  - Eval `1394` and summary `1395` remain dependency-pending.
- Epoch-1 evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_mixed_contrelqfusion_research/2026-05-15/10-30-04/epoch_1.pth`
    - size: `764865537` bytes.
  - Latest log reached `Epoch [2/12][100/1000]`.
  - `research/night_gen_phase1/results/S5_contrelqfusion/slurm_1393.err`: `0` bytes.
- Interpretation:
  - The continuous reliability query fusion branch has passed the first epoch/checkpoint health marker.
  - This is not a metric decision; final train/eval/summary must complete before comparing to S0 and S5 gates.

### S5_contrelqfusion final checkpoint and eval start - 2026-05-15 19:09 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1393` is no longer in `squeue`.
  - Eval job `1394` is RUNNING on `livenode02`, started `2026-05-15 18:40:50 UTC`.
  - Summary job `1395` remains dependency-pending.
- Final checkpoint evidence:
  - Checkpoint exists:
    - `outputs/racformer_train2k_mixed_contrelqfusion_research/2026-05-15/10-30-04/epoch_12.pth`
    - size: `764865729` bytes.
  - Train log reached `Epoch [12/12][1000/1000]` and `Saving checkpoint at 12 epochs`.
  - `research/night_gen_phase1/results/S5_contrelqfusion/slurm_1393.err`: `1622` bytes, inspected as tqdm/progress-bar output only.
- Eval evidence:
  - Eval wrapper is using the expected final checkpoint:
    - `WEIGHTS=outputs/racformer_train2k_mixed_contrelqfusion_research/2026-05-15/10-30-04/epoch_12.pth`
  - Eval stderr reports:
    - `running inference on 6019 samples`
    - `loading NuScenes for per-split eval`
  - No `summary_metrics.md/json` or `eval/eval_by_condition.json` exists yet.
- Interpretation:
  - Training completed cleanly enough to start full-val eval.
  - This is still not a metric decision; wait for summary job `1395` and then compare against S0 and S5 gates.

### S5_contrelqfusion final summary and decision - 2026-05-15 19:26 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Jobs `1393`, `1394`, and `1395` are no longer in `squeue`.
  - Summary files exist:
    - `research/night_gen_phase1/results/S5_contrelqfusion/summary_metrics.md`
    - `research/night_gen_phase1/results/S5_contrelqfusion/summary_metrics.json`
    - `research/night_gen_phase1/results/S5_contrelqfusion/eval/eval_by_condition.json`
- Metrics:
  - day: `0.2598 / 0.3305`, delta vs S0 `-5.55 pp / -4.41 pp`.
  - night: `0.1673 / 0.2094`, delta vs S0 `+1.85 pp / -0.57 pp`.
  - rain: `0.2308 / 0.3251`, delta vs S0 `-4.35 pp / -4.62 pp`.
  - overall: `0.2539 / 0.3269`, delta vs S0 `-5.01 pp / -4.29 pp`.
  - Gate verdict: FAIL.
- Gate check:
  - night mAP >= `+1.0 pp`: PASS (`+1.85 pp`).
  - day mAP >= `-1.0 pp`: FAIL (`-5.55 pp`).
  - overall mAP >= `-1.5 pp`: FAIL (`-5.01 pp`).
  - night NDS >= `-0.5 pp`: FAIL (`-0.57 pp`).
- S5 comparison:
  - day mAP/NDS: `-0.02 pp / -0.15 pp`.
  - night mAP/NDS: `-0.68 pp / +0.10 pp`.
  - rain mAP/NDS: `+1.18 pp / +0.11 pp`.
  - overall mAP/NDS: `+0.04 pp / -0.09 pp`.
- Interpretation:
  - Continuous reliability query fusion did not fix the mixed-condition S5 failure mode.
  - It is essentially neutral vs S5, preserves S5's severe day/overall collapse, and misses the S0 publication gate.
  - Do not run the staged cue ablations for this failed parent branch unless a new diagnostic question specifically needs them.
- Decision:
  - Reject Branch A as currently implemented.
  - Move to a branch-choice audit for Branch C versus Branch D before spending more GPU budget.

## Chronological Archive

Everything below this point is audit history. It is not a live queue and should not be acted on unless the Stop-Go Dashboard above is updated first.

## Analysis Helpers

### Loss weight sweep comparator

- Script: `research/night_gen_phase1/compare_loss_weight_sweep.py`
- SLURM wrapper: `run_loss_weight_sweep_compare.sbatch`
- Pending compare job: `1264`, dependency `afterok:1247:1263`
- Output markdown: `research/night_gen_phase1/results/loss_weight_sweep_summary.md`
- Output JSON: `research/night_gen_phase1/results/loss_weight_sweep_summary.json`

This script compares S0, S3 ratio18p75 weight 1.0, w05, and w025 against the shared gate. It is safe to run before w05/w025 finish; unfinished stages are marked `PENDING`.

Current caveat: the original S3 seed20260425 ratio18p75 run passes the narrow gate in this comparator, but seed20260502 did not reproduce the night gain. Do not present the weight-1.0 result as paper-worthy without replication.

## Completed Or Rejected

### S3 fixed-partition generated night, seed20260425, ratio18p75

Result:

- Day mAP/NDS: `0.3082946194 / 0.3717686587`
- Night mAP/NDS: `0.1680892081 / 0.2314156374`
- Rain mAP/NDS: `0.2642961759 / 0.3620450679`
- Overall mAP/NDS: `0.2990321017 / 0.3681380218`

Interpretation: promising night gain, but day/overall regressions and replication risk. Do not treat this as publishable by itself.

### S3 fixed-partition generated night, seed20260502, ratio18p75

Result:

- Day mAP/NDS: `0.3050154571 / 0.3662316265`
- Night mAP/NDS: `0.1493829746 / 0.2130162502`
- Rain mAP/NDS: `0.2644216906 / 0.3634460261`
- Overall mAP/NDS: `0.2958551337 / 0.3633082857`

Interpretation: did not reproduce the seed20260425 night gain. Do not rerun this exact setting.

### Flawed first w05 loss weighting run

- Jobs: `1243` / `1244`
- Output dir to ignore: `outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_research/2026-05-11/23-11-01`
- Reason: reviewer caught a normalization bug. Generated samples were downweighted, but the denominator was also effectively weighted, weakening the intended change.

Do not use metrics from this run.

### DriveGEN

Status: technical fallback exists, but current visual quality is weak.

What worked:

- Clone: `/srv/nfs/shared/gnmp/DriveGEN`, shallow commit `ca330ca`
- Environment: `/srv/nfs/shared/gnmp/.conda/envs/driveGEN`
- Public model path: `sd2-community/stable-diffusion-2-1-base`
- Bbox exporter: `research/night_gen_phase1/build_drivegen_bboxes.py`
- One-image 800x448 smoke succeeded:
  - Stage 1 job: `1259`
  - Stage 2 job: `1260`
  - Output: `/srv/nfs/shared/gnmp/DriveGEN/experiments/night_pilot/temp_data_2.1_base_seed20260425_r18p75_first2_800x448/nus_res/night/CAM_BACK_n008-2018-08-30-15-52-26-0400__CAM_BACK__1535659414187558.jpg`

What failed or is weak:

- Official `stabilityai/stable-diffusion-2-1-base` returned unauthenticated 404 on the cluster; use `sd2-community/stable-diffusion-2-1-base`.
- 1600x896 stage 2 OOMed on a 24 GB RTX 4090 even with xformers/offload.
- 800x448 output is visually strange enough that it should not be scaled into a training augmentation run unless a manual 12-image QC improves substantially.
- 2026-05-12 manual visual judgement: reject the current DriveGEN output for augmentation. It is night-like, but it hallucinates/reshapes vehicles, adds heavy blur/glare, and likely breaks source-label geometry.

Decision: DriveGEN is not the main branch right now. Keep it as a possible negative-control or backup generator, not as publishable evidence until visual geometry preservation is verified. Do not scale the current one-image output into training.

### CycleGAN

Not run. Deprioritized because unpaired style transfer is less likely than DriveGEN to preserve 3D object geometry and camera-consistent labels. Only revisit as a weak negative-control baseline if the paper needs a simple style-transfer comparison.

## Staged But Not Applied

### Adaptive fusion gate

Paper inspiration: RobuRCDet/SAMFusion-style adaptive radar-camera weighting.

Files staged on remote:

- `research/night_gen_phase1/staged_adaptfusion/adaptive_fusion_gate.patch`
- `research/night_gen_phase1/staged_adaptfusion/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research.py`
- `research/night_gen_phase1/staged_adaptfusion/run_t11_adaptfusion_seed20260425_ratio18p75_w05.sbatch`
- `research/night_gen_phase1/staged_adaptfusion/ADAPTIVE_FUSION_GATE_PLAN.md`

Implementation intent:

- Hook point: `models/racformer_transformer.py` decoder fusion where image, radar, and LSS query features are all `[B, Q, 256]`.
- Adds an optional `adaptive_fusion_gate=False` path.
- When enabled: `Linear(3C,C) -> ReLU -> Linear(C,3)`, final layer zero-initialized.
- Gate uses `sigmoid(logits) * 2`, so initialization is neutral multiplier 1.0 for all three streams.
- `git apply --check research/night_gen_phase1/staged_adaptfusion/adaptive_fusion_gate.patch` already passed.

Decision rule:

- If corrected w05 passes the promotion gate, prefer replication/ablation before applying this.
- If corrected w05 fails or is marginal, apply this staged patch and run T11 as the next architecture branch.

## Paper Inspirations Already Checked

Do not repeat broad searches for these unless looking for implementation details.

- DriveGEN, CVPR 2025: training-free controllable diffusion generation for OOD 3D detection.
  - arXiv: `https://arxiv.org/abs/2503.11122`
- RobuRCDet, ICLR 2025: radar 3D Gaussian expansion plus weather-adaptive fusion.
  - arXiv: `https://arxiv.org/abs/2502.13071`
- SAMFusion, 2025: sensor-adaptive multimodal fusion with distance/visibility weighting.
  - arXiv: `https://arxiv.org/abs/2508.16408`
- RCDINO, 2025: DINOv2 semantic features for radar-camera 3D detection.
  - arXiv: `https://arxiv.org/abs/2508.15353`
- Camera-radar distillation/adverse-weather distillation papers: useful for framing, but likely heavier than the current time budget unless w05 and adaptive fusion both fail.

## Next Actions

1. Wait for jobs `1245 -> 1246 -> 1247`.
2. Wait for jobs `1261 -> 1262 -> 1263`.
3. Read `research/night_gen_phase1/results/S3_seed20260425_ratio18p75_w05/summary_metrics.md`.
4. Read `research/night_gen_phase1/results/S3_seed20260425_ratio18p75_w025/summary_metrics.md`.
5. Run `conda run -n racformerfix --no-capture-output python -u research/night_gen_phase1/compare_loss_weight_sweep.py`.
6. If either loss-weighted run passes: document result, run one focused replication or ablation before claiming paper value.
7. If both fail or are marginal: apply staged adaptive fusion gate and submit T11.
8. Keep DriveGEN paused unless a QC-specific 12-image pilot is explicitly chosen; do not scale the current output quality into training.

## Progress Poll - 2026-05-12 03:36 UTC

- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; last observed training log reached epoch `3/12`, iter `800/1000`; stderr size remains `0` bytes. Eval `1246` and summary `1247` still dependency-pending.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; last observed training log reached epoch `1/12`, iter `400/1000`; stderr size remains `0` bytes. Eval `1262` and summary `1263` still dependency-pending.
- `loss_weight_sweep_summary.md` currently contains pending placeholders for `w05` and `w025`; compare job `1264` is still dependency-pending, so do not treat that table as final.
- No `summary_metrics.md/json` exists yet for `S3_seed20260425_ratio18p75_w05` or `S3_seed20260425_ratio18p75_w025`. Next action remains waiting for train/eval summaries, then applying the documented promotion gate.

## Targeted Paper Search - 2026-05-12 03:38 UTC

Reason: active GPU jobs are still running, so I searched for non-duplicative implementation inspiration in adverse-condition radar-camera / multimodal 3D detection.

- ContextualFusion (`https://arxiv.org/abs/2404.14780`): context-based gated fusion for adverse operation; reports strong night-time nuScenes gains. This reinforces the staged adaptive fusion direction and suggests a future `day/night/rain` condition signal if feature-only gating is too weak.
- SpaRC (`https://arxiv.org/abs/2411.19860`): sparse frustum fusion plus range-adaptive radar aggregation. Relevant for a stronger architecture story, but likely too invasive for the immediate NB2 rescue path.
- CRT-Fusion (`https://arxiv.org/abs/2411.03013`): camera-radar temporal fusion using motion information. Interesting if we need a temporal branch, but requires more plumbing than the current loss-weight/adaptive-gate experiments.
- CVFusion (`https://arxiv.org/abs/2507.04587`): cross-view two-stage 4D radar-camera fusion. Useful framing around proposal refinement; less directly applicable to nuScenes radar without a larger design change.
- D3PD (`https://doi.org/10.1016/j.patcog.2025.112350`): dynamic sampling/fusion and distillation for camera-radar BEV perception. Keep as a possible framing source if we later add a teacher/distillation branch, but do not start it before current jobs finish.

Decision: no new implementation now. Wait for `w05/w025`; if both are weak, prefer the already-staged adaptive fusion gate before attempting larger temporal/sparse/distillation ideas.

## Progress Poll - 2026-05-12 03:39 UTC

- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `3/12`, iter `950/1000`; stderr size remains `0` bytes. Eval `1246`, summary `1247`, and compare `1264` remain dependency-pending.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `1/12`, iter `500/1000`; stderr size remains `0` bytes. Eval `1262`, summary `1263`, and compare `1264` remain dependency-pending.
- No final `summary_metrics.md/json` files exist for either loss-weighted run yet, so no promotion-gate decision can be made from this poll.

## Staged Fallback Prep - 2026-05-12 03:46 UTC

- Adaptive-fusion staged train script was corrected to call `train.py --config configs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research.py`; the prior staged positional-config/`--work-dir` form would not match this repo `train.py` parser.
- Added staged adaptive-fusion eval, summary sbatch, and summarizer artifacts under `research/night_gen_phase1/staged_adaptfusion/`.
- Validation passed: `git apply --check research/night_gen_phase1/staged_adaptfusion/adaptive_fusion_gate.patch`, `bash -n` on staged train/eval/summary sbatch files, and `conda run -n racformerfix --no-capture-output python -m py_compile research/night_gen_phase1/staged_adaptfusion/summarize_adaptfusion.py`.
- The staged summarizer reports both S0 gate status and adaptive-fusion-vs-w05 gate status. Do not apply or submit T11 until the active `w05/w025` loss-weight chains finish and the promotion gate is evaluated.

## Progress Poll - 2026-05-12 03:47 UTC

- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; checkpoint saved at epoch `3`, latest observed train log reached epoch `4/12`, iter `250/1000`; stderr size remains `0` bytes.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `1/12`, iter `800/1000`; stderr size remains `0` bytes.
- Eval/summary jobs `1246/1247` and `1262/1263`, plus compare job `1264`, remain dependency-pending. No final `summary_metrics.md/json` files exist yet for either loss-weighted run.

## W05 Eval Fallback Prep - 2026-05-12 03:52 UTC

- Stored queued job `1246` uses generic `run_t9_eval.sbatch`; the stored script is fine if `WEIGHTS` was exported as a concrete `epoch_12.pth` path, but it cannot resolve `__LATEST__` for `S3_seed20260425_ratio18p75_w05`.
- To avoid losing time if `1246` fails after train, staged reproducible fallback scripts at repo root: `run_t10_s3_seed20260425_ratio18p75_w05_eval.sbatch` and `run_t10_s3_seed20260425_ratio18p75_w05_summary.sbatch`.
- Fallback scripts use explicit `CFG_DIR=outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_research`, resolve latest work dir, require `epoch_12.pth`, and call `summarize_w05.py`; both pass `bash -n` on the remote.
- Do not submit these unless queued job `1246`/`1247` fails or is canceled; current dependency chain remains untouched.

## Progress Poll - 2026-05-12 03:53 UTC

- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `4/12`, iter `450/1000`; stderr size remains `0` bytes.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; checkpoint saved at epoch `1`; stderr size remains `0` bytes.
- Eval/summary jobs and compare job remain dependency-pending. No final `summary_metrics.md/json` files exist yet for either loss-weighted run.

## W05 Eval Chain Replacement - 2026-05-12 03:56 UTC

- Risk found: old pending `w05` eval job `1246` was pinned to `livenode02`, which is fully allocated by `w025` train `1261` (`CPUAlloc=16/CPUTot=16`). That would delay `w05` evaluation after train `1245` finishes on `livenode03`.
- Canceled stale pending jobs `1246`, `1247`, and old compare `1264`; running train jobs `1245` and `1261` were not touched.
- Submitted clean replacement chain: `1265` (`w05` eval, afterok:`1245`, nodelist `livenode03`) -> `1266` (`w05_summary`, afterok:`1265`) -> `1267` (`loss_weight_compare`, afterok:`1266` and afterok:`1263`).
- Stored job scripts verified with `scontrol write batch_script`: `1265` uses `run_t10_s3_seed20260425_ratio18p75_w05_eval.sbatch` with explicit `CFG_DIR=outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_research`; `1266` calls `summarize_w05.py`; `1267` runs `compare_loss_weight_sweep.py`.
- Current active chain is now `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.

## Progress Poll - 2026-05-12 03:57 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `4/12`, iter `600/1000`; stderr size remains `0` bytes; checkpoints through epoch `3` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `2/12`, iter `200/1000`; stderr size remains `0` bytes; checkpoint through epoch `1` exists.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 03:58 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `4/12`, iter `650/1000`; stderr size remains `0` bytes; checkpoints through epoch `3` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `2/12`, iter `250/1000`; stderr size remains `0` bytes; checkpoint through epoch `1` exists.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 03:59 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `4/12`, iter `700/1000`; stderr size remains `0` bytes; checkpoints through epoch `3` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `2/12`, iter `250/1000`; stderr size remains `0` bytes; checkpoint through epoch `1` exists.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:00 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `4/12`, iter `750/1000`; stderr size remains `0` bytes; checkpoints through epoch `3` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `2/12`, iter `300/1000`; stderr size remains `0` bytes; checkpoint through epoch `1` exists.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:01 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `4/12`, iter `750/1000`; stderr size remains `0` bytes; checkpoints through epoch `3` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `2/12`, iter `350/1000`; stderr size remains `0` bytes; checkpoint through epoch `1` exists.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:02 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `4/12`, iter `800/1000`; stderr size remains `0` bytes; checkpoints through epoch `3` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `2/12`, iter `400/1000`; stderr size remains `0` bytes; checkpoint through epoch `1` exists.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Paper Inspiration Scan - 2026-05-12 04:10 UTC

Narrow web scan performed because DriveGEN visual QC is rejected and the active loss-weight runs are still training. Do not repeat these queries unless looking for implementation details.

- WCRD / camera-radar distillation for adverse weather (Displays 2026, DOI `10.1016/j.displa.2025.103320`): clear-weather teacher -> adverse-weather student, spatial/semantic/prediction distillation, gated camera-radar fusion, and weather-like image degradation. Relevant as a heavier fallback if adaptive fusion is not enough; likely too expensive to launch before the current loss-weight/adaptive-gate path is evaluated.
- MoME / Immortal (CVPR 2025): multi-modal expert decoding with adaptive query routing for robust fusion under sensor failures (`https://openaccess.thecvf.com/content/CVPR2025/html/Park_Resilient_Sensor_Fusion_Under_Adverse_Sensor_Failures_via_Multi-Modal_Expert_CVPR_2025_paper.html`). Supports the staged adaptive-fusion/query-gating direction, though the paper is LiDAR-camera rather than radar-camera.
- RICCARDO (CVPR 2025): radar-hit distribution modeling around monocular detections for camera-radar 3D detection (`https://cvpr.thecvf.com/virtual/2025/poster/33054`). Interesting but not a quick NB2 patch because it needs a radar-hit distribution/kernel branch.
- V2X-R / MDD (CVPR 2025): weather-robust radar-conditioned denoising diffusion for adverse weather robustness (`https://openaccess.thecvf.com/content/CVPR2025/html/Huang_V2X-R_Cooperative_LiDAR-4D_Radar_Fusion_with_Denoising_Diffusion_for_3D_CVPR_2025_paper.html`). Useful as conceptual backing for radar-conditioned denoising, not a near-term RaCFormer experiment because it is LiDAR/4D radar/V2X.
- ZFusion (CVPR 2025 workshop): feature-pyramid double deformable cross-attention fuser for camera and 4D radar (`https://cvpr.thecvf.com/virtual/2025/35800`). Supports multi-scale cross-attention as a future direction, but less directly actionable for current RaCFormer NB2 than adaptive query/fusion gating.

Decision from scan: do not spend the next GPU slot on more unconstrained image generation. The most defensible next branch after `w05/w025` is still the staged adaptive fusion gate; if that fails, consider a compact teacher-student/adverse-degradation distillation branch inspired by WCRD.

## Progress Poll - 2026-05-12 04:13 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `5/12`, iter `100/1000`; stderr size remains `0` bytes; checkpoints through epoch `4` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `2/12`, iter `700/1000`; stderr size remains `0` bytes; checkpoint through epoch `1` exists.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.
- Staged adaptive-fusion documentation was updated to reference the replacement active chain (`1265/1266/1267`) instead of stale `1246/1247`, and staged validation still passes.

## Pending Eval Chain Audit - 2026-05-12 04:16 UTC

- `bash -n` passed for `run_t10_s3_seed20260425_ratio18p75_w025_eval.sbatch`, `run_t10_s3_seed20260425_ratio18p75_w025_summary.sbatch`, `run_t10_s3_seed20260425_ratio18p75_w05_eval.sbatch`, `run_t10_s3_seed20260425_ratio18p75_w05_summary.sbatch`, and `run_loss_weight_sweep_compare.sbatch`.
- Eval scripts resolve their latest work dir under the intended `CFG_DIR` and require `epoch_12.pth` before running.
- Stored SLURM job chain is correct:
  - `1262` w025 eval: `afterok:1261`, `ReqNodeList=livenode02`, command `run_t10_s3_seed20260425_ratio18p75_w025_eval.sbatch`.
  - `1263` w025 summary: `afterok:1262`, `ReqNodeList=livenode02`, command `run_t10_s3_seed20260425_ratio18p75_w025_summary.sbatch`.
  - `1265` w05 eval: `afterok:1245`, `ReqNodeList=livenode03`, command `run_t10_s3_seed20260425_ratio18p75_w05_eval.sbatch`.
  - `1266` w05 summary: `afterok:1265`, `ReqNodeList=livenode03`, command `run_t10_s3_seed20260425_ratio18p75_w05_summary.sbatch`.
  - `1267` compare: `afterok:1266,1263`, command `run_loss_weight_sweep_compare.sbatch`.
- No action needed unless a train/eval/summary job fails or summaries are missing after a job reports COMPLETED.

## Summary Logic Audit - 2026-05-12 04:19 UTC

- `summarize_w05.py`, `summarize_w025.py`, and `compare_loss_weight_sweep.py` were inspected for split paths and gate thresholds.
- `py_compile` passed under `racformerfix` for all three scripts.
- Existing baseline/reference metric files parse correctly from the same paths the scripts use:
  - S0 day/night/rain/overall mAP: `0.3152649818`, `0.1487749875`, `0.2743174671`, `0.3039905911`.
  - S0 day/night/rain/overall NDS: `0.3745762709`, `0.2150977574`, `0.3713314930`, `0.3697754272`.
  - seed20260425 ratio18p75 w1.0 day/night/rain/overall mAP: `0.3082946194`, `0.1680892081`, `0.2642961759`, `0.2990321017`.
  - seed20260425 ratio18p75 w1.0 day/night/rain/overall NDS: `0.3717686587`, `0.2314156374`, `0.3620450679`, `0.3681380218`.
- Gate logic in both summarizers and the comparator matches the current promotion rule: night mAP >= +1.0 pp, day mAP >= -1.0 pp, overall mAP >= -1.5 pp, night NDS >= -0.5 pp vs S0.
- Conclusion: the comparison machinery is ready; wait for `w05/w025` eval summaries before making the next experiment decision.

## Completion Audit Snapshot - 2026-05-12 04:22 UTC

Objective restated as concrete deliverables:

1. Inspect local handoff context for RaCFormer/NB2 and current failure modes.
2. Use subagents for repo/research review.
3. Use SSH MCP on `cluster_live_tail` to inspect remote RaCFormer state.
4. Use `livenode02` and `livenode03` GPU capacity for useful experiments.
5. Evaluate no-Vertex alternatives including DriveGEN and CycleGAN.
6. Keep a markdown experiment log to avoid repeating work.
7. Produce a paper-worthy, decision-grade RaCFormer/NB2 result, or continue to the next defensible branch if not achieved.

Prompt-to-artifact checklist:

| Requirement | Evidence inspected | Status |
|---|---|---|
| Local handoffs inspected | Local files found in `/home/gabriel/LIVE`: `HANDOFF.md`, `PHASE1_HANDOFF.md`, `RACFORMER_AUTORESEARCH_HANDOFF.md`, `RRPN/AGENTS.md`; earlier subagent summaries extracted S0/S3 metrics, NB2 failure, validation caveats, RRPN caveats. | Done |
| Subagents used | Completed/closed subagents reviewed repo state, DriveGEN/CycleGAN feasibility, loss-weight bug, corrected loss-weight path, and adaptive-fusion hook. | Done |
| Remote RaCFormer inspected | SSH MCP target `cluster_live_tail`, cwd `/srv/nfs/shared/gnmp/RaCFormer`, branch `main`, head `869407e`; dirty state intentionally contains research configs/scripts/docs and model edits for generated-sample weighting. | Done |
| livenode02/livenode03 used | Active jobs: `1261` running on `livenode02`, `1245` running on `livenode03`; pending eval/summary chains pinned to the same nodes; comparator `1267` waits for both summaries. | In progress |
| DriveGEN considered | Clone `/srv/nfs/shared/gnmp/DriveGEN`; one-image 800x448 smoke succeeded; generated image visually rejected for training augmentation; `DRIVEGEN_NIGHT_PILOT.md` updated. | Done, rejected for scaling |
| CycleGAN considered | Deprioritized as weak unpaired style-transfer negative control; not run because label/geometry preservation risk is worse than DriveGEN for 3D detection. | Done, not main path |
| Markdown experiment log maintained | Main tracker `research/night_gen_phase1/RACFORMER_NB2_EXPERIMENT_TRACKER.md`; DriveGEN note `DriveGEN/experiments/night_pilot/DRIVEGEN_NIGHT_PILOT.md`; adaptive-fusion plan `research/night_gen_phase1/staged_adaptfusion/ADAPTIVE_FUSION_GATE_PLAN.md`. | Done, ongoing |
| Decision-grade metrics for active loss-weight runs | Missing: `summary_metrics.md/json` absent for `S3_seed20260425_ratio18p75_w05` and `S3_seed20260425_ratio18p75_w025`; train jobs still running. | Missing |
| Paper-worthy result | Missing: no completed new full-val metric table yet; no replicated or stronger result beyond the already-known non-reproducible S3 seed1 result. | Not achieved |

Current conclusion: objective is not complete. Continue monitoring `w05/w025` until summaries exist, evaluate the gate, then either promote/replicate the best loss-weight result or apply the staged adaptive-fusion fallback.

## Adaptive Fusion Config Audit - 2026-05-12 04:28 UTC

- Staged adaptive config was parse-checked by temporarily placing it under `configs/` beside its real base configs, then removing the temp file.
- Important constraint: do not parse/run `research/night_gen_phase1/staged_adaptfusion/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research.py` in place; its `_base_ = ["./racformer_train2k_genaug_seed20260425_ratio18p75_w05_research.py"]` is intentionally valid only after copying into `configs/`.
- Parse check confirmed: `adaptive_fusion_gate=True`, expected output dir basename `outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research`, `load_from` inherited from the RaCFormer base pretrain, `total_epochs=12`, `eval_config.interval=12`, manifest path `research/night_gen_phase1/manifests/phase1_t10_seed20260425_ratio18p75_manifest.json`, `generated_sample_weight=0.5`, and `generated_sample_weight` present in `Collect3D.meta_keys`.
- Temp check file `configs/.adaptfusion_config_parse_check.py` was removed after validation.

## Progress Poll - 2026-05-12 04:32 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `5/12`, iter `500/1000`; stderr size remains `0` bytes; checkpoints through epoch `4` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; checkpoint saved at epoch `2`; latest observed train log reached epoch `3/12`, iter `100/1000`; stderr size remains `0` bytes.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:36 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `5/12`, iter `550/1000`; stderr size remains `0` bytes; checkpoints through epoch `4` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `3/12`, iter `100/1000`; stderr size remains `0` bytes; checkpoints through epoch `2` exist.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:37 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log remains epoch `5/12`, iter `550/1000`; stderr size remains `0` bytes; checkpoints through epoch `4` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `3/12`, iter `150/1000`; stderr size remains `0` bytes; checkpoints through epoch `2` exist.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Active Run Backup Config Audit - 2026-05-12 04:39 UTC

- Verified running job backup configs, not just source configs:
  - `outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_research/2026-05-11/23-22-47/backup/configs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_research.py` has manifest `phase1_t10_seed20260425_ratio18p75_manifest.json`, `generated_sample_weight=0.5`, `generated_sample_weight` in metadata, `total_epochs=12`, and `eval_config.interval=12`.
  - `outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w025_research/2026-05-12/00-25-43/backup/configs/racformer_train2k_genaug_seed20260425_ratio18p75_w025_research.py` has the same manifest, `generated_sample_weight=0.25`, `generated_sample_weight` in metadata, `total_epochs=12`, and `eval_config.interval=12`.
- This confirms the active jobs captured the intended loss-weight sweep configs at launch time.

## Progress Poll - 2026-05-12 04:40 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `5/12`, iter `650/1000`; stderr size remains `0` bytes; checkpoints through epoch `4` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `3/12`, iter `200/1000`; stderr size remains `0` bytes; checkpoints through epoch `2` exist.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:41 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log remains epoch `5/12`, iter `650/1000`; stderr size remains `0` bytes; checkpoints through epoch `4` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `3/12`, iter `250/1000`; stderr size remains `0` bytes; checkpoints through epoch `2` exist.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:43 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster time at poll: `2026-05-12T04:25:27+00:00`; both train logs had fresh mtimes, so neither run appears stalled.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `5/12`, iter `700/1000`; stderr size remains `0` bytes; checkpoints through epoch `4` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log remains epoch `3/12`, iter `250/1000`; stderr size remains `0` bytes; checkpoints through epoch `2` exist.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:26 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 04:26:17 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log remains epoch `5/12`, iter `700/1000`; stderr size remains `0` bytes; checkpoints through epoch `4` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `3/12`, iter `300/1000`; stderr size remains `0` bytes; checkpoints through epoch `2` exist.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:27 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 04:27:11 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `5/12`, iter `750/1000`; stderr size remains `0` bytes; checkpoints through epoch `4` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `3/12`, iter `350/1000`; stderr size remains `0` bytes; checkpoints through epoch `2` exist.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:28 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 04:27:58 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `5/12`, iter `800/1000`; stderr size remains `0` bytes; checkpoints through epoch `4` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log remains epoch `3/12`, iter `350/1000`; stderr size remains `0` bytes; checkpoints through epoch `2` exist.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:29 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 04:28:45 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log remains epoch `5/12`, iter `800/1000`; stderr size remains `0` bytes; checkpoints through epoch `4` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `3/12`, iter `400/1000`; stderr size remains `0` bytes; checkpoints through epoch `2` exist.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:30 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 04:29:31 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `5/12`, iter `850/1000`; stderr size remains `0` bytes; checkpoints through epoch `4` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `3/12`, iter `450/1000`; stderr size remains `0` bytes; checkpoints through epoch `2` exist.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:30:15 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 04:30:15 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `5/12`, iter `900/1000`; stderr size remains `0` bytes; checkpoints through epoch `4` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log remains epoch `3/12`, iter `450/1000`; stderr size remains `0` bytes; checkpoints through epoch `2` exist.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:31 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 04:31:05 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log remains epoch `5/12`, iter `900/1000`; stderr size remains `0` bytes; checkpoints through epoch `4` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `3/12`, iter `500/1000`; stderr size remains `0` bytes; checkpoints through epoch `2` exist.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## DriveGEN Visual Recheck - 2026-05-12 04:33 UTC

- User visually inspected the 800x448 DriveGEN night pilot and reported that the generated image looked very weird.
- Rechecked the same generated output locally after downloading it from:
  `/srv/nfs/shared/gnmp/DriveGEN/experiments/night_pilot/temp_data_2.1_base_seed20260425_r18p75_first2_800x448/nus_res/night/CAM_BACK_n008-2018-08-30-15-52-26-0400__CAM_BACK__1535659414187558.jpg`
- Judgement: reject the current DriveGEN output for training augmentation. It is not merely low quality; it invents or reshapes visible vehicles, adds strong cinematic glare/motion effects, and likely invalidates the source 3D/image-label geometry.
- Decision: do not spend the next GPU slot scaling DriveGEN. Keep it only as a technical fallback or possible negative-control branch after a deliberate QC-only pilot, not as the active path toward paper-worthy metrics.

## Progress Poll - 2026-05-12 04:34 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 04:33:47 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`; latest observed train log reached epoch `6/12`, iter `50/1000`; stderr size remains `0` bytes; checkpoints through epoch `5` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`; latest observed train log reached epoch `3/12`, iter `650/1000`; stderr size remains `0` bytes; checkpoints through epoch `2` exist.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:36 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 04:36:33 UTC`; host `cluster-live`; repo path `/srv/nfs/shared/gnmp/RaCFormer`; HEAD `869407e`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `2:14:13`; latest observed train log reached epoch `6/12`, iter `100/1000`; stderr size remains `0` bytes; checkpoints through epoch `5` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `1:11:18`; latest observed train log reached epoch `3/12`, iter `700/1000`; stderr size remains `0` bytes; checkpoints through epoch `2` exist.
- Existing `results/loss_weight_sweep_summary.md/json` is stale for the active corrected runs: it still lists `w05` and `w025` as `PENDING` and predates jobs `1266`, `1263`, and `1267`.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Queued Eval/Summary Dependency Audit - 2026-05-12 04:38 UTC

- Target nodes are fully allocated by the two active train jobs: `livenode02` is running `1261`; `livenode03` is running `1245`. No additional GPU workload should be submitted while these are active.
- `1265` eval is pending `afterok:1245`, pinned to `livenode03`, command `run_t10_s3_seed20260425_ratio18p75_w05_eval.sbatch`, and writes eval logs under `results/S3_seed20260425_ratio18p75_w05/`.
- `1266` summary is pending `afterok:1265`, pinned to `livenode03`, command `run_t10_s3_seed20260425_ratio18p75_w05_summary.sbatch`.
- `1262` eval is pending `afterok:1261`, pinned to `livenode02`, command `run_t10_s3_seed20260425_ratio18p75_w025_eval.sbatch`, and writes eval logs under `results/S3_seed20260425_ratio18p75_w025/`.
- `1263` summary is pending `afterok:1262`, pinned to `livenode02`, command `run_t10_s3_seed20260425_ratio18p75_w025_summary.sbatch`.
- `1267` comparator is pending `afterok:1266,afterok:1263`, command `run_loss_weight_sweep_compare.sbatch`; this is the job expected to overwrite the stale `results/loss_weight_sweep_summary.md/json`.

## Progress Poll - 2026-05-12 04:39 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 04:39:37 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `2:17:16`; latest observed train log reached epoch `6/12`, iter `250/1000`; stderr remains empty by previous size check.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `1:14:21`; latest observed train log reached epoch `3/12`, iter `800/1000`; stderr remains empty by previous size check.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:41 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 04:41:34 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `2:19:13`; latest observed train log reached epoch `6/12`, iter `300/1000`; checkpoints through epoch `5` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `1:16:18`; latest observed train log reached epoch `3/12`, iter `900/1000`; checkpoints through epoch `2` exist.
- Watch item: one `w025` minibatch showed loss `27.37` at epoch `3/12`, iter `900/1000`. No traceback, OOM, stderr, or repeated pattern observed, so no intervention yet.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:45 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 04:45:22 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `2:23:01`; latest observed train log reached epoch `6/12`, iter `450/1000`; checkpoints through epoch `5` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `1:20:06`; latest observed train log reached epoch `4/12`, iter `50/1000`; checkpoints through epoch `3` exist.
- The previous `w025` loss spike did not stop training: epoch `3/12` finished, checkpoint `epoch_3.pth` was saved, and epoch `4/12` started normally.
- Stderr sizes remain `0` bytes for both active train jobs.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:46 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 04:46:15 UTC`; host `cluster-live`; repo path `/srv/nfs/shared/gnmp/RaCFormer`; HEAD `869407e`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `2:23:54`; latest observed train log reached epoch `6/12`, iter `500/1000`; checkpoints through epoch `5` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `1:20:59`; latest observed train log reached epoch `4/12`, iter `100/1000`; checkpoints through epoch `3` exist.
- Stderr sizes remain `0` bytes for both active train jobs.
- Existing `results/loss_weight_sweep_summary.md/json` is still stale for the active corrected runs, with `w05/w025` marked `PENDING`.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 04:53 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 04:53:29 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `2:31:08`; latest observed train log reached epoch `6/12`, iter `750/1000`; checkpoints through epoch `5` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `1:28:13`; latest observed train log reached epoch `4/12`, iter `350/1000`; checkpoints through epoch `3` exist.
- Watch item: one `w05` minibatch showed loss `26.90` at epoch `6/12`, iter `750/1000`, driven by `loss_cls=2.09`. No traceback, OOM, stderr, or repeated pattern observed, so no intervention yet.
- Stderr sizes remain `0` bytes for both active train jobs.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 05:03 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 05:03:16 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `2:40:55`; latest observed train log reached epoch `7/12`, iter `150/1000`; checkpoint `epoch_6.pth` saved at `2026-05-12 04:58`.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `1:38:00`; latest observed train log reached epoch `4/12`, iter `750/1000`; checkpoints through epoch `3` exist.
- The previous `w05` loss spike did not stop training: epoch `6/12` completed, checkpoint `epoch_6.pth` was saved, and epoch `7/12` started normally.
- Watch item: one additional `w025` minibatch showed loss `27.06` at epoch `4/12`, iter `650/1000`, driven by `loss_cls=2.01`; the next two logged iterations recovered to `17.68` and `17.12`.
- Stderr sizes remain `0` bytes for both active train jobs.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 05:12 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 05:12:03 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `2:49:42`; latest observed train log reached epoch `7/12`, iter `500/1000`; checkpoints through epoch `6` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `1:46:47`; latest observed train log reached epoch `5/12`, iter `50/1000`; checkpoint `epoch_4.pth` saved at `2026-05-12 05:09`.
- The previous `w025` loss spikes did not stop training: epoch `4/12` completed, checkpoint `epoch_4.pth` was saved, and epoch `5/12` started normally.
- Watch item: `w025` had another isolated high-loss minibatch at epoch `4/12`, iter `900/1000` (`loss=28.93`, `loss_cls=2.53`), followed by normal loss at iter `950/1000` and `1000/1000`.
- Stderr sizes remain `0` bytes for both active train jobs.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 05:13 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 05:13:02 UTC`; host `cluster-live`; repo path `/srv/nfs/shared/gnmp/RaCFormer`; HEAD `869407e`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `2:50:41`; latest observed train log reached epoch `7/12`, iter `550/1000`; checkpoints through epoch `6` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `1:47:46`; latest observed train log reached epoch `5/12`, iter `100/1000`; checkpoints through epoch `4` exist.
- Watch item: `w05` had another isolated high-loss minibatch at epoch `7/12`, iter `550/1000` (`loss=34.30`, `loss_cls=3.34`). This is a watch-only signal unless it repeats or causes stderr/job failure.
- Stderr sizes remain `0` bytes for both active train jobs.
- Existing `results/loss_weight_sweep_summary.md/json` is still stale for the active corrected runs, with `w05/w025` marked `PENDING`.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 05:18 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 05:18:12 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `2:55:51`; latest observed train log reached epoch `7/12`, iter `700/1000`; checkpoints through epoch `6` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `1:52:56`; latest observed train log reached epoch `5/12`, iter `300/1000`; checkpoints through epoch `4` exist.
- The `w05` high-loss minibatch at epoch `7/12`, iter `550/1000` recovered on subsequent logged batches: iter `600/1000` loss `17.99`, iter `650/1000` loss `17.45`, iter `700/1000` loss `17.53`.
- Stderr sizes remain `0` bytes for both active train jobs.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 05:28 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 05:28:08 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `3:05:47`; latest observed train log reached epoch `8/12`, iter `100/1000`; checkpoint `epoch_7.pth` saved at `2026-05-12 05:24`.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `2:02:52`; latest observed train log reached epoch `5/12`, iter `700/1000`; checkpoints through epoch `4` exist.
- The previous `w05` loss spike did not stop training: epoch `7/12` completed, checkpoint `epoch_7.pth` was saved, and epoch `8/12` started normally.
- Watch item: `w025` had another isolated high-loss minibatch at epoch `5/12`, iter `500/1000` (`loss=32.33`, `loss_cls=2.92`), followed by normal losses at iter `550/1000`, `600/1000`, `650/1000`, and `700/1000`.
- Stderr sizes remain `0` bytes for both active train jobs.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 05:38 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 05:38:00 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `3:15:39`; latest observed train log reached epoch `8/12`, iter `450/1000`; checkpoints through epoch `7` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `2:12:44`; latest observed train log reached epoch `6/12`, iter `50/1000`; checkpoint `epoch_5.pth` saved at `2026-05-12 05:35`.
- The previous `w025` high-loss minibatch did not stop training: epoch `5/12` completed, checkpoint `epoch_5.pth` was saved, and epoch `6/12` started normally.
- Stderr sizes remain `0` bytes for both active train jobs.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 05:53 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 05:53:32 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `3:31:11`; latest observed train log reached epoch `9/12`, iter `50/1000`; checkpoint `epoch_8.pth` saved at `2026-05-12 05:50`.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `2:28:16`; latest observed train log reached epoch `6/12`, iter `650/1000`; checkpoints through epoch `5` exist.
- The previous `w05` high-loss watch did not recur as an issue; epoch `8/12` completed, checkpoint `epoch_8.pth` was saved, and epoch `9/12` started normally.
- `w025` epoch `6/12` losses are stable in the observed window; no new high-loss watch item in this poll.
- Stderr sizes remain `0` bytes for both active train jobs.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 06:04 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 06:04:28 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `3:42:07`; latest observed train log reached epoch `9/12`, iter `500/1000`; checkpoints through epoch `8` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `2:39:12`; latest observed train log reached epoch `7/12`, iter `100/1000`; checkpoint `epoch_6.pth` saved at `2026-05-12 06:01`.
- Watch item: `w05` had one high-loss minibatch at epoch `9/12`, iter `100/1000` (`loss=33.96`, `loss_cls=3.61`), followed by normal losses through iter `500/1000`.
- Watch item: `w025` had one high-loss minibatch at epoch `6/12`, iter `750/1000` (`loss=25.81`, `loss_cls=2.07`), followed by normal losses through epoch completion and epoch `7/12` start.
- Stderr sizes remain `0` bytes for both active train jobs.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 06:21 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 06:20:57 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `3:58:36`; latest observed train log reached epoch `10/12`, iter `100/1000`; checkpoint `epoch_9.pth` saved at `2026-05-12 06:17`.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `2:55:41`; latest observed train log reached epoch `7/12`, iter `700/1000`; checkpoints through epoch `6` exist.
- The previous `w05` high-loss watch did not stop training: epoch `9/12` completed, checkpoint `epoch_9.pth` was saved, and epoch `10/12` started normally.
- Watch item: `w025` had one high-loss minibatch at epoch `7/12`, iter `550/1000` (`loss=33.67`, `loss_cls=3.32`), followed by normal losses through iter `700/1000`.
- Stderr sizes remain `0` bytes for both active train jobs.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 06:33 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 06:32:34 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `4:10:13`; latest observed train log reached epoch `10/12`, iter `550/1000`; checkpoints through epoch `9` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `3:07:18`; latest observed train log reached epoch `8/12`, iter `150/1000`; checkpoint `epoch_7.pth` saved at `2026-05-12 06:27`.
- The previous `w025` high-loss watch did not stop training: epoch `7/12` completed, checkpoint `epoch_7.pth` was saved, and epoch `8/12` started normally.
- Watch item: `w05` had one high-loss minibatch at epoch `10/12`, iter `250/1000` (`loss=54.31`, `loss_cls=6.19`), followed by normal losses through iter `550/1000`.
- Stderr sizes remain `0` bytes for both active train jobs.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 06:47 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 06:46:40 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `4:24:19`; latest observed train log reached epoch `11/12`, iter `100/1000`; checkpoint `epoch_10.pth` saved at `2026-05-12 06:43`.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `3:21:24`; latest observed train log reached epoch `8/12`, iter `700/1000`; checkpoints through epoch `7` exist.
- The previous `w05` high-loss watch did not stop training: epoch `10/12` completed, checkpoint `epoch_10.pth` was saved, and epoch `11/12` started normally.
- `w025` epoch `8/12` losses are stable in the observed window; no new high-loss watch item in this poll.
- Stderr sizes remain `0` bytes for both active train jobs.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 06:58 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 06:58:09 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `4:35:48`; latest observed train log reached epoch `11/12`, iter `550/1000`; checkpoints through epoch `10` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `3:32:53`; latest observed train log reached epoch `9/12`, iter `150/1000`; checkpoint `epoch_8.pth` saved at `2026-05-12 06:53`.
- The previous `w025` epoch `8/12` completed cleanly, checkpoint `epoch_8.pth` was saved, and epoch `9/12` started normally.
- Watch item: `w025` had one high-loss minibatch at epoch `9/12`, iter `100/1000` (`loss=33.33`, `loss_cls=3.59`), followed by normal loss at iter `150/1000`.
- Stderr sizes remain `0` bytes for both active train jobs.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 07:12 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 07:12:27 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `4:50:06`; latest observed train log reached final train epoch `12/12`, iter `100/1000`; checkpoint `epoch_11.pth` saved at `2026-05-12 07:09`.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `3:47:11`; latest observed train log reached epoch `9/12`, iter `700/1000`; checkpoints through epoch `8` exist.
- The previous `w05` epoch `11/12` completed, checkpoint `epoch_11.pth` was saved, and final epoch `12/12` started normally.
- Watch item: `w05` had one high-loss minibatch at epoch `11/12`, iter `1000/1000` (`loss=26.42`, `loss_cls=1.49`), but checkpointing and final epoch start succeeded.
- `w025` epoch `9/12` recovered after the earlier watch item and showed normal losses through iter `700/1000`.
- Stderr sizes remain `0` bytes for both active train jobs.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 07:32 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 07:32:11 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `5:09:50`; latest observed train log reached final train epoch `12/12`, iter `850/1000`; checkpoints through epoch `11` exist.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `4:06:55`; latest observed train log reached epoch `10/12`, iter `450/1000`; checkpoint `epoch_9.pth` saved at `2026-05-12 07:19`.
- `w025` epoch `9/12` completed and checkpointed; epoch `10/12` is in progress.
- Stderr sizes remain `0` bytes for both active train jobs.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 07:35 UTC

- Active chain remains `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then `1267`.
- Cluster timestamp: `2026-05-12 07:35:07 UTC`.
- `w05` corrected generated-sample weight run: train job `1245` still RUNNING on `livenode03`, elapsed `5:12:46`; latest observed train log reached final train epoch `12/12`, iter `1000/1000`.
- `w05` checkpoint `epoch_12.pth` appeared during the poll, but size was only `68157440` bytes at observation time, while prior complete checkpoints are `764836945` bytes. Treat the checkpoint as still being written until size stabilizes and job `1245` exits successfully.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `4:09:51`; latest observed train log reached epoch `10/12`, iter `550/1000`; checkpoints through epoch `9` exist.
- Stderr sizes remain `0` bytes for both active train jobs.
- Eval jobs `1265` and `1262`, summary jobs `1266` and `1263`, and comparator job `1267` are still pending on dependencies.
- Existing `results/loss_weight_sweep_summary.md/json` remains stale from `2026-05-12 03:29` and still lists `w05/w025` as `PENDING`.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 07:37 UTC

- Active chain is now `1265 -> 1266` for `w05`, `1261 -> 1262 -> 1263` for `w025`, then `1267`.
- Cluster timestamp: `2026-05-12 07:37:52 UTC`.
- `w05` train job `1245` has completed successfully enough for the dependency to release: eval job `1265` is RUNNING on `livenode03`, elapsed `0:42`.
- `w05` checkpoint `epoch_12.pth` is now stable at `764836945` bytes, matching prior complete checkpoints.
- `w05` eval log is active and running inference on `6019` samples; observed progress reached `38/6019` samples with no traceback or OOM.
- `w025` sibling generated-sample weight run: train job `1261` still RUNNING on `livenode02`, elapsed `4:12:36`; latest observed train log reached epoch `10/12`, iter `650/1000`; checkpoints through epoch `9` exist.
- `w05` summary job `1266`, `w025` eval/summary jobs `1262/1263`, and comparator job `1267` remain pending on dependencies.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 07:41 UTC

- Active chain remains `1265 -> 1266` for `w05`, `1261 -> 1262 -> 1263` for `w025`, then `1267`.
- Cluster timestamp: `2026-05-12 07:41:37 UTC`.
- `w05` eval job `1265` is RUNNING on `livenode03`, elapsed `4:27`; concise eval progress reached `943/6019`, about `4.0 task/s`, ETA about `1277s`.
- `w05` eval stderr only contains normal startup messages: config, model build, and inference start.
- `w025` train job `1261` is RUNNING on `livenode02`, elapsed `4:16:21`; latest observed train log reached epoch `10/12`, iter `800/1000`; checkpoints through epoch `9` exist.
- `w05` summary job `1266`, `w025` eval/summary jobs `1262/1263`, and comparator job `1267` remain pending on dependencies.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 07:45 UTC

- Active chain remains `1265 -> 1266` for `w05`, `1261 -> 1262 -> 1263` for `w025`, then `1267`.
- Cluster timestamp: `2026-05-12 07:45:32 UTC`.
- `w05` eval job `1265` is RUNNING on `livenode03`, elapsed `8:22`; concise eval progress reached `1887/6019`, about `4.0 task/s`, ETA about `1035s`.
- `w05` eval stderr still only contains normal startup messages; no traceback, OOM, or metric output yet.
- `w025` train job `1261` is RUNNING on `livenode02`, elapsed `4:20:16`; latest observed train log reached epoch `10/12`, iter `950/1000`; checkpoints through epoch `9` exist.
- `w025` latest observed epoch `10/12` losses remain stable; no new high-loss watch item in the latest window.
- `w05` summary job `1266`, `w025` eval/summary jobs `1262/1263`, and comparator job `1267` remain pending on dependencies.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 07:52 UTC

- Active chain remains `1265 -> 1266` for `w05`, `1261 -> 1262 -> 1263` for `w025`, then `1267`.
- Cluster timestamp: `2026-05-12 07:52:07 UTC`.
- `w05` eval job `1265` is RUNNING on `livenode03`, elapsed `14:57`; concise eval progress reached `3455/6019`, about `4.0 task/s`, ETA about `643s`.
- `w05` eval stderr still only contains normal startup messages; no traceback, OOM, or metric output yet.
- `w025` train job `1261` is RUNNING on `livenode02`, elapsed `4:26:51`; epoch `10/12` completed and full-size checkpoint `epoch_10.pth` exists (`764836945` bytes).
- `w025` started epoch `11/12`; latest observed train log reached iter `200/1000`, with stable losses in the latest window.
- `w05` summary job `1266`, `w025` eval/summary jobs `1262/1263`, and comparator job `1267` remain pending on dependencies.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 08:00 UTC

- Active chain remains `1265 -> 1266` for `w05`, `1261 -> 1262 -> 1263` for `w025`, then `1267`.
- Cluster timestamp: `2026-05-12 08:00:50 UTC`.
- `w05` eval job `1265` is RUNNING on `livenode03`, elapsed `23:40`; concise eval progress reached `5535/6019`, about `4.0 task/s`, ETA about `121s`.
- `w05` eval stderr still only contains normal startup messages; no traceback, OOM, or metric output yet.
- `w025` train job `1261` is RUNNING on `livenode02`, elapsed `4:35:34`; latest observed train log reached epoch `11/12`, iter `550/1000`; checkpoints through epoch `10` exist.
- `w025` epoch `11/12` observed losses remain stable in the latest window.
- `w05` summary job `1266`, `w025` eval/summary jobs `1262/1263`, and comparator job `1267` remain pending on dependencies.
- No final `summary_metrics.md/json` or per-split `metrics_summary.json` files exist yet for either loss-weighted run, so no promotion-gate decision is possible from this poll.

## Progress Poll - 2026-05-12 08:05 UTC

- Active chain remains `1265 -> 1266` for `w05`, `1261 -> 1262 -> 1263` for `w025`, then `1267`.
- Cluster timestamp: `2026-05-12 08:05:07 UTC`.
- `w05` eval job `1265` is still RUNNING on `livenode03`, elapsed `27:57`.
- `w05` full-val inference completed all `6019/6019` samples and wrote `eval/submission_overall/pts_bbox/results_nusc.json`; the job is now in nuScenes bbox evaluation.
- `w05` summary job `1266` has not released yet and no `summary_metrics.*` artifacts exist, so the eval job is not complete.
- `w025` train job `1261` is RUNNING on `livenode02`, elapsed `4:39:51`; latest observed train log reached epoch `11/12`, iter `700/1000`; checkpoints through epoch `10` exist.
- `w025` epoch `11/12` observed losses remain stable in the latest window.
- `w025` eval/summary jobs `1262/1263` and comparator job `1267` remain pending on dependencies.
- No final promotion-gate decision is possible from this poll.

## Result - S3 seed20260425 ratio18p75 w05 - 2026-05-12 08:10 UTC

- Stage: `S3_seed20260425_ratio18p75_w05`.
- Generated-keyframe sample weight: `0.5`.
- Summary artifacts:
  - `research/night_gen_phase1/results/S3_seed20260425_ratio18p75_w05/summary_metrics.md`
  - `research/night_gen_phase1/results/S3_seed20260425_ratio18p75_w05/summary_metrics.json`
- Metrics:
  - day: mAP `0.3083104470`, NDS `0.3680101238`; delta vs S0 `-0.70 pp` mAP, `-0.66 pp` NDS.
  - night: mAP `0.1514892487`, NDS `0.2092292122`; delta vs S0 `+0.27 pp` mAP, `-0.59 pp` NDS.
  - rain: mAP `0.2680655070`, NDS `0.3645405659`; delta vs S0 `-0.63 pp` mAP, `-0.68 pp` NDS.
  - overall: mAP `0.2992690380`, NDS `0.3637460535`; delta vs S0 `-0.47 pp` mAP, `-0.60 pp` NDS.
- Gate verdict: `FAIL`.
- Reason: night mAP gain is only `+0.27 pp`, below the required `+1.0 pp`; night NDS is `-0.59 pp`, just below the allowed `-0.5 pp` floor. Day and overall mAP stayed within allowed regression, but this does not preserve the original night improvement.
- Decision: do not promote `generated_sample_weight=0.5`. Continue waiting for the already-running `w025` chain before making a loss-weight sweep decision. Do not submit the staged adaptive-fusion fallback until `w025` and comparator job `1267` finish.

## Progress Poll - 2026-05-12 08:17 UTC

- Active chain is now `1261 -> 1262 -> 1263` for `w025`, then `1267`.
- Cluster timestamp: `2026-05-12 08:17:45 UTC`.
- `w05` eval and summary jobs have completed and are no longer in the queue.
- `w025` train job `1261` is RUNNING on `livenode02`, elapsed `4:52:29`.
- `w025` epoch `11/12` completed and full-size checkpoint `epoch_11.pth` exists (`764836945` bytes).
- Watch item: `w025` had one high-loss minibatch at epoch `11/12`, iter `1000/1000` (`loss=25.48`, increased bbox DN loss), but checkpointing succeeded and final epoch `12/12` started normally.
- Latest observed final-epoch train log reached epoch `12/12`, iter `200/1000`, with losses back in the normal range.
- `w025` eval/summary jobs `1262/1263` and comparator job `1267` remain pending on dependencies.
- No `w025` promotion-gate decision is possible yet.

## Progress Poll - 2026-05-12 08:26 UTC

- Active chain remains `1261 -> 1262 -> 1263` for `w025`, then `1267`.
- Cluster timestamp: `2026-05-12 08:26:25 UTC`.
- `w025` train job `1261` is RUNNING on `livenode02`, elapsed `5:01:09`.
- Latest observed train log reached final epoch `12/12`, iter `550/1000`; last complete checkpoint remains `epoch_11.pth`.
- Watch item: one final-epoch minibatch at iter `500/1000` had `loss=23.03`, but the next logged batch at iter `550/1000` recovered to `loss=14.95`.
- `w025` eval/summary jobs `1262/1263` and comparator job `1267` remain pending on dependencies.
- No `w025` promotion-gate decision is possible yet.

## Progress Poll - 2026-05-12 08:38 UTC

- Active chain remains `1261 -> 1262 -> 1263` for `w025`, then `1267`.
- Cluster timestamp: `2026-05-12 08:38:07 UTC`.
- `w025` final train epoch `12/12` completed and full-size checkpoint `epoch_12.pth` exists (`764836945` bytes).
- Final observed training minibatch at epoch `12/12`, iter `1000/1000` had normal loss `14.70`; checkpoint saving logged at `2026-05-12 05:37:41`.
- `w025` train job `1261` was still listed RUNNING at this poll, so eval job `1262` had not released yet.
- `w025` eval/summary jobs `1262/1263` and comparator job `1267` remain pending on dependencies.
- No `w025` promotion-gate decision is possible yet.

## Progress Poll - 2026-05-12 08:40 UTC

- Active chain is now `1262 -> 1263` for `w025`, then `1267`.
- Cluster timestamp: `2026-05-12 08:40:26 UTC`.
- `w025` train job `1261` exited and eval job `1262` is RUNNING on `livenode02`, elapsed `0:49`.
- The train job's post-checkpoint 300-step internal validation/report phase completed (`300/300`) before dependency release.
- `w025` eval logs exist:
  - `research/night_gen_phase1/results/S3_seed20260425_ratio18p75_w025/eval_slurm_1262.out`
  - `research/night_gen_phase1/results/S3_seed20260425_ratio18p75_w025/eval_slurm_1262.err`
- `w025` summary job `1263` and comparator job `1267` remain pending on dependencies.
- No `w025` promotion-gate decision is possible yet.

## Progress Poll - 2026-05-12 08:50 UTC

- Active chain remains `1262 -> 1263` for `w025`, then `1267`.
- Cluster timestamp: `2026-05-12 08:50:01 UTC`.
- `w025` eval job `1262` is RUNNING on `livenode02`, elapsed `10:24`.
- Concise eval progress reached `2408/6019`, about `4.1 task/s`, ETA about `890s`.
- Eval stderr still only contains normal startup/inference messages.
- `w025` summary job `1263` and comparator job `1267` remain pending on dependencies.
- No `w025` summary artifacts exist yet, so no promotion-gate decision is possible.

## Progress Poll - 2026-05-12 08:53 UTC

- Active chain remains `1262 -> 1263` for `w025`, then `1267`.
- Cluster timestamp: `2026-05-12 08:53:00 UTC`.
- `w025` eval job `1262` is RUNNING on `livenode02`, elapsed `13:23`.
- Concise eval progress reached `3135/6019`, about `4.1 task/s`, ETA about `710s`.
- Eval stderr still only contains normal startup/inference messages.
- `w025` summary job `1263` and comparator job `1267` remain pending on dependencies.
- No `w025` summary artifacts exist yet, so no promotion-gate decision is possible.

## Progress Poll - 2026-05-12 09:00 UTC

- Active chain remains `1262 -> 1263` for `w025`, then `1267`.
- Cluster timestamp: `2026-05-12 09:00:37 UTC`.
- `w025` eval job `1262` is RUNNING on `livenode02`, elapsed `21:00`.
- Concise eval progress reached `4978/6019`, about `4.1 task/s`, ETA about `257s`.
- Eval stderr still only contains normal startup/inference messages.
- `w025` summary job `1263` and comparator job `1267` remain pending on dependencies.
- No `w025` summary artifacts exist yet, so no promotion-gate decision is possible.

## Result - S3 seed20260425 ratio18p75 w025 - 2026-05-12 09:16 UTC

- Stage: `S3_seed20260425_ratio18p75_w025`.
- Generated-keyframe sample weight: `0.25`.
- Summary artifacts:
  - `research/night_gen_phase1/results/S3_seed20260425_ratio18p75_w025/summary_metrics.md`
  - `research/night_gen_phase1/results/S3_seed20260425_ratio18p75_w025/summary_metrics.json`
- Metrics:
  - day: mAP `0.2930102434`, NDS `0.3619689982`; delta vs S0 `-2.23 pp` mAP, `-1.26 pp` NDS.
  - night: mAP `0.1348438170`, NDS `0.2000884870`; delta vs S0 `-1.39 pp` mAP, `-1.50 pp` NDS.
  - rain: mAP `0.2560373268`, NDS `0.3603799634`; delta vs S0 `-1.83 pp` mAP, `-1.10 pp` NDS.
  - overall: mAP `0.2840318370`, NDS `0.3582017824`; delta vs S0 `-2.00 pp` mAP, `-1.16 pp` NDS.
- Gate verdict: `FAIL`.
- Reason: this setting fails every mAP gate and worsens night NDS; it is not a usable regularizer for NB2 generated-keyframe replay.

## Decision - Loss Weight Sweep Closed - 2026-05-12 09:16 UTC

- Comparator artifact: `research/night_gen_phase1/results/loss_weight_sweep_summary.md`.
- Sweep table:
  - `S3_seed20260425_ratio18p75` weight `1.0`: `PASS` on seed `20260425`, but not reproducible on seed `20260502`.
  - `S3_seed20260425_ratio18p75_w05` weight `0.5`: `FAIL`; preserves day/overall mAP but loses the night mAP gain and slightly breaches night NDS floor.
  - `S3_seed20260425_ratio18p75_w025` weight `0.25`: `FAIL`; collapses day, night, rain, and overall metrics.
- Interpretation: generated-keyframe downweighting does not recover a robust paper-worthy result. Do not rerun more loss-weight values without a new mechanism or evidence.
- Next branch: use the already-staged adaptive fusion gate fallback, because it changes adverse-condition fusion behavior directly and does not rely on the visually unreliable DriveGEN-style image synthesis path.
- Cluster state at decision: `livenode02` and `livenode03` idle; no active jobs for remote user `gnmp`.

## Submission - Adaptive Fusion Gate Fallback - 2026-05-12 09:25 UTC

- Stage: `S3_seed20260425_ratio18p75_w05_adaptfusion`.
- Hypothesis: an identity-initialized decoder fusion gate can learn condition-specific image/radar/LSS weighting and recover night robustness better than further generated-sample loss reweighting.
- Rationale: broad paper scan favored adaptive radar-camera fusion over unconstrained image synthesis after DriveGEN visual QC failed; this is the smallest local RobuRCDet/WCRD-inspired change that acts on fusion instead of image pixels.
- Code/config changes applied:
  - `models/racformer_transformer.py`: opt-in `adaptive_fusion_gate` argument, three-channel gate over aligned image/radar/LSS decoder features, final gate layer zero-initialized so initial multiplier is identity (`sigmoid(0) * 2 = 1`).
  - `configs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research.py`: enables `adaptive_fusion_gate=True` on top of corrected `w05`.
  - `run_t11_adaptfusion_seed20260425_ratio18p75_w05*.sbatch`: train/eval/summary chain.
- Validation before submission:
  - `git apply --check research/night_gen_phase1/staged_adaptfusion/adaptive_fusion_gate.patch` passed.
  - `python -m py_compile models/racformer_transformer.py` passed.
  - `bash -n` passed for train/eval/summary scripts.
  - Config parse assertions passed for `adaptive_fusion_gate=True`, `generated_sample_weight=0.5`, `generated_sample_weight` in `Collect3D.meta_keys`, expected output directory, and `total_epochs=12`.
  - CPU model build/init smoke passed; verified the gate exists and final layer weights/biases are zero.
- SLURM note: initial submission failed before creating any jobs because the staged train script had `#SBATCH --gres=gpu:1`, but this cluster reports `GRES=(null)`. Removed that line from both the root and staged train script and aligned `cpus-per-task=16` with the working T10 scripts.
- Submitted dependency chain:
  - Train job: `1268` (`t11_adaptfusion_r18p75_w05`) on `livenode03`.
  - Eval job: `1269` afterok `1268`.
  - Summary job: `1270` afterok `1269`.
- Initial queue state: `1268` pending with reason `(None)`, `1269/1270` pending on dependencies.
- Gate to inspect after summary:
  - Against S0: night mAP >= `+1.0 pp`, day mAP >= `-1.0 pp`, overall mAP >= `-1.5 pp`, night NDS >= `-0.5 pp`.
  - Against corrected w05: night mAP retained/improved, day mAP >= `-1.0 pp`, overall mAP >= `-1.5 pp`, night NDS >= `-0.5 pp`.

## Progress Poll - 2026-05-12 09:24 UTC

- Active chain: `1268 -> 1269 -> 1270`.
- Train job `1268` is RUNNING on `livenode03`, elapsed `1:03`.
- Eval job `1269` and summary job `1270` are pending on dependencies.
- Output artifacts started:
  - `research/night_gen_phase1/results/S3_seed20260425_ratio18p75_w05_adaptfusion/slurm_1268.out`
  - `research/night_gen_phase1/results/S3_seed20260425_ratio18p75_w05_adaptfusion/slurm_1268.err`
- Train stderr is empty at this poll.
- Train stdout reached runner startup, work dir creation, and manifest-loader first-sample messages; no traceback or CUDA failure observed.
- Work dir: `outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research/2026-05-12/06-23-27`.

## Progress Poll - 2026-05-12 09:26 UTC

- Active chain remains `1268 -> 1269 -> 1270`.
- Train job `1268` is RUNNING on `livenode03`, elapsed `3:44`.
- Eval job `1269` and summary job `1270` remain pending on dependencies.
- Work dir is active: `outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research/2026-05-12/06-23-27`.
- Runtime smoke criterion met: training reached at least epoch `1/12`, iter `100/1000`, so the new adaptive fusion gate has executed in the train forward/backward path.
- Observed logs:
  - iter `50/1000`: loss `42.19`, mem `15449M`, ETA `5:12:59`.
  - iter `100/1000`: loss `35.45`, mem `15449M`, ETA `5:08:42`.
- Train stderr remains empty; stdout warnings about no BEV receptive-field points match prior training behavior and are not a new failure signal.

## Progress Poll - 2026-05-12 09:28 UTC

- Active chain remains `1268 -> 1269 -> 1270`.
- Train job `1268` is RUNNING on `livenode03`, elapsed `5:07`.
- Eval job `1269` and summary job `1270` remain pending on dependencies.
- Latest train log reached epoch `1/12`, iter `150/1000`.
- Observed loss trend:
  - iter `50/1000`: loss `42.19`.
  - iter `100/1000`: loss `35.45`.
  - iter `150/1000`: loss `32.59`.
- Memory remains stable at `15449M`.
- No traceback, OOM, or stderr output observed. This confirms the adaptive-fusion gate survives a longer train forward/backward window.

## Correction - Adaptive Eval Config - 2026-05-12 09:35 UTC

- Issue found before eval started: `run_t11_adaptfusion_seed20260425_ratio18p75_w05_eval.sbatch` used the generic `configs/racformer_eval_fullval_research.py`.
- Why this matters: `eval_by_condition.py` calls `load_checkpoint(..., strict=True)`, while the adaptive-fusion checkpoint will contain `fusion_gate` parameters that require an eval-time model built with `adaptive_fusion_gate=True`.
- Fix applied:
  - Patched root and staged eval scripts to use `configs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research.py` with `--full-val`. The eval wrapper still coerces the validation data to the full 6,019-sample val set.
  - `bash -n run_t11_adaptfusion_seed20260425_ratio18p75_w05_eval.sbatch` passed.
  - Config assertion confirmed `cfg.model['pts_bbox_head']['transformer']['adaptive_fusion_gate'] is True`.
- Queue correction:
  - Canceled pending, unstarted eval/summary jobs `1269` and `1270`.
  - Submitted replacement eval job `1271` with dependency `afterok:1268`.
  - Submitted replacement summary job `1272` with dependency `afterok:1271`.
- Active chain is now `1268 -> 1271 -> 1272`.

## Progress Poll - 2026-05-12 09:35 UTC

- Active chain: `1268 -> 1271 -> 1272`.
- Train job `1268` is RUNNING on `livenode03`, elapsed `12:41`.
- Replacement eval job `1271` and summary job `1272` are pending on dependencies.
- `livenode02` is idle; `livenode03` is allocated.
- Latest train log reached epoch `1/12`, iter `450/1000`.
- Loss trend remains stable/decreasing:
  - iter `200/1000`: loss `31.41`.
  - iter `250/1000`: loss `28.11`.
  - iter `300/1000`: loss `27.05`.
  - iter `350/1000`: loss `26.70`.
  - iter `400/1000`: loss `26.93`.
  - iter `450/1000`: loss `26.83`.
- Memory remains stable around `15449-15452M`.

## Side Eval - Legacy Full Camera Dropout Checkpoints - 2026-05-12 09:41 UTC

- Purpose: gather condition-wise evidence from an existing full-training camera-dropout checkpoint while `livenode02` is idle. This is not the train2k NB2 gate result and should not be mixed with the active NB2/adaptive-fusion claim.
- Existing checkpoints:
  - Full baseline: `outputs/racformer_r50_nuimg_704x256_f8/2026-01-25/20-49-49/epoch_36.pth`.
  - Full camera dropout 20%: `outputs/racformer_r50_nuimg_704x256_f8_dropout/2026-01-25/22-45-16/epoch_36.pth`.
- Pre-submit validation:
  - Both checkpoints strict-load with the current non-adaptive eval architecture.
  - `bash -n` passed for all three staged scripts.
  - `python -m py_compile` passed for `research/night_gen_phase1/staged_existing_dropout_eval/summarize_full_dropout_eval.py`.
- Staged files:
  - `research/night_gen_phase1/staged_existing_dropout_eval/run_full_baseline_epoch36_eval.sbatch`.
  - `research/night_gen_phase1/staged_existing_dropout_eval/run_full_camdrop20_epoch36_eval.sbatch`.
  - `research/night_gen_phase1/staged_existing_dropout_eval/run_full_dropout_epoch36_compare.sbatch`.
  - `research/night_gen_phase1/staged_existing_dropout_eval/summarize_full_dropout_eval.py`.
- Submitted dependency chain on `livenode02`:
  - Full baseline eval job `1273`.
  - Full camdrop20 eval job `1274` afterok `1273`.
  - Comparison job `1275` afterok `1274`.
- Initial queue state: `1273` pending with reason `(None)`, `1274/1275` pending on dependencies.
- Output comparison artifacts expected:
  - `research/night_gen_phase1/results/full_dropout_epoch36_comparison.md`.
  - `research/night_gen_phase1/results/full_dropout_epoch36_comparison.json`.

## Progress Poll - 2026-05-12 09:43 UTC

- Active adaptive chain: `1268 -> 1271 -> 1272`.
- Active side-eval chain: `1273 -> 1274 -> 1275`.
- `1268` is RUNNING on `livenode03`, elapsed `19:57`.
- `1273` is RUNNING on `livenode02`, elapsed `2:01`; `1274/1275` remain dependency-held.
- Full baseline eval startup is healthy:
  - Config: `configs/racformer_eval_fullval_research.py`.
  - Weights: `outputs/racformer_r50_nuimg_704x256_f8/2026-01-25/20-49-49/epoch_36.pth`.
  - Full validation set: `6019` samples.
  - Latest observed progress: inference started and reached early samples at about `3.7 task/s`.
- Adaptive train latest log reached epoch `1/12`, iter `750/1000`.
- Adaptive loss remains stable/decreasing:
  - iter `500/1000`: loss `27.23`.
  - iter `550/1000`: loss `25.70`.
  - iter `600/1000`: loss `25.02`.
  - iter `650/1000`: loss `24.38`.
  - iter `700/1000`: loss `25.42`.
  - iter `750/1000`: loss `23.52`.
- Adaptive memory remains stable around `15545-15548M`.

## Progress Poll - 2026-05-12 09:45 UTC

- Active adaptive chain remains `1268 -> 1271 -> 1272`.
- Active side-eval chain remains `1273 -> 1274 -> 1275`.
- Cluster state:
  - `livenode02` allocated to full baseline condition eval job `1273`.
  - `livenode03` allocated to adaptive train job `1268`.
- `1273` side eval progress: `1024/6019` full-val samples, about `4.1 task/s`, ETA about `1230s`.
- No side-eval metric JSON exists yet, so no camdrop comparison decision is available.
- Adaptive train latest parsed log reached epoch `1/12`, iter `850/1000`; latest observed loss `24.09`.
- No traceback, OOM, or unexpected stderr observed for either active job.

## Progress Poll - 2026-05-12 09:50 UTC

- Active adaptive chain remains `1268 -> 1271 -> 1272`.
- Active side-eval chain remains `1273 -> 1274 -> 1275`.
- `1268` is RUNNING on `livenode03`, elapsed `27:46`.
- `1273` is RUNNING on `livenode02`, elapsed `9:50`.
- Adaptive train milestone:
  - `epoch_1.pth` exists in `outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research/2026-05-12/06-23-27`.
  - Epoch `1/12`, iter `1000/1000` completed; loss `30.61`.
  - Checkpoint save logged at `2026-05-12 06:49:22`.
  - Epoch `2/12` started; iter `50/1000` logged loss `22.54`.
- Full baseline side eval progress: `2256/6019` samples, about `4.1 task/s`, ETA about `924s`.
- No side-eval metrics or camdrop comparison artifacts exist yet.
- Train stderr for `1268` remains empty.

## Progress Poll - 2026-05-12 09:58 UTC

- Active adaptive chain remains `1268 -> 1271 -> 1272`.
- Active side-eval chain remains `1273 -> 1274 -> 1275`.
- `1268` is RUNNING on `livenode03`, elapsed `35:33`.
- `1273` is RUNNING on `livenode02`, elapsed `17:37`.
- Full baseline side eval progress: `4147/6019` samples, about `4.1 task/s`, ETA about `460s`.
- Adaptive train latest log reached epoch `2/12`, iter `350/1000`.
- Adaptive epoch 2 loss observations:
  - iter `50/1000`: loss `22.54`.
  - iter `100/1000`: loss `22.01`.
  - iter `150/1000`: loss `22.60`.
  - iter `200/1000`: loss `24.15`.
  - iter `250/1000`: loss `23.29`.
  - iter `300/1000`: loss `23.34`.
  - iter `350/1000`: loss `22.87`.
- No side-eval metrics or camdrop comparison artifacts exist yet.

## Progress Poll - 2026-05-12 10:09 UTC

- Active adaptive chain remains `1268 -> 1271 -> 1272`.
- Active side-eval chain remains `1273 -> 1274 -> 1275`.
- `1268` is RUNNING on `livenode03`, elapsed `46:12`.
- `1273` is RUNNING on `livenode02`, elapsed `28:16`.
- Full baseline side eval has produced overall metrics and is now inside condition-split evaluation:
  - Overall mAP `0.5144223384`.
  - Overall NDS `0.5934656087`.
  - Overall artifacts:
    - `research/night_gen_phase1/results/full_baseline_epoch36_condition/eval/submission_overall/pts_bbox/results_nusc.json`.
    - `research/night_gen_phase1/results/full_baseline_epoch36_condition/eval/submission_overall/pts_bbox/metrics_summary.json`.
    - `research/night_gen_phase1/results/full_baseline_epoch36_condition/eval/submission_overall/pts_bbox/metrics_details.json`.
- Latest parsed side-eval condition progress: `3684/4449` samples in the current split-stage evaluation.
- Adaptive train latest log reached epoch `2/12`, iter `750/1000`.
- Adaptive epoch 2 remains stable:
  - iter `400/1000`: loss `23.28`.
  - iter `450/1000`: loss `22.37`.
  - iter `500/1000`: loss `23.06`.
  - iter `550/1000`: loss `22.31`.
  - iter `600/1000`: loss `22.53`.
  - iter `650/1000`: loss `21.47`.
  - iter `700/1000`: loss `21.72`.
  - iter `750/1000`: loss `21.69`.
- No camdrop metrics or full-dropout comparison artifacts exist yet.

## Result - Full Baseline Epoch36 Condition Eval - 2026-05-12 10:13 UTC

- Stage: `full_baseline_epoch36_condition`.
- Weights: `outputs/racformer_r50_nuimg_704x256_f8/2026-01-25/20-49-49/epoch_36.pth`.
- Job: `1273`.
- Status: completed successfully and released camdrop eval job `1274`.
- Metrics:
  - overall: mAP `0.5144223384`, NDS `0.5934656087`.
  - day: mAP `0.5118604584`, NDS `0.5933699446`.
  - night: mAP `0.3105706715`, NDS `0.3846498727`.
  - rain: mAP `0.5498265663`, NDS `0.6246686630`.
- Artifacts:
  - `research/night_gen_phase1/results/full_baseline_epoch36_condition/eval/submission_overall/pts_bbox/metrics_summary.json`.
  - `research/night_gen_phase1/results/full_baseline_epoch36_condition/eval/eval_day/metrics_summary.json`.
  - `research/night_gen_phase1/results/full_baseline_epoch36_condition/eval/eval_night/metrics_summary.json`.
  - `research/night_gen_phase1/results/full_baseline_epoch36_condition/eval/eval_rain/metrics_summary.json`.
  - `research/night_gen_phase1/results/full_baseline_epoch36_condition/eval/eval_by_condition.json`.
- Side-eval chain now: `1274 -> 1275`.
- `1274` camdrop20 eval is RUNNING on `livenode02`; latest observed progress `615/6019`, about `4.0 task/s`.
- Adaptive train job `1268` remains RUNNING on `livenode03`, latest parsed log at epoch `2/12`, iter `900/1000`, loss `21.43`.

## Progress Poll - 2026-05-12 10:28 UTC

- Active adaptive chain remains `1268 -> 1271 -> 1272`.
- Active side-eval chain remains `1274 -> 1275`.
- `1274` camdrop20 eval is RUNNING on `livenode02`, elapsed `17:47`.
- Camdrop20 full-val inference progress: `4191/6019`, about `4.1 task/s`, ETA about `451s`.
- No camdrop20 metrics or comparison artifacts exist yet.
- Adaptive train job `1268` is RUNNING on `livenode03`, elapsed `1:05:25`.
- Adaptive `epoch_2.pth` exists in `outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research/2026-05-12/06-23-27`.
- Adaptive train latest log reached epoch `3/12`, iter `450/1000`.
- Adaptive loss observations:
  - epoch `2/12`, iter `1000/1000`: loss `21.63`; checkpoint saved at epoch 2.
  - epoch `3/12`, iter `50/1000`: loss `20.99`.
  - epoch `3/12`, iter `100/1000`: loss `21.80`.
  - epoch `3/12`, iter `150/1000`: loss `20.13`.
  - epoch `3/12`, iter `200/1000`: loss `21.44`.
  - epoch `3/12`, iter `250/1000`: loss `22.01`.
  - epoch `3/12`, iter `300/1000`: loss `20.92`.
  - epoch `3/12`, iter `350/1000`: loss `25.21`.
  - epoch `3/12`, iter `400/1000`: loss `21.79`.
  - epoch `3/12`, iter `450/1000`: loss `21.73`.

## Result - Full CamDrop20 Epoch36 Side Eval - 2026-05-12 10:42 UTC

- Stage: `full_camdrop20_epoch36_condition`.
- Weights: `outputs/racformer_r50_nuimg_704x256_f8_dropout/2026-01-25/22-45-16/epoch_36.pth`.
- Jobs: `1274` eval, `1275` comparison summary.
- Status: completed.
- Comparison artifacts:
  - `research/night_gen_phase1/results/full_dropout_epoch36_comparison.md`.
  - `research/night_gen_phase1/results/full_dropout_epoch36_comparison.json`.
- Metrics vs full baseline epoch36:
  - day: baseline mAP/NDS `0.5119 / 0.5934`; camdrop20 `0.5137 / 0.5933`; delta `+0.19 pp / -0.01 pp`.
  - night: baseline `0.3106 / 0.3846`; camdrop20 `0.2828 / 0.3599`; delta `-2.78 pp / -2.48 pp`.
  - rain: baseline `0.5498 / 0.6247`; camdrop20 `0.5453 / 0.6265`; delta `-0.46 pp / +0.19 pp`.
  - overall: baseline `0.5144 / 0.5935`; camdrop20 `0.5149 / 0.5925`; delta `+0.05 pp / -0.10 pp`.
- Gate verdict: `FAIL`.
- Decision: do not pursue hard camera dropout as the next NB2 paper direction; it preserves day/overall but hurts the exact night robustness target.
- Resource state after side comparison: `livenode02` is free for another evidence-only checkpoint eval while adaptive NB2 train `1268` continues on `livenode03`.

## Submission - Full NightAug Epoch36 Side Eval - 2026-05-12 10:46 UTC

- Purpose: evaluate an existing full-training night-augmentation checkpoint on the same condition splits while `livenode02` is free. This is an evidence-only side branch, not the train2k NB2 gate result.
- Weights: `outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-02-01/01-41-59/epoch_36.pth`.
- Eval config: `configs/racformer_r50_nuimg_704x256_f8_nightaug.py` with `--full-val`.
- Reason for config choice: the nightaug config changes non-parameter model behavior (`d_region_list`), so evaluating with the matching config is safer than using the generic eval config.
- Validation before submission:
  - `bash -n` passed for eval and compare scripts.
  - `python -m py_compile` passed for `research/night_gen_phase1/staged_existing_nightaug_eval/summarize_full_nightaug_eval.py`.
  - The nightaug checkpoint strict-loads with the nightaug eval config.
- Staged files:
  - `research/night_gen_phase1/staged_existing_nightaug_eval/run_full_nightaug_epoch36_eval.sbatch`.
  - `research/night_gen_phase1/staged_existing_nightaug_eval/run_full_nightaug_epoch36_compare.sbatch`.
  - `research/night_gen_phase1/staged_existing_nightaug_eval/summarize_full_nightaug_eval.py`.
- Submitted chain:
  - Eval job `1276`.
  - Compare job `1277` afterok `1276`.
- Initial queue state: `1276` pending with reason `(None)`, `1277` dependency-held.

## Progress Poll - 2026-05-12 10:49 UTC

- Active adaptive chain remains `1268 -> 1271 -> 1272`.
- Active nightaug side-eval chain: `1276 -> 1277`.
- `1276` is RUNNING on `livenode02`, elapsed `2:28`.
- NightAug full-val inference progress: `457/6019`, about `4.0 task/s`, ETA about `1388s`.
- NightAug stderr shows normal eval startup:
  - config `configs/racformer_r50_nuimg_704x256_f8_nightaug.py`.
  - weights `outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-02-01/01-41-59/epoch_36.pth`.
  - full val pkl `nuscenes_infos_val_sweep.pkl`, no val cap.
- Adaptive train job `1268` is RUNNING on `livenode03`, elapsed `1:26:20`.
- Adaptive `epoch_3.pth` exists and epoch `4/12` has started.
- Latest adaptive train log reached epoch `4/12`, iter `250/1000`; recent losses:
  - epoch `3/12`, iter `1000/1000`: loss `20.19`.
  - epoch `4/12`, iter `50/1000`: loss `20.42`.
  - epoch `4/12`, iter `100/1000`: loss `20.42`.
  - epoch `4/12`, iter `150/1000`: loss `19.79`.
  - epoch `4/12`, iter `200/1000`: loss `19.27`.
  - epoch `4/12`, iter `250/1000`: loss `19.54`.

## Result - Full NightAug Epoch36 Side Eval - 2026-05-12 11:19 UTC

- Stage: `full_nightaug_epoch36_condition`.
- Weights: `outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-02-01/01-41-59/epoch_36.pth`.
- Jobs: `1276` eval, `1277` comparison summary.
- Status: completed.
- Comparison artifacts:
  - `research/night_gen_phase1/results/full_nightaug_epoch36_comparison.md`.
  - `research/night_gen_phase1/results/full_nightaug_epoch36_comparison.json`.
- Metrics vs full baseline epoch36:
  - day: baseline mAP/NDS `0.5119 / 0.5934`; nightaug `0.4937 / 0.5808`; delta `-1.81 pp / -1.26 pp`.
  - night: baseline `0.3106 / 0.3846`; nightaug `0.3123 / 0.3871`; delta `+0.17 pp / +0.24 pp`.
  - rain: baseline `0.5498 / 0.6247`; nightaug `0.5238 / 0.6123`; delta `-2.60 pp / -1.23 pp`.
  - overall: baseline `0.5144 / 0.5935`; nightaug `0.4938 / 0.5813`; delta `-2.06 pp / -1.22 pp`.
- Gate verdict: `FAIL`.
- Decision: do not use the existing full NightAug branch as the next NB2 direction. It barely improves night and loses too much day/rain/overall accuracy.

## Progress Poll - 2026-05-12 11:19 UTC

- Active adaptive chain remains `1268 -> 1271 -> 1272`.
- NightAug side-eval chain `1276 -> 1277` completed and is no longer in the queue.
- Queue state:
  - `1268` adaptive train RUNNING on `livenode03`, elapsed `1:56:08`.
  - `1271` adaptive full-val condition eval PENDING on dependency.
  - `1272` adaptive summary PENDING on dependency.
- Adaptive train latest log reached epoch `5/12`, iter `450/1000`.
- Adaptive checkpoint state:
  - `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, and `epoch_4.pth` exist in `outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research/2026-05-12/06-23-27`.
- Recent adaptive loss observations:
  - epoch `4/12`, iter `1000/1000`: loss `19.62`; checkpoint saved at epoch 4.
  - epoch `5/12`, iter `50/1000`: loss `19.36`.
  - epoch `5/12`, iter `100/1000`: loss `19.86`.
  - epoch `5/12`, iter `150/1000`: loss `19.48`.
  - epoch `5/12`, iter `200/1000`: loss `19.98`.
  - epoch `5/12`, iter `250/1000`: loss `19.46`.
  - epoch `5/12`, iter `300/1000`: loss `19.32`.
  - epoch `5/12`, iter `350/1000`: loss `19.33`.
  - epoch `5/12`, iter `400/1000`: loss `18.42`.
  - epoch `5/12`, iter `450/1000`: loss `18.67`.
- Current side-branch conclusion:
  - DriveGEN remains rejected for scaling because visual geometry preservation is poor.
  - Full CamDrop20 failed because it hurts night.
  - Full NightAug failed because the night gain is tiny and day/rain/overall regress too much.
  - Keep the active NB2 adaptive-fusion train/eval chain as the only currently promising branch.

## Submission - Full 3Cam3Rad Epoch36 Side Eval - 2026-05-12 11:27 UTC

- Purpose: evaluate the only remaining existing full-training checkpoint while `livenode02` is free. This is a bounded side-evidence check for whether a front-sensor-only model improves adverse-condition robustness; it is not a train2k NB2 gate result.
- Weights: `outputs/racformer_r50_nuimg_704x256_f8_3cam_3rad/2025-12-13/10-19-58/epoch_36.pth`.
- Eval config: `configs/racformer_r50_nuimg_704x256_f8_3cam_3rad.py` with `--full-val`.
- Validation before submission:
  - `bash -n` passed for eval and compare sbatch files.
  - `python -m py_compile` passed for `research/night_gen_phase1/staged_existing_3cam3rad_eval/summarize_full_3cam3rad_eval.py`.
  - Config parse confirmed `num_cams=3` in both model and transformer config, full-val ann file `nuscenes_infos_val_sweep.pkl`, and no validation cap.
  - Strict checkpoint load was not run on the login context because importing dataset loaders requires the compute-node NuScenes mount; the eval job itself will fail fast under the same strict `load_checkpoint(..., strict=True)` path if shapes do not match.
- Staged files:
  - `research/night_gen_phase1/staged_existing_3cam3rad_eval/run_full_3cam3rad_epoch36_eval.sbatch`.
  - `research/night_gen_phase1/staged_existing_3cam3rad_eval/run_full_3cam3rad_epoch36_compare.sbatch`.
  - `research/night_gen_phase1/staged_existing_3cam3rad_eval/summarize_full_3cam3rad_eval.py`.
- Submitted chain:
  - Eval job `1278`.
  - Compare job `1279` afterok `1278`.
- Initial queue state:
  - `1278` pending with reason `(None)` on `livenode02`.
  - `1279` dependency-held.
  - Adaptive NB2 chain remains `1268 -> 1271 -> 1272` on `livenode03`.

## Result - Full 3Cam3Rad Epoch36 Side Eval - 2026-05-12 11:51 UTC

- Stage: `full_3cam3rad_epoch36_condition`.
- Weights: `outputs/racformer_r50_nuimg_704x256_f8_3cam_3rad/2025-12-13/10-19-58/epoch_36.pth`.
- Jobs: `1278` eval, `1279` comparison summary.
- Status: completed.
- Comparison artifacts:
  - `research/night_gen_phase1/results/full_3cam3rad_epoch36_comparison.md`.
  - `research/night_gen_phase1/results/full_3cam3rad_epoch36_comparison.json`.
- Metrics vs full baseline epoch36:
  - day: baseline mAP/NDS `0.5119 / 0.5934`; 3Cam3Rad `0.3508 / 0.4951`; delta `-16.10 pp / -9.82 pp`.
  - night: baseline `0.3106 / 0.3846`; 3Cam3Rad `0.2652 / 0.3439`; delta `-4.54 pp / -4.07 pp`.
  - rain: baseline `0.5498 / 0.6247`; 3Cam3Rad `0.4117 / 0.5422`; delta `-13.81 pp / -8.25 pp`.
  - overall: baseline `0.5144 / 0.5935`; 3Cam3Rad `0.3583 / 0.4961`; delta `-15.61 pp / -9.73 pp`.
- Gate verdict: `FAIL`.
- Decision: do not use the front-sensor-only full checkpoint as a robustness direction. It severely reduces all splits, including the night target.

## Progress Poll - 2026-05-12 11:51 UTC

- Only active jobs now are the adaptive-fusion NB2 chain:
  - `1268` adaptive train RUNNING on `livenode03`, elapsed `2:27:53`.
  - `1271` adaptive full-val condition eval PENDING on dependency.
  - `1272` adaptive summary PENDING on dependency.
- Adaptive checkpoint state: `epoch_1.pth` through `epoch_5.pth` exist in `outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research/2026-05-12/06-23-27`.
- Latest adaptive train log reached epoch `6/12`, iter `650/1000`.
- Recent adaptive loss observations:
  - epoch `5/12`, iter `1000/1000`: loss `19.21`; checkpoint saved at epoch 5.
  - epoch `6/12`, iter `50/1000`: loss `18.82`.
  - epoch `6/12`, iter `100/1000`: loss `17.76`.
  - epoch `6/12`, iter `150/1000`: loss `18.91`.
  - epoch `6/12`, iter `200/1000`: loss `18.67`.
  - epoch `6/12`, iter `250/1000`: loss `17.90`.
  - epoch `6/12`, iter `300/1000`: loss `19.07`.
  - epoch `6/12`, iter `350/1000`: loss `18.58`.
  - epoch `6/12`, iter `400/1000`: loss `18.43`.
  - epoch `6/12`, iter `450/1000`: loss `18.44`.
  - epoch `6/12`, iter `500/1000`: loss `18.96`.
  - epoch `6/12`, iter `550/1000`: loss `18.79`.
  - epoch `6/12`, iter `600/1000`: loss `18.44`.
  - epoch `6/12`, iter `650/1000`: loss `18.31`.
- Side-check conclusion after evaluating all existing full checkpoints:
  - CamDrop20: failed; hurts night.
  - NightAug: failed; tiny night gain, too much day/rain/overall loss.
  - 3Cam3Rad: failed; severe all-split regression.
  - Do not launch more old-checkpoint side evals without a new specific hypothesis. Wait for adaptive-fusion NB2 train/eval results.

## Progress Poll - 2026-05-12 11:57 UTC

- Active adaptive-fusion NB2 chain remains `1268 -> 1271 -> 1272`.
- Queue state:
  - `1268` adaptive train RUNNING on `livenode03`, elapsed `2:34:33`.
  - `1271` adaptive full-val condition eval PENDING on dependency.
  - `1272` adaptive summary PENDING on dependency.
- `livenode02` is idle. Decision: keep it idle for now because all old full-checkpoint side branches are exhausted and the tracker already says not to launch more without a new specific hypothesis.
- Adaptive train latest log reached epoch `6/12`, iter `900/1000`.
- Adaptive checkpoint state: `epoch_1.pth` through `epoch_5.pth` exist.
- Recent adaptive loss observations:
  - epoch `6/12`, iter `700/1000`: loss `17.51`.
  - epoch `6/12`, iter `750/1000`: loss `27.16`.
  - epoch `6/12`, iter `800/1000`: loss `18.13`.
  - epoch `6/12`, iter `850/1000`: loss `18.34`.
  - epoch `6/12`, iter `900/1000`: loss `18.74`.
- Watch item: one isolated epoch-6 spike at iter `750/1000`, with immediate recovery in the next logged iterations and no stderr output. No intervention.
- Rechecked adaptive eval/summary scripts:
  - Eval waits for `epoch_12.pth` under `outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research/*/*`.
  - Eval uses `configs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research.py`, so strict checkpoint loading should include `fusion_gate` parameters.
  - Summary reports both gate vs S0 and gate vs corrected w05.

## Progress Poll - 2026-05-12 12:14 UTC

- Active adaptive-fusion NB2 chain remains `1268 -> 1271 -> 1272`.
- Queue state:
  - `1268` adaptive train RUNNING on `livenode03`, elapsed `2:51:16`.
  - `1271` adaptive full-val condition eval PENDING on dependency.
  - `1272` adaptive summary PENDING on dependency.
- Adaptive checkpoint state: `epoch_1.pth` through `epoch_6.pth` exist.
- Latest adaptive train log reached epoch `7/12`, iter `550/1000`.
- Recent adaptive loss observations:
  - epoch `6/12`, iter `1000/1000`: loss `19.55`; checkpoint saved at epoch 6.
  - epoch `7/12`, iter `50/1000`: loss `16.39`.
  - epoch `7/12`, iter `100/1000`: loss `17.98`.
  - epoch `7/12`, iter `150/1000`: loss `18.40`.
  - epoch `7/12`, iter `200/1000`: loss `19.29`.
  - epoch `7/12`, iter `250/1000`: loss `17.33`.
  - epoch `7/12`, iter `300/1000`: loss `18.37`.
  - epoch `7/12`, iter `350/1000`: loss `17.62`.
  - epoch `7/12`, iter `400/1000`: loss `17.09`.
  - epoch `7/12`, iter `450/1000`: loss `18.56`.
  - epoch `7/12`, iter `500/1000`: loss `17.41`.
  - epoch `7/12`, iter `550/1000`: loss `34.35`.
- Watch item: second isolated spike, now at epoch `7/12`, iter `550/1000`; no stderr output and no intervention unless the pattern persists or the run fails.

## Progress Poll - 2026-05-12 12:45 UTC

- Active adaptive-fusion NB2 chain remains `1268 -> 1271 -> 1272`.
- Queue state:
  - `1268` adaptive train RUNNING on `livenode03`, elapsed `3:22:35`.
  - `1271` adaptive full-val condition eval PENDING on dependency.
  - `1272` adaptive summary PENDING on dependency.
- Adaptive checkpoint state: `epoch_1.pth` through `epoch_7.pth` exist.
- Latest adaptive train log reached epoch `8/12`, iter `750/1000`.
- Recent adaptive loss observations:
  - epoch `7/12`, iter `1000/1000`: loss `17.20`; checkpoint saved at epoch 7.
  - epoch `8/12`, iter `50/1000`: loss `17.19`.
  - epoch `8/12`, iter `100/1000`: loss `18.20`.
  - epoch `8/12`, iter `150/1000`: loss `17.60`.
  - epoch `8/12`, iter `200/1000`: loss `16.45`.
  - epoch `8/12`, iter `250/1000`: loss `17.73`.
  - epoch `8/12`, iter `300/1000`: loss `18.78`.
  - epoch `8/12`, iter `350/1000`: loss `16.77`.
  - epoch `8/12`, iter `400/1000`: loss `17.72`.
  - epoch `8/12`, iter `450/1000`: loss `16.52`.
  - epoch `8/12`, iter `500/1000`: loss `17.66`.
  - epoch `8/12`, iter `550/1000`: loss `17.01`.
  - epoch `8/12`, iter `600/1000`: loss `17.88`.
  - epoch `8/12`, iter `650/1000`: loss `17.05`.
  - epoch `8/12`, iter `700/1000`: loss `16.86`.
  - epoch `8/12`, iter `750/1000`: loss `18.34`.
- No new stderr output. The prior epoch-7 spike recovered and epoch 8 has been stable so far.

## Progress Poll - 2026-05-12 13:07 UTC

- Active adaptive-fusion NB2 chain remains `1268 -> 1271 -> 1272`.
- Queue state:
  - `1268` adaptive train RUNNING on `livenode03`, elapsed `3:44:07`.
  - `1271` adaptive full-val condition eval PENDING on dependency.
  - `1272` adaptive summary PENDING on dependency.
- Adaptive checkpoint state: `epoch_1.pth` through `epoch_8.pth` exist.
- Latest adaptive train log reached epoch `9/12`, iter `550/1000`.
- Recent adaptive loss observations:
  - epoch `8/12`, iter `1000/1000`: loss `17.44`; checkpoint saved at epoch 8.
  - epoch `9/12`, iter `50/1000`: loss `16.56`.
  - epoch `9/12`, iter `100/1000`: loss `34.02`.
  - epoch `9/12`, iter `150/1000`: loss `16.92`.
  - epoch `9/12`, iter `200/1000`: loss `16.43`.
  - epoch `9/12`, iter `250/1000`: loss `17.43`.
  - epoch `9/12`, iter `300/1000`: loss `18.00`.
  - epoch `9/12`, iter `350/1000`: loss `17.15`.
  - epoch `9/12`, iter `400/1000`: loss `15.95`.
  - epoch `9/12`, iter `450/1000`: loss `16.39`.
  - epoch `9/12`, iter `500/1000`: loss `16.61`.
  - epoch `9/12`, iter `550/1000`: loss `16.02`.
- Watch item: another isolated spike at epoch `9/12`, iter `100/1000`, with immediate recovery. Stderr remains empty.

## Progress Poll - 2026-05-12 13:15 UTC

- Local tracker mirror moved out of `/tmp` to `/home/gabriel/LIVE/RACFORMER_NB2_EXPERIMENT_TRACKER.md`.
- Remote canonical tracker remains `research/night_gen_phase1/RACFORMER_NB2_EXPERIMENT_TRACKER.md`.
- Active adaptive-fusion NB2 chain remains `1268 -> 1271 -> 1272`.
- Queue state:
  - `1268` adaptive train RUNNING on `livenode03`, elapsed `3:52:27`.
  - `1271` adaptive full-val condition eval PENDING on dependency.
  - `1272` adaptive summary PENDING on dependency.
- Adaptive checkpoint state: `epoch_1.pth` through `epoch_8.pth` exist.
- Latest adaptive train log reached epoch `9/12`, iter `900/1000`.
- Recent adaptive loss observations:
  - epoch `9/12`, iter `600/1000`: loss `16.10`.
  - epoch `9/12`, iter `650/1000`: loss `17.18`.
  - epoch `9/12`, iter `700/1000`: loss `16.81`.
  - epoch `9/12`, iter `750/1000`: loss `17.11`.
  - epoch `9/12`, iter `800/1000`: loss `17.29`.
  - epoch `9/12`, iter `850/1000`: loss `17.36`.
  - epoch `9/12`, iter `900/1000`: loss `17.54`.
- No new stderr output; epoch 9 remains stable after the earlier isolated spike.

## Progress Poll - 2026-05-12 13:33 UTC

- Active adaptive-fusion NB2 chain remains `1268 -> 1271 -> 1272`.
- Queue state:
  - `1268` adaptive train RUNNING on `livenode03`, elapsed `4:09:57`.
  - `1271` adaptive full-val condition eval PENDING on dependency.
  - `1272` adaptive summary PENDING on dependency.
- Adaptive checkpoint state: `epoch_1.pth` through `epoch_9.pth` exist.
- Latest adaptive train log reached epoch `10/12`, iter `550/1000`.
- Recent adaptive loss observations:
  - epoch `9/12`, iter `1000/1000`: loss `16.96`; checkpoint saved at epoch 9.
  - epoch `10/12`, iter `50/1000`: loss `17.05`.
  - epoch `10/12`, iter `100/1000`: loss `15.99`.
  - epoch `10/12`, iter `150/1000`: loss `17.27`.
  - epoch `10/12`, iter `200/1000`: loss `16.35`.
  - epoch `10/12`, iter `250/1000`: loss `54.29`.
  - epoch `10/12`, iter `300/1000`: loss `15.67`.
  - epoch `10/12`, iter `350/1000`: loss `17.17`.
  - epoch `10/12`, iter `400/1000`: loss `16.47`.
  - epoch `10/12`, iter `450/1000`: loss `16.50`.
  - epoch `10/12`, iter `500/1000`: loss `16.61`.
  - epoch `10/12`, iter `550/1000`: loss `15.58`.
- Watch item: large isolated spike at epoch `10/12`, iter `250/1000`, driven by bbox terms in the logged line, with immediate recovery. No stderr output.

## Progress Poll - 2026-05-12 13:49 UTC

- Active adaptive-fusion NB2 chain remains `1268 -> 1271 -> 1272`.
- Queue state:
  - `1268` adaptive train RUNNING on `livenode03`, elapsed `4:25:31`.
  - `1271` adaptive full-val condition eval PENDING on dependency.
  - `1272` adaptive summary PENDING on dependency.
- Adaptive checkpoint state: `epoch_1.pth` through `epoch_10.pth` exist.
- Latest adaptive train log reached epoch `11/12`, iter `150/1000`.
- Recent adaptive loss observations:
  - epoch `10/12`, iter `700/1000`: loss `16.78`.
  - epoch `10/12`, iter `750/1000`: loss `16.41`.
  - epoch `10/12`, iter `800/1000`: loss `16.13`.
  - epoch `10/12`, iter `850/1000`: loss `16.51`.
  - epoch `10/12`, iter `900/1000`: loss `20.39`.
  - epoch `10/12`, iter `950/1000`: loss `15.97`.
  - epoch `10/12`, iter `1000/1000`: loss `17.66`; checkpoint saved at epoch 10.
  - epoch `11/12`, iter `50/1000`: loss `16.79`.
  - epoch `11/12`, iter `100/1000`: loss `16.21`.
  - epoch `11/12`, iter `150/1000`: loss `15.21`.
- No new stderr output. Epoch 10 checkpointing completed normally.

## Progress Poll - 2026-05-12 14:11 UTC

- Active adaptive-fusion NB2 chain remains `1268 -> 1271 -> 1272`.
- Queue state:
  - `1268` adaptive train RUNNING on `livenode03`, elapsed `4:48:43`.
  - `1271` adaptive full-val condition eval PENDING on dependency.
  - `1272` adaptive summary PENDING on dependency.
- Adaptive checkpoint state: `epoch_1.pth` through `epoch_11.pth` exist.
- Latest adaptive train log reached epoch `12/12`, iter `50/1000`.
- Recent adaptive loss observations:
  - epoch `11/12`, iter `900/1000`: loss `16.05`.
  - epoch `11/12`, iter `950/1000`: loss `15.75`.
  - epoch `11/12`, iter `1000/1000`: loss `26.38`; checkpoint saved at epoch 11.
  - epoch `12/12`, iter `50/1000`: loss `16.64`.
- No stderr output. Epoch 11 checkpointing completed normally; the end-of-epoch loss spike recovered at the epoch 12 start.

## Progress Poll - 2026-05-12 14:16 UTC

- Active adaptive-fusion NB2 chain remains `1268 -> 1271 -> 1272`.
- Queue state:
  - `1268` adaptive train RUNNING on `livenode03`, elapsed `4:52:55`.
  - `1271` adaptive full-val condition eval PENDING on dependency.
  - `1272` adaptive summary PENDING on dependency.
- Adaptive checkpoint state: `epoch_1.pth` through `epoch_11.pth` exist.
- Latest adaptive train log reached epoch `12/12`, iter `250/1000`.
- Recent adaptive loss observations:
  - epoch `12/12`, iter `50/1000`: loss `16.64`.
  - epoch `12/12`, iter `100/1000`: loss `16.62`.
  - epoch `12/12`, iter `150/1000`: loss `15.59`.
  - epoch `12/12`, iter `200/1000`: loss `15.80`.
  - epoch `12/12`, iter `250/1000`: loss `15.72`.
- No stderr output. Eval remains dependency-blocked until train exits after epoch 12.

## Progress Poll - 2026-05-12 14:28 UTC

- Active adaptive-fusion NB2 chain remains `1268 -> 1271 -> 1272`.
- Queue state:
  - `1268` adaptive train RUNNING on `livenode03`, elapsed `5:05:23`.
  - `1271` adaptive full-val condition eval PENDING on dependency.
  - `1272` adaptive summary PENDING on dependency.
- Adaptive checkpoint state: `epoch_1.pth` through `epoch_11.pth` exist.
- Latest adaptive train log reached epoch `12/12`, iter `650/1000`.
- Recent adaptive loss observations:
  - epoch `12/12`, iter `300/1000`: loss `16.36`.
  - epoch `12/12`, iter `350/1000`: loss `15.63`.
  - epoch `12/12`, iter `400/1000`: loss `16.20`.
  - epoch `12/12`, iter `450/1000`: loss `16.49`.
  - epoch `12/12`, iter `500/1000`: loss `23.67`.
  - epoch `12/12`, iter `550/1000`: loss `15.84`.
  - epoch `12/12`, iter `600/1000`: loss `29.61`.
  - epoch `12/12`, iter `650/1000`: loss `16.46`.
- Watch item: two isolated epoch-12 spikes at iters `500` and `600`, with immediate recovery. No stderr output.

## Progress Poll - 2026-05-12 14:39 UTC

- Adaptive train job `1268` completed and released the dependency.
- Queue state:
  - `1271` adaptive full-val condition eval RUNNING on `livenode03`, elapsed `1:03`.
  - `1272` adaptive summary PENDING on dependency.
- Adaptive checkpoint state: final `epoch_12.pth` exists at
  `outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research/2026-05-12/06-23-27/epoch_12.pth`.
- Final train loss observations:
  - epoch `12/12`, iter `700/1000`: loss `16.64`.
  - epoch `12/12`, iter `750/1000`: loss `15.54`.
  - epoch `12/12`, iter `800/1000`: loss `15.77`.
  - epoch `12/12`, iter `850/1000`: loss `15.96`.
  - epoch `12/12`, iter `900/1000`: loss `15.68`.
  - epoch `12/12`, iter `950/1000`: loss `16.06`.
  - epoch `12/12`, iter `1000/1000`: loss `15.67`.
- Eval command is using `configs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research.py`
  with `epoch_12.pth`; it started inference on `6019` validation samples.
- Train stderr contains only tqdm/progress-bar output from final validation bookkeeping, not a traceback.

## Progress Poll - 2026-05-12 14:50 UTC

- Queue state:
  - `1271` adaptive full-val condition eval RUNNING on `livenode03`, elapsed `11:57`.
  - `1272` adaptive summary PENDING on dependency.
- Eval progress: stdout shows roughly `2730/6019` validation samples processed at about `4.0` samples/sec, with ETA about `14` minutes.
- Eval stderr still contains only startup/info lines:
  - config is `configs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research.py`.
  - weights are final adaptive `epoch_12.pth`.
  - full validation size is `6019` samples.
- No result metrics files yet; wait for eval completion and summary job `1272`.

## Progress Poll - 2026-05-12 15:06 UTC

- Queue state:
  - `1271` adaptive full-val condition eval RUNNING on `livenode03`, elapsed `27:59`.
  - `1272` adaptive summary PENDING on dependency.
- Eval artifacts:
  - `eval/submission_overall/pts_bbox/results_nusc.json` has been written.
  - `eval_slurm_1271.out` shows the validation pass near completion/end (`~5960/6019` in the sampled tail).
- No final `summary_metrics.md` yet. Likely waiting on nuScenes metric computation and/or remaining condition-split evaluation.
- Eval stderr still has only startup/info lines, no traceback.

## Result - S3 seed20260425 ratio18p75 w05 adaptive fusion - 2026-05-12 15:11 UTC

- Eval job `1271` completed and summary job `1272` wrote:
  `research/night_gen_phase1/results/S3_seed20260425_ratio18p75_w05_adaptfusion/summary_metrics.md`.
- Metrics:
  - day: mAP `0.3077072813`, NDS `0.3714759683`; vs S0 mAP `-0.76 pp`, NDS `-0.31 pp`.
  - night: mAP `0.1598210056`, NDS `0.2211167581`; vs S0 mAP `+1.10 pp`, NDS `+0.60 pp`.
  - rain: mAP `0.2717316245`, NDS `0.3708414365`; vs S0 mAP `-0.26 pp`, NDS `-0.05 pp`.
  - overall: mAP `0.2989939713`, NDS `0.3679375857`; vs S0 mAP `-0.50 pp`, NDS `-0.18 pp`.
- Gate verdict vs S0: PASS.
  - night mAP clears +1.0 pp by a narrow margin (`+1.10 pp`).
  - day and overall degradation stay well inside the gate.
  - night NDS improves rather than regressing.
- Gate verdict vs corrected w05: PASS.
  - night mAP `+0.83 pp`, night NDS `+1.19 pp`, rain mAP/NDS positive, overall NDS positive.
  - overall mAP is effectively flat vs w05 (`-0.03 pp`).
- Decision: promising, defensible branch. Do not claim paper result from a single seed; next best action is replication on the held-out seed/config before adding new mechanisms.

## Staged Replication - S3 seed20260502 ratio18p75 w05 adaptive fusion - 2026-05-12 15:20 UTC

- Rationale: seed20260425 adaptive fusion passes the NB2 gate, but the earlier non-adaptive S3 pass failed seed replication.
  A second-seed replication is the highest-value next check before adding any new mechanism.
- New remote files staged:
  - `configs/racformer_train2k_genaug_seed20260502_ratio18p75_w05_research.py`
  - `configs/racformer_train2k_genaug_seed20260502_ratio18p75_w05_adaptfusion_research.py`
  - `research/night_gen_phase1/staged_adaptfusion/run_t12_adaptfusion_seed20260502_ratio18p75_w05.sbatch`
  - `research/night_gen_phase1/staged_adaptfusion/run_t12_adaptfusion_seed20260502_ratio18p75_w05_eval.sbatch`
  - `research/night_gen_phase1/staged_adaptfusion/run_t12_adaptfusion_seed20260502_ratio18p75_w05_summary.sbatch`
  - `research/night_gen_phase1/staged_adaptfusion/summarize_adaptfusion_seed20260502.py`
- Config validation:
  - Manifest path resolves to `research/night_gen_phase1/manifests/phase1_t10_seed20260502_ratio18p75_generated.json`.
  - `generated_sample_weight=0.5`.
  - `adaptive_fusion_gate=True`.
  - `total_epochs=12`.
  - sbatch scripts pass `bash -n`; summarizer passes `py_compile`.
- Reference for the seed20260502 replication summary is the existing non-adaptive
  `S3_seed20260502_ratio18p75`, since there is no same-seed w05-only ablation yet.
- Submission correction:
  - A malformed embedded `awk` submit command accidentally created train job `1282` before failing.
  - A retry created duplicate pending chain `1283 -> 1284 -> 1285`.
  - Duplicate pending jobs `1283`, `1284`, and `1285` were cancelled before running.
- Active replication chain:
  - `1282` train RUNNING on `livenode03`.
  - `1286` full-val condition eval PENDING afterok `1282`.
  - `1287` summary PENDING afterok `1286`.
- Early train log: normal model construction/pretrain load; stderr empty.

## Progress Poll - S3 seed20260502 ratio18p75 w05 adaptive fusion - 2026-05-12 15:33 UTC

- Active replication chain remains `1282 -> 1286 -> 1287`.
- Queue state:
  - `1282` train RUNNING on `livenode03`, elapsed `12:43`.
  - `1286` full-val condition eval PENDING on dependency.
  - `1287` summary PENDING on dependency.
- Work dir: `outputs/racformer_train2k_genaug_seed20260502_ratio18p75_w05_adaptfusion_research/2026-05-12/12-21-12`.
- Latest train log reached epoch `1/12`, iter `450/1000`.
- Early loss observations:
  - epoch `1/12`, iter `50/1000`: loss `44.96`.
  - epoch `1/12`, iter `100/1000`: loss `36.11`.
  - epoch `1/12`, iter `150/1000`: loss `32.26`.
  - epoch `1/12`, iter `200/1000`: loss `29.73`.
  - epoch `1/12`, iter `250/1000`: loss `29.43`.
  - epoch `1/12`, iter `300/1000`: loss `27.03`.
  - epoch `1/12`, iter `350/1000`: loss `25.07`.
  - epoch `1/12`, iter `400/1000`: loss `28.28`.
  - epoch `1/12`, iter `450/1000`: loss `25.56`.
- Manifest loader first-six probe showed `0 hits / 6 misses`, but this is not a key/path failure:
  - manifest has `2244` usable camera entries.
  - overlap with `nuscenes_infos_train_2k_day.pkl`: `2244/12000` camera slots, `374` samples.
  - all `2244` generated paths exist.
  - first train sample has zero selected cameras for this seed, explaining the early miss-only probe.
- Train stderr remains empty.

## Progress Poll - S3 seed20260502 ratio18p75 w05 adaptive fusion - 2026-05-12 15:51 UTC

- Active replication chain remains `1282 -> 1286 -> 1287`.
- Queue state:
  - `1282` train RUNNING on `livenode03`, elapsed `30:45`.
  - `1286` full-val condition eval PENDING on dependency.
  - `1287` summary PENDING on dependency.
- Checkpoint state: `epoch_1.pth` exists.
- Latest train log reached epoch `2/12`, iter `150/1000`.
- Recent loss observations:
  - epoch `1/12`, iter `750/1000`: loss `22.92`.
  - epoch `1/12`, iter `800/1000`: loss `24.30`.
  - epoch `1/12`, iter `850/1000`: loss `24.46`.
  - epoch `1/12`, iter `900/1000`: loss `24.87`.
  - epoch `1/12`, iter `950/1000`: loss `23.44`.
  - epoch `1/12`, iter `1000/1000`: loss `30.39`; checkpoint saved at epoch 1.
  - epoch `2/12`, iter `50/1000`: loss `23.31`.
  - epoch `2/12`, iter `100/1000`: loss `23.18`.
  - epoch `2/12`, iter `150/1000`: loss `22.38`.
- No stderr output. Epoch-1 end spike recovered at epoch 2 start.

## Progress Poll - S3 seed20260502 ratio18p75 w05 adaptive fusion - 2026-05-12 16:23 UTC

- Active replication chain remains `1282 -> 1286 -> 1287`.
- Queue state:
  - `1282` train RUNNING on `livenode03`, elapsed `1:02:05`.
  - `1286` full-val condition eval PENDING on dependency.
  - `1287` summary PENDING on dependency.
- Checkpoint state: `epoch_1.pth` and `epoch_2.pth` exist.
- Latest train log reached epoch `3/12`, iter `350/1000`.
- Recent loss observations:
  - epoch `2/12`, iter `800/1000`: loss `22.01`.
  - epoch `2/12`, iter `850/1000`: loss `22.30`.
  - epoch `2/12`, iter `900/1000`: loss `21.74`.
  - epoch `2/12`, iter `950/1000`: loss `21.27`.
  - epoch `2/12`, iter `1000/1000`: loss `21.73`; checkpoint saved at epoch 2.
  - epoch `3/12`, iter `50/1000`: loss `21.04`.
  - epoch `3/12`, iter `100/1000`: loss `21.96`.
  - epoch `3/12`, iter `150/1000`: loss `21.04`.
  - epoch `3/12`, iter `200/1000`: loss `21.66`.
  - epoch `3/12`, iter `250/1000`: loss `22.05`.
  - epoch `3/12`, iter `300/1000`: loss `20.12`.
  - epoch `3/12`, iter `350/1000`: loss `25.84`.
- Watch item: one epoch-3 loss spike at iter `350`; no stderr output.

## Progress Poll - S3 seed20260502 ratio18p75 w05 adaptive fusion - 2026-05-12 17:24 UTC

- Active replication chain remains `1282 -> 1286 -> 1287`.
- Queue state:
  - `1282` train RUNNING on `livenode03`, elapsed `2:03:06`.
  - `1286` full-val condition eval PENDING on dependency.
  - `1287` summary PENDING on dependency.
- Checkpoint state: `epoch_1.pth` through `epoch_4.pth` exist.
- Latest train log reached epoch `5/12`, iter `700/1000`.
- Recent loss observations:
  - epoch `4/12`, iter `650/1000`: loss `28.18`.
  - epoch `4/12`, iter `700/1000`: loss `19.90`.
  - epoch `4/12`, iter `750/1000`: loss `18.61`.
  - epoch `4/12`, iter `800/1000`: loss `19.18`.
  - epoch `4/12`, iter `850/1000`: loss `19.91`.
  - epoch `4/12`, iter `900/1000`: loss `30.40`.
  - epoch `4/12`, iter `950/1000`: loss `20.17`.
  - epoch `4/12`, iter `1000/1000`: loss `20.07`; checkpoint saved at epoch 4.
  - epoch `5/12`, iter `500/1000`: loss `32.80`.
  - epoch `5/12`, iter `550/1000`: loss `19.31`.
  - epoch `5/12`, iter `600/1000`: loss `19.04`.
  - epoch `5/12`, iter `650/1000`: loss `19.51`.
  - epoch `5/12`, iter `700/1000`: loss `19.59`.
- Watch item: isolated spikes at epoch `4/12` iters `650` and `900`, and epoch `5/12` iter `500`, each with immediate recovery. No stderr output.

## Tracker Sync Incident - 2026-05-12 17:30 UTC

- Full-file `ssh_upload` attempts for the tracker timed out and left the remote canonical tracker truncated at `32768` bytes.
- Local repo-root tracker remained intact at `/home/gabriel/LIVE/RACFORMER_NB2_EXPERIMENT_TRACKER.md` (`129577` bytes before this note).
- Restored remote canonical tracker by uploading base64 chunks under
  `research/night_gen_phase1/tracker_restore_chunks/`, decoding to a temp file, verifying byte count, and then replacing the canonical tracker.
- Remote canonical tracker after restore: `129577` bytes and contains the 2026-05-12 17:24 UTC replication poll.
- Backup of the truncated remote copy retained at
  `research/night_gen_phase1/RACFORMER_NB2_EXPERIMENT_TRACKER.md.truncated_20260512_1726`.
- Operational note: avoid full-file `ssh_upload` for this tracker if the connection is flaky; use small append updates or chunked restore.

## Progress Poll - S3 seed20260502 ratio18p75 w05 adaptive fusion - 2026-05-12 18:33 UTC

- Active replication chain remains `1282 -> 1286 -> 1287`.
- Queue state:
  - `1282` train RUNNING on `livenode03`, elapsed `3:11:46`.
  - `1286` full-val condition eval PENDING on dependency.
  - `1287` summary PENDING on dependency.
- Checkpoint state: `epoch_1.pth` through `epoch_7.pth` exist.
- Latest train log reached epoch `8/12`, iter `300/1000`.
- Recent loss observations:
  - epoch `6/12`, iter `750/1000`: loss `27.93`.
  - epoch `6/12`, iter `800/1000`: loss `18.24`.
  - epoch `6/12`, iter `850/1000`: loss `18.25`.
  - epoch `6/12`, iter `900/1000`: loss `18.91`.
  - epoch `6/12`, iter `950/1000`: loss `19.20`.
  - epoch `6/12`, iter `1000/1000`: loss `18.72`; checkpoint saved at epoch 6.
  - epoch `7/12`, iter `550/1000`: loss `33.48`.
  - epoch `7/12`, iter `600/1000`: loss `18.02`.
  - epoch `7/12`, iter `650/1000`: loss `18.16`.
  - epoch `7/12`, iter `1000/1000`: loss `17.81`; checkpoint saved at epoch 7.
  - epoch `8/12`, iter `50/1000`: loss `17.57`.
  - epoch `8/12`, iter `100/1000`: loss `18.06`.
  - epoch `8/12`, iter `150/1000`: loss `17.24`.
  - epoch `8/12`, iter `200/1000`: loss `16.71`.
  - epoch `8/12`, iter `250/1000`: loss `18.41`.
  - epoch `8/12`, iter `300/1000`: loss `18.71`.
- Watch item: isolated spikes at epoch `6/12` iter `750` and epoch `7/12` iter `550`, each with immediate recovery. No stderr output.

## Progress Poll - S3 seed20260502 ratio18p75 w05 adaptive fusion - 2026-05-12 19:34 UTC

- Active replication chain remains `1282 -> 1286 -> 1287`.
- Queue state:
  - `1282` train RUNNING on `livenode03`, elapsed `4:13:25`.
  - `1286` full-val condition eval PENDING on dependency.
  - `1287` summary PENDING on dependency.
- Checkpoint state: `epoch_1.pth` through `epoch_9.pth` exist.
- Latest train log reached epoch `10/12`, iter `700/1000`.
- Recent loss observations:
  - epoch `8/12`, iter `1000/1000`: loss `17.25`; checkpoint saved at epoch 8.
  - epoch `9/12`, iter `50/1000`: loss `17.19`.
  - epoch `9/12`, iter `100/1000`: loss `34.96`.
  - epoch `9/12`, iter `150/1000`: loss `17.77`.
  - epoch `9/12`, iter `1000/1000`: loss `16.65`; checkpoint saved at epoch 9.
  - epoch `10/12`, iter `50/1000`: loss `16.49`.
  - epoch `10/12`, iter `100/1000`: loss `17.09`.
  - epoch `10/12`, iter `150/1000`: loss `17.27`.
  - epoch `10/12`, iter `200/1000`: loss `16.75`.
  - epoch `10/12`, iter `250/1000`: loss `55.42`.
  - epoch `10/12`, iter `300/1000`: loss `16.44`.
  - epoch `10/12`, iter `350/1000`: loss `16.90`.
  - epoch `10/12`, iter `400/1000`: loss `16.44`.
  - epoch `10/12`, iter `450/1000`: loss `16.49`.
  - epoch `10/12`, iter `500/1000`: loss `17.24`.
  - epoch `10/12`, iter `550/1000`: loss `16.11`.
  - epoch `10/12`, iter `600/1000`: loss `16.36`.
  - epoch `10/12`, iter `650/1000`: loss `16.87`.
  - epoch `10/12`, iter `700/1000`: loss `16.62`.
- Watch item: isolated spikes at epoch `9/12` iter `100` and epoch `10/12` iter `250`, each with immediate recovery. No stderr output.

## Progress Poll - S3 seed20260502 ratio18p75 w05 adaptive fusion - 2026-05-12 20:21 UTC

- Active replication chain remains `1282 -> 1286 -> 1287`.
- Queue state:
  - `1282` train RUNNING on `livenode03`, elapsed `5:00:00`.
  - `1286` full-val condition eval PENDING on dependency.
  - `1287` summary PENDING on dependency.
- Checkpoint state: `epoch_1.pth` through `epoch_11.pth` exist.
- Latest train log reached epoch `12/12`, iter `450/1000`.
- Recent loss observations:
  - epoch `10/12`, iter `1000/1000`: loss `16.50`; checkpoint saved at epoch 10.
  - epoch `11/12`, iter `50/1000`: loss `17.49`.
  - epoch `11/12`, iter `500/1000`: loss `15.64`.
  - epoch `11/12`, iter `950/1000`: loss `15.77`.
  - epoch `11/12`, iter `1000/1000`: loss `26.55`; checkpoint saved at epoch 11.
  - epoch `12/12`, iter `50/1000`: loss `16.46`.
  - epoch `12/12`, iter `100/1000`: loss `16.14`.
  - epoch `12/12`, iter `150/1000`: loss `16.18`.
  - epoch `12/12`, iter `200/1000`: loss `15.36`.
  - epoch `12/12`, iter `250/1000`: loss `15.27`.
  - epoch `12/12`, iter `300/1000`: loss `16.24`.
  - epoch `12/12`, iter `350/1000`: loss `16.06`.
  - epoch `12/12`, iter `400/1000`: loss `16.35`.
  - epoch `12/12`, iter `450/1000`: loss `17.02`.
- No stderr output. Eval remains dependency-blocked until final checkpoint/exit.

## Progress Poll - S3 seed20260502 ratio18p75 w05 adaptive fusion - 2026-05-12 20:42 UTC

- Train job `1282` completed and released the dependency.
- Queue state:
  - `1286` full-val condition eval RUNNING on `livenode03`, elapsed `6:00`.
  - `1287` summary PENDING on dependency.
- Final checkpoint state: `epoch_12.pth` exists at
  `outputs/racformer_train2k_genaug_seed20260502_ratio18p75_w05_adaptfusion_research/2026-05-12/12-21-12/epoch_12.pth`.
- Final train loss observations:
  - epoch `12/12`, iter `500/1000`: loss `23.46`.
  - epoch `12/12`, iter `550/1000`: loss `16.43`.
  - epoch `12/12`, iter `600/1000`: loss `29.96`.
  - epoch `12/12`, iter `650/1000`: loss `16.29`.
  - epoch `12/12`, iter `700/1000`: loss `16.93`.
  - epoch `12/12`, iter `750/1000`: loss `16.17`.
  - epoch `12/12`, iter `800/1000`: loss `16.42`.
  - epoch `12/12`, iter `850/1000`: loss `15.95`.
  - epoch `12/12`, iter `900/1000`: loss `16.01`.
  - epoch `12/12`, iter `950/1000`: loss `16.89`.
  - epoch `12/12`, iter `1000/1000`: loss `16.64`.
- Eval command is using `configs/racformer_train2k_genaug_seed20260502_ratio18p75_w05_adaptfusion_research.py`
  with final `epoch_12.pth`; it started inference on `6019` validation samples.
- Eval progress at poll: roughly `1280/6019` samples at about `4.0` samples/sec, ETA about `20` minutes.
- Train stderr contains only tqdm/progress-bar output from final validation bookkeeping, not a traceback.

## Result - S3 seed20260502 ratio18p75 w05 adaptive fusion - 2026-05-12 21:08 UTC

- Eval job `1286` completed and summary job `1287` wrote:
  `research/night_gen_phase1/results/S3_seed20260502_ratio18p75_w05_adaptfusion/summary_metrics.md`.
- Metrics:
  - day: mAP `0.3069704490`, NDS `0.3660246402`; vs S0 mAP `-0.83 pp`, NDS `-0.86 pp`.
  - night: mAP `0.1341373592`, NDS `0.2037173738`; vs S0 mAP `-1.46 pp`, NDS `-1.14 pp`.
  - rain: mAP `0.2807183868`, NDS `0.3728058191`; vs S0 mAP `+0.64 pp`, NDS `+0.15 pp`.
  - overall: mAP `0.2974233826`, NDS `0.3620743481`; vs S0 mAP `-0.66 pp`, NDS `-0.77 pp`.
- Gate verdict vs S0: FAIL.
  - Night mAP fails strongly (`-1.46 pp` vs S0, target was `+1.0 pp`).
  - Night NDS also fails (`-1.14 pp` vs S0, target was no worse than `-0.5 pp`).
- Gate verdict vs seed20260502 non-adaptive: FAIL.
  - Night mAP `-1.52 pp`, night NDS `-0.93 pp`.
  - Rain improves substantially (`+1.63 pp` mAP vs seed20260502 non-adaptive), but this does not rescue the night objective.
- Decision: adaptive+w05 seed20260425 pass does not replicate. Do not claim this as paper-worthy.
- Interpretation: the combined generated-night weighting + adaptive gate appears seed-sensitive and may be learning augmentation artifacts; the replicated branch shifts benefit toward rain while damaging night.
- Next action: stop repeating generated-image seed variants unless there is a new mechanism. Search targeted paper directions for a stronger, non-synthesis-first path.

## Tracker File Location Note - 2026-05-12 21:18 UTC

- Active local tracker is `/home/gabriel/LIVE/RACFORMER_NB2_EXPERIMENT_TRACKER.md`.
- Active remote canonical tracker is `/srv/nfs/shared/gnmp/RaCFormer/research/night_gen_phase1/RACFORMER_NB2_EXPERIMENT_TRACKER.md`.
- The old `/tmp/RACFORMER_NB2_EXPERIMENT_TRACKER.md` copy was stale; it ended at the 2026-05-12 13:07 UTC progress poll.
- Preserved that stale temp copy in the local repo root as `RACFORMER_NB2_EXPERIMENT_TRACKER.stale_tmp_20260512_1007.md` so nothing important depends on `/tmp`.
- Operational rule: do not use `/tmp` for active experiment tracking.

## Staged Run - S5 condition-aware fusion - 2026-05-12 21:31 UTC

- Motivation: generated-image/NB2 + adaptive gate did not replicate; DriveGEN pilot looked visually unreliable. Stop repeating generated-image seed variants without a new mechanism.
- Paper inspiration searched because the branch is stuck:
  - ContextualFusion (`https://arxiv.org/abs/2404.14780`): day/night/rain operational context gating; reports strong night-time nuScenes gains.
  - RobuRCDet (`https://arxiv.org/abs/2502.13071`): weather-adaptive radar-camera fusion under noisy/adverse conditions.
  - SpaRC (`https://arxiv.org/abs/2411.19860`): sparse radar-camera alignment/range-adaptive aggregation, more invasive than current patch.
  - RCDINO (`https://arxiv.org/abs/2508.15353`): DINO semantic enhancement for radar-camera, but older local DINOv3 crop experiments were negative.
- Hypothesis: a ContextualFusion-style explicit `day/night/rain` gate trained on the S5 mixed/oversampled subset can keep S5's night mAP gain while recovering day/rain/overall performance.
- Prior branches not repeated:
  - S5 real-night oversampling: night mAP +2.53 pp vs S0, but day -5.53 pp, rain -5.54 pp, overall -5.04 pp; failed gate.
  - S1 SimulateNight: night mAP +0.98 pp and night NDS +0.91 pp, but day -3.29 pp, rain -3.58 pp, overall -2.97 pp; weak/failed.
  - Brightness inference gating: invalidated in older repo notes; official mAP worsened from 0.5418 to 0.5171.
- Remote files staged:
  - `models/racformer_transformer.py`: added opt-in `condition_fusion_gate=False` path; learned gate uses `scene_condition` metadata and is identity-initialized.
  - `configs/racformer_r50_nuimg_704x256_f8.py`: added `scene_condition` to train/test `meta_keys`.
  - `configs/racformer_train2k_mixed_conditionfusion_research.py`: enables only `condition_fusion_gate=True` on top of S5.
  - `research/night_gen_phase1/staged_condition_gate/run_s5_conditionfusion*.sbatch`.
  - `research/night_gen_phase1/staged_condition_gate/summarize_conditionfusion_s5.py`.
- Remote backups before patch:
  - `models/racformer_transformer.py.bak.conditionfusion_20260512_212540`
  - `configs/racformer_r50_nuimg_704x256_f8.py.bak.conditionfusion_20260512_212540`
- Smoke validation:
  - Job `1292` on `livenode02` passed.
  - Dataset length `2000`.
  - First sample `scene_condition=day`.
  - `img_metas` contains `scene_condition`.
  - Image tensor shape `(48, 3, 256, 704)`.
- Config/model validation:
  - `condition_fusion_gate=True`.
  - Train and eval `meta_keys` include `scene_condition`.
  - `build_model` succeeds and decoder layer has `condition_fusion_gate=True`.
- Active SLURM chain:
  - Train `1293` (`s5_conditionfusion`) on `livenode02`.
  - Eval `1294` after train.
  - Summary `1295` after eval.
- Gate:
  - Publication target vs S0 unchanged: night mAP >= +1.0 pp, day mAP >= -1.0 pp, overall mAP >= -1.5 pp, night NDS >= -0.5 pp.
  - Diagnostic vs S5: day/rain/overall mAP recover by >= +2.0 pp while night mAP stays within -0.5 pp.
- Decision rule: if this fails, do not keep trying scalar/context gating variants blindly; next plausible branch must be a stronger architecture change (radar densification/range-adaptive aggregation) or a full-size confirmation only if a mini gate passes.

## Progress Poll - S5 condition-aware fusion - 2026-05-12 21:33 UTC

- Active chain remains `1293 -> 1294 -> 1295`.
- Train job `1293` is RUNNING on `livenode02`.
- First train log reached epoch `1/12`, iter `50/1000`.
- Loss snapshot: total loss `49.15`, `loss_depth=1.96`, `loss_cls=1.53`, `loss_bbox=2.89`; ETA about `5:07:32`; GPU memory `15089M`.
- `slurm_1293.err` is empty.
- No checkpoint yet; expected after epoch 1.

## Progress Poll - S5 condition-aware fusion - 2026-05-12 21:57 UTC

- Active chain remains `1293 -> 1294 -> 1295`.
- Train job `1293` is RUNNING on `livenode02`; eval `1294` and summary `1295` remain dependency-pending.
- Epoch 1 completed and saved:
  `outputs/racformer_train2k_mixed_conditionfusion_research/2026-05-12/18-29-20/epoch_1.pth`.
- Latest train log reached epoch `2/12`, iter `50/1000`.
- End of epoch 1 loss snapshot: iter `1000/1000`, total loss `27.76`, `loss_depth=1.36`, `loss_cls=0.89`, `loss_bbox=1.90`.
- Watch item: isolated epoch-1 spike at iter `600/1000` with total loss `117.17`; it immediately recovered to `30.73` at iter `650/1000`, so no intervention.
- `slurm_1293.err` remains empty.

## Progress Poll - S5 condition-aware fusion - 2026-05-12 22:23 UTC

- Active chain remains `1293 -> 1294 -> 1295`.
- Train job `1293` is RUNNING on `livenode02`.
- Epoch 2 completed and saved:
  `outputs/racformer_train2k_mixed_conditionfusion_research/2026-05-12/18-29-20/epoch_2.pth`.
- Latest train log reached epoch `3/12`, iter `50/1000`.
- End of epoch 2 loss snapshot: iter `1000/1000`, total loss `24.78`, `loss_depth=1.30`, `loss_cls=0.87`, `loss_bbox=1.62`.
- Epoch 3 began with a higher loss at iter `50/1000` (`45.79`), similar to prior recovered spikes; keep watching.
- `slurm_1293.err` remains empty.

## Progress Poll - S5 condition-aware fusion - 2026-05-12 22:49 UTC

- Active chain remains `1293 -> 1294 -> 1295`.
- Train job `1293` is RUNNING on `livenode02`.
- Epoch 3 completed and saved:
  `outputs/racformer_train2k_mixed_conditionfusion_research/2026-05-12/18-29-20/epoch_3.pth`.
- Latest train log reached epoch `4/12`, iter `50/1000`.
- End of epoch 3 loss snapshot: iter `1000/1000`, total loss `23.74`, `loss_depth=1.29`, `loss_cls=0.82`, `loss_bbox=1.58`.
- The epoch-3 high-start recovered; epoch 4 began at total loss `24.16`.
- `slurm_1293.err` remains empty.

## Progress Poll - S5 condition-aware fusion - 2026-05-12 23:49 UTC

- Active chain remains `1293 -> 1294 -> 1295`.
- Train job `1293` is RUNNING on `livenode02`; eval and summary remain dependency-pending.
- Checkpoints through epoch 5 exist:
  - `epoch_4.pth`
  - `epoch_5.pth`
- Latest train log reached epoch `6/12`, iter `400/1000`.
- End of epoch 5 loss snapshot: iter `1000/1000`, total loss `33.73`; it was a spike relative to surrounding epoch-5 losses around `22-23`.
- Epoch 6 has recovered after a spike at iter `250/1000` (`86.25`), with subsequent losses `21.96`, `21.74`, then `31.51` at iter `400/1000`.
- `slurm_1293.err` remains empty.
- Eval job `1294` has not started yet.

## Progress Poll - S5 condition-aware fusion - 2026-05-13 00:50 UTC

- Active chain remains `1293 -> 1294 -> 1295`.
- Train job `1293` is RUNNING on `livenode02`; eval and summary remain dependency-pending.
- Checkpoints through epoch 7 exist:
  - `epoch_6.pth`
  - `epoch_7.pth`
- Latest train log reached epoch `8/12`, iter `750/1000`.
- Recent epoch-8 losses are stable around `20-22` after a smaller spike at iter `300/1000` (`31.61`).
- `slurm_1293.err` remains empty.
- Eval job `1294` has not started yet.

## Tracker location note - 2026-05-13 02:52 UTC

- Canonical local tracker is `RACFORMER_NB2_EXPERIMENT_TRACKER.md` in `/home/gabriel/LIVE`.
- `/tmp/RACFORMER_NB2_EXPERIMENT_TRACKER.md` is intentionally not used; `/tmp` can be cleaned by the system.
- The old `/tmp` copy was preserved under the workspace as
  `RACFORMER_NB2_EXPERIMENT_TRACKER.stale_tmp_20260512_1007.md` for reference only.
- Continue appending local notes to the workspace tracker and remote notes to the canonical remote tracker:
  `/srv/nfs/shared/gnmp/RaCFormer/research/night_gen_phase1/RACFORMER_NB2_EXPERIMENT_TRACKER.md`.

## Progress Poll - S5 condition-aware fusion - 2026-05-13 02:52 UTC

- Train job `1293` completed and saved final checkpoint:
  `outputs/racformer_train2k_mixed_conditionfusion_research/2026-05-12/18-29-20/epoch_12.pth`.
- Active chain is now eval `1294` RUNNING on `livenode02`; summary job `1295` remains dependency-pending.
- Eval `1294` is using the epoch-12 checkpoint and running full inference on `6019` validation samples.
- Summary files do not exist yet:
  `research/night_gen_phase1/results/S5_conditionfusion/summary_metrics.md/json`.
- No gate judgement yet. Wait for summary metrics before deciding whether condition-aware fusion helped S5.

## Fallback design prep while S5 eval runs - 2026-05-13 02:56 UTC

- S5 condition-fusion eval `1294` was still RUNNING on `livenode02`; progress reached about `3401/6019` at
  `4.0 task/s`, ETA about `653s`. Summary job `1295` remained dependency-pending.
- No new GPU job launched while the S5 gate is unresolved.
- Web sources checked for a stronger fallback branch:
  - SpaRC (`https://arxiv.org/abs/2411.19860`): sparse frustum fusion, range-adaptive radar aggregation, and local self-attention.
  - RobuRCDet (`https://arxiv.org/abs/2502.13071`): 3D Gaussian Expansion for radar voxel robustness and camera-confidence-guided fusion.
  - ContextualFusion (`https://arxiv.org/abs/2404.14780`): context-gated fusion; already being tested by current S5 condition-fusion branch.
- Repo inspection notes:
  - Remote repo remains dirty with staged research changes; do not revert or overwrite unrelated modifications.
  - `models/racformer_transformer.py::BEVSampling` is the lowest-risk insertion point for a SpaRC-like range-adaptive
    query/radar aggregation change because both radar and LSS query features already pass through this sampler.
  - `models/racformer.py::extract_pts_feat` is the lowest-risk insertion point for a RobuRCDet-like radar BEV expansion/smoothing
    branch because radar points are voxelized/scattered before `radar_bev_conv`.
- Decision rule unchanged: if S5 condition-fusion fails, do not keep trying scalar/context gates. Next bounded branch should be
  radar/range-adaptive aggregation or radar BEV expansion, with smoke/mini validation before full validation.

## Result - S5 condition-aware fusion - 2026-05-13 03:15 UTC

- Jobs `1294 -> 1295` completed and wrote:
  - `research/night_gen_phase1/results/S5_conditionfusion/summary_metrics.md`
  - `research/night_gen_phase1/results/S5_conditionfusion/summary_metrics.json`
  - `research/night_gen_phase1/results/S5_conditionfusion/eval/eval_by_condition.json`
- Metrics:
  - day: mAP `0.2582515742`, NDS `0.3374292901`; vs S0 `-5.70 pp` mAP, `-3.71 pp` NDS; vs S5 `-0.17 pp` mAP, `+0.54 pp` NDS.
  - night: mAP `0.1740704513`, NDS `0.2204298652`; vs S0 `+2.53 pp` mAP, `+0.53 pp` NDS; vs S5 `-0.00 pp` mAP, `+1.20 pp` NDS.
  - rain: mAP `0.2255817113`, NDS `0.3261973648`; vs S0 `-4.87 pp` mAP, `-4.51 pp` NDS; vs S5 `+0.66 pp` mAP, `+0.22 pp` NDS.
  - overall: mAP `0.2530287625`, NDS `0.3332221146`; vs S0 `-5.10 pp` mAP, `-3.66 pp` NDS; vs S5 `-0.06 pp` mAP, `+0.55 pp` NDS.
- Publication gate vs S0: FAIL. Night improves, but day and overall regress far beyond the allowed limits.
- Diagnostic recovery gate vs S5: FAIL. The condition gate keeps S5's night benefit but does not recover day/rain/overall mAP.
- Decision: do not pursue more scalar/adaptive/context fusion gates on this S5 branch. Next hypothesis must be a different mechanism,
  preferably radar/range-adaptive aggregation or radar BEV expansion, and should start with smoke/mini validation before a full run.

## Local staging - radar BEV expansion candidate - 2026-05-13 03:20 UTC

- Staged locally only; no remote code changed and no new GPU job submitted.
- Plan file:
  `remote_patch_work/staged_radar_bev_expansion/RADAR_BEV_EXPANSION_PLAN.md`.
- Local staged files:
  - `remote_patch_work/models/racformer.py`
  - `remote_patch_work/configs/racformer_train2k_day_radarbevexp_research.py`
- Hypothesis: add a RobuRCDet-style fixed Gaussian local expansion over encoded radar BEV features, with a zero-initialized residual projection, on top of the S0 day-only training setup.
- Rationale: isolate architecture from the S5 oversampling data distribution; if it helps night/rain while preserving day, it is a cleaner paper direction than real-night oversampling or synthetic-image augmentation.
- Validation: local syntax check passed:
  `python -m py_compile remote_patch_work/models/racformer.py remote_patch_work/configs/racformer_train2k_day_radarbevexp_research.py`.
- Proposed gate: same S0 publication gate. If approved, upload/apply the patch remotely, run smoke/model-build first, then train/eval/summary on a single node.

## Local staging update - radar BEV expansion scripts - 2026-05-13 03:22 UTC

- Added local-only scripts under `remote_patch_work/staged_radar_bev_expansion/`:
  - `smoke_s0_radarbevexp_model.sbatch`
  - `run_s0_radarbevexp.sbatch`
  - `run_s0_radarbevexp_eval.sbatch`
  - `run_s0_radarbevexp_summary.sbatch`
  - `summarize_s0_radarbevexp.py`
- Updated `RADAR_BEV_EXPANSION_PLAN.md` with the remote validation sequence.
- Validation passed:
  - `bash -n` on all staged sbatch scripts.
  - `python -m py_compile` on staged `racformer.py`, config, and summarizer.
- Remote cluster state at inspection: `livenode02` and `livenode03` idle; no new job submitted from this staging update.

## Local staging review - radar BEV expansion - 2026-05-13 03:32 UTC

- Code review finding: the staged radar BEV expansion config is not strict-checkpoint compatible with pre-expansion RaCFormer checkpoints because it adds trainable `radar_bev_expansion.residual_proj.*` parameters. This does not block training/evaluating a new checkpoint, but it should not be used to strictly evaluate old S0 weights.
- Local cleanup applied:
  - Gaussian kernels in `RadarBEVExpansion` are now registered with `persistent=False`, avoiding unnecessary `kernel_3/5/7` checkpoint keys.
  - Plan/config wording now says the baseline forward output is preserved at initialization, not that the full architecture/checkpoint state is identical.
  - Smoke script now asserts the Gaussian kernels are non-persistent and residual projection is zero-initialized.
- Validation passed after cleanup:
  - `python -m py_compile remote_patch_work/models/racformer.py remote_patch_work/configs/racformer_train2k_day_radarbevexp_research.py remote_patch_work/staged_radar_bev_expansion/summarize_s0_radarbevexp.py`
  - `bash -n remote_patch_work/staged_radar_bev_expansion/smoke_s0_radarbevexp_model.sbatch remote_patch_work/staged_radar_bev_expansion/run_s0_radarbevexp.sbatch remote_patch_work/staged_radar_bev_expansion/run_s0_radarbevexp_eval.sbatch remote_patch_work/staged_radar_bev_expansion/run_s0_radarbevexp_summary.sbatch`
- No remote code changed and no new GPU job submitted from this review cleanup.

## Cluster node constraint - 2026-05-13 03:36 UTC

- Do not use `livenode01`; it has a known NVIDIA driver problem.
- Future SLURM scripts and job submissions must target only `livenode02` or `livenode03`.
- Current local radar-BEV staged scripts already target `livenode03`; no staged radar script targets `livenode01`.

## Open implementation audit - adverse radar-camera papers - 2026-05-13 03:39 UTC

- Cloned/read on cluster under `/srv/nfs/shared/gnmp/paper_impls`; outside the active RaCFormer repo.
- SpaRC:
  - Source pages checked: `https://phi-wol.github.io/sparc/`, `https://github.com/phi-wol/sparc`.
  - Clone has only `README.md` and `figs/architecture.png`; README says official PyTorch implementation is coming soon. No implementation code is available to adapt right now.
  - Keep the paper idea in reserve, but do not claim code-level adoption from SpaRC.
- ContextualFusion:
  - Source pages checked: `https://arxiv.org/abs/2404.14780`, `https://github.com/ssuralcmu/ContextualFusion`.
  - Repo has BEVFusion-style fusers/gating modules. Key files inspected:
    - `mmdet3d/models/gating/gating.py`
    - `mmdet3d/models/fusers/conv_3conditions_trainable.py`
    - `mmdet3d/models/fusers/conv_trainable_sigmoid_bounded.py`
    - `mmdet3d/models/fusion_models/bevfusion_contextualfusion.py`
  - Implementation is mostly scalar/context-driven modality weighting around BEV fusers; that family now has direct negative evidence in our S5 condition-aware fusion run, so do not continue with more context-gate variants unless there is a materially different mechanism.
- RobuRCDet:
  - Source pages checked: `https://arxiv.org/abs/2502.13071`, `https://github.com/Jingtong0527/RobuRCDet`.
  - Repo has concrete code for 3D Gaussian Expansion and confidence-guided fusion.
  - Key files inspected:
    - `layers/backbones/pts_backbone_3DGE.py`
    - `layers/modules/conf_guided_multimodal_cross_attention.py`
    - `layers/fuser/multimodal_feature_fusion.py`
    - `models/roburcdet.py`
  - The 3DGE implementation is not directly portable as-is due hardcoded CUDA/shape assumptions, but it supports the current local staged hypothesis: add a bounded Gaussian local expansion on encoded radar BEV in RaCFormer.
  - Confidence-guided multimodal cross-attention is a possible later branch, but it is larger and more invasive than radar BEV expansion.

## Remote smoke submission - S0 radar BEV expansion - 2026-05-13 03:42 UTC

- Applied staged radar BEV expansion files on cluster `/srv/nfs/shared/gnmp/RaCFormer`.
- Backup created before upload: `models/racformer.py.bak.radarbevexp_20260513_033229`.
- Remote files changed/added:
  - `models/racformer.py`
  - `configs/racformer_train2k_day_radarbevexp_research.py`
  - `research/night_gen_phase1/staged_radar_bev_expansion/`
- Remote validation before smoke:
  - uploaded `models/racformer.py` hash `166ca7c08fc63494dc3698deb870c00cad072205924bdd832fdf0144827256b2`.
  - staged sbatch files do not target `livenode01`.
  - `conda run -n racformerfix python -m py_compile ...` passed.
  - `bash -n` on smoke/train/eval/summary sbatch scripts passed.
- Node constraint honored: `livenode02` and `livenode03` were idle; `livenode01` was not used.
- Submitted smoke/model-build job `1297` on `livenode03` only. No full train submitted yet.

## Smoke result - S0 radar BEV expansion - 2026-05-13 03:45 UTC

- Smoke/model-build job `1297` completed on `livenode03`.
- Output log:
  `research/night_gen_phase1/results/S0_radarbevexp/smoke_slurm_1297.out`
- Error log:
  `research/night_gen_phase1/results/S0_radarbevexp/smoke_slurm_1297.err`
- Passed checks:
  - Config built: `configs/racformer_train2k_day_radarbevexp_research.py`
  - `radar_bev_expansion` present with kernel sizes `(3, 5, 7)`
  - residual projection weight/bias remain exactly zero after `model.init_weights()`
  - fixed Gaussian kernels are non-persistent and absent from `model.state_dict()`
- `smoke_slurm_1297.err` contains mmcv init logging only; no traceback/failure.
- Decision: smoke passes. Full train/eval/summary chain may be submitted on `livenode02` or `livenode03` only; do not use `livenode01`.

## Full chain submission - S0 radar BEV expansion - 2026-05-13 03:48 UTC

- Submitted dependency chain on `livenode03`:
  - train `1298`: `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp.sbatch`
  - eval `1299`: `afterok:1298`, `run_s0_radarbevexp_eval.sbatch`
  - summary `1300`: `afterok:1299`, `run_s0_radarbevexp_summary.sbatch`
- Node constraint honored: all scripts are pinned to `livenode03`; no `livenode01` use.
- Expected result files after completion:
  - `research/night_gen_phase1/results/S0_radarbevexp/eval/eval_by_condition.json`
  - `research/night_gen_phase1/results/S0_radarbevexp/summary_metrics.md`
  - `research/night_gen_phase1/results/S0_radarbevexp/summary_metrics.json`

## Progress poll - S0 radar BEV expansion - 2026-05-13 03:50 UTC

- Train job `1298` is RUNNING on `livenode03`; eval `1299` and summary `1300` remain dependency-pending.
- Train startup loaded the usual image-pretrain checkpoint non-strictly. This is expected and separate from strict-loading old S0 detector checkpoints.
- `slurm_1298.err` contains a PyTorch `torch.meshgrid` future warning from `RadarBEVExpansion`; no traceback or training failure observed.
- Latest parsed train output reached runner startup: `workflow: [('train', 1)], max: 12 epochs`.

## Progress poll - S0 radar BEV expansion - 2026-05-13 03:52 UTC

- Train job `1298` is RUNNING on `livenode03`; eval `1299` and summary `1300` remain dependency-pending.
- Latest parsed training progress: epoch `1/12`, iter `50/1000`, loss `47.38`, ETA about `5:13:32`, GPU memory about `15398M`.
- `slurm_1298.err` still only contains the `torch.meshgrid` future warning; no traceback/failure.

## Eval/summary path check - S0 radar BEV expansion - 2026-05-13 03:57 UTC

- Verified eval script path before train completion:
  - `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp_eval.sbatch` selects latest `outputs/racformer_train2k_day_radarbevexp_research/*/*/epoch_12.pth`.
  - It calls `research/night_gen_phase1/eval_by_condition.py` with `--full-val`.
  - `eval_by_condition.py` asserts CUDA availability and loads the newly trained checkpoint with `strict=True`, which is appropriate for the new radar-BEV checkpoint.
- Verified S0 baseline metric files required by `summarize_s0_radarbevexp.py` exist and match known baselines:
  - day mAP/NDS `0.3152649818 / 0.3745762709`
  - night mAP/NDS `0.1487749875 / 0.2150977574`
  - rain mAP/NDS `0.2743174671 / 0.3713314930`
  - overall mAP/NDS `0.3039905911 / 0.3697754272`
- No script issue found in the dependency chain so far.

## Open implementation audit addendum - CRN and RCBEVDet - 2026-05-13 04:04 UTC

- Additional sources checked while S0 radar-BEV expansion trains:
  - CRN paper/repo: `https://github.com/youngskkim/CRN`, paper page `https://huggingface.co/papers/2304.00670`
  - RCBEVDet paper/repo: `https://github.com/VDIGPKU/RCBEVDet`
- Cloned/extracted under `/srv/nfs/shared/gnmp/paper_impls`, outside the active RaCFormer repo:
  - `CRN` at git head `5e9d2fa2f91c714b297e75ca666d27fc4ad0d13d`
  - `RCBEVDet` at git head `15c83ccdd5a8cfd3b7c0390eacd8664cf842d513`; code extracted from `rcbevdet-master.zip`
- CRN implementation notes:
  - `models/camera_radar_net_det.py` sends radar context/occupancy into image BEV lifting, then fuses BEV features with `MFAFuser`.
  - `layers/fuser/multimodal_feature_aggregation.py` builds BEV queries from normalized camera/radar BEV features and applies stacked `DeformableCrossAttention`.
  - `layers/modules/multimodal_deformable_cross_attention.py` concatenates projected image/radar values and uses learned offsets/attention weights for multimodal BEV deformable attention.
  - Adoption note: RaCFormer already performs query-level sampling from image/LSS/radar BEV in `models/racformer_transformer.py`; a CRN-style full BEV-grid fuser would be larger and likely less isolated than the current radar-BEV expansion branch.
- RCBEVDet implementation notes:
  - `configs/rcbevdet/rcbevdet-256x704-r50-BEV128-9kf-depth-cbgs12e-circlelarger.py` uses `RadarBEVNet`, `PointPillarsScatterRCS`, radar BEV backbone/neck, and bidirectional deformable BEV fusion.
  - `mmdet3d/models/backbones/radar_encoder.py` has a dual point/transformer radar encoder with point embedding, injector/extractor cross-attention, and self-attention over radar points.
  - `mmdet3d/models/middle_encoders/pillar_scatter.py::PointPillarsScatterRCS` builds an RCS-derived heatmap and feature map, then compresses it with scattered radar BEV features.
  - `mmdet3d/models/detectors/bevdet_rc.py` reduces radar BEV channels, applies deformable attention in both radar-query->camera and camera-query->radar directions, then fuses with `RadarConvFuser`.
  - Adoption note: the lowest-risk RaCFormer follow-up, if current S0 radar-BEV expansion fails, is likely an RCS-aware radar BEV attention/heatmap branch after `radar_bev_conv`, not a full BEVDet-style detector rewrite.

## Progress poll - S0 radar BEV expansion - 2026-05-13 04:05 UTC

- Train job `1298` is still RUNNING on `livenode03`; eval `1299` and summary `1300` remain dependency-pending.
- Latest parsed progress: epoch `1/12`, iter `250/1000`, loss `31.21`, ETA about `5:03:29`.
- Loss is decreasing through early training; no new stderr beyond the known `torch.meshgrid` future warning.

## Progress poll - S0 radar BEV expansion - 2026-05-13 04:09 UTC

- Train job `1298` is RUNNING on `livenode03`; eval `1299` and summary `1300` remain dependency-pending.
- `livenode02` is idle; keep it idle while this one-hypothesis experiment is unresolved.
- Latest parsed progress: epoch `1/12`, iter `300/1000`, loss `29.50`, ETA about `5:03:05`.
- No summary metrics exist yet; no gate decision can be made.
- Stderr remains only the known `torch.meshgrid` future warning.

## Progress poll - S0 radar BEV expansion - 2026-05-13 04:10 UTC

- Node constraint rechecked: train job `1298` is RUNNING on `livenode03`; eval `1299` and summary `1300` remain dependency-pending. No `livenode01` use.
- Latest parsed progress: epoch `1/12`, iter `400/1000`, loss `29.41`, ETA about `4:59:14`, memory about `15400M`.
- Result directory currently has only `slurm_1298.out` and `slurm_1298.err`; no `summary_metrics.md/json` yet.
- `sacct` is unavailable on this cluster because Slurm accounting storage is disabled; use `squeue` and logs for active monitoring.
- Stderr remains only the known `torch.meshgrid` future warning.

## Progress and fallback prep - S0 radar BEV expansion - 2026-05-13 03:52 UTC

- Active train job `1298` remains RUNNING on `livenode03`; eval `1299` and summary `1300` are still dependency-pending. `livenode02` is idle; no `livenode01` use.
- Latest parsed train progress: epoch `1/12`, iter `600/1000`, loss `27.76`, ETA about `4:54:00`, memory about `15491M`.
- No eval or summary artifact exists yet, so there is no gate decision.
- Read-only fallback preparation, not submitted:
  - RaCFormer radar BEV path is in `models/racformer.py`: `extract_pts_feat()` voxelizes radar points, handles empty radar sweeps with a zero `[B, 64, 128, 128]` tensor, runs `radar_middle_encoder`, then `radar_bev_conv` to `embed_dims=256` before temporal stacking.
  - RaCFormer already carries image-space `radar_rcs` into `img_lss_view_transformer`, and point-level radar input uses seven selected radar channels from `loaders/pipelines/loading.py`.
  - RCBEVDet implementation `PointPillarsScatterRCS` builds an RCS-derived BEV heatmap and feature map, projects them with `rcs_att`, concatenates with scattered radar BEV features, and compresses back to radar channels.
  - Lowest-risk fallback if this S0 radar-BEV expansion fails: add an optional RCS-aware BEV heatmap/residual branch after `radar_bev_conv`, keyed by a new config field, while preserving current empty-radar behavior and strict eval for newly trained checkpoints.
  - Main risks: mapping point-level RCS consistently after RaCFormer zeroes z and voxelizes; avoiding CPU loops over many voxels; avoiding another scalar fusion gate family, since previous adaptive/condition gates failed.

## Explorer addendum - RCS BEV fallback - 2026-05-13 03:55 UTC

- Local read-only explorer confirmed the same fallback without SSH/GPU use.
- Local prototype worth reusing conceptually, not wholesale: `minimal_radar_camera_fusion/models/radar_pillar_bev.py` builds an RCS-aware Gaussian BEV scatter from point feature index `3`, with sigma `0.5 + 1.5 * sigmoid(rcs / 20)`.
- Suggested fallback branch shape:
  - Build `rcs_heatmap` as `[B, 1, 128, 128]` from voxel `coors` and `voxels[..., 3]`.
  - Apply after `radar_bev_conv`, preserving `[B, 256, 128, 128]` input/output.
  - Use identity-safe initialization, e.g. zero-init residual projection or zero learned scale; do not multiply by raw sigmoid attention at init.
- Coordinate risk to verify in smoke: voxel `coors` should be treated as `[batch, z, y, x]`; a swapped `x/y` silently transposes the RCS attention map.

## Progress poll - S0 radar BEV expansion - 2026-05-13 04:04 UTC

- Train job `1298` remains RUNNING on `livenode03`; eval `1299` and summary `1300` remain dependency-pending. No `livenode01` use.
- Epoch 1 completed and saved checkpoint:
  `outputs/racformer_train2k_day_radarbevexp_research/2026-05-13/00-35-54/epoch_1.pth`
- Latest parsed progress: epoch `2/12`, iter `50/1000`, loss `24.64`, ETA about `4:42:22`, memory about `15675M`.
- Result directory still has only train stdout/stderr; no `eval_by_condition.json` or `summary_metrics.md/json` yet.
- Stderr remains only the known `torch.meshgrid` future warning. Stdout includes repeated `warning ---> no points within the predefined bev receptive field`; this has not stopped training.

## Progress poll - S0 radar BEV expansion - 2026-05-13 04:05 UTC

- Train job `1298` remains RUNNING on `livenode03`; eval `1299` and summary `1300` remain dependency-pending. No `livenode01` use.
- `livenode02` is idle and `livenode03` is allocated to `s0_radarbevexp`.
- Latest parsed progress: epoch `2/12`, iter `100/1000`, loss `24.95`, ETA about `4:41:18`, memory about `15675M`.
- No eval/summary artifacts exist yet; wait for later checkpoints or the dependency chain result before making a gate decision.

## Parallel livenode02 screening submission - S0 radar BEV expansion epoch 1 - 2026-05-13 04:09 UTC

- Submitted a safe parallel full-val screening eval of the already-saved epoch-1 checkpoint on `livenode02`; no model-code edits and no changes to the active epoch-12 train/eval chain.
- Jobs:
  - epoch-1 eval `1301`: `s0_radarbevexp_e1_eval`, RUNNING on `livenode02`
  - epoch-1 summary `1302`: `afterok:1301`, dependency-pending
  - main train `1298`: still RUNNING on `livenode03`
  - main eval `1299` and summary `1300`: still dependency-pending on epoch-12 train
- Staged files added:
  - `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp_epoch1_eval_livenode02.sbatch`
  - `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp_epoch1_summary_livenode02.sbatch`
  - `research/night_gen_phase1/staged_radar_bev_expansion/summarize_s0_radarbevexp_epoch1.py`
- Output paths:
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch1/eval/`
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch1/summary_metrics.md`
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch1/summary_metrics.json`
- Purpose: use idle `livenode02` for early full-val sanity screening only. Final publication gate remains the epoch-12 dependency-chain eval.

## Strict-load fix and livenode02 retry - S0 radar BEV expansion epoch 1 - 2026-05-13 04:16 UTC

- Epoch-1 screening eval job `1301` failed quickly before inference:
  - strict checkpoint load rejected unexpected `radar_bev_expansion.kernel_3`, `kernel_5`, and `kernel_7`
  - stale dependent summary job `1302` was canceled
- Diagnosis: the fixed Gaussian kernels were registered with `persistent=False` in the eval-time model, while the saved checkpoint contains those keys. This would likely have broken the final epoch-12 eval too.
- Fix applied:
  - backed up remote `models/racformer.py` to `models/racformer.py.bak.kernelpersistent_20260513_0413`
  - changed `RadarBEVExpansion.register_buffer(..., persistent=True)` for the fixed kernels
  - local and remote `py_compile` passed
  - forward behavior is unchanged; this only makes strict checkpoint loading compatible with the saved checkpoint keys
- Resubmitted livenode02 screening:
  - retry eval `1303`: RUNNING on `livenode02`
  - retry summary `1304`: `afterok:1303`, dependency-pending
- Retry evidence: `eval_slurm_1303.out` shows successful checkpoint load and full-val inference progress (`121/6019` samples observed). The previous strict-load error is gone.
- Main train remains independent and continues on `livenode03`; latest parsed progress at this poll was epoch `2/12`, iter `500/1000`, loss `25.16`, ETA about `4:31:04`.

## Progress poll - parallel epoch-1 eval and main train - 2026-05-13 04:21 UTC

- Job placement:
  - epoch-1 screening eval `1303`: RUNNING on `livenode02`
  - epoch-1 screening summary `1304`: dependency-pending
  - main train `1298`: RUNNING on `livenode03`
  - main epoch-12 eval `1299` and summary `1300`: dependency-pending
- Epoch-1 eval progress: full-val inference reached about `1747/6019` samples at roughly `4.0 task/s`; no new stderr beyond normal startup warnings.
- Main train progress: epoch `2/12`, iter `750/1000`, loss `24.94`, ETA about `4:24:42`.
- Only checkpoint currently saved is `epoch_1.pth`; no eval/summary metrics yet.

## Progress poll - parallel epoch-1 eval and main train - 2026-05-13 04:37 UTC

- Epoch-1 screening eval `1303` remains RUNNING on `livenode02`; summary `1304` remains dependency-pending.
- Eval progress: `5591/6019` full-val samples, about `4.0 task/s`, ETA about `106s`.
- Main train `1298` remains RUNNING on `livenode03`; epoch `2` completed and saved:
  `outputs/racformer_train2k_day_radarbevexp_research/2026-05-13/00-35-54/epoch_2.pth`
- Latest main train progress: epoch `3/12`, iter `350/1000`, loss `31.97`, ETA about `4:09:21`.
- No screening summary metrics yet, but they should appear after eval `1303` finishes and summary `1304` runs.

## Screening result - S0 radar BEV expansion epoch 1 - 2026-05-13 04:50 UTC

- Epoch-1 full-val screening eval `1303` completed on `livenode02`; summary `1304` completed.
- This is not the final publication gate; final decision remains the epoch-12 dependency-chain eval.
- Metrics vs S0:
  - day `0.0666 / 0.1287` (`-24.86 pp` mAP, `-24.59 pp` NDS)
  - night `0.0343 / 0.0792` (`-11.45 pp` mAP, `-13.59 pp` NDS)
  - rain `0.0545 / 0.1398` (`-21.98 pp` mAP, `-23.15 pp` NDS)
  - overall `0.0649 / 0.1276` (`-23.91 pp` mAP, `-24.22 pp` NDS)
- Screening verdict: FAIL, as expected for an early checkpoint; do not use this as a final decision.
- Useful side effect: the strict-load bug was found and fixed before the final epoch-12 eval.
- `livenode02` is free again and can be used for the next saved-checkpoint screening pass while `livenode03` continues training.

## Parallel livenode02 screening submission - S0 radar BEV expansion epoch 2 - 2026-05-13 04:54 UTC

- Local and remote checks passed before submission:
  - `bash -n` passed for the epoch-2 eval and summary sbatch files.
  - `py_compile` passed for `summarize_s0_radarbevexp_screening.py`.
  - checkpoint exists: `outputs/racformer_train2k_day_radarbevexp_research/2026-05-13/00-35-54/epoch_2.pth` (732M).
- `livenode02` was idle before submission; `livenode03` was still running the main epoch-12 train job `1298`.
- Submitted safe parallel full-val screening of the already-saved epoch-2 checkpoint on `livenode02`:
  - epoch-2 eval `1305`: `s0_radarbevexp_e2_eval`, RUNNING on `livenode02`
  - epoch-2 summary `1306`: `afterok:1305`, dependency-pending
- New staged files:
  - `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp_epoch2_eval_livenode02.sbatch`
  - `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp_epoch2_summary_livenode02.sbatch`
  - `research/night_gen_phase1/staged_radar_bev_expansion/summarize_s0_radarbevexp_screening.py`
- Output paths:
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch2/eval/`
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch2/summary_metrics.md`
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch2/summary_metrics.json`
- This is still screening only. The final publication gate remains the epoch-12 dependency-chain eval/summary.

## Parallel livenode02 screening queue - S0 radar BEV expansion epoch 3 - 2026-05-13 04:57 UTC

- `epoch_3.pth` exists: `outputs/racformer_train2k_day_radarbevexp_research/2026-05-13/00-35-54/epoch_3.pth` (732M).
- Local and remote `bash -n` passed for the epoch-3 eval and summary sbatch files.
- Queued the epoch-3 screening chain behind the epoch-2 summary, so `livenode02` continues screening only after the active epoch-2 chain completes:
  - epoch-3 eval `1307`: dependency `afterok:1306`, pinned to `livenode02`
  - epoch-3 summary `1308`: dependency `afterok:1307`, pinned to `livenode02`
- New staged files:
  - `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp_epoch3_eval_livenode02.sbatch`
  - `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp_epoch3_summary_livenode02.sbatch`
- Output paths:
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch3/eval/`
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch3/summary_metrics.md`
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch3/summary_metrics.json`
- This remains early screening only. The epoch-12 dependency-chain eval is still the final gate.

## Screening result - S0 radar BEV expansion epoch 2 - 2026-05-13 05:31 UTC

- Epoch-2 full-val screening eval `1305` completed on `livenode02`; summary `1306` completed with no summary stderr.
- Metrics vs S0:
  - day `0.0954 / 0.1746` (`-21.99 pp` mAP, `-19.99 pp` NDS)
  - night `0.0519 / 0.1006` (`-9.69 pp` mAP, `-11.45 pp` NDS)
  - rain `0.0695 / 0.2103` (`-20.48 pp` mAP, `-16.10 pp` NDS)
  - overall `0.0903 / 0.1726` (`-21.37 pp` mAP, `-19.72 pp` NDS)
- Screening verdict: FAIL. This is better than epoch 1 but still far below S0, so do not treat it as evidence for the hypothesis yet.
- Epoch-3 screening eval `1307` has started on `livenode02`; summary `1308` remains dependency-pending.
- Main epoch-12 train `1298` continues on `livenode03`; at this poll it was in epoch `5/12`.
- Decision: let epoch-3 screening finish for trend context. Do not queue epoch-4 screening until epoch-3 metrics show whether the early curve is recovering fast enough to justify more livenode02 eval time.

## Screening result - S0 radar BEV expansion epoch 3 - 2026-05-13 06:05 UTC

- Epoch-3 full-val screening eval `1307` completed on `livenode02`; summary `1308` completed with no summary stderr.
- Metrics vs S0:
  - day `0.1192 / 0.2206` (`-19.61 pp` mAP, `-15.39 pp` NDS)
  - night `0.0491 / 0.0943` (`-9.97 pp` mAP, `-12.08 pp` NDS)
  - rain `0.0940 / 0.2329` (`-18.04 pp` mAP, `-13.85 pp` NDS)
  - overall `0.1126 / 0.2185` (`-19.14 pp` mAP, `-15.13 pp` NDS)
- Screening verdict: FAIL.
- Trend vs epoch 2:
  - day and overall improved, but night mAP/NDS regressed slightly.
  - The checkpoint is still too undertrained to support or reject the final epoch-12 hypothesis.
- Main train `1298` remains RUNNING on `livenode03`; latest checkpoint observed: `epoch_5.pth`.
- Decision: skip epoch-4 screening. Because `livenode02` is idle and `epoch_5.pth` is already available, run a single epoch-5 mid-training full-val screen for a more useful trend point, then stop screening unless it shows meaningful recovery.

## Parallel livenode02 screening submission - S0 radar BEV expansion epoch 5 - 2026-05-13 06:06 UTC

- Local and remote `bash -n` passed for epoch-5 eval and summary sbatch files.
- Confirmed checkpoint exists: `outputs/racformer_train2k_day_radarbevexp_research/2026-05-13/00-35-54/epoch_5.pth` (732M).
- `livenode02` was idle before submission; main train `1298` remains RUNNING on `livenode03`.
- Submitted one mid-training screen, skipping epoch 4:
  - epoch-5 eval `1309`: `s0_radarbevexp_e5_eval`, pinned to `livenode02`
  - epoch-5 summary `1310`: `afterok:1309`, pinned to `livenode02`
- New staged files:
  - `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp_epoch5_eval_livenode02.sbatch`
  - `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp_epoch5_summary_livenode02.sbatch`
- Output paths:
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch5/eval/`
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch5/summary_metrics.md`
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch5/summary_metrics.json`
- This is still screening only. Unless epoch 5 shows meaningful recovery, do not queue more checkpoint screens before the final epoch-12 eval.

## Screening result - S0 radar BEV expansion epoch 5 - 2026-05-13 06:43 UTC

- Epoch-5 full-val screening eval `1309` completed on `livenode02`; summary `1310` completed.
- Metrics vs S0:
  - day `0.1853 / 0.2579` (`-13.00 pp` mAP, `-11.67 pp` NDS)
  - night `0.0949 / 0.1390` (`-5.38 pp` mAP, `-7.61 pp` NDS)
  - rain `0.1451 / 0.2296` (`-12.92 pp` mAP, `-14.18 pp` NDS)
  - overall `0.1776 / 0.2524` (`-12.64 pp` mAP, `-11.73 pp` NDS)
- Screening verdict: FAIL.
- Trend vs epoch 3:
  - night mAP recovered from `0.0491` to `0.0949`.
  - overall mAP recovered from `0.1126` to `0.1776`.
  - still far below S0, but the recovery is large enough to justify one later-checkpoint screen while `livenode02` is idle.
- Main train `1298` remains RUNNING on `livenode03`; latest checkpoint observed: `epoch_7.pth`.
- Decision: run an epoch-7 screen on `livenode02`. Do not queue more screens after epoch 7 unless it is much closer to S0; otherwise wait for final epoch-12 eval.

## Parallel livenode02 screening submission - S0 radar BEV expansion epoch 7 - 2026-05-13 06:44 UTC

- Local and remote `bash -n` passed for epoch-7 eval and summary sbatch files.
- Confirmed checkpoint exists: `outputs/racformer_train2k_day_radarbevexp_research/2026-05-13/00-35-54/epoch_7.pth` (732M).
- `livenode02` was idle before submission; main train `1298` remains RUNNING on `livenode03`.
- Submitted one later-checkpoint screen:
  - epoch-7 eval `1311`: `s0_radarbevexp_e7_eval`, pinned to `livenode02`
  - epoch-7 summary `1312`: `afterok:1311`, pinned to `livenode02`
- New staged files:
  - `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp_epoch7_eval_livenode02.sbatch`
  - `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp_epoch7_summary_livenode02.sbatch`
- Output paths:
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch7/eval/`
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch7/summary_metrics.md`
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch7/summary_metrics.json`
- This is still screening only. Unless epoch 7 is much closer to S0, wait for the final epoch-12 dependency-chain eval rather than queueing more screens.

## Screening result - S0 radar BEV expansion epoch 7 - 2026-05-13 07:21 UTC

- Epoch-7 full-val screening eval `1311` completed on `livenode02`; summary `1312` completed.
- Metrics vs S0:
  - day `0.2391 / 0.3112` (`-7.61 pp` mAP, `-6.34 pp` NDS)
  - night `0.1102 / 0.1661` (`-3.86 pp` mAP, `-4.90 pp` NDS)
  - rain `0.1970 / 0.3010` (`-7.73 pp` mAP, `-7.03 pp` NDS)
  - overall `0.2312 / 0.3071` (`-7.28 pp` mAP, `-6.26 pp` NDS)
- Screening verdict: FAIL.
- Trend vs epoch 5:
  - night mAP improved from `0.0949` to `0.1102`.
  - overall mAP improved from `0.1776` to `0.2312`.
  - gap remains large; this is not evidence of a publishable result yet.
- Main train `1298` remains RUNNING on `livenode03`; latest observed progress was epoch `9/12`.
- Decision: do not claim success. Wait for final epoch-12 dependency-chain eval unless an epoch-9 checkpoint becomes available with enough time for one last trend screen on idle `livenode02`.

## Parallel livenode02 screening submission - S0 radar BEV expansion epoch 9 - 2026-05-13 07:34 UTC

- Local and remote `bash -n` passed for epoch-9 eval and summary sbatch files.
- Confirmed checkpoint exists: `outputs/racformer_train2k_day_radarbevexp_research/2026-05-13/00-35-54/epoch_9.pth` (732M).
- `livenode02` was idle before submission; main train `1298` remains RUNNING on `livenode03`.
- Submitted final planned trend screen:
  - epoch-9 eval `1313`: `s0_radarbevexp_e9_eval`, pinned to `livenode02`
  - epoch-9 summary `1314`: `afterok:1313`, pinned to `livenode02`
- New staged files:
  - `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp_epoch9_eval_livenode02.sbatch`
  - `research/night_gen_phase1/staged_radar_bev_expansion/run_s0_radarbevexp_epoch9_summary_livenode02.sbatch`
- Output paths:
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch9/eval/`
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch9/summary_metrics.md`
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch9/summary_metrics.json`
- This is still screening only. After epoch 9, wait for the final epoch-12 dependency-chain eval instead of queueing more checkpoint screens.

## Screening result - S0 radar BEV expansion epoch 9 - 2026-05-13 08:10 UTC

- Epoch-9 full-val screening eval `1313` completed on `livenode02`; summary `1314` completed.
- Metrics vs S0:
  - day `0.2762 / 0.3360` (`-3.91 pp` mAP, `-3.86 pp` NDS)
  - night `0.1406 / 0.1841` (`-0.82 pp` mAP, `-3.10 pp` NDS)
  - rain `0.2371 / 0.3248` (`-3.72 pp` mAP, `-4.65 pp` NDS)
  - overall `0.2683 / 0.3316` (`-3.57 pp` mAP, `-3.82 pp` NDS)
- Screening verdict: FAIL.
- Trend vs epoch 7:
  - night mAP improved from `0.1102` to `0.1406`, now close to S0 night mAP but still below the +1 pp target.
  - day and overall remain too far below S0 to pass the paper gate.
- Main train `1298` remains RUNNING on `livenode03`; latest observed progress was epoch `11/12`.
- Decision: stop checkpoint screens. Wait for final epoch-12 dependency-chain eval/summary (`1299`/`1300`) before accepting or rejecting the radar-BEV expansion hypothesis.

## Parallel livenode02 screening submission - S0 radar BEV expansion epoch 11 - 2026-05-13 08:35 UTC

- User reminded to check whether idle `livenode02` can be used for parallel work.
- Safe parallel item identified: full-val screening of the already-written epoch-11 checkpoint. This is read-only, pinned to `livenode02`, and does not modify model code while epoch 12 continues training on `livenode03`.
- Local `bash -n` passed for epoch-11 eval and summary sbatch files.
- Confirmed checkpoint exists: `outputs/racformer_train2k_day_radarbevexp_research/2026-05-13/00-35-54/epoch_11.pth`.
- Submitted:
  - epoch-11 eval `1315`: `s0_radarbevexp_e11_eval`, pinned to `livenode02`
  - epoch-11 summary `1316`: `afterok:1315`, pinned to `livenode02`
- Main train `1298` remains RUNNING on `livenode03`.
- Output paths:
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch11/eval/`
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch11/summary_metrics.md`
  - `research/night_gen_phase1/results/S0_radarbevexp_epoch11/summary_metrics.json`
- This is still screening only. Final decision remains the epoch-12 dependency-chain eval/summary (`1299`/`1300`).

## Screening result - S0 radar BEV expansion epoch 11 - 2026-05-13 09:08 UTC

- Epoch-11 full-val screening eval `1315` completed on `livenode02`; summary `1316` completed.
- Metrics vs S0:
  - day `0.2961 / 0.3619` (`-1.91 pp` mAP, `-1.27 pp` NDS)
  - night `0.1490 / 0.2090` (`+0.02 pp` mAP, `-0.61 pp` NDS)
  - rain `0.2626 / 0.3503` (`-1.17 pp` mAP, `-2.11 pp` NDS)
  - overall `0.2890 / 0.3581` (`-1.50 pp` mAP, `-1.16 pp` NDS)
- Screening verdict: FAIL.
- Interpretation:
  - Night mAP has recovered to roughly S0, but still misses the required `+1.0 pp` night target by about `0.98 pp`.
  - Day mAP still misses the `-1.0 pp` floor, and night NDS misses the `-0.5 pp` floor.
  - Overall mAP is borderline at `-1.50 pp`, but this is not enough without the night gain and day/NDS gates.
- Main final eval `1299` is RUNNING on `livenode03`.
- Decision: keep waiting for epoch-12 final summary before rejecting the radar-BEV expansion branch, but probability of passing is now low.

## Paper implementation audit - BEV-Radar - 2026-05-13 09:12 UTC

- Web search found open implementation: `https://github.com/Etah0409/BEV-Radar`.
- Cloned on the cluster under `/srv/nfs/shared/gnmp/paper_impls/BEV-Radar`.
- Clone head: `1e725443d11132b3e9b757ea3cf3621fef1d0359`.
- Relevant files inspected:
  - `configs/bev_fusion/radar_bev_fusion.py`
  - `configs/bev_fusion/radar_bev_fusion_trans.py`
  - `mmdet3d/models/fusion_models/radarbevfusion.py`
  - `mmdet3d/models/fusion_models/radarbevfusion_v2.py`
  - `mmdet3d/models/fusers/conv.py`
  - `mmdet3d/models/heads/bbox/radarprehead.py`
- Useful implementation ideas:
  - BEV-level radar-camera fusion is kept simple: concatenate BEV feature maps, apply a conv fuser, then concatenate a residual copy and refine with additional conv blocks.
  - Radar branch uses PointPillars-style voxelization and scatter into BEV, matching RaCFormer’s existing radar path closely enough to borrow small ideas.
  - V2 exposes radar features to the detection head, but that is a larger head-contract change than needed for the next guarded experiment.
- Decision:
  - Do not port BEV-Radar wholesale.
  - Use it as support for a small BEV-level fallback branch together with RCBEVDet’s RCS-aware radar BEV idea: zero-init residual from radar occupancy + RCS BEV statistics after `radar_bev_conv`.
  - Local fallback patch prepared in `/home/gabriel/LIVE/remote_patch_work` only; do not upload over active remote source until final epoch-12 eval no longer depends on it.

## Final result - S0 radar BEV expansion epoch 12 - 2026-05-13 09:24 UTC

- Main train `1298`, eval `1299`, and summary `1300` completed.
- Metrics vs S0:
  - day `0.3019 / 0.3638` (`-1.34 pp` mAP, `-1.08 pp` NDS)
  - night `0.1452 / 0.2056` (`-0.36 pp` mAP, `-0.95 pp` NDS)
  - rain `0.2690 / 0.3569` (`-0.53 pp` mAP, `-1.44 pp` NDS)
  - overall `0.2944 / 0.3598` (`-0.96 pp` mAP, `-0.99 pp` NDS)
- Gate verdict: FAIL.
- Decision:
  - Reject the fixed Gaussian radar-BEV expansion branch.
  - It recovered day/overall relative to epoch 11 but did not produce the required night gain; night mAP is below S0 and night NDS regresses beyond the gate.
  - Move to the next distinct hypothesis: zero-init radar occupancy + RCS BEV residual, inspired by RCBEVDet RCS-aware BEV encoding and BEV-Radar’s simple BEV fusion.

## Staged experiment - S0 RCS BEV residual - 2026-05-13 09:30 UTC

- Hypothesis: radar occupancy and RCS statistics in BEV provide a less destructive radar-side cue than broad Gaussian feature expansion.
- Implementation:
  - Added optional `RadarRCSBEVResidual` in `models/racformer.py`.
  - Builds a two-channel BEV map from voxel occupancy and normalized RCS (`tanh(mean_rcs / 32.0)`), then applies a small zero-init residual conv branch after `radar_bev_conv`.
  - Keeps the empty-radar path valid by producing an all-zero RCS/occupancy map.
  - Config: `configs/racformer_train2k_day_rcsbev_research.py`.
  - Staged scripts: `research/night_gen_phase1/staged_rcs_bev_residual/`.
- Remote safety:
  - Current model backed up before upload: `models/racformer.py.bak.rcsbev_20260513_092940`.
  - Remote py_compile passed via `conda run -n racformerfix`.
  - Remote `bash -n` passed for smoke/train/eval/summary sbatch files.
  - Jobs are pinned to `livenode02`; no `livenode01`.
- Smoke submitted:
  - smoke job `1317`: `s0_rcsbev_smoke`, pinned to `livenode02`.
- Decision: wait for smoke before submitting the train/eval/summary chain.

## Smoke fix - S0 RCS BEV residual - 2026-05-13 09:32 UTC

- Smoke `1317` failed before exercising the model because the smoke script omitted repo-local registry imports:
  - error: `KeyError: 'RaCFormer is not in the models registry'`
- Fix:
  - Added `import models` and `import loaders` to `smoke_s0_rcsbev_model.sbatch`.
  - Switched smoke model build to match the prior radar-BEV smoke style: `build_model(cfg.model, test_cfg=cfg.get("test_cfg"))`.
- Resubmitted smoke:
  - smoke job `1318`: `s0_rcsbev_smoke`, pinned to `livenode02`.
- This was a smoke-script issue, not evidence about the RCS branch behavior.

## Smoke pass and submission - S0 RCS BEV residual - 2026-05-13 09:33 UTC

- Corrected smoke `1318` passed on `livenode02`.
- Smoke output:
  - `radar_rcs_bev_residual (128, 128) 3 32.0`
  - `state_keys 4`
- Submitted train/eval/summary chain, all pinned to `livenode02`:
  - train `1319`: `s0_rcsbev`
  - eval `1320`: `afterok:1319`
  - summary `1321`: `afterok:1320`
- Output paths:
  - `research/night_gen_phase1/results/S0_rcsbev/`
  - `outputs/racformer_train2k_day_rcsbev_research/`
- This is a new distinct hypothesis, not a continuation of the failed Gaussian expansion branch.

## Train launch fix - S0 RCS BEV residual - 2026-05-13 09:37 UTC

- Train job `1319` failed immediately before completing an iteration.
- Error:
  - `RuntimeError: Index put requires the source and destination dtypes match, got Half for the destination and Float for the source.`
- Cause:
  - RCS BEV map destination followed `radar_bev` dtype under fp16, while the normalized RCS tensor was promoted back to fp32.
- Fix:
  - Cast normalized RCS back to the destination dtype before scatter assignment.
  - Expanded smoke to run a dummy half-precision `RadarRCSBEVResidual.forward()` and assert zero-init output equality.
- Previous dependent eval/summary from `1319` did not run.

## Smoke fix 2 - S0 RCS BEV residual - 2026-05-13 09:39 UTC

- Expanded smoke `1322` failed in the smoke script before the forward check:
  - error: `AttributeError: 'ConfigDict' object has no attribute 'embed_dims'`
- Cause:
  - The smoke attempted to read `cfg.model.pts_bbox_head.embed_dims`, but this config stores the head dimensions differently; the built model exposes the resolved value.
- Fix:
  - Use `model.pts_bbox_head.embed_dims` when constructing the dummy BEV tensor.
- No train job was submitted from `1322`.

## Smoke fix 3 - S0 RCS BEV residual - 2026-05-13 09:42 UTC

- Expanded smoke `1323` caught another dtype issue:
  - error: `"clamp_min_scalar_cpu" not implemented for 'Half'`
- Fix:
  - Normalize RCS/counts in fp32 and cast only the final normalized RCS tensor to the BEV destination dtype.
  - Run the dummy half-precision forward smoke on CUDA, matching the actual training path.
- No train job was submitted from `1323`.

## Smoke fix 4 - S0 RCS BEV residual - 2026-05-13 09:44 UTC

- CUDA half-forward smoke `1324` caught the residual conv dtype boundary:
  - error: `Input type (c10::Half) and bias type (float) should be the same`
- Fix:
  - Run the residual conv in its parameter dtype and cast the zero-init residual back to the radar BEV dtype before addition.
- No train job was submitted from `1324`.

## Smoke pass and resubmission - S0 RCS BEV residual - 2026-05-13 09:46 UTC

- Smoke `1325` passed on `livenode02`.
- Smoke output:
  - `radar_rcs_bev_residual (128, 128) 3 32.0`
  - `state_keys 4`
  - `half_forward_zero_init True`
- Relaunched train/eval/summary chain, all pinned to `livenode02`:
  - train `1326`: `s0_rcsbev`
  - eval `1327`: `afterok:1326`
  - summary `1328`: `afterok:1327`
- Decision: monitor early train logs to confirm the first real iterations run before treating this as a stable long job.

## Early train status - S0 RCS BEV residual - 2026-05-13 09:51 UTC

- Train `1326` is RUNNING on `livenode02`.
- Early logs reached epoch `1/12`, iteration `100/1000`.
- No stderr output observed.
- Current output directory:
  - `outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/`
- Decision:
  - Treat as stable long-running train job.
  - Next evidence point is checkpoint screening or final eval/summary from dependent jobs `1327` and `1328`.

## Parallel branch staging - S0 occupancy-dominant BEV residual - 2026-05-13 09:54 UTC

- `livenode03` was idle while full RCS residual train `1326` continued on `livenode02`.
- Safe parallel option chosen: config-only ablation using the same already-smoked source code, so the active `S0_rcsbev` source dependency is not changed.
- Hypothesis: raw RCS may be noisy; retaining radar occupancy while effectively muting RCS may improve night transfer with less feature noise.
- Config:
  - `configs/racformer_train2k_day_rcsoccbev_research.py`
  - `radar_rcs_bev_residual=dict(output_shape=(128,128), rcs_index=3, rcs_scale=1000000.0)`
- Staged files:
  - `research/night_gen_phase1/staged_rcs_occ_residual/smoke_s0_rcsoccbev_model.sbatch`
  - `research/night_gen_phase1/staged_rcs_occ_residual/run_s0_rcsoccbev_livenode03.sbatch`
  - `research/night_gen_phase1/staged_rcs_occ_residual/run_s0_rcsoccbev_eval_livenode03.sbatch`
  - `research/night_gen_phase1/staged_rcs_occ_residual/run_s0_rcsoccbev_summary_livenode03.sbatch`
  - `research/night_gen_phase1/staged_rcs_occ_residual/summarize_s0_rcsoccbev.py`
- Local and remote syntax checks passed.
- Smoke submitted:
  - smoke job `1329`: `s0_rcsoccbev_smoke`, pinned to `livenode03`.
- No `livenode01` use.

## Smoke pass and submission - S0 occupancy-dominant BEV residual - 2026-05-13 09:57 UTC

- Smoke `1329` passed on `livenode03`.
- Smoke output:
  - `radar_occ_bev_residual (128, 128) 3 1000000.0`
  - `half_forward_zero_init True`
- Submitted train/eval/summary chain, all pinned to `livenode03`:
  - train `1330`: `s0_rcsoccbev`
  - eval `1331`: `afterok:1330`
  - summary `1332`: `afterok:1331`
- Parallel state:
  - Full RCS branch train `1326` remains RUNNING on `livenode02`.
  - Occupancy-dominant ablation train `1330` starts on `livenode03`.
- No `livenode01` use.

## Early parallel train status - RCS vs occupancy-dominant BEV residuals - 2026-05-13 10:01 UTC

- `S0_rcsbev` train `1326`:
  - RUNNING on `livenode02`.
  - Reached epoch `1/12`, iteration `500/1000`.
  - No stderr output observed.
- `S0_rcsoccbev` train `1330`:
  - RUNNING on `livenode03`.
  - Reached epoch `1/12`, iteration `100/1000`.
  - No stderr output observed.
- Both allowed GPU nodes are occupied; no `livenode01` use.
- Decision:
  - Let both train.
  - Use non-GPU time for paper/code fallback research only; do not modify active remote source while both dependent evals rely on it.

## Parallel capacity check - allowed nodes occupied - 2026-05-13 10:04 UTC

- User reminder handled: checked whether idle `livenode02` could be used for parallel work.
- SLURM state:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, latest log at epoch `1/12`, iteration `650/1000`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, latest log at epoch `1/12`, iteration `250/1000`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Both train stderr files are empty.
- No checkpoints have appeared yet.
- Decision:
  - Do not submit additional GPU work right now; both permitted nodes are occupied.
  - Recheck for idle capacity after the first checkpoint or when either train exits.
  - Continue only CPU-side paper/code fallback inspection while waiting, without changing active remote source.

## CPU-side fallback audit - RCTrans radar dense encoder - 2026-05-13 10:07 UTC

- Reason:
  - Both allowed GPU nodes are occupied, so use wait time for read-only inspection only.
  - Prepare a next candidate if both current BEV-residual branches fail.
- Open implementation inspected:
  - `/srv/nfs/shared/gnmp/paper_impls/RCTrans`
  - `projects/mmdet3d_plugin/models/backbones/pointpillars.py`
  - `projects/mmdet3d_plugin/models/detectors/rcdetr.py`
- Relevant mechanism:
  - `Radar_dense_encoder_tf` is applied after radar voxel/middle encoding and before radar backbone/neck.
  - Architecture is a U-Net-like BEV densifier:
    - `DoubleConv -> Down(64,128) -> Down(128,256) -> Down(256,512)`
    - bottleneck self-attention over a learned 2D grid position embedding
    - three upsampling blocks back to 64 channels
- Adaptation note for RaCFormer:
  - Current RaCFormer insertion point would be after `radar_bev_conv`, before optional residual/expansion or fusion.
  - A direct port is heavier and riskier than the current zero-init residual branch.
  - If current RCS/occupancy residuals fail, consider a small zero-init "radar dense residual" inspired by RCTrans rather than a full U-Net/attention port.
- Decision:
  - Do not implement now; active source remains unchanged while jobs `1326` and `1330` run.

## Active train poll - both allowed nodes still occupied - 2026-05-13 10:06 UTC

- Host check:
  - SSH target `cluster_live_tail`, repo `/srv/nfs/shared/gnmp/RaCFormer`.
  - Cluster host reported `cluster-live`.
- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `19:46`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `9:49`.
  - `1327`, `1328`, `1331`, `1332`: dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: epoch `1/12`, iteration `700/1000`, no stderr.
  - `S0_rcsoccbev`: epoch `1/12`, iteration `350/1000`, no stderr.
- Checkpoint state:
  - No `epoch_*.pth` checkpoint found yet for either branch.
- Decision:
  - Do not launch more GPU work; both allowed nodes are occupied.
  - Recheck after epoch 1 checkpoint or if either train exits.

## CPU-side paper/code audit - SGDet3D - 2026-05-13 10:11 UTC

- Reason:
  - User asked to search for paper inspiration when stuck and to check open implementations before adopting ideas.
  - Both permitted GPU nodes are currently occupied, so only read-only/code-inspection work is appropriate.
- Web/source:
  - SGDet3D paper/repo: `https://github.com/shawnnnkb/SGDet3D`
  - README states `2025.02.25 all code released`.
  - Scope mismatch: SGDet3D targets 4D radar-camera datasets (`VoD`, `TJ4DRadSet`), not nuScenes NB2 radar-camera. Treat as inspiration only.
- Clone:
  - Remote clone path: `/srv/nfs/shared/gnmp/paper_impls/SGDet3D`
  - Commit inspected: `83f8c7d76f2b6d3e84ca948dee5f9e4ba472b64c`
- Relevant implementation files:
  - `projects/SGDet3D/mmdet3d_plugin/models/voxel_encoder/pillar_encoder.py`
  - `projects/SGDet3D/mmdet3d_plugin/models/necks/BEVCross_modal_attention.py`
  - `projects/SGDet3D/mmdet3d_plugin/models/detectors/SGDet3D.py`
  - `projects/SGDet3D/mmdet3d_plugin/models/img2bev/forward_projection/GeometryDepth_Net.py`
- Transferable mechanisms:
  - Radar voxel encoder decorates radar point features with local cluster offsets, local pillar xy offsets, and `features[:, :, 3:5]` center offsets when `with_velocity_snr_center=True`.
  - Their `Cross_Modal_Fusion` computes per-modality spatial attention using channel mean/max maps, then gates image BEV by radar attention and radar BEV by image attention before concatenation + conv reduction.
  - Their geometry-depth branch uses radar-projected depth to sharpen camera depth, but that is much more invasive than a RaCFormer residual branch.
- Adaptation note for RaCFormer:
  - If both current `S0_rcsbev` and `S0_rcsoccbev` branches fail, a conservative next variant is an identity-safe radar BEV residual map with occupancy + RCS + compensated velocity channels (`vx_comp`, `vy_comp`) rather than only occupancy/mean RCS.
  - Avoid porting SGDet3D object-oriented attention or radar-depth supervision wholesale; it is dataset/API-heavy and too invasive for this branch.
- Decision:
  - Do not edit active source while jobs `1326` and `1330` run.
  - Keep SGDet3D as a fallback idea: velocity/RCS-centered radar BEV residual or reciprocal spatial attention, not a full transplant.

## Visual QC check - DriveGEN pilot remains rejected - 2026-05-13 10:18 UTC

- Reason:
  - User reported that the DriveGEN-generated image looked visually weird and asked to judge before relying on that branch.
- Files inspected locally:
  - Original frame downloaded from `/srv/nfs/shared/shared/nuscenes/samples/CAM_BACK/n008-2018-08-30-15-52-26-0400__CAM_BACK__1535659414187558.jpg`
  - DriveGEN output downloaded from `/srv/nfs/shared/gnmp/DriveGEN/experiments/night_pilot/temp_data_2.1_base_seed20260425_r18p75_first2_800x448/nus_res/night/CAM_BACK_n008-2018-08-30-15-52-26-0400__CAM_BACK__1535659414187558.jpg`
- Visual judgement:
  - Not just a day-to-night style change.
  - The centered Toyota/minivan is changed into a different front-facing vehicle.
  - The left-side scene becomes a large trailer-like object, changing scene layout/object geometry.
  - Strong glare/light streaks and blur alter road appearance.
- Decision:
  - Keep DriveGEN paused for training data.
  - Do not scale the current DriveGEN pilot into RaCFormer training.
  - Only revisit DriveGEN if a new QC-gated setting/prompt/model can preserve object geometry against original frames.

## Active train poll - nearing first checkpoint - 2026-05-13 10:12 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `25:35`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `15:38`.
  - `1327`, `1328`, `1331`, `1332`: dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: epoch `1/12`, iteration `950/1000`, no stderr.
  - `S0_rcsoccbev`: epoch `1/12`, iteration `550/1000`, no stderr.
- Checkpoint config:
  - Base config has `checkpoint_config = dict(interval=1, max_keep_ckpts=4)`.
  - No `epoch_*.pth` checkpoint found yet, so `1326` has not completed epoch 1.
- Decision:
  - Continue waiting; no safe parallel GPU slot is available.
  - Next poll should verify whether `epoch_1.pth` appears for `S0_rcsbev`.

## Checkpoint poll - S0 RCS BEV residual epoch 1 saved - 2026-05-13 10:14 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `27:22`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `17:25`.
- Latest train logs:
  - `S0_rcsbev`: reached epoch `1/12`, iteration `1000/1000`.
  - `S0_rcsoccbev`: reached epoch `1/12`, iteration `650/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_1.pth`
  - Size: `765690049` bytes.
- Errors:
  - Both train stderr files remain `0` bytes.
- Decision:
  - Do not launch a checkpoint screen now; both permitted GPU nodes are occupied.
  - Keep `epoch_1.pth` as a possible diagnostic if a node frees unexpectedly.
  - Let the existing full train/eval/summary chain continue.

## Active train poll - S0 RCS entered epoch 2 - 2026-05-13 10:16 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `28:56`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `18:59`.
  - Dependent eval/summary jobs remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: epoch `2/12`, iteration `50/1000`.
  - `S0_rcsoccbev`: epoch `1/12`, iteration `700/1000`.
- Checkpoints:
  - `S0_rcsbev` still has only `epoch_1.pth`.
  - No `S0_rcsoccbev` checkpoint yet.
- Errors:
  - Both train stderr files remain `0` bytes.
- Decision:
  - No new GPU work; both allowed nodes are occupied.
  - Continue CPU-only fallback review and wait for either `S0_rcsoccbev` epoch 1 checkpoint or node release.

## Local-only fallback staging - S0 RCS + velocity BEV residual - 2026-05-13 10:20 UTC

- Status:
  - Staged locally only under `/home/gabriel/LIVE/remote_patch_work`.
  - Not uploaded to the remote RaCFormer repo.
  - Not smoked on cluster.
  - Not submitted to SLURM.
- Reason:
  - Both GPU nodes are occupied.
  - If `S0_rcsbev` and `S0_rcsoccbev` fail, the next conservative branch should be ready without editing from scratch.
- Hypothesis:
  - A zero-init radar BEV residual using occupancy + RCS + compensated velocity (`vx_comp`, `vy_comp`) may retain the robustness of occupancy priors while adding motion cues useful for night without changing the head/transformer.
- Local code/config staged:
  - `remote_patch_work/models/racformer.py`
    - `RadarRCSBEVResidual` remains backward-compatible by default.
    - New optional args: `extra_indices`, `extra_scales`.
    - Current two-channel RCS configs still instantiate a 2-channel map.
    - Velocity config instantiates a 4-channel map: occupancy, RCS, `vx_comp`, `vy_comp`.
  - `remote_patch_work/configs/racformer_train2k_day_rcsvelbev_research.py`
  - `remote_patch_work/staged_rcs_vel_residual/smoke_s0_rcsvelbev_model.sbatch`
  - `remote_patch_work/staged_rcs_vel_residual/run_s0_rcsvelbev_livenode02.sbatch`
  - `remote_patch_work/staged_rcs_vel_residual/run_s0_rcsvelbev_eval_livenode02.sbatch`
  - `remote_patch_work/staged_rcs_vel_residual/run_s0_rcsvelbev_summary_livenode02.sbatch`
  - `remote_patch_work/staged_rcs_vel_residual/summarize_s0_rcsvelbev.py`
- Local verification:
  - `python -m py_compile` passed for modified model, new config, and summary script.
  - `bash -n` passed for staged sbatch scripts.
  - `rg` check found no `livenode01` reference in the staged branch.
- Decision:
  - Do not upload or run this while current remote train/eval chains depend on the active source.
  - Revisit only if both active branches fail the publication gate.

## Active train poll - occupancy branch near epoch 1 checkpoint - 2026-05-13 10:21 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `33:40`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `23:43`.
- Latest train logs:
  - `S0_rcsbev`: epoch `2/12`, iteration `250/1000`.
  - `S0_rcsoccbev`: epoch `1/12`, iteration `850/1000`.
- Checkpoints:
  - `S0_rcsbev`: `epoch_1.pth` present.
  - `S0_rcsoccbev`: no checkpoint yet.
- Errors:
  - Both train stderr files remain `0` bytes.
- Decision:
  - Still no idle GPU capacity on allowed nodes.
  - Wait for `S0_rcsoccbev` epoch 1 checkpoint or a job/node state change.

## Active train poll - occupancy branch at 950/1000 - 2026-05-13 10:22 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `35:10`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `25:13`.
- Latest train log:
  - `S0_rcsoccbev`: epoch `1/12`, iteration `950/1000`.
- Checkpoints:
  - `S0_rcsbev`: `epoch_1.pth` present.
  - `S0_rcsoccbev`: no checkpoint yet.
- Errors:
  - Both train stderr files remain `0` bytes.
- Decision:
  - No idle GPU capacity.
  - Next poll should confirm `S0_rcsoccbev` epoch `1/12` completion and checkpoint creation.

## Checkpoint poll - S0 occupancy-dominant BEV residual epoch 1 saved - 2026-05-13 10:25 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `38:01`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `28:04`.
  - Dependent eval/summary jobs remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: epoch `2/12`, iteration `400/1000`.
  - `S0_rcsoccbev`: epoch `2/12`, iteration `50/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_1.pth`
  - Size: `765690049` bytes.
- Existing checkpoint:
  - `outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_1.pth`
- Errors:
  - Both train stderr files remain `0` bytes.
- Decision:
  - Still no idle GPU capacity on permitted nodes.
  - Do not run midpoint screens while both full trainings are active.
  - Let both trains continue toward final eval chains.

## CPU-side paper/code audit - RCM-Fusion - 2026-05-13 10:27 UTC

- Reason:
  - User asked to search for paper inspiration and inspect open implementations before adopting ideas.
  - Both permitted GPU nodes remain occupied, so this was CPU/read-only audit work.
- Web/source:
  - `https://github.com/mjseong0414/RCM-Fusion`
  - README describes radar-camera multi-level fusion on nuScenes with a Radar Guided BEV Encoder and Radar Grid Point Refinement.
- Clone:
  - Remote clone path: `/srv/nfs/shared/gnmp/paper_impls/RCM-Fusion`
  - Commit inspected: `23c828bb0cefdc98700c5d7f6e4e2afede97cc68`
- Relevant files inspected:
  - `projects/mmdet3d_plugin/rcm_fusion/modules/radar_guided_bev_encoder.py`
  - `projects/mmdet3d_plugin/rcm_fusion/modules/radar_camera_gating.py`
  - `projects/mmdet3d_plugin/models/fusion_layers/instance_level_fusion.py`
  - `projects/mmdet3d_plugin/rcm_fusion/dense_heads/feature_level_fusion.py`
- Findings:
  - `RadarGuidedBEVEncoder` is a BEVFormer-style transformer sequence that injects radar BEV as `pts_bev` into self-attention and then applies `RadarCameraGating` before FFN.
  - `RadarCameraGating` computes channel-wise 1D-conv sigmoid weights from `query_c + query_r`, then combines camera/radar tokens as `query_c * cam_weight + query_r * rad_weight`.
  - `InstanceLevelFusion` uses radar points associated with proposals, then generates extra radar grid points shifted along estimated object velocity direction before point pooling/refinement.
- Adaptation note for RaCFormer:
  - Full `RadarGuidedBEVEncoder` is too coupled to RCM-Fusion's BEVFormer-style head; do not transplant wholesale.
  - `RadarCameraGating` resembles prior adaptive/context gates, which have already been unstable in this RaCFormer/NB2 track.
  - The most useful low-risk idea is the velocity-aware radar prior: RCM-Fusion explicitly uses velocity-direction grid expansion in its instance refinement, which supports the locally staged `S0_rcsvelbev` fallback.
- Decision:
  - Do not implement RCM-Fusion directly now.
  - Keep the staged velocity/RCS BEV residual as the conservative adaptation if current active branches fail.

## CPU-side paper/code audit - SIFormer - 2026-05-13 10:31 UTC

- Reason:
  - Web search found a newer 2026 radar-camera paper with source listed as `github.com/shawnnnkb/SIFormer`.
  - User asked to check open implementations before adopting paper ideas.
- Web/source:
  - `https://github.com/shawnnnkb/SIFormer`
  - Paper topic: instance awareness via cross-view correlation for 4D radar and camera.
- Clone:
  - Remote clone path: `/srv/nfs/shared/gnmp/paper_impls/SIFormer`
  - Commit inspected: `ecad02ffa47ef64f5f95fac4cf73b9eb58c86e6d`
- Search results:
  - README describes CVC, SSI, and IEA modules.
  - Repo project tree currently exposes:
    - `projects/SIFormer/preprocess/*`
    - `projects/RadarPillarNet/*`
    - general vendored mmdet3d configs/files.
  - Grep for `CVC`, `IEA`, `Sparse Scene Integration`, `Cross-View Correlation`, and `Instance Enhance` only found README text, not implementable module code.
- Decision:
  - Not actionable for immediate RaCFormer adaptation.
  - Do not spend GPU or implementation time on SIFormer unless the repo later exposes the actual model modules.

## Active train poll - post paper-code audits - 2026-05-13 10:29 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `41:41`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `31:44`.
- Latest train logs:
  - `S0_rcsbev`: epoch `2/12`, iteration `550/1000`.
  - `S0_rcsoccbev`: epoch `2/12`, iteration `150/1000`.
- Checkpoints:
  - `S0_rcsbev`: `epoch_1.pth` present.
  - `S0_rcsoccbev`: `epoch_1.pth` present.
- Errors:
  - Both train stderr files remain `0` bytes.
- Decision:
  - Both allowed GPU nodes remain occupied and healthy.
  - No further GPU submission in this pass.

## Active train poll - no final summaries yet - 2026-05-13 10:30 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `42:36`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `32:39`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: epoch `2/12`, iteration `600/1000`.
  - `S0_rcsoccbev`: epoch `2/12`, iteration `200/1000`.
- Checkpoints:
  - `S0_rcsbev`: `epoch_1.pth` present.
  - `S0_rcsoccbev`: `epoch_1.pth` present.
- Errors:
  - Both train stderr files remain `0` bytes.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - No new GPU work; both allowed nodes are occupied.
  - Next useful action is another poll after a meaningful transition: epoch 2 checkpoint, job completion, eval start, or node release.

## Active train poll - still healthy, no idle node - 2026-05-13 10:31 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `43:48`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `33:51`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: epoch `2/12`, iteration `650/1000`.
  - `S0_rcsoccbev`: epoch `2/12`, iteration `250/1000`.
- Checkpoints:
  - `S0_rcsbev`: `epoch_1.pth` present.
  - `S0_rcsoccbev`: `epoch_1.pth` present.
- Errors:
  - Both train stderr files remain `0` bytes.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Do not submit additional GPU work; both permitted nodes are still occupied.

## Checkpoint poll - S0 RCS BEV residual epoch 2 saved - 2026-05-13 10:41 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `54:05`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `44:08`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: reached epoch `2/12`, iteration `1000/1000`, then entered epoch `3/12`, iteration `50/1000`.
  - `S0_rcsoccbev`: epoch `2/12`, iteration `650/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_2.pth`
  - Size: `765690177` bytes.
- Existing checkpoints:
  - `S0_rcsbev`: `epoch_1.pth`
  - `S0_rcsoccbev`: `epoch_1.pth`
- Errors:
  - Both train stderr files remain `0` bytes.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - No new GPU work; both allowed nodes remain occupied.
  - Next meaningful transition is `S0_rcsoccbev` epoch 2 checkpoint or a job/eval state change.

## Checkpoint poll - S0 occupancy-dominant BEV residual epoch 2 saved - 2026-05-13 10:51 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `1:04:06`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `54:09`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: epoch `3/12`, iteration `400/1000`.
  - `S0_rcsoccbev`: reached epoch `2/12`, iteration `1000/1000`, then entered epoch `3/12`, iteration `50/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_2.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth`, `epoch_2.pth`
  - `S0_rcsoccbev`: `epoch_1.pth`, `epoch_2.pth`
- Errors:
  - Both train stderr files remain `0` bytes.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - No new GPU work; both allowed nodes remain occupied.
  - Let both full train/eval/summary chains continue.

## Checkpoint poll - S0 RCS BEV residual epoch 3 saved - 2026-05-13 11:07 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `1:19:53`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `1:09:56`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: reached epoch `3/12`, iteration `1000/1000`.
  - `S0_rcsoccbev`: epoch `3/12`, iteration `650/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_3.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`
  - `S0_rcsoccbev`: `epoch_1.pth`, `epoch_2.pth`
- Errors:
  - Both train stderr files remain `0` bytes.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - No new GPU work; both allowed nodes remain occupied.
  - Next meaningful transition is `S0_rcsoccbev` epoch 3 checkpoint or a job/eval state change.

## Checkpoint poll - both active branches reached epoch 4 - 2026-05-13 11:19 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `1:32:29`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `1:22:32`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: epoch `4/12`, iteration `500/1000`.
  - `S0_rcsoccbev`: reached epoch `3/12`, iteration `1000/1000`, then entered epoch `4/12`, iteration `100/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_3.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`
  - `S0_rcsoccbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`
- Errors:
  - Both train stderr files remain `0` bytes.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - `livenode02` is not idle; it is occupied by `S0_rcsbev`.
  - Do not submit additional GPU work on allowed nodes until one branch finishes or a node is actually free.

## CPU-side paper/code audit - RICCARDO - 2026-05-13 11:28 UTC

- Reason:
  - Both permitted GPU nodes remain occupied by active full-training branches.
  - User asked to search for paper inspirations and inspect open implementations before adopting ideas.
- Web/source:
  - `https://github.com/longyunf/riccardo`
  - Paper/topic: RICCARDO, CVPR 2025, radar hit prediction and convolution for camera-radar 3D object detection on nuScenes.
- Clone:
  - Remote clone path: `/srv/nfs/shared/gnmp/paper_impls/riccardo`
  - Commit inspected: `09583406d74ba03b004065a8c4df83ff53200faa`
- Relevant files inspected:
  - `README.md`
  - `lib/my_model/distr_network.py`
  - `lib/my_model/stage3_network.py`
  - `lib/my_model/my_data_gen1.py`
  - `lib/my_model/my_data_gen2.py`
  - `scripts/gen_eval_cand.py`
  - `scripts/gen_pos_cand.py`
  - `scripts/gen_eval_offset.py`
  - `scripts/train_stage3.py`
- Findings:
  - RICCARDO is a three-stage pipeline: generate SparseBEV monocular detections; predict a 129x129 object-conditioned radar hit distribution; convolve observed accumulated radar returns against that predicted distribution over radial shifts; train a stage-3 MLP to choose a 65-bin radial offset and update object range/score.
  - The radar association code uses range/tangential windows around monocular detections, class-dependent point thresholds, Doppler compatibility checks, multi-sweep timestamp compensation, and a radial-shift convolution curve.
  - Stage-3 is post-detection range refinement, not a simple backbone/fusion module. It depends on precomputed candidate files, pretrained RICCARDO stage checkpoints, and SparseBEV-style detection tensors.
- Adaptation note for RaCFormer:
  - Do not wholesale-port RICCARDO during the current NB2 rescue loop; it would require a separate post-processing data pipeline and extra trained stages.
  - The useful low-risk idea is object/range-conditioned radar support: if BEV residual branches fail, a smaller RaCFormer-compatible fallback could add a cheap radial-support score around predicted queries or proposal centers, using existing radar points and velocity compatibility.
  - For the immediate next architecture fallback, the already staged velocity/RCS BEV residual remains lower-risk than RICCARDO's full object-conditioned refinement.
- Decision:
  - Keep RICCARDO as framing/inspiration for a future post-hoc range-refinement ablation.
  - Do not spend GPU time on it before the active `S0_rcsbev` and `S0_rcsoccbev` chains finish.

## Checkpoint poll - S0 RCS BEV residual epoch 4 saved - 2026-05-13 11:31 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `1:44:45`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `1:34:48`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: reached epoch `4/12`, iteration `1000/1000`.
  - `S0_rcsoccbev`: epoch `4/12`, iteration `600/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_4.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `epoch_4.pth`
  - `S0_rcsoccbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`
- Errors:
  - Both train stderr files remain `0` bytes.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Both allowed GPU nodes remain occupied; no new GPU work.
  - Next meaningful transition is the `S0_rcsoccbev` epoch 4 checkpoint or a training/eval state change.

## CPU-side paper/code audit - D3PD - 2026-05-13 11:46 UTC

- Reason:
  - Active `S0_rcsbev` and `S0_rcsoccbev` trainings still occupy `livenode02` and `livenode03`.
  - Web search found D3PD, a recent camera-radar BEV distillation/dynamic-fusion paper with an explicit open-code URL.
- Web/source:
  - Paper page: `https://www.sciencedirect.com/science/article/pii/S0031320325010118`
  - Code URL from paper page: `https://github.com/no-Name128/D3PD`
- Clone:
  - Remote clone path: `/srv/nfs/shared/gnmp/paper_impls/D3PD`
  - Commit inspected: `e2f3a8e69e18e8a1eab1e63feae49d4524997539`
- Relevant files inspected:
  - `README.md`
  - `docs/getting_started.md`
  - `projects/configs/d3pd/d3pd-r50_sf_radar-detfeatsdistill.py`
  - `projects/configs/d3pd/d3pd-r101_sfd_samd_dcrd_smfd.py`
  - `projects/mmdet3d/models/detectors/d3pd.py`
  - `projects/mmdet3d/models/detectors/d3pd_v3.py`
  - `projects/mmdet3d/models/necks/fusion.py`
  - `projects/mmdet3d/models/losses/distill_loss.py`
- Findings:
  - D3PD is a BEVDet/CenterPoint-style radar-camera detector with a frozen LiDAR teacher and multiple distillation terms: sparse BEV feature distillation, radar multi-scale distillation, detection-result distillation, mask-focused distillation, and optional sampling-feature distillation.
  - Its radar branch uses radar voxelization, `PillarFeatureNet`, `PointPillarsScatter`, `SECOND`, and `SECONDFPN`; configs select radar dims `[0, 1, 2, 8, 9, 18]`, i.e. position plus selected radar attributes rather than the RaCFormer NB2 `[x, y, z, rcs, vx_comp, vy_comp, time]` convention.
  - `RC_BEV_Fusion` combines image/radar BEV through `BiDirectionWeightFusion` and `DualWeight_Fusion`; this is another learned scalar/spatial weighting family, similar in risk profile to already failed adaptive/context-gate branches.
  - `RC_BEV_Fusion_Sampling` predicts offsets from image/radar BEV and grid-samples image BEV features, but it is tightly coupled to the BEVDet feature layout and D3PD distillation plumbing.
- Adaptation note for RaCFormer:
  - Do not port the full D3PD training framework in the current loop; it needs a LiDAR teacher checkpoint, BEVDet/CenterPoint heads, and loss plumbing that does not match RaCFormer query fusion.
  - The useful low-risk idea is not another learned fusion gate; it is D3PD's distillation framing around sparse radar BEV and object-focused BEV regions. If current RCS branches fail and we have more time, a possible paper-quality follow-up would be a RaCFormer-specific radar BEV distillation/regularization branch against an existing strong teacher, not a direct D3PD port.
  - For immediate GPU use, the staged velocity/RCS BEV residual remains lower-risk and better aligned with current code than D3PD's learned BEV fusion.
- Decision:
  - Keep D3PD as paper framing and possible future distillation direction.
  - Do not spend current `livenode02`/`livenode03` GPU time on D3PD before active RCS/occupancy branches finish.

## Checkpoint poll - S0 occupancy-dominant BEV residual epoch 4 saved - 2026-05-13 11:42 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `1:57:24`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `1:47:27`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: epoch `5/12`, iteration `450/1000`.
  - `S0_rcsoccbev`: reached epoch `4/12`, saved checkpoint, then entered epoch `5/12`, iteration `50/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_4.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `epoch_4.pth`
  - `S0_rcsoccbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `epoch_4.pth`
- Errors:
  - Both train stderr files remain `0` bytes.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Both allowed GPU nodes remain occupied; no new GPU work.
  - Next meaningful transition is the `S0_rcsbev` epoch 5 checkpoint or a training/eval state change.

## Pending eval-chain audit - RCS and occupancy residual branches - 2026-05-13 11:50 UTC

- Reason:
  - Active trains are still running, so this was CPU/read-only verification of the already queued eval/summary chain.
  - Goal was to catch stale eval scripts before jobs `1326`/`1330` finish.
- SLURM dependency state:
  - `1327` `s0_rcsbev_eval`: pending `afterok:1326`, pinned to `livenode02`, 6h limit.
  - `1328` `s0_rcsbev_summary`: pending `afterok:1327`, pinned to `livenode02`, 20m limit.
  - `1331` `s0_rcsoccbev_eval`: pending `afterok:1330`, pinned to `livenode03`, 6h limit.
  - `1332` `s0_rcsoccbev_summary`: pending `afterok:1331`, pinned to `livenode03`, 20m limit.
- Script audit:
  - `bash -n` passed for:
    - `research/night_gen_phase1/staged_rcs_bev_residual/run_s0_rcsbev_eval_livenode02.sbatch`
    - `research/night_gen_phase1/staged_rcs_bev_residual/run_s0_rcsbev_summary_livenode02.sbatch`
    - `research/night_gen_phase1/staged_rcs_occ_residual/run_s0_rcsoccbev_eval_livenode03.sbatch`
    - `research/night_gen_phase1/staged_rcs_occ_residual/run_s0_rcsoccbev_summary_livenode03.sbatch`
  - `conda run -n racformerfix --no-capture-output python -m py_compile` passed for:
    - `research/night_gen_phase1/eval_by_condition.py`
    - `research/night_gen_phase1/staged_rcs_bev_residual/summarize_s0_rcsbev.py`
    - `research/night_gen_phase1/staged_rcs_occ_residual/summarize_s0_rcsoccbev.py`
  - Grep found no `livenode01` and no stale `__LATEST__` token in the staged RCS/occupancy scripts.
- Config audit:
  - `configs/racformer_train2k_day_rcsbev_research.py` sets `radar_rcs_bev_residual=dict(output_shape=(128, 128), rcs_index=3, rcs_scale=32.0)`.
  - `configs/racformer_train2k_day_rcsoccbev_research.py` sets `radar_rcs_bev_residual=dict(output_shape=(128, 128), rcs_index=3, rcs_scale=1000000.0)`.
- Decision:
  - The pending eval/summary chain is syntactically ready and correctly restricted to `livenode02`/`livenode03`.
  - Keep waiting for epoch 12 and final summaries; do not submit backup eval jobs unless the queued chain fails.

## Checkpoint poll - S0 RCS BEV residual epoch 5 saved - 2026-05-13 12:00 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `2:12:58`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `2:03:01`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: saved epoch `5/12`, then entered epoch `6/12`, iteration `50/1000`.
  - `S0_rcsoccbev`: epoch `5/12`, iteration `650/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_5.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `epoch_4.pth`, `epoch_5.pth`
  - `S0_rcsoccbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `epoch_4.pth`
- Errors:
  - Both train stderr files remain `0` bytes.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Both allowed GPU nodes remain occupied; no new GPU work.
  - Next meaningful transition is the `S0_rcsoccbev` epoch 5 checkpoint or a training/eval state change.

## Checkpoint poll - S0 occupancy-dominant BEV residual epoch 5 saved - 2026-05-13 12:08 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `2:21:29`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `2:11:32`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Node state:
  - `livenode02`: allocated, occupied by `S0_rcsbev`.
  - `livenode03`: allocated, occupied by `S0_rcsoccbev`.
- Latest train logs:
  - `S0_rcsbev`: epoch `6/12`, iteration `400/1000`.
  - `S0_rcsoccbev`: saved epoch `5/12`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_5.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `epoch_4.pth`, `epoch_5.pth`
  - `S0_rcsoccbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `epoch_4.pth`, `epoch_5.pth`
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Do not start new GPU work on `livenode02`; it is not idle.
  - Parallelizable work for now is limited to CPU/read-only preparation, tracker updates, and paper/open-implementation inspection.
  - Next meaningful transition is the `S0_rcsbev` epoch 6 checkpoint or a training/eval state change.

## Checkpoint poll - S0 RCS BEV residual epoch 6 saved - 2026-05-13 12:24 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `2:37:25`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `2:27:28`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: reached epoch `6/12`, iteration `1000/1000`.
  - `S0_rcsoccbev`: epoch `6/12`, iteration `600/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_6.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `epoch_4.pth`, `epoch_5.pth`, `epoch_6.pth`
  - `S0_rcsoccbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `epoch_4.pth`, `epoch_5.pth`
- Errors:
  - Both train stderr tails remain empty.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Keep both GPU lanes unchanged; no extra work on `livenode02`.
  - Next meaningful transition is the `S0_rcsoccbev` epoch 6 checkpoint or a training/eval state change.

## Checkpoint poll - S0 occupancy-dominant BEV residual epoch 6 saved - 2026-05-13 12:34 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `2:47:17`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `2:37:20`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: epoch `7/12`, iteration `400/1000`.
  - `S0_rcsoccbev`: reached epoch `6/12`, iteration `1000/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_6.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `epoch_4.pth`, `epoch_5.pth`, `epoch_6.pth`
  - `S0_rcsoccbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `epoch_4.pth`, `epoch_5.pth`, `epoch_6.pth`
- Errors:
  - Both train stderr tails remain empty.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - `livenode02` is still allocated to `S0_rcsbev`; do not start parallel GPU work there.
  - Next meaningful transition is the `S0_rcsbev` epoch 7 checkpoint or a training/eval state change.

## Checkpoint poll - S0 RCS BEV residual epoch 7 saved - 2026-05-13 12:50 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `3:03:24`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `2:53:27`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: reached epoch `7/12`, iteration `1000/1000`.
  - `S0_rcsoccbev`: epoch `7/12`, iteration `600/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_7.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `epoch_4.pth`, `epoch_5.pth`, `epoch_6.pth`, `epoch_7.pth`
  - `S0_rcsoccbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `epoch_4.pth`, `epoch_5.pth`, `epoch_6.pth`
- Errors:
  - Both train stderr tails remain empty.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Keep the active GPU allocation unchanged; `livenode02` is not a free parallel lane.
  - Next meaningful transition is the `S0_rcsoccbev` epoch 7 checkpoint or a training/eval state change.

## Checkpoint poll - S0 occupancy-dominant BEV residual epoch 7 saved - 2026-05-13 13:02 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `3:14:51`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `3:04:54`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: epoch `8/12`, iteration `450/1000`.
  - `S0_rcsoccbev`: saved epoch `7/12`, then entered epoch `8/12`, iteration `50/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_7.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `epoch_4.pth`, `epoch_5.pth`, `epoch_6.pth`, `epoch_7.pth`
  - `S0_rcsoccbev`: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `epoch_4.pth`, `epoch_5.pth`, `epoch_6.pth`, `epoch_7.pth`
- Errors:
  - Both train stderr tails remain empty.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Both allowed nodes remain allocated; no new GPU work.
  - Next meaningful transition is the `S0_rcsbev` epoch 8 checkpoint or a training/eval state change.

## Checkpoint poll - S0 RCS BEV residual epoch 8 saved - 2026-05-13 13:18 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `3:31:48`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `3:21:51`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: saved epoch `8/12`, then entered epoch `9/12`, iteration `100/1000`.
  - `S0_rcsoccbev`: epoch `8/12`, iteration `700/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_8.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth` through `epoch_8.pth`
  - `S0_rcsoccbev`: `epoch_1.pth` through `epoch_7.pth`
- Watch item:
  - Latest `S0_rcsbev` log line after epoch 8 shows a transient spike: total loss `35.16`, `loss_cls` `3.64`, and decoder cls losses around `3.64`.
  - No stderr content and the job is still running, so do not cancel from log noise alone; final eval metrics remain decisive.
- Errors:
  - Both train stderr tails remain empty.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Continue the active chain; no new GPU job on `livenode02`.
  - Next meaningful transition is the `S0_rcsoccbev` epoch 8 checkpoint or a training/eval state change.

## Checkpoint poll - S0 occupancy-dominant BEV residual epoch 8 saved - 2026-05-13 13:29 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `3:42:23`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `3:32:26`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: epoch `9/12`, iteration `500/1000`, loss back to `18.55`.
  - `S0_rcsoccbev`: saved epoch `8/12`, then entered epoch `9/12`, iteration `100/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_8.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth` through `epoch_8.pth`
  - `S0_rcsoccbev`: `epoch_1.pth` through `epoch_8.pth`
- Watch item:
  - The RCS branch spike at epoch `9/12` iteration `100/1000` appears transient; by iteration `500/1000`, total loss is back to `18.55`.
  - The occupancy branch now shows the same epoch `9/12` iteration `100/1000` pattern: total loss `35.32`, `loss_cls` `3.66`, decoder cls losses around `3.65` to `3.66`.
  - Because both branches show the same single-point pattern and stderr is empty, treat it as a training/logging watch item rather than an immediate failure.
- Errors:
  - Both train stderr tails remain empty.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Continue both active chains; do not start new GPU work on `livenode02` or `livenode03`.
  - Next meaningful transition is the `S0_rcsbev` epoch 9 checkpoint or a training/eval state change.

## Checkpoint poll - S0 RCS BEV residual epoch 9 saved - 2026-05-13 13:43 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `3:56:01`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `3:46:04`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: reached epoch `9/12`, iteration `1000/1000`, loss `18.16`.
  - `S0_rcsoccbev`: epoch `9/12`, iteration `600/1000`, loss `17.23`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_9.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth` through `epoch_9.pth`
  - `S0_rcsoccbev`: `epoch_1.pth` through `epoch_8.pth`
- Watch item:
  - The epoch-9 early loss spike settled on both active branches:
    - `S0_rcsbev`: loss `18.16` by epoch `9/12`, iteration `1000/1000`.
    - `S0_rcsoccbev`: loss `17.23` by epoch `9/12`, iteration `600/1000`.
  - Continue treating the spike as a transient training/logging pattern unless eval metrics fail.
- Errors:
  - Both train stderr tails remain empty.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Continue both active chains; no parallel GPU work on allocated nodes.
  - Next meaningful transition is the `S0_rcsoccbev` epoch 9 checkpoint or a training/eval state change.

## Checkpoint poll - S0 occupancy-dominant BEV residual epoch 9 saved - 2026-05-13 13:54 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `4:07:31`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `3:57:34`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: epoch `10/12`, iteration `450/1000`, loss `17.75`.
  - `S0_rcsoccbev`: saved epoch `9/12`, then entered epoch `10/12`, iteration `50/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_9.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth` through `epoch_9.pth`
  - `S0_rcsoccbev`: `epoch_1.pth` through `epoch_9.pth`
- Errors:
  - Both train stderr tails remain empty.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Continue both active chains; no new GPU work while `livenode02` and `livenode03` are allocated.
  - Next meaningful transition is the `S0_rcsbev` epoch 10 checkpoint or a training/eval state change.

## Checkpoint poll - S0 RCS BEV residual epoch 10 saved - 2026-05-13 14:10 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `4:23:03`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `4:13:06`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: saved epoch `10/12`, then entered epoch `11/12`, iteration `50/1000`.
  - `S0_rcsoccbev`: epoch `10/12`, iteration `650/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_10.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth` through `epoch_10.pth`
  - `S0_rcsoccbev`: `epoch_1.pth` through `epoch_9.pth`
- Errors:
  - Both train stderr tails remain empty.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Continue both active chains; no extra GPU work on `livenode02`.
  - Next meaningful transition is the `S0_rcsoccbev` epoch 10 checkpoint or a training/eval state change.

## Checkpoint poll - S0 occupancy-dominant BEV residual epoch 10 saved - 2026-05-13 14:21 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `4:34:34`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `4:24:37`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: epoch `11/12`, iteration `500/1000`, loss `16.99`.
  - `S0_rcsoccbev`: saved epoch `10/12`, then entered epoch `11/12`, iteration `100/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_10.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth` through `epoch_10.pth`
  - `S0_rcsoccbev`: `epoch_1.pth` through `epoch_10.pth`
- Errors:
  - Both train stderr tails remain empty.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Continue both active chains; no new GPU work on `livenode02` or `livenode03`.
  - Next meaningful transition is the `S0_rcsbev` epoch 11 checkpoint or a training/eval state change.

## Checkpoint poll - S0 RCS BEV residual epoch 11 saved - 2026-05-13 14:36 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `4:49:10`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `4:39:13`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: saved epoch `11/12`, then entered epoch `12/12`, iteration `50/1000`.
  - `S0_rcsoccbev`: epoch `11/12`, iteration `650/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_11.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth` through `epoch_11.pth`
  - `S0_rcsoccbev`: `epoch_1.pth` through `epoch_10.pth`
- Errors:
  - Both train stderr tails remain empty.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Continue both active chains.
  - Next meaningful RCS transition is `epoch_12.pth` followed by dependency handoff to eval job `1327` on `livenode02`.
  - Next occupancy transition is `S0_rcsoccbev` epoch 11 checkpoint.

## Checkpoint poll - S0 occupancy-dominant BEV residual epoch 11 saved - 2026-05-13 14:48 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: RUNNING on `livenode02`, elapsed `5:00:51`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `4:50:54`.
  - Eval/summary jobs `1327`, `1328`, `1331`, and `1332` remain dependency-pending.
- Latest train logs:
  - `S0_rcsbev`: epoch `12/12`, iteration `500/1000`, total loss `24.63`.
  - `S0_rcsoccbev`: saved epoch `11/12`, then entered epoch `12/12`, iteration `100/1000`.
- New checkpoint:
  - `outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_11.pth`
  - Size: `765690177` bytes.
- Current checkpoint inventory:
  - `S0_rcsbev`: `epoch_1.pth` through `epoch_11.pth`
  - `S0_rcsoccbev`: `epoch_1.pth` through `epoch_11.pth`
- Watch item:
  - `S0_rcsbev` shows another transient-looking classification spike at epoch `12/12`, iteration `500/1000`: total loss `24.63`, `loss_cls` `1.84`, decoder cls losses around `1.85`.
  - Stderr remains empty and both jobs are running; continue to final checkpoint and eval.
- Errors:
  - Both train stderr tails remain empty.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Continue to final checkpoints and dependency-triggered evals.
  - Do not start parallel GPU work on `livenode02`; it is still occupied by the final RCS train/eval chain.

## Final train checkpoint and eval handoff - S0 RCS BEV residual - 2026-05-13 15:03 UTC

- SLURM state on permitted nodes:
  - `1326` `s0_rcsbev`: completed and left the queue.
  - `1327` `s0_rcsbev_eval`: RUNNING on `livenode02`, elapsed `0:53`.
  - `1328` `s0_rcsbev_summary`: pending `afterok:1327`.
  - `1330` `s0_rcsoccbev`: RUNNING on `livenode03`, elapsed `5:05:54`.
  - `1331`/`1332`: occupancy eval/summary remain dependency-pending.
- Final RCS train checkpoint:
  - `outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_12.pth`
  - Timestamp: `2026-05-13 15:00 UTC`
  - Size: `765690177` bytes.
- Latest logs:
  - `S0_rcsbev`: finished epoch `12/12`, iteration `1000/1000`, loss `17.14`.
  - `S0_rcsoccbev`: epoch `12/12`, iteration `650/1000`, loss `17.33`.
- Eval log files:
  - `research/night_gen_phase1/results/S0_rcsbev/eval_slurm_1327.out`
  - `research/night_gen_phase1/results/S0_rcsbev/eval_slurm_1327.err`
  - Do not look for `slurm_1327.out`; the eval wrapper uses the `eval_slurm_*` prefix.
- Errors:
  - RCS train stderr has progress/noise from final validation/checkpoint work, not a Python traceback.
  - Occupancy train stderr tail remains empty.
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Keep `livenode02` reserved for `1327` eval and then `1328` summary.
  - Continue monitoring occupancy final checkpoint on `livenode03`.

## Final train checkpoint and eval handoff - S0 occupancy-dominant BEV residual - 2026-05-13 15:13 UTC

- SLURM state on permitted nodes:
  - `1327` `s0_rcsbev_eval`: RUNNING on `livenode02`, elapsed `10:54`.
  - `1328` `s0_rcsbev_summary`: pending `afterok:1327`.
  - `1330` `s0_rcsoccbev`: completed and left the queue.
  - `1331` `s0_rcsoccbev_eval`: RUNNING on `livenode03`, elapsed `0:12`.
  - `1332` `s0_rcsoccbev_summary`: pending `afterok:1331`.
- Final occupancy train checkpoint:
  - `outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_12.pth`
  - Timestamp: `2026-05-13 15:10 UTC`
  - Size: `765690177` bytes.
- Latest logs:
  - `S0_rcsoccbev`: finished epoch `12/12`, iteration `1000/1000`, loss `17.13`.
  - `S0_rcsbev_eval`: running full inference over `6019` samples.
- Eval log files:
  - RCS eval: `research/night_gen_phase1/results/S0_rcsbev/eval_slurm_1327.out` and `research/night_gen_phase1/results/S0_rcsbev/eval_slurm_1327.err`
  - Occupancy eval: `research/night_gen_phase1/results/S0_rcsoccbev/eval_slurm_1331.out` and `research/night_gen_phase1/results/S0_rcsoccbev/eval_slurm_1331.err`
- Summaries:
  - No `summary_metrics.md` exists yet for either branch.
- Decision:
  - Both allowed nodes are now occupied by eval jobs, not idle.
  - Wait for `1328`/`1332` summaries; do not launch duplicate evals or new GPU experiments.

## Final eval summary - S0 RCS BEV residual rejected - 2026-05-13 15:35 UTC

- Summary file:
  - `research/night_gen_phase1/results/S0_rcsbev/summary_metrics.md`
- Metrics:
  - Day: mAP `0.3152`, NDS `0.3753`, vs S0 `-0.00 pp` mAP / `+0.07 pp` NDS.
  - Night: mAP `0.1408`, NDS `0.1996`, vs S0 `-0.79 pp` mAP / `-1.55 pp` NDS.
  - Rain: mAP `0.2755`, NDS `0.3726`, vs S0 `+0.12 pp` mAP / `+0.13 pp` NDS.
  - Overall: mAP `0.3043`, NDS `0.3697`, vs S0 `+0.03 pp` mAP / `-0.00 pp` NDS.
- Gate verdict:
  - FAIL.
  - Fails the night NDS gate (`-1.55 pp`, threshold `>= -0.5 pp`) and does not meet the required night mAP gain (`-0.79 pp`, threshold `>= +1.0 pp`).
- Decision:
  - Reject `S0_rcsbev` as a promotion candidate.
  - Do not scale the RCS BEV residual branch.
  - Continue waiting for `S0_rcsoccbev` summary before choosing the next fallback.

## Fallback activation - S0 RCS+velocity BEV residual smoke submitted - 2026-05-13 15:45 UTC

- Reason:
  - `S0_rcsbev` failed the publication gate.
  - `livenode02` became idle while `S0_rcsoccbev` eval continued on `livenode03`.
  - This uses the user's requested parallel lane, but only through the guarded smoke-first path.
- Hypothesis:
  - SGDet3D/RCM-Fusion-inspired occupancy + RCS + compensated velocity BEV statistics may add motion context without another learned camera/radar gate.
- Remote backup:
  - `models/racformer.py.bak.rcsvelbev_20260513_1536`
- Uploaded files:
  - `models/racformer.py`
  - `configs/racformer_train2k_day_rcsvelbev_research.py`
  - `research/night_gen_phase1/staged_rcs_vel_residual/smoke_s0_rcsvelbev_model.sbatch`
  - `research/night_gen_phase1/staged_rcs_vel_residual/run_s0_rcsvelbev_livenode02.sbatch`
  - `research/night_gen_phase1/staged_rcs_vel_residual/run_s0_rcsvelbev_eval_livenode02.sbatch`
  - `research/night_gen_phase1/staged_rcs_vel_residual/run_s0_rcsvelbev_summary_livenode02.sbatch`
  - `research/night_gen_phase1/staged_rcs_vel_residual/summarize_s0_rcsvelbev.py`
- Verification before submit:
  - `bash -n` passed for all staged velocity sbatch wrappers.
  - `conda run -n racformerfix --no-capture-output python -m py_compile` passed for model, config, and summarizer.
  - Grep found no `livenode01` and no stale `__LATEST__` token.
- Submitted smoke:
  - `1333` `s0_rcsvelbev_smoke` on `livenode02`.
- Decision:
  - Do not submit full velocity training until smoke passes.
  - Continue waiting for `S0_rcsoccbev` summary on `livenode03`.

## Final eval summary - S0 occupancy-dominant BEV residual rejected - 2026-05-13 15:47 UTC

- Summary file:
  - `research/night_gen_phase1/results/S0_rcsoccbev/summary_metrics.md`
- Metrics:
  - Day: mAP `0.3126`, NDS `0.3761`, vs S0 `-0.27 pp` mAP / `+0.15 pp` NDS.
  - Night: mAP `0.1381`, NDS `0.2140`, vs S0 `-1.06 pp` mAP / `-0.11 pp` NDS.
  - Rain: mAP `0.2725`, NDS `0.3760`, vs S0 `-0.19 pp` mAP / `+0.47 pp` NDS.
  - Overall: mAP `0.3030`, NDS `0.3719`, vs S0 `-0.10 pp` mAP / `+0.21 pp` NDS.
- Gate verdict:
  - FAIL.
  - Fails the required night mAP gain (`-1.06 pp`, threshold `>= +1.0 pp`).
  - Passes the day, overall, and night-NDS guardrails, but that is not enough for promotion.
- Decision:
  - Reject `S0_rcsoccbev` as a promotion candidate.
  - Do not scale occupancy-only BEV residual.
  - Continue the already-started `S0_rcsvelbev` fallback on `livenode02`.

## Fallback activation - S0 RCS+velocity BEV residual train chain submitted - 2026-05-13 15:47 UTC

- Smoke result:
  - Job `1333` completed.
  - `smoke_slurm_1333.out` reports `radar_rcsvel_bev_residual (128, 128) (3, 4, 5) (32.0, 20.0, 20.0)`.
  - `state_keys 4`.
  - `half_forward_zero_init True`.
- Submitted chain on `livenode02`:
  - Train: `1334` `s0_rcsvelbev`, RUNNING.
  - Eval: `1335` `s0_rcsvelbev_eval`, pending `afterok:1334`.
  - Summary: `1336` `s0_rcsvelbev_summary`, pending `afterok:1335`.
- Decision:
  - `S0_rcsvelbev` is now the active fallback experiment.
  - Continue monitoring first checkpoint and stderr before assuming the train is stable.

## Parallel screening prep - S0 RCS+velocity epoch-1 eval on livenode03 - 2026-05-13 15:55 UTC

- Reason:
  - User reminded to check whether idle permitted nodes can be used in parallel.
  - `S0_rcsvelbev` is training on `livenode02`; `livenode03` is idle.
  - A separate epoch-1 screen can run on `livenode03` after `epoch_1.pth` appears without mutating the shared source or touching the final `S0_rcsvelbev` result directory.
- Added local and remote staged files:
  - `research/night_gen_phase1/staged_rcs_vel_residual/run_s0_rcsvelbev_epoch1_eval_livenode03.sbatch`
  - `research/night_gen_phase1/staged_rcs_vel_residual/run_s0_rcsvelbev_epoch1_summary_livenode03.sbatch`
  - `research/night_gen_phase1/staged_rcs_vel_residual/summarize_s0_rcsvelbev_epoch1.py`
- Validation:
  - Local `bash -n` passed for both new sbatch wrappers.
  - Local `python3 -m py_compile` passed for the new summarizer.
  - Remote `bash -n` passed for both new sbatch wrappers.
  - Remote `conda run -n racformerfix --no-capture-output python -m py_compile` passed for the new summarizer.
  - Grep found no `livenode01` and no stale `__LATEST__` token.
- Current active train state:
  - `1334` `s0_rcsvelbev` remains RUNNING on `livenode02`.
  - Latest observed log reached epoch `1/12`, iteration `250/1000`, with empty stderr.
  - No `epoch_1.pth` exists yet, so the epoch-1 screen has not been submitted.
- Decision:
  - Submit the epoch-1 eval/summary chain on `livenode03` only after `epoch_1.pth` exists.
  - Treat the epoch-1 result as a sanity signal, not final promotion evidence.

## CPU-side paper/code audit - CRKD - 2026-05-13 16:02 UTC

- Reason:
  - While waiting for the velocity checkpoint, searched for non-duplicative open implementations relevant to radar-camera robustness/distillation.
  - CRKD has an open implementation and was not already cloned under `paper_impls`.
- Web references checked:
  - Project/code: `https://github.com/Song-Jingyu/CRKD`
  - Project page: `https://song-jingyu.github.io/CRKD/`
  - CVPR 2024 paper page: `https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_CRKD_Enhanced_Camera-Radar_Object_Detection_with_Cross-modality_Knowledge_Distillation_CVPR_2024_paper.html`
- Remote clone:
  - `/srv/nfs/shared/gnmp/paper_impls/CRKD`
  - Commit: `c7e6893`
- Relevant implementation details:
  - `mmdet3d/models/fusers/gated.py` implements a two-sensor `GatedFuser` with per-sensor sigmoid feature weights followed by conv+BN+ReLU fusion.
  - `mmdet3d/models/fusion_models/feature_response_distiller.py` builds frozen teacher modules, initialized student modules, student detection loss, teacher soft-response loss, and fused-feature affinity KD.
  - `mmdet3d/models/losses/mask_feat_loss.py` masks feature distillation to BEV regions derived from GT boxes.
  - `mmdet3d/models/losses/affinity_loss.py` distills multi-scale BEV feature affinities after optional channel adaptation.
- Applicability to RaCFormer:
  - Too heavy for the active fallback because it expects separate trained teacher and student checkpoints and a BEVFusion-style detector layout.
  - Useful if all lightweight residual branches fail: adapt only the idea of box/region-masked BEV consistency or response distillation, not the full CRKD training stack.
  - The gated-fuser idea overlaps with already-rejected adaptive/context fusion branches, so do not repeat it without a stronger training signal such as distillation.
- Decision:
  - Do not apply CRKD now.
  - Keep it as a documented backup for a distillation-based branch if the velocity residual fails.

## Parallel screening submitted - S0 RCS+velocity epoch-1 eval - 2026-05-13 16:14 UTC

- Trigger:
  - `S0_rcsvelbev` saved `epoch_1.pth` while continuing full training on `livenode02`.
- Checkpoint:
  - `outputs/racformer_train2k_day_rcsvelbev_research/2026-05-13/12-47-35/epoch_1.pth`
  - Timestamp: `2026-05-13 16:13 UTC`
  - Size: `765745345` bytes.
- Main train:
  - `1334` `s0_rcsvelbev` remains RUNNING on `livenode02`.
  - Latest observed epoch-1 terminal log: `Epoch [1/12][1000/1000]`, loss `33.29`.
  - Main train stderr remains empty.
- Parallel screen on `livenode03`:
  - Eval: `1337` `s0_rcsvelbev_e1_eval`, RUNNING on `livenode03`.
  - Summary: `1338` `s0_rcsvelbev_e1_summary`, pending `afterok:1337`.
  - Eval writes to `research/night_gen_phase1/results/S0_rcsvelbev_epoch1/eval/`.
  - Eval log confirms it is using the explicit epoch-1 checkpoint above.
- Decision:
  - Continue full training regardless of epoch-1 screen unless the screen reveals a severe configuration/runtime issue.
  - Use the epoch-1 metrics only as a sanity signal, not as final promotion evidence.

## Parallel screening result - S0 RCS+velocity epoch-1 eval rejected as early signal - 2026-05-13 16:50 UTC

- Jobs:
  - Eval `1337` completed on `livenode03`.
  - Summary `1338` completed on `livenode03`.
- Summary file:
  - `research/night_gen_phase1/results/S0_rcsvelbev_epoch1/summary_metrics.md`
- Metrics:
  - Day: mAP `0.0450`, NDS `0.1167`, vs S0 `-27.03 pp` mAP / `-25.79 pp` NDS.
  - Night: mAP `0.0159`, NDS `0.0620`, vs S0 `-13.29 pp` mAP / `-15.31 pp` NDS.
  - Rain: mAP `0.0393`, NDS `0.1325`, vs S0 `-23.50 pp` mAP / `-23.88 pp` NDS.
  - Overall: mAP `0.0443`, NDS `0.1175`, vs S0 `-25.97 pp` mAP / `-25.23 pp` NDS.
- Gate verdict:
  - FAIL, as expected for an epoch-1 checkpoint.
- Main train status:
  - `1334` `s0_rcsvelbev` remains RUNNING on `livenode02`.
  - Latest observed log reached epoch `3/12`, iteration `600/1000`, with empty stderr.
  - Checkpoints `epoch_1.pth` and `epoch_2.pth` exist.
- Decision:
  - Do not treat epoch-1 metrics as a final rejection.
  - Do not keep launching repeated low-value early screens just because `livenode03` is idle; each full-val screen writes large result JSONs and does not change the decision before mature checkpoints.
  - Continue the main `S0_rcsvelbev` train/eval/summary chain.

## Parallel ablation staged - S0 occupancy + velocity BEV residual - 2026-05-13 17:05 UTC

- Reason:
  - `livenode03` is idle after the epoch-1 screen.
  - The active source already supports `extra_indices`/`extra_scales`, so a config-only ablation can run without mutating the shared model code used by `S0_rcsvelbev`.
  - Prior RCS-only and occupancy-dominant branches failed; this tests whether velocity helps while muting the RCS statistic.
- Hypothesis:
  - Radar occupancy + compensated velocity (`vx_comp`, `vy_comp`) may add motion context for night robustness while avoiding the noisy RCS channel.
- New files:
  - `configs/racformer_train2k_day_occvelbev_research.py`
  - `research/night_gen_phase1/staged_occ_vel_residual/smoke_s0_occvelbev_model.sbatch`
  - `research/night_gen_phase1/staged_occ_vel_residual/run_s0_occvelbev_livenode03.sbatch`
  - `research/night_gen_phase1/staged_occ_vel_residual/run_s0_occvelbev_eval_livenode03.sbatch`
  - `research/night_gen_phase1/staged_occ_vel_residual/run_s0_occvelbev_summary_livenode03.sbatch`
  - `research/night_gen_phase1/staged_occ_vel_residual/summarize_s0_occvelbev.py`
- Config:
  - `radar_rcs_bev_residual.output_shape=(128, 128)`
  - `rcs_index=3`, `rcs_scale=1000000.0` to mute RCS
  - `extra_indices=(4, 5)`, `extra_scales=(20.0, 20.0)` for compensated velocity
- Validation before submit:
  - Local and remote `bash -n` passed for all staged sbatch wrappers.
  - Local and remote `python -m py_compile` passed for the config/summarizer.
  - Grep found no `livenode01` and no stale `__LATEST__` token.
- Decision:
  - Submit smoke on `livenode03`.
  - Submit the full train/eval/summary chain on `livenode03` only if smoke passes.

## Parallel ablation submitted - S0 occupancy + velocity BEV residual - 2026-05-13 17:07 UTC

- Smoke result:
  - Job `1339` completed on `livenode03`.
  - `smoke_slurm_1339.out` reports `radar_occvel_bev_residual (128, 128) (3, 4, 5) (1000000.0, 20.0, 20.0)`.
  - `state_keys 4`.
  - `half_forward_zero_init True`.
- Submitted chain on `livenode03`:
  - Train: `1340` `s0_occvelbev`, RUNNING.
  - Eval: `1341` `s0_occvelbev_eval`, pending `afterok:1340`.
  - Summary: `1342` `s0_occvelbev_summary`, pending `afterok:1341`.
- Concurrent active chain:
  - `1334` `s0_rcsvelbev` continues RUNNING on `livenode02`; `1335` and `1336` remain dependency-pending.
- Decision:
  - Monitor `1340` through first checkpoint and stderr.
  - Do not mutate `models/racformer.py` while either active train/eval chain is in progress.

## Checkpoint poll - dual velocity branches healthy - 2026-05-13 17:16 UTC

- SLURM state on permitted nodes:
  - `1334` `s0_rcsvelbev`: RUNNING on `livenode02`, elapsed `1:29:10`.
  - `1335`/`1336`: RCS+velocity eval/summary remain dependency-pending.
  - `1340` `s0_occvelbev`: RUNNING on `livenode03`, elapsed `0:11:53`.
  - `1341`/`1342`: occupancy+velocity eval/summary remain dependency-pending.
- `S0_rcsvelbev`:
  - Checkpoints through `epoch_3.pth` exist.
  - Latest observed log reached epoch `4/12`, iteration `400/1000`, loss `21.45`.
  - Stderr tail remains empty.
- `S0_occvelbev`:
  - No checkpoint yet, as expected for early epoch 1.
  - Latest observed log reached epoch `1/12`, iteration `400/1000`, loss `29.95`.
  - Stderr tail remains empty.
- Summaries:
  - No final `summary_metrics.md` exists yet for either active branch.
- Decision:
  - Continue both active chains.
  - Next useful transition is `S0_occvelbev` epoch-1 checkpoint or `S0_rcsvelbev` epoch-4/5 checkpoint.

## Checkpoint poll - S0 occvel epoch 1 saved - 2026-05-13 17:31 UTC

- SLURM state on permitted nodes:
  - `1334` `s0_rcsvelbev`: RUNNING on `livenode02`, elapsed `1:45:14`.
  - `1335`/`1336`: RCS+velocity eval/summary remain dependency-pending.
  - `1340` `s0_occvelbev`: RUNNING on `livenode03`, elapsed `0:27:57`.
  - `1341`/`1342`: occupancy+velocity eval/summary remain dependency-pending.
- `S0_rcsvelbev`:
  - Checkpoints through `epoch_4.pth` exist.
  - Latest observed log reached epoch `4/12`, iteration `1000/1000`, loss `21.19`.
  - Watch item: transient classification-loss spike at epoch `4/12`, iteration `900/1000`, loss `32.54`; job continued, completed epoch 4, and stderr remained empty.
- `S0_occvelbev`:
  - `epoch_1.pth` exists at `outputs/racformer_train2k_day_occvelbev_research/2026-05-13/14-04-52/epoch_1.pth`.
  - Latest observed log reached epoch `2/12`, iteration `50/1000`, loss `25.00`.
  - Stderr tail remains empty.
- Summaries:
  - No final `summary_metrics.md` exists yet for either active branch.
- Decision:
  - Continue both active chains.
  - Do not launch more early full-val screens unless a checkpoint pattern suggests a specific actionable risk.

## Checkpoint poll - dual velocity branches mid-train - 2026-05-13 18:02 UTC

- SLURM state on permitted nodes:
  - `1334` `s0_rcsvelbev`: RUNNING on `livenode02`, elapsed `2:16:31`.
  - `1335`/`1336`: RCS+velocity eval/summary remain dependency-pending.
  - `1340` `s0_occvelbev`: RUNNING on `livenode03`, elapsed `0:59:14`.
  - `1341`/`1342`: occupancy+velocity eval/summary remain dependency-pending.
- `S0_rcsvelbev`:
  - Checkpoints through `epoch_5.pth` exist.
  - Latest observed log reached epoch `6/12`, iteration `200/1000`, loss `19.88`.
  - Stderr tail remains empty.
- `S0_occvelbev`:
  - Checkpoints through `epoch_2.pth` exist.
  - Latest observed log reached epoch `3/12`, iteration `200/1000`, loss `22.94`.
  - Stderr tail remains empty.
- Summaries:
  - No final `summary_metrics.md` exists yet for either active branch.
- Decision:
  - Continue both active train/eval/summary chains.
  - Next useful transition is `S0_rcsvelbev` epoch-7/8 checkpoint or `S0_occvelbev` epoch-4/5 checkpoint.

## Checkpoint poll - dual velocity branches still healthy - 2026-05-13 18:33 UTC

- SLURM state on permitted nodes:
  - `1334` `s0_rcsvelbev`: RUNNING on `livenode02`, elapsed `2:47:43`.
  - `1335`/`1336`: RCS+velocity eval/summary remain dependency-pending.
  - `1340` `s0_occvelbev`: RUNNING on `livenode03`, elapsed `1:30:26`.
  - `1341`/`1342`: occupancy+velocity eval/summary remain dependency-pending.
- `S0_rcsvelbev`:
  - Checkpoints through `epoch_6.pth` exist.
  - Latest observed log reached epoch `7/12`, iteration `400/1000`, loss `19.18`.
  - Stderr tail remains empty.
- `S0_occvelbev`:
  - Checkpoints through `epoch_3.pth` exist.
  - Latest observed log reached epoch `4/12`, iteration `400/1000`, loss `21.29`.
  - Stderr tail remains empty.
- Summaries:
  - No final `summary_metrics.md` exists yet for either active branch.
- Decision:
  - Continue both active train/eval/summary chains.
  - Next useful transition is `S0_rcsvelbev` epoch-9/10 checkpoint or `S0_occvelbev` epoch-6/7 checkpoint.

## Checkpoint poll - RCS velocity epoch 8 verified - 2026-05-13 19:18 UTC

- SLURM state on permitted nodes:
  - `1334` `s0_rcsvelbev`: RUNNING on `livenode02`, elapsed `3:30:42`.
  - `1335`/`1336`: RCS+velocity eval/summary remain dependency-pending.
  - `1340` `s0_occvelbev`: RUNNING on `livenode03`, elapsed `2:13:25`.
  - `1341`/`1342`: occupancy+velocity eval/summary remain dependency-pending.
- `S0_rcsvelbev`:
  - `epoch_8.pth` is now a valid nonzero checkpoint: `765745473` bytes.
  - Checkpoints currently verified through `epoch_8.pth`.
  - Latest observed log reached epoch `9/12`, iteration `50/1000`, loss `18.19`.
  - Stderr tail remains empty.
- `S0_occvelbev`:
  - Checkpoints currently verified through `epoch_5.pth`.
  - Latest complete visible log reached epoch `5/12`, iteration `1000/1000`, with losses around `20-21`.
  - Stderr tail remains empty.
- Summaries:
  - No final `summary_metrics.md` exists yet for either active branch.
- Decision:
  - Continue both active train/eval/summary chains.
  - Do not mutate `models/racformer.py` while these chains are still running.

## Paper/code audit - radar BEV fallbacks checked - 2026-05-13 19:21 UTC

- Reason:
  - User asked to search paper inspirations if stuck and to check open implementations before adopting paper ideas.
  - Active branches already overlap radar BEV/velocity ideas, so source-code audit is useful before choosing the next fallback.
- Web/code sources checked:
  - SGDet3D: `https://github.com/shawnnnkb/SGDet3D`, remote clone `/srv/nfs/shared/gnmp/paper_impls/SGDet3D`, commit `83f8c7d`.
  - RCTrans: `https://github.com/liyih/RCTrans`, remote clone `/srv/nfs/shared/gnmp/paper_impls/RCTrans`, commit `47884e8`.
  - RCBEVDet: `https://github.com/VDIGPKU/RCBEVDet`, remote clone `/srv/nfs/shared/gnmp/paper_impls/RCBEVDet`, released zip already extracted.
- Relevant implementation details:
  - SGDet3D `RadarPillarFeatureNet` adds per-pillar centered velocity/SNR-like channels before PFN encoding.
  - SGDet3D also builds sparse image-plane `radar_depth`, but this is dataset/pipeline-heavy for RaCFormer and not a quick patch.
  - RCTrans has a radar voxel pipeline plus `Radar_dense_encoder_tf`, a U-Net-like radar dense encoder with self-attention at the lowest BEV scale.
  - RCBEVDet builds image-plane radar maps with depth, raw velocity, compensated velocity, velocity magnitudes, and RCS, then fuses radar/image BEV with deformable attention.
- Decision:
  - Do not transplant full SGDet3D/RCTrans/RCBEVDet modules while active jobs are running.
  - If both current velocity branches fail, the most realistic next paper-inspired patch is a small learned radar BEV adapter or centered-velocity normalization, not full deformable cross-attention.

## Checkpoint poll - dual velocity branches progressing - 2026-05-13 19:21 UTC

- SLURM state on permitted nodes:
  - `1334` `s0_rcsvelbev`: RUNNING on `livenode02`, elapsed `3:34:10`.
  - `1335`/`1336`: RCS+velocity eval/summary remain dependency-pending.
  - `1340` `s0_occvelbev`: RUNNING on `livenode03`, elapsed `2:16:53`.
  - `1341`/`1342`: occupancy+velocity eval/summary remain dependency-pending.
- Capacity check:
  - `livenode02` and `livenode03` are both `alloc`.
  - Each active train job reserves `16` CPUs and about `128 GB` memory on its node.
  - No additional non-conflicting parallel SLURM branch should be launched right now.
- `S0_rcsvelbev`:
  - Checkpoints currently verified through `epoch_8.pth`.
  - Latest observed log reached epoch `9/12`, iteration `150/1000`, loss `18.94`.
  - Watch item: one classification-loss spike at epoch `9/12`, iteration `100/1000`, loss `35.47`; next logged step recovered and stderr stayed empty.
- `S0_occvelbev`:
  - Checkpoints currently verified through `epoch_5.pth`.
  - Latest observed log reached epoch `6/12`, iteration `200/1000`, loss `19.97`.
  - Stderr tail remains empty.
- Summaries:
  - No final `summary_metrics.md` exists yet for either active branch.
- Decision:
  - Continue both active chains.
  - Next useful checkpoint transition is `S0_rcsvelbev` epoch 9/10 and `S0_occvelbev` epoch 6/7.

## Local fallback staged only - occupancy velocity time BEV - 2026-05-13 19:28 UTC

- File:
  - Local only: `remote_patch_work/configs/racformer_train2k_day_occveltimebev_research.py`.
- Hypothesis:
  - Occupancy plus compensated velocity plus sweep-relative time may keep motion/temporal context while muting noisy RCS.
  - Uses existing `RadarRCSBEVResidual` support, so it is config-only.
- Config:
  - `rcs_index=3`, `rcs_scale=1000000.0` to mute RCS.
  - `extra_indices=(4, 5, 6)` for compensated velocity x/y plus time.
  - `extra_scales=(20.0, 20.0, 1.0)`.
- Validation:
  - `python -m py_compile remote_patch_work/configs/racformer_train2k_day_occveltimebev_research.py` passed locally.
- Status:
  - Not uploaded.
  - Not submitted.
  - Do not submit until one of `livenode02`/`livenode03` is free and current summaries justify another config-only fallback.

## Checkpoint poll - epoch boundaries reached - 2026-05-13 19:44 UTC

- SLURM state on permitted nodes:
  - `1334` `s0_rcsvelbev`: RUNNING on `livenode02`, elapsed `3:57:12`.
  - `1335`/`1336`: RCS+velocity eval/summary remain dependency-pending.
  - `1340` `s0_occvelbev`: RUNNING on `livenode03`, elapsed `2:39:55`.
  - `1341`/`1342`: occupancy+velocity eval/summary remain dependency-pending.
- `S0_rcsvelbev`:
  - Checkpoints now verified through `epoch_9.pth`.
  - `epoch_9.pth` size: `765745473` bytes.
  - Latest observed log reached epoch `10/12`, iteration `50/1000`, loss `18.49`.
  - Stderr remains empty.
- `S0_occvelbev`:
  - Checkpoints now verified through `epoch_6.pth`.
  - `epoch_6.pth` size: `765745473` bytes.
  - Latest observed log reached epoch `7/12`, iteration `100/1000`, loss `19.84`.
  - Stderr remains empty.
- Summaries:
  - No final `summary_metrics.md` exists yet for either active branch.
- Decision:
  - Continue both active train/eval/summary chains.
  - Next useful transition is `S0_rcsvelbev` epoch 10/11 or train completion, and `S0_occvelbev` epoch 7/8.

## Checkpoint poll - RCS velocity epoch 10 and occ velocity epoch 7 - 2026-05-13 20:10 UTC

- SLURM state on permitted nodes:
  - `1334` `s0_rcsvelbev`: RUNNING on `livenode02`, elapsed `4:23:23`.
  - `1335`/`1336`: RCS+velocity eval/summary remain dependency-pending.
  - `1340` `s0_occvelbev`: RUNNING on `livenode03`, elapsed `3:06:06`.
  - `1341`/`1342`: occupancy+velocity eval/summary remain dependency-pending.
- `S0_rcsvelbev`:
  - Checkpoints now verified through `epoch_10.pth`.
  - `epoch_10.pth` size: `765745473` bytes.
  - Latest observed log reached epoch `11/12`, iteration `50/1000`, loss `18.15`.
  - Watch item: transient loss spike at epoch `10/12`, iteration `900/1000`, loss `25.95`; next steps recovered and checkpoint saved.
  - Stderr remains empty.
- `S0_occvelbev`:
  - Checkpoints now verified through `epoch_7.pth`.
  - `epoch_7.pth` size: `765745473` bytes.
  - Latest observed log reached epoch `8/12`, iteration `100/1000`, loss `19.26`.
  - Stderr remains empty.
- Summaries:
  - No final `summary_metrics.md` exists yet for either active branch.
- Decision:
  - Continue both active train/eval/summary chains.
  - `S0_rcsvelbev` is now closest to train completion; next useful transition is `epoch_11.pth` or eval start.

## Checkpoint poll - RCS velocity final epoch started - 2026-05-13 20:36 UTC

- SLURM state on permitted nodes:
  - `1334` `s0_rcsvelbev`: RUNNING on `livenode02`, elapsed `4:49:08`.
  - `1335`/`1336`: RCS+velocity eval/summary remain dependency-pending.
  - `1340` `s0_occvelbev`: RUNNING on `livenode03`, elapsed `3:31:51`.
  - `1341`/`1342`: occupancy+velocity eval/summary remain dependency-pending.
- `S0_rcsvelbev`:
  - Checkpoints now verified through `epoch_11.pth`.
  - `epoch_11.pth` size: `765745473` bytes.
  - Latest observed log reached epoch `12/12`, iteration `50/1000`, loss `18.02`.
  - Watch item: transient loss spike at epoch `11/12`, iteration `1000/1000`, loss `27.95`; next logged step recovered and checkpoint saved.
  - Stderr remains empty.
- `S0_occvelbev`:
  - Checkpoints now verified through `epoch_8.pth`.
  - `epoch_8.pth` size: `765745473` bytes.
  - Latest observed log reached epoch `9/12`, iteration `50/1000`, loss `18.31`.
  - Watch item: transient loss spike at epoch `8/12`, iteration `950/1000`, loss `22.59`; next logged step recovered and checkpoint saved.
  - Stderr remains empty.
- Summaries:
  - No final `summary_metrics.md` exists yet for either active branch.
- Decision:
  - Continue both active chains.
  - Next useful transition is `S0_rcsvelbev` training completion and eval job `1335` starting on `livenode02`.

## Checkpoint poll - RCS velocity final checkpoint saved - 2026-05-13 21:02 UTC

- SLURM state on permitted nodes:
  - `1334` `s0_rcsvelbev`: still RUNNING on `livenode02`, elapsed `5:14:52`.
  - `1335`/`1336`: RCS+velocity eval/summary remain dependency-pending.
  - `1340` `s0_occvelbev`: RUNNING on `livenode03`, elapsed `3:57:35`.
  - `1341`/`1342`: occupancy+velocity eval/summary remain dependency-pending.
- `S0_rcsvelbev`:
  - Final training checkpoint `epoch_12.pth` is saved and nonzero.
  - `epoch_12.pth` size: `765745473` bytes.
  - Final training log reached epoch `12/12`, iteration `1000/1000`, loss `17.28`.
  - After saving the checkpoint, job `1334` entered a 300-sample post-train validation/test phase inside the same SLURM job, so dependency eval `1335` has not started yet.
  - No final `summary_metrics.md` yet.
- `S0_occvelbev`:
  - Last verified checkpoint remains `epoch_8.pth`.
  - Job remains active on `livenode03`.
- Decision:
  - Wait for `1334` to fully exit and for dependency eval `1335` to start.
  - Do not manually submit another eval unless the dependency chain fails.

## Eval handoff - S0 RCS velocity eval running - 2026-05-13 21:06 UTC

- SLURM state on permitted nodes:
  - `1335` `s0_rcsvelbev_eval`: RUNNING on `livenode02`, elapsed `3:43`.
  - `1336` `s0_rcsvelbev_summary`: dependency-pending.
  - `1340` `s0_occvelbev`: RUNNING on `livenode03`, elapsed `4:01:39`.
  - `1341`/`1342`: occupancy+velocity eval/summary remain dependency-pending.
- `S0_rcsvelbev`:
  - Train job `1334` exited and dependency eval `1335` started automatically.
  - Eval is using `outputs/racformer_train2k_day_rcsvelbev_research/2026-05-13/12-47-35/epoch_12.pth`.
  - Eval reports inference on `6019` samples.
  - `eval_slurm_1335.err` currently contains normal setup logs only.
  - No `summary_metrics.md` yet.
- `S0_occvelbev`:
  - Checkpoints now verified through `epoch_9.pth`.
  - `epoch_9.pth` size: `765745473` bytes.
  - Latest observed log reached epoch `10/12`, iteration `200/1000`, loss `18.36`.
  - Stderr remains empty.
- Decision:
  - Monitor eval `1335` until summary job `1336` runs.
  - Continue letting `S0_occvelbev` train in parallel on `livenode03`.

## Final result - S0 RCS velocity BEV residual - 2026-05-13 21:38 UTC

- Branch:
  - `S0_rcsvelbev`
  - Config: `configs/racformer_train2k_day_rcsvelbev_research.py`
  - Checkpoint: `outputs/racformer_train2k_day_rcsvelbev_research/2026-05-13/12-47-35/epoch_12.pth`
- Metrics vs S0:
  - day: `0.3159 / 0.3749`, delta `+0.06 pp / +0.03 pp`
  - night: `0.1506 / 0.2150`, delta `+0.18 pp / -0.01 pp`
  - rain: `0.2806 / 0.3682`, delta `+0.63 pp / -0.31 pp`
  - overall: `0.3062 / 0.3697`, delta `+0.22 pp / -0.00 pp`
- Gate verdict:
  - FAIL.
  - Main miss: night mAP improved only `+0.18 pp`, below the `+1.0 pp` target.
  - Day and overall were safe; night NDS was effectively flat.
- Decision:
  - Do not promote as paper result.
  - Keep as evidence that lightweight RCS+velocity BEV residual is stable but insufficient.

## Parallel fallback smoke submitted - S0 occupancy velocity time BEV - 2026-05-13 21:38 UTC

- Reason:
  - `livenode02` became idle after `S0_rcsvelbev` eval/summary completed.
  - User asked to parallelize using idle permitted nodes when possible.
  - This is a config-only fallback using already-loaded radar feature dim `6` for sweep-relative time, while muting RCS.
- Files:
  - Config uploaded: `configs/racformer_train2k_day_occveltimebev_research.py`
  - Staged scripts uploaded: `research/night_gen_phase1/staged_occ_vel_time_residual/`
- Validation before submit:
  - Local `bash -n` and `python -m py_compile` passed.
  - Remote `bash -n` and `conda run -n racformerfix python -m py_compile` passed.
  - Guard grep found no `livenode01`, no `__LATEST__`, and no accidental `S0_occvelbev`/`rcsvel` stage-name reuse.
- Smoke:
  - Submitted job `1343` on `livenode02`.
  - Expected logs: `research/night_gen_phase1/results/S0_occveltimebev/smoke_slurm_1343.out/err`.
- Status:
  - Full train is not submitted yet.
  - Submit full train/eval/summary chain only if smoke passes.

## Parallel fallback submitted - S0 occupancy velocity time BEV - 2026-05-13 21:39 UTC

- Smoke result:
  - Job `1343` completed on `livenode02`.
  - Output: `radar_occveltime_bev_residual (128, 128) (3, 4, 5, 6) (1000000.0, 20.0, 20.0, 1.0)`.
  - Output: `state_keys 4`.
  - Output: `half_forward_zero_init True`.
  - Smoke stderr contained expected MMEngine/MMCV init logging only.
- Submitted chain on permitted idle node `livenode02`:
  - Train `1344` `s0_occveltimebev`: RUNNING on `livenode02`.
  - Eval `1345` `s0_occveltimebev_eval`: dependency-pending after train.
  - Summary `1346` `s0_occveltimebev_summary`: dependency-pending after eval.
- Concurrent branch:
  - `1340` `s0_occvelbev` remains RUNNING on `livenode03`.
- Decision:
  - Monitor `S0_occvelbev` and `S0_occveltimebev` in parallel.
  - Do not launch additional jobs unless another permitted node becomes idle and a clear non-overlapping hypothesis is available.

## Checkpoint poll - S0 occvel final epoch active - 2026-05-13 22:05 UTC

- SLURM state on permitted nodes:
  - `1340` `s0_occvelbev`: RUNNING on `livenode03`, elapsed `5:00:46`.
  - `1341`/`1342`: occupancy+velocity eval/summary remain dependency-pending.
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed `25:55`.
  - `1345`/`1346`: occupancy+velocity+time eval/summary remain dependency-pending.
- `S0_occvelbev`:
  - Checkpoints now verified through `epoch_11.pth`.
  - `epoch_11.pth` size: `765745473` bytes.
  - Latest observed log reached epoch `12/12`, iteration `500/1000`, loss `24.94`.
  - Watch item: transient classification-loss spike at epoch `12/12`, iteration `500/1000`; stderr remains empty.
- `S0_occveltimebev`:
  - No checkpoint yet.
  - Latest observed log reached epoch `1/12`, iteration `950/1000`, loss `25.62`.
  - Stderr remains empty.
- Decision:
  - Continue both chains.
  - Next useful transitions are `S0_occvelbev` training completion/eval start and `S0_occveltimebev` epoch-1 checkpoint.

## Checkpoint poll - S0 occveltime epoch 1 saved - 2026-05-13 22:12 UTC

- SLURM state on permitted nodes:
  - `1340` `s0_occvelbev`: RUNNING on `livenode03`, elapsed `5:08:26`.
  - `1341`/`1342`: occupancy+velocity eval/summary remain dependency-pending.
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed `33:35`.
  - `1345`/`1346`: occupancy+velocity+time eval/summary remain dependency-pending.
- `S0_occvelbev`:
  - Checkpoints remain verified through `epoch_11.pth`.
  - Latest observed log reached epoch `12/12`, iteration `800/1000`, loss `17.65`.
  - Watch item: transient loss spike at epoch `12/12`, iteration `600/1000`, loss `35.00`; later logs recovered and stderr stayed empty.
- `S0_occveltimebev`:
  - Checkpoints now verified through `epoch_1.pth`.
  - `epoch_1.pth` size: `765772993` bytes.
  - Latest observed log reached epoch `2/12`, iteration `250/1000`, loss `24.63`.
  - Stderr remains empty.
- Decision:
  - Continue both chains.
  - Next useful transition is `S0_occvelbev` training completion/eval start.

## Checkpoint poll - S0 occvel final checkpoint saved - 2026-05-13 22:19 UTC

- SLURM state on permitted nodes:
  - `1340` `s0_occvelbev`: still RUNNING on `livenode03`, elapsed `5:15:07`.
  - `1341`/`1342`: occupancy+velocity eval/summary remain dependency-pending.
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed `40:16`.
  - `1345`/`1346`: occupancy+velocity+time eval/summary remain dependency-pending.
- `S0_occvelbev`:
  - Final training checkpoint `epoch_12.pth` is saved and nonzero.
  - `epoch_12.pth` size: `765745473` bytes.
  - Final training log reached epoch `12/12`, iteration `1000/1000`, loss `17.46`.
  - After saving the checkpoint, job `1340` entered a 300-sample post-train validation/test phase inside the same SLURM job, so dependency eval `1341` has not started yet.
- `S0_occveltimebev`:
  - Last verified checkpoint remains `epoch_1.pth`.
  - Job remains active on `livenode02`.
- Decision:
  - Wait for `1340` to fully exit and for dependency eval `1341` to start.

## Eval handoff - S0 occvel eval running - 2026-05-13 22:24 UTC

- SLURM state on permitted nodes:
  - `1341` `s0_occvelbev_eval`: RUNNING on `livenode03`, elapsed `4:20`.
  - `1342` `s0_occvelbev_summary`: dependency-pending.
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed `45:01`.
  - `1345`/`1346`: occupancy+velocity+time eval/summary remain dependency-pending.
- `S0_occvelbev`:
  - Train job `1340` exited and dependency eval `1341` started automatically.
  - Eval is using `outputs/racformer_train2k_day_occvelbev_research/2026-05-13/14-04-52/epoch_12.pth`.
  - Eval reports inference on `6019` samples.
  - Latest compact progress: about `905/6019`.
  - `eval_slurm_1341.err` currently contains normal setup logs only.
- `S0_occveltimebev`:
  - Last verified checkpoint remains `epoch_1.pth`.
  - Latest observed log reached epoch `2/12`, iteration `700/1000`, loss `23.87`.
  - Stderr remains empty.
- Decision:
  - Monitor eval `1341` until summary job `1342` runs.
  - Continue `S0_occveltimebev` training on `livenode02`.

## Parallelism poll - both permitted nodes allocated - 2026-05-13 22:38 UTC

- SLURM state on permitted nodes:
  - `1341` `s0_occvelbev_eval`: RUNNING on `livenode03`, elapsed `18:39`.
  - `1342` `s0_occvelbev_summary`: dependency-pending.
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed `59:20`.
  - `1345`/`1346`: occupancy+velocity+time eval/summary remain dependency-pending.
- Node availability:
  - `livenode02`: allocated, `16/0/0/16`.
  - `livenode03`: allocated, `16/0/0/16`.
  - `livenode01`: intentionally not used due known driver problem.
- `S0_occvelbev`:
  - Eval progress reached `4265/6019`, about `3.9 task/s`, ETA about `447s`.
  - No summary metrics file yet.
- `S0_occveltimebev`:
  - Checkpoints now verified through `epoch_2.pth`.
  - `epoch_1.pth` size: `765772993` bytes.
  - `epoch_2.pth` size: `765773121` bytes.
  - Latest observed log reached epoch `3/12`, iteration `250/1000`, loss `23.81`.
  - Stderr remains empty.
- Decision:
  - No additional parallel job is safe right now because both allowed nodes are allocated.
  - Next useful transition is `S0_occvelbev` summary completion; then evaluate gate before launching a new hypothesis.

## Eval poll - S0 occvel bbox metrics running - 2026-05-13 22:52 UTC

- SLURM state on permitted nodes:
  - `1341` `s0_occvelbev_eval`: still RUNNING on `livenode03`, elapsed about `32:06`.
  - `1342` `s0_occvelbev_summary`: dependency-pending.
  - `1344` `s0_occveltimebev`: still RUNNING on `livenode02`, elapsed about `1:12:47`.
  - `1345`/`1346`: occupancy+velocity+time eval/summary remain dependency-pending.
- `S0_occvelbev`:
  - Full inference loop completed `6019/6019`.
  - Evaluation wrote `research/night_gen_phase1/results/S0_occvelbev/eval/submission_overall/pts_bbox/results_nusc.json`.
  - Job is now in `Evaluating bboxes of pts_bbox`.
  - No summary metrics file yet.
- `S0_occveltimebev`:
  - Checkpoints remain verified through `epoch_2.pth`.
  - Latest observed training logs were in epoch `3/12`.
- Decision:
  - Continue waiting for `1341` to exit and `1342` to summarize.
  - Do not launch another branch while `livenode02` and `livenode03` remain allocated.

## Final result - S0 occupancy velocity BEV residual - 2026-05-13 22:56 UTC

- Branch: `S0_occvelbev`
- Config: `configs/racformer_train2k_day_occvelbev_research.py`
- Summary file: `research/night_gen_phase1/results/S0_occvelbev/summary_metrics.md`
- Hypothesis:
  - Keep the same zero-init radar BEV residual structure, mute RCS, and expose occupancy plus compensated radar velocity channels.
  - Intended effect: retain geometry while removing noisy RCS and letting velocity improve night split behavior.
- Metrics:
  - day: `0.2990 / 0.3645`, delta `-1.63 pp / -1.01 pp`.
  - night: `0.1348 / 0.1977`, delta `-1.40 pp / -1.74 pp`.
  - rain: `0.2665 / 0.3647`, delta `-0.78 pp / -0.66 pp`.
  - overall: `0.2896 / 0.3603`, delta `-1.44 pp / -0.94 pp`.
- Gate verdict: FAIL.
  - Night mAP misses by `2.40 pp` relative to the `+1.0 pp` required gain.
  - Day mAP misses by `0.63 pp` relative to the `-1.0 pp` allowed regression.
  - Night NDS misses by `1.24 pp` relative to the `-0.5 pp` allowed regression.
  - Overall mAP barely stays within tolerance, but it does not compensate for night/day failures.
- Decision:
  - Do not pursue the velocity-only BEV residual as a paper candidate.
  - Keep `S0_occveltimebev` running because it is already staged and uses the same allowed-model-source path with time added as an isolated hypothesis.
  - Recheck permitted-node availability after `S0_occvelbev` completion; if `livenode03` is idle, only start another branch after selecting a non-overlapping hypothesis.

## Parallel fallback smoke submitted - S0 RCS velocity time BEV - 2026-05-13 22:59 UTC

- Trigger:
  - `S0_occvelbev` failed the gate.
  - `livenode03` became idle while `S0_occveltimebev` continues on `livenode02`.
- Hypothesis:
  - `S0_rcsvelbev` was the only radar-stat BEV branch with a positive night mAP delta.
  - Add sweep-relative time to `RCS + vx_comp + vy_comp` instead of muting RCS, so the branch can separate recent and stale radar support while preserving the RCS cue.
- Scope:
  - Config-only branch using existing `RadarRCSBEVResidual.extra_indices` support.
  - No mutation of `models/racformer.py` while active jobs run.
  - Uses `livenode03`; no use of `livenode01`.
- Local and remote files staged:
  - `configs/racformer_train2k_day_rcsveltimebev_research.py`.
  - `research/night_gen_phase1/staged_rcs_vel_time_residual/smoke_s0_rcsveltimebev_model.sbatch`.
  - `research/night_gen_phase1/staged_rcs_vel_time_residual/run_s0_rcsveltimebev_livenode03.sbatch`.
  - `research/night_gen_phase1/staged_rcs_vel_time_residual/run_s0_rcsveltimebev_eval_livenode03.sbatch`.
  - `research/night_gen_phase1/staged_rcs_vel_time_residual/run_s0_rcsveltimebev_summary_livenode03.sbatch`.
  - `research/night_gen_phase1/staged_rcs_vel_time_residual/summarize_s0_rcsveltimebev.py`.
- Validation before submit:
  - Local `bash -n` passed for staged sbatch files.
  - Local `python -m py_compile` passed for config and summarizer.
  - Remote `bash -n` passed for staged sbatch files.
  - Remote `conda run -n racformerfix python -m py_compile` passed for config and summarizer.
  - Guard grep found no `livenode01`, no `__LATEST__`, and no accidental `S0_occveltimebev`/`S0_occvelbev` stage-name reuse.
- Smoke job:
  - Submitted `1347` on `livenode03`.
  - Expected logs: `research/night_gen_phase1/results/S0_rcsveltimebev/smoke_slurm_1347.out/err`.
- Decision:
  - If smoke passes, submit train/eval/summary dependency chain on `livenode03`.
  - Continue monitoring `S0_occveltimebev` on `livenode02`.

## Parallel fallback submitted - S0 RCS velocity time BEV - 2026-05-13 23:01 UTC

- Smoke result:
  - Job `1347` completed successfully.
  - Output confirmed `radar_rcsveltime_bev_residual (128, 128) (3, 4, 5, 6) (32.0, 20.0, 20.0, 1.0)`.
  - `state_keys 4`.
  - `half_forward_zero_init True`.
  - Stderr contained only expected MMEngine/MMCV init-weight logs.
- Submitted full dependency chain on `livenode03`:
  - Train: `1348` `s0_rcsveltimebev`.
  - Eval: `1349` `s0_rcsveltimebev_eval`, `afterok:1348`.
  - Summary: `1350` `s0_rcsveltimebev_summary`, `afterok:1349`.
- Concurrent branch:
  - `S0_occveltimebev` remains active on `livenode02` as job `1344`.
  - `1345`/`1346` remain dependency-pending for its eval/summary.
- Decision:
  - Monitor both time branches in parallel.
  - Next useful transition is `1348` dispatching on `livenode03` or `S0_occveltimebev` reaching the next checkpoint.

## Dispatch confirmation - dual time branches running - 2026-05-13 23:03 UTC

- SLURM state:
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed about `1:23:46`.
  - `1345`/`1346`: dependency-pending after `1344`.
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `1:20`.
  - `1349`/`1350`: dependency-pending after `1348`.
- `S0_rcsveltimebev`:
  - Confirmed host: `livenode03`.
  - GPU: NVIDIA GeForce RTX 4090, detected by training script.
  - Work dir: `outputs/racformer_train2k_day_rcsveltimebev_research/2026-05-13/20-02-10`.
  - Stderr size: `0`.
- `S0_occveltimebev`:
  - Checkpoints now verified through `epoch_3.pth`.
  - Latest observed training log reached epoch `4/12`, iteration `150/1000`, loss `22.09`.
  - Stderr remains empty.
- Decision:
  - Both permitted nodes are intentionally allocated.
  - Next useful transition is `S0_rcsveltimebev` epoch-1 checkpoint or `S0_occveltimebev` epoch-4 checkpoint.

## Checkpoint poll - S0 occveltime epoch 4 saved - 2026-05-13 23:25 UTC

- SLURM state:
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed about `1:46:14`.
  - `1345`/`1346`: dependency-pending.
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `23:48`.
  - `1349`/`1350`: dependency-pending.
- `S0_occveltimebev`:
  - Checkpoints now verified through `epoch_4.pth`.
  - `epoch_4.pth` size: `765773121` bytes.
  - Reached epoch `4/12`, iteration `1000/1000`, loss `21.01`.
  - Entered epoch `5/12`, iteration `50/1000`, loss `20.65`.
  - Stderr remains empty.
- `S0_rcsveltimebev`:
  - No checkpoint yet.
  - Latest observed log reached epoch `1/12`, iteration `850/1000`, loss `25.85`.
  - Stderr remains empty.
- Decision:
  - Continue both branches.
  - Next useful transition is `S0_rcsveltimebev` epoch-1 checkpoint.

## Checkpoint poll - S0 rcsveltime epoch 1 saved - 2026-05-13 23:32 UTC

- SLURM state:
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed about `1:52:48`.
  - `1345`/`1346`: dependency-pending.
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `30:22`.
  - `1349`/`1350`: dependency-pending.
- `S0_rcsveltimebev`:
  - Checkpoints now verified through `epoch_1.pth`.
  - `epoch_1.pth` size: `765772993` bytes.
  - Reached epoch `1/12`, iteration `1000/1000`, loss `33.41`; this was a transient classification spike.
  - Entered epoch `2/12`; by iteration `100/1000`, loss recovered to `24.98`.
  - Stderr remains empty.
- `S0_occveltimebev`:
  - Checkpoints remain verified through `epoch_4.pth`.
  - Latest observed log reached epoch `5/12`, iteration `300/1000`, loss `21.26`.
  - Stderr remains empty.
- Decision:
  - Continue both time branches.
  - Next useful transition is `S0_occveltimebev` epoch-5 checkpoint or `S0_rcsveltimebev` epoch-2 checkpoint.

## Mid-train poll and fallback paper scan - 2026-05-13 23:43 UTC

- SLURM state:
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed about `2:04:32`.
  - `1345`/`1346`: dependency-pending.
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `42:06`.
  - `1349`/`1350`: dependency-pending.
- `S0_occveltimebev`:
  - Checkpoints remain verified through `epoch_4.pth`.
  - Latest observed log reached epoch `5/12`, iteration `750/1000`, loss `21.03`.
  - Transient classification spike at epoch `5/12`, iteration `500/1000`, loss `34.51`; recovered by iteration `550/1000`, loss `20.43`.
  - Stderr remains empty.
- `S0_rcsveltimebev`:
  - Checkpoints remain verified through `epoch_1.pth`.
  - Latest observed log reached epoch `2/12`, iteration `550/1000`, loss `24.81`.
  - Stderr remains empty.
- Bounded paper/code fallback scan, for use only if both active time branches fail:
  - `SGDet3D` (IEEE RA-L 2025, open code `https://github.com/shawnnnkb/SGDet3D`): inspect `projects/SGDet3D/mmdet3d_plugin/models/detectors/SGDet3D.py`, `models/necks/MRF3Net.py`, `models/necks/BEVCross_modal_attention.py`, and `models/voxel_encoder/`. Minimal non-repeated idea: radar pre-encoder normalization / centered pillar features before radar encoding, optionally appending velocity magnitude and point count. Ranked first because it changes radar token quality before fusion instead of adding another BEV-stat residual.
  - `CRT-Fusion` (NeurIPS 2024/arXiv 2024, open code `https://github.com/mjseong0414/CRT-Fusion`): inspect `mmdet3d/models/detectors/crtfusion.py`, `mmdet3d/models/necks/view_transformer_crtfusion.py`, and phase configs. Minimal idea: one-step radar temporal carry into radar tokens using previous-frame radar BEV and compensated velocity/time.
  - `TransCAR` (IEEE 2023, open code `https://github.com/pangsu0613/TransCAR`): inspect decoder/dense-head radar refinement modules. Minimal idea: radar-only top-K uncertain query refinement after normal decoder update.
- Decision:
  - Do not launch any fallback while `livenode02` and `livenode03` are allocated.
  - If both current time branches fail, prefer the SGDet3D-style radar pre-encoder normalization as the next small branch.

## Checkpoint poll - occveltime epoch 5 and rcsveltime epoch 2 saved - 2026-05-13 23:59 UTC

- SLURM state:
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed about `2:20:25`.
  - `1345`/`1346`: dependency-pending.
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `57:59`.
  - `1349`/`1350`: dependency-pending.
- `S0_occveltimebev`:
  - Checkpoints now verified through `epoch_5.pth`.
  - `epoch_5.pth` size: `765773121` bytes.
  - Latest observed log reached epoch `6/12`, iteration `350/1000`, loss `20.30`.
  - Stderr remains empty.
- `S0_rcsveltimebev`:
  - Checkpoints now verified through `epoch_2.pth`.
  - `epoch_2.pth` size: `765773121` bytes.
  - Reached epoch `2/12`, iteration `1000/1000`, loss `23.22`.
  - Entered epoch `3/12`; latest observed log reached iteration `200/1000`, loss `22.96`.
  - Stderr remains empty.
- Decision:
  - Continue both branches.
  - Next useful transition is `S0_occveltimebev` epoch-6 checkpoint or `S0_rcsveltimebev` epoch-3 checkpoint.

## Checkpoint poll - occveltime epoch 6 and rcsveltime epoch 3 saved - 2026-05-14 00:30 UTC

- SLURM state:
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed about `2:51:16`.
  - `1345`/`1346`: dependency-pending.
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `1:28:50`.
  - `1349`/`1350`: dependency-pending.
- `S0_occveltimebev`:
  - Checkpoints now verified through `epoch_6.pth`.
  - `epoch_6.pth` size: `765773121` bytes.
  - Latest observed log reached epoch `7/12`, iteration `500/1000`, loss `19.40`.
  - Stderr remains empty.
- `S0_rcsveltimebev`:
  - Checkpoints now verified through `epoch_3.pth`.
  - `epoch_3.pth` size: `765773121` bytes.
  - Latest observed log reached epoch `4/12`, iteration `350/1000`, loss `21.84`.
  - Stderr remains empty.
- Decision:
  - Continue both branches.
  - Next useful transition is `S0_occveltimebev` epoch-7/8 checkpoint or `S0_rcsveltimebev` epoch-4 checkpoint.

## Checkpoint poll - occveltime epoch 7 and rcsveltime epoch 4 saved - 2026-05-14 01:01 UTC

- SLURM state:
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed about `3:22:00`.
  - `1345`/`1346`: dependency-pending.
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `1:59:34`.
  - `1349`/`1350`: dependency-pending.
- `S0_occveltimebev`:
  - Checkpoints now verified through `epoch_7.pth`.
  - `epoch_7.pth` size: `765773121` bytes.
  - Latest observed log reached epoch `8/12`, iteration `700/1000`, loss `18.20`.
  - Stderr remains empty.
- `S0_rcsveltimebev`:
  - Checkpoints now verified through `epoch_4.pth`.
  - `epoch_4.pth` size: `765773121` bytes.
  - Latest observed log reached epoch `5/12`, iteration `550/1000`, loss `20.52`.
  - Transient classification spike at epoch `5/12`, iteration `500/1000`, loss `34.53`; recovered by iteration `550/1000`.
  - Stderr remains empty.
- Decision:
  - Continue both branches.
  - Next useful transition is `S0_occveltimebev` epoch-8/9 checkpoint or `S0_rcsveltimebev` epoch-5 checkpoint.

## Checkpoint poll - occveltime epoch 8 and rcsveltime epoch 5 saved - 2026-05-14 01:27 UTC

- SLURM state:
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed about `3:47:42`.
  - `1345`/`1346`: dependency-pending.
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `2:25:16`.
  - `1349`/`1350`: dependency-pending.
- `S0_occveltimebev`:
  - Checkpoints now verified through `epoch_8.pth`.
  - `epoch_8.pth` size: `765773121` bytes.
  - Latest observed log reached epoch `9/12`, iteration `700/1000`, loss `18.24`.
  - Stderr remains empty.
- `S0_rcsveltimebev`:
  - Checkpoints now verified through `epoch_5.pth`.
  - `epoch_5.pth` size: `765773121` bytes.
  - Latest observed log reached epoch `6/12`, iteration `550/1000`, loss `20.66`.
  - Stderr remains empty.
- Decision:
  - Continue both branches.
  - Next useful transition is `S0_occveltimebev` epoch-9/10 checkpoint or train completion handoff.

## Checkpoint poll - occveltime epoch 9 and rcsveltime epoch 6 saved - 2026-05-14 01:52 UTC

- SLURM state:
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed about `4:13:33`.
  - `1345`/`1346`: dependency-pending.
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `2:51:07`.
  - `1349`/`1350`: dependency-pending.
- `S0_occveltimebev`:
  - Checkpoints now verified through `epoch_9.pth`.
  - `epoch_9.pth` size: `765773121` bytes.
  - Latest observed log reached epoch `10/12`, iteration `700/1000`, loss `17.93`.
  - Stderr remains empty.
- `S0_rcsveltimebev`:
  - Checkpoints now verified through `epoch_6.pth`.
  - `epoch_6.pth` size: `765773121` bytes.
  - Latest observed log reached epoch `7/12`, iteration `500/1000`, loss `19.27`.
  - Stderr remains empty.
- Decision:
  - Continue both branches.
  - Next useful transition is `S0_occveltimebev` epoch-10/11 checkpoint or final train handoff.

## Checkpoint poll - occveltime epoch 10 and rcsveltime epoch 7 saved - 2026-05-14 02:13 UTC

- SLURM state:
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed about `4:34:19`.
  - `1345`/`1346`: dependency-pending.
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `3:11:53`.
  - `1349`/`1350`: dependency-pending.
- `S0_occveltimebev`:
  - Checkpoints now verified through `epoch_10.pth`.
  - `epoch_10.pth` size: `765773121` bytes.
  - Latest observed log reached epoch `11/12`, iteration `450/1000`, loss `17.46`.
  - Eval job `1345` has not started yet.
  - Stderr remains empty.
- `S0_rcsveltimebev`:
  - Checkpoints now verified through `epoch_7.pth`.
  - `epoch_7.pth` size: `765773121` bytes.
  - Latest observed log reached epoch `8/12`, iteration `300/1000`, loss `19.72`.
  - Stderr remains empty.
- Decision:
  - Continue both branches.
  - Poll more tightly for `S0_occveltimebev` final checkpoint and eval handoff.

## Final-epoch poll - S0 occveltime epoch 11 saved - 2026-05-14 02:29 UTC

- SLURM state:
  - `1344` `s0_occveltimebev`: RUNNING on `livenode02`, elapsed about `4:50:07`.
  - `1345`/`1346`: dependency-pending.
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `3:27:41`.
  - `1349`/`1350`: dependency-pending.
- `S0_occveltimebev`:
  - Checkpoints now verified through `epoch_11.pth`.
  - `epoch_11.pth` size: `765773121` bytes.
  - Reached epoch `11/12`, iteration `1000/1000`, loss `27.80`; this was another transient classification spike.
  - Entered final epoch `12/12`; by iteration `100/1000`, loss recovered to `17.70`.
  - Eval job `1345` has not started yet.
  - Stderr remains empty.
- `S0_rcsveltimebev`:
  - Checkpoints remain verified through `epoch_7.pth`.
  - Latest observed log reached epoch `8/12`, iteration `900/1000`, loss `18.78`.
  - Stderr remains empty.
- Decision:
  - Continue both branches.
  - Next useful transition is `S0_occveltimebev` final checkpoint or eval handoff.

## Eval handoff - S0 occveltime eval running - 2026-05-14 02:55 UTC

- SLURM state:
  - `1345` `s0_occveltimebev_eval`: RUNNING on `livenode02`, elapsed about `0:36`.
  - `1346` `s0_occveltimebev_summary`: dependency-pending.
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `3:53:30`.
  - `1349`/`1350`: dependency-pending.
- `S0_occveltimebev`:
  - Train job `1344` finished and dependency eval `1345` started automatically.
  - Final checkpoint `epoch_12.pth` is saved and nonzero.
  - `epoch_12.pth` size: `765773121` bytes.
  - Final training log reached epoch `12/12`, iteration `1000/1000`, loss `17.18`.
  - Eval is using `outputs/racformer_train2k_day_occveltimebev_research/2026-05-13/18-39-42/epoch_12.pth`.
  - Eval reports inference on `6019` samples.
- `S0_rcsveltimebev`:
  - Checkpoints now verified through `epoch_8.pth`.
  - `epoch_8.pth` size: `765773121` bytes.
  - Latest observed log reached epoch `9/12`, iteration `900/1000`, loss `18.71`.
  - Stderr remains empty.
- Decision:
  - Monitor `1345` until `1346` summarizes and evaluate against S0 gate.
  - Continue `S0_rcsveltimebev` training on `livenode03`.

## Eval progress - S0 occveltime inference running - 2026-05-14 03:03 UTC

- SLURM state:
  - `1345` `s0_occveltimebev_eval`: RUNNING on `livenode02`, elapsed about `8:24`.
  - `1346` `s0_occveltimebev_summary`: dependency-pending.
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `4:01:18`.
  - `1349`/`1350`: dependency-pending.
- `S0_occveltimebev`:
  - Eval progress reached about `1877/6019`, around `4.0 task/s`.
  - Eval stderr contains normal setup logs only.
  - No summary metrics file yet.
- `S0_rcsveltimebev`:
  - Checkpoints now verified through `epoch_9.pth`.
  - `epoch_9.pth` size: `765773121` bytes.
  - Latest observed log reached epoch `10/12`, iteration `200/1000`, loss `18.14`.
  - Stderr remains empty.
- Decision:
  - Continue `S0_occveltimebev` eval and wait for summary.
  - Continue `S0_rcsveltimebev` training.

## Candidate result - S0 occupancy velocity time BEV residual passed gate - 2026-05-14 03:34 UTC

- Branch: `S0_occveltimebev`
- Config: `configs/racformer_train2k_day_occveltimebev_research.py`
- Summary file: `research/night_gen_phase1/results/S0_occveltimebev/summary_metrics.md`
- Hypothesis:
  - Mute RCS, keep radar occupancy plus compensated velocity, and add sweep-relative time as a zero-init BEV residual.
  - Intended effect: keep lightweight motion and recency context while avoiding noisy RCS.
- Metrics:
  - day: `0.3093 / 0.3733`, delta `-0.60 pp / -0.13 pp`.
  - night: `0.1637 / 0.2228`, delta `+1.49 pp / +0.77 pp`.
  - rain: `0.2780 / 0.3695`, delta `+0.37 pp / -0.18 pp`.
  - overall: `0.3011 / 0.3693`, delta `-0.29 pp / -0.05 pp`.
- Gate verdict: PASS.
  - Night mAP exceeds the `+1.0 pp` required gain by `+0.49 pp`.
  - Day mAP remains within the allowed `-1.0 pp` regression.
  - Overall mAP remains within the allowed `-1.5 pp` regression.
  - Night NDS improves rather than regresses.
- Evidence:
  - Eval completed full `6019` sample pass and wrote `eval/eval_by_condition.json`.
  - Summary job `1346` wrote `summary_metrics.md`.
- Decision:
  - Treat this as the current paper-candidate branch.
  - Do not mark the overall objective complete from one seed/run alone; next useful work is a replication or confirmation if `livenode02` is idle, while `S0_rcsveltimebev` continues on `livenode03`.

## Replication submitted - S0 occveltime seed1 - 2026-05-14 03:39 UTC

- Trigger:
  - `S0_occveltimebev` passed the S0 promotion gate.
  - `livenode02` became idle after eval/summary completed.
  - `train.py` hard-codes seed 0, so a deterministic rerun would not be meaningful.
- Scope:
  - Seed-1 confirmation run using a copied trainer at `research/night_gen_phase1/staged_occ_vel_time_seed1/train_seeded.py`.
  - Shared `train.py` and `models/racformer.py` were not mutated.
  - Config and output names are isolated under `S0_occveltimebev_seed1`.
  - Uses `livenode02`; no use of `livenode01`.
- Local and remote files staged:
  - `configs/racformer_train2k_day_occveltimebev_seed1_research.py`.
  - `research/night_gen_phase1/staged_occ_vel_time_seed1/train_seeded.py`.
  - `research/night_gen_phase1/staged_occ_vel_time_seed1/smoke_s0_occveltimebev_seed1_model.sbatch`.
  - `research/night_gen_phase1/staged_occ_vel_time_seed1/run_s0_occveltimebev_seed1_livenode02.sbatch`.
  - `research/night_gen_phase1/staged_occ_vel_time_seed1/run_s0_occveltimebev_seed1_eval_livenode02.sbatch`.
  - `research/night_gen_phase1/staged_occ_vel_time_seed1/run_s0_occveltimebev_seed1_summary_livenode02.sbatch`.
  - `research/night_gen_phase1/staged_occ_vel_time_seed1/summarize_s0_occveltimebev_seed1.py`.
- Validation:
  - Local `bash -n` passed for staged sbatch files.
  - Local `python -m py_compile` passed for seeded trainer, config, and summarizer.
  - Remote `bash -n` passed for staged sbatch files.
  - Remote `conda run -n racformerfix python -m py_compile` passed for seeded trainer, config, and summarizer.
  - Guard grep found no `livenode01`, no `__LATEST__`, and no stale seed-0 output/stage names in operational fields.
  - Smoke job `1351` passed: `seed 1`, correct residual stats, and `half_forward_zero_init True`.
- Submitted full dependency chain:
  - Train: `1352` `s0_occveltime_s1`.
  - Eval: `1353` `s0_occveltime_s1_eval`, `afterok:1352`.
  - Summary: `1354` `s0_occveltime_s1_summary`, `afterok:1353`.
- Concurrent branch:
  - `S0_rcsveltimebev` continues on `livenode03` as job `1348`.
- Decision:
  - Monitor seed-1 replication and `S0_rcsveltimebev` in parallel.
  - A seed-1 pass would make `S0_occveltimebev` much stronger as a paper candidate.

## Seed1 replication fix and resubmission - 2026-05-14 03:47 UTC

- Issue:
  - First seed1 train job `1352` failed immediately before training.
  - Error: `ModuleNotFoundError: No module named 'utils'` from the copied trainer.
  - Dependent jobs `1353` and `1354` were left pending with unsatisfied dependencies.
- Fix:
  - Patched only the copied staged trainer:
    - `research/night_gen_phase1/staged_occ_vel_time_seed1/train_seeded.py`
    - Added `sys.path.insert(0, os.getcwd())` before repo-local imports.
  - Shared `train.py`, `models/racformer.py`, and active `S0_rcsveltimebev` code were not changed.
- Cleanup and validation:
  - Canceled stale dependency jobs `1353` and `1354`.
  - Login-node `--help` confirmed the `utils` import issue was fixed, then hit the repo's existing nuScenes import side effect on the login node.
  - Compute-node trainer-entry smoke job `1355` ran on `livenode02` and reached argparse with no stderr.
  - Previous model/config smoke job `1351` had already verified seed `1`, residual stats, CUDA availability, and zero-init behavior.
- Resubmitted full dependency chain on idle `livenode02`:
  - Train: `1356` `s0_occveltime_s1`.
  - Eval: `1357` `s0_occveltime_s1_eval`, `afterok:1356`.
  - Summary: `1358` `s0_occveltime_s1_summary`, `afterok:1357`.
- Concurrent branch:
  - `S0_rcsveltimebev` continues on `livenode03` as `1348 -> 1349 -> 1350`.
- Decision:
  - This uses idle `livenode02` as the replication lane while `livenode03` continues the independent RCS+velocity+time branch.
  - Next check: confirm `1356` starts cleanly and watch both summaries.

## Parallel lane poll - seed1 clean start - 2026-05-14 03:49 UTC

- Queue:
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `0:46`.
  - `1357` `s0_occveltime_s1_eval`: dependency-pending on `1356`.
  - `1358` `s0_occveltime_s1_summary`: dependency-pending on `1357`.
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `4:45:20`.
  - `1349`/`1350`: dependency-pending on the RCS+velocity+time train/eval chain.
- Seed1 clean-start evidence:
  - `1356` entered training setup and created checkpoint output under `outputs/racformer_train2k_day_occveltimebev_seed1_research/2026-05-14/00-46-41`.
  - Stderr for `1356` was empty at the poll.
  - The previous `utils` import failure is not recurring.
- RCS+velocity+time branch:
  - `1348` reached epoch `11/12`, iteration `900/1000`.
  - Stderr for `1348` remained `0` bytes.
- Decision:
  - Keep both allowed nodes occupied: `livenode02` for seed-1 replication, `livenode03` for the independent RCS+velocity+time branch.
  - Next meaningful decision point is the first completed summary among `1358` and `1350`.

## Parallel lane poll - both GPU lanes healthy - 2026-05-14 03:51 UTC

- Queue:
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `4:46:37`.
  - `1349`/`1350`: dependency-pending.
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `2:03`.
  - `1357`/`1358`: dependency-pending.
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
  - No use of `livenode01`.
- RCS+velocity+time branch:
  - Latest observed progress: epoch `11/12`, iteration `950/1000`.
  - `slurm_1348.err`: `0` bytes.
  - Summary not ready.
- Seed1 replication:
  - `1356` remains running with `0` stderr bytes.
  - Summary not ready.
- Decision:
  - No intervention needed. Continue monitoring for `1348` train completion and `1349` eval start; seed1 remains a longer-running replication lane.

## Active scoring path validation - 2026-05-14 03:53 UTC

- Scope:
  - Checked the CPU-side scripts that will score the two active branches after GPU jobs finish.
- Files checked:
  - `research/night_gen_phase1/staged_rcs_vel_time_residual/run_s0_rcsveltimebev_eval_livenode03.sbatch`
  - `research/night_gen_phase1/staged_rcs_vel_time_residual/run_s0_rcsveltimebev_summary_livenode03.sbatch`
  - `research/night_gen_phase1/staged_rcs_vel_time_residual/summarize_s0_rcsveltimebev.py`
  - `research/night_gen_phase1/staged_occ_vel_time_seed1/run_s0_occveltimebev_seed1_eval_livenode02.sbatch`
  - `research/night_gen_phase1/staged_occ_vel_time_seed1/run_s0_occveltimebev_seed1_summary_livenode02.sbatch`
  - `research/night_gen_phase1/staged_occ_vel_time_seed1/summarize_s0_occveltimebev_seed1.py`
  - `research/night_gen_phase1/eval_by_condition.py`
- Validation:
  - `bash -n` passed for all four active eval/summary sbatch scripts.
  - `conda run -n racformerfix --no-capture-output python -m py_compile` passed for both summarizers and `eval_by_condition.py`.
- Path sanity:
  - `S0_rcsveltimebev` eval resolves latest work dir under `outputs/racformer_train2k_day_rcsveltimebev_research` and requires `epoch_12.pth`.
  - `S0_occveltimebev_seed1` eval resolves latest work dir under `outputs/racformer_train2k_day_occveltimebev_seed1_research` and requires `epoch_12.pth`.
  - Both summarizers compare against `S0` and apply the same publication gate:
    - night mAP >= `+1.0 pp`
    - day mAP >= `-1.0 pp`
    - overall mAP >= `-1.5 pp`
    - night NDS >= `-0.5 pp`
- Decision:
  - No scoring-path fix needed before the dependent eval/summary jobs run.

## Parallel lane poll - RCS final epoch starting - 2026-05-14 03:56 UTC

- Queue:
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `4:48:56`.
  - `1349`/`1350`: dependency-pending.
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `4:22`.
  - `1357`/`1358`: dependency-pending.
- RCS+velocity+time branch:
  - `epoch_11.pth` exists, size `765773121` bytes.
  - Latest observed log reached epoch `11/12`, iteration `1000/1000`.
  - `slurm_1348.err`: `0` bytes.
- Seed1 replication:
  - Latest observed log reached epoch `1/12`, iteration `100/1000`.
  - `slurm_1356.err`: `0` bytes.
- Decision:
  - Continue waiting for `epoch_12.pth` and the dependent `1349` full-val eval.

## Active script guard grep - 2026-05-14 03:57 UTC

- Checked active configs and staged scripts for operational hazards:
  - `configs/racformer_train2k_day_rcsveltimebev_research.py`
  - `configs/racformer_train2k_day_occveltimebev_seed1_research.py`
  - `research/night_gen_phase1/staged_rcs_vel_time_residual/`
  - `research/night_gen_phase1/staged_occ_vel_time_seed1/`
- Results:
  - No `livenode01` references.
  - No `__LATEST__` placeholders.
  - Active stage/output names point at `S0_rcsveltimebev` and `S0_occveltimebev_seed1`.
- Decision:
  - No guardrail fix needed. Continue monitoring the active dependency chains.

## Parallel lane poll - RCS epoch 12 underway - 2026-05-14 03:59 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
  - No allowed GPU is idle; no use of `livenode01`.
- Queue:
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `4:50:44`.
  - `1349`/`1350`: dependency-pending.
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `6:10`.
  - `1357`/`1358`: dependency-pending.
- RCS+velocity+time branch:
  - Latest observed progress: epoch `12/12`, iteration `100/1000`.
  - `slurm_1348.err`: `0` bytes.
  - No summary file yet.
- Seed1 replication:
  - Latest observed progress: epoch `1/12`, iteration `150/1000`.
  - `slurm_1356.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - No intervention or new experiment launch because both allowed GPUs are occupied.
  - Next useful check is `epoch_12.pth` plus `1349` eval start for `S0_rcsveltimebev`.

## Parallel lane poll - RCS final epoch progress - 2026-05-14 04:00 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
  - No allowed GPU is idle; no use of `livenode01`.
- Queue:
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `4:51:55`.
  - `1349`/`1350`: dependency-pending.
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `7:21`.
  - `1357`/`1358`: dependency-pending.
- RCS+velocity+time branch:
  - Latest observed progress: epoch `12/12`, iteration `150/1000`.
  - `slurm_1348.err`: `0` bytes.
  - No summary file yet.
- Seed1 replication:
  - Latest observed progress: epoch `1/12`, iteration `250/1000`.
  - `slurm_1356.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - Continue monitoring. The next meaningful state change should be `epoch_12.pth` and `1349` eval start for `S0_rcsveltimebev`.

## Parallel lane poll - RCS final epoch mid-progress - 2026-05-14 04:04 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
  - No allowed GPU is idle; no use of `livenode01`.
- Queue:
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `4:55:58`.
  - `1349`/`1350`: dependency-pending.
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `11:24`.
  - `1357`/`1358`: dependency-pending.
- RCS+velocity+time branch:
  - Latest observed progress: epoch `12/12`, iteration `300/1000`.
  - Job log ETA at iteration `300/1000`: about `18:04` remaining.
  - `slurm_1348.err`: `0` bytes.
  - No summary file yet.
- Seed1 replication:
  - Latest observed progress: epoch `1/12`, iteration `400/1000`.
  - `slurm_1356.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - Continue waiting for `S0_rcsveltimebev` training completion and full-val eval job `1349`.

## Parallel lane poll - RCS final epoch halfway - 2026-05-14 04:10 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
  - No allowed GPU is idle; no use of `livenode01`.
- Queue:
  - `1348` `s0_rcsveltimebev`: RUNNING on `livenode03`, elapsed about `5:02:00`.
  - `1349`/`1350`: dependency-pending.
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `17:26`.
  - `1357`/`1358`: dependency-pending.
- RCS+velocity+time branch:
  - Latest observed progress: epoch `12/12`, iteration `500/1000`.
  - Job log ETA at iteration `500/1000`: about `12:54` remaining.
  - `slurm_1348.err`: `0` bytes.
  - No summary file yet.
- Seed1 replication:
  - Latest observed progress: epoch `1/12`, iteration `650/1000`.
  - `slurm_1356.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - Continue waiting for `epoch_12.pth`, then full-val eval `1349`.

## RCS branch train complete and eval running - 2026-05-14 04:18 UTC

- Queue:
  - `1349` `s0_rcsveltimebev_eval`: RUNNING on `livenode03`, elapsed about `0:40`.
  - `1350` `s0_rcsveltimebev_summary`: dependency-pending.
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `31:39`.
  - `1357`/`1358`: dependency-pending.
- RCS+velocity+time branch:
  - Train job `1348` finished and printed `END 2026-05-14T01:17:16-03:00`.
  - Final checkpoint exists:
    - `outputs/racformer_train2k_day_rcsveltimebev_research/2026-05-13/20-02-10/epoch_12.pth`
    - size `765773121` bytes.
  - Eval job `1349` selected the intended checkpoint:
    - `WEIGHTS=outputs/racformer_train2k_day_rcsveltimebev_research/2026-05-13/20-02-10/epoch_12.pth`
  - Eval stderr confirms:
    - config `configs/racformer_train2k_day_rcsveltimebev_research.py`
    - val pkl `/srv/nfs/shared/gnmp/RaCFormer/nuscenes_infos_val_sweep.pkl`
    - full validation over `6019` samples.
  - Train stderr was nonzero only because of a completed tqdm-style progress stream, not an error traceback.
- Seed1 replication:
  - Running on `livenode02`.
  - Latest observed progress: epoch `2/12`, iteration `150/1000`.
  - `slurm_1356.err`: `0` bytes.
- Decision:
  - Monitor `1349` full-val progress and wait for summary job `1350`.
  - Do not start a new branch while both allowed GPUs remain allocated.

## RCS eval progress - 2026-05-14 04:24 UTC

- Queue:
  - `1349` `s0_rcsveltimebev_eval`: RUNNING on `livenode03`, elapsed about `6:29`.
  - `1350` `s0_rcsveltimebev_summary`: dependency-pending.
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `37:28`.
  - `1357`/`1358`: dependency-pending.
- RCS+velocity+time eval:
  - Full-val inference is progressing normally at about `3.9 task/s`.
  - Latest observed progress: about `1405/6019`.
  - Eval ETA at poll: about `1173s` remaining.
  - Eval stderr contains info logs only:
    - config and checkpoint path,
    - model build,
    - running inference on `6019` samples.
  - Summary not ready.
- Seed1 replication:
  - Latest observed progress: epoch `2/12`, iteration `400/1000`.
  - `slurm_1356.err`: `0` bytes.
- Decision:
  - Continue monitoring `1349` until full-val completes and summary job `1350` writes metrics.

## RCS eval late progress - 2026-05-14 04:35 UTC

- Queue:
  - `1349` `s0_rcsveltimebev_eval`: RUNNING on `livenode03`, elapsed about `17:05`.
  - `1350` `s0_rcsveltimebev_summary`: dependency-pending.
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `48:04`.
  - `1357`/`1358`: dependency-pending.
- RCS+velocity+time eval:
  - Full-val inference remains healthy at about `3.9 task/s`.
  - Latest observed progress: about `3912/6019`.
  - Eval ETA at poll: about `534s` remaining.
  - Eval stderr still contains info logs only.
  - Summary not ready.
- Seed1 replication:
  - Latest observed progress: epoch `2/12`, iteration `800/1000`.
  - `slurm_1356.err`: `0` bytes.
- Decision:
  - Continue waiting. Next poll should plausibly see eval completion or summary job `1350` running/completed.

## RCS + velocity + time result - failed gate - 2026-05-14 04:56 UTC

- Branch: `S0_rcsveltimebev`
- Config: `configs/racformer_train2k_day_rcsveltimebev_research.py`
- Summary files:
  - `research/night_gen_phase1/results/S0_rcsveltimebev/summary_metrics.md`
  - `research/night_gen_phase1/results/S0_rcsveltimebev/summary_metrics.json`
- Eval evidence:
  - Full-val inference ran on `6019` samples.
  - Per-split eval wrote metrics for:
    - `eval_day/metrics_summary.json`
    - `eval_night/metrics_summary.json`
    - `eval_rain/metrics_summary.json`
    - `submission_overall/pts_bbox/metrics_summary.json`
  - `eval_by_condition.json` was written.
  - Summary job `1350` wrote `summary_metrics.md`.
- Metrics:
  - day: `0.3106 / 0.3746`, delta `-0.47 pp / -0.00 pp`.
  - night: `0.1477 / 0.2107`, delta `-0.11 pp / -0.44 pp`.
  - rain: `0.2730 / 0.3719`, delta `-0.13 pp / +0.06 pp`.
  - overall: `0.3009 / 0.3703`, delta `-0.31 pp / +0.05 pp`.
- Gate verdict: FAIL.
  - Overall mAP/NDS are acceptable.
  - Day regression is acceptable.
  - Night NDS is still within the allowed regression.
  - Night mAP fails the required `+1.0 pp` gain; it is slightly below S0.
- Interpretation:
  - Adding RCS back into the successful velocity+time family does not help the night target.
  - This strengthens the current direction: RCS should remain muted; occupancy plus compensated velocity plus sweep-time remains the paper-candidate branch pending seed confirmation.
- Decision:
  - Do not promote `S0_rcsveltimebev`.
  - Check whether `livenode03` is free for a bounded follow-up ablation while seed1 continues on `livenode02`.

## New ablation submitted - S0 occupancy + time - 2026-05-14 05:00 UTC

- Trigger:
  - `S0_rcsveltimebev` failed the night mAP gate.
  - `livenode03` became idle while seed1 replication continued on `livenode02`.
- Hypothesis:
  - Remove compensated velocity from the passing `S0_occveltimebev` feature set while keeping RCS muted.
  - Tests whether the seed-0 night gain needs velocity or is mostly occupancy plus sweep-recency.
- Branch: `S0_occtimebev`
- Config:
  - `configs/racformer_train2k_day_occtimebev_research.py`
  - `rcs_index=3`, `rcs_scale=1000000.0`
  - `extra_indices=(6,)`, `extra_scales=(1.0,)`
  - Expected residual channels: occupancy + muted RCS + time, `in_channels=3`.
- Staged files:
  - `research/night_gen_phase1/staged_occ_time_residual/smoke_s0_occtimebev_model.sbatch`
  - `research/night_gen_phase1/staged_occ_time_residual/run_s0_occtimebev_livenode03.sbatch`
  - `research/night_gen_phase1/staged_occ_time_residual/run_s0_occtimebev_eval_livenode03.sbatch`
  - `research/night_gen_phase1/staged_occ_time_residual/run_s0_occtimebev_summary_livenode03.sbatch`
  - `research/night_gen_phase1/staged_occ_time_residual/summarize_s0_occtimebev.py`
- Validation:
  - Local `bash -n` passed for staged sbatch files.
  - Local `python -m py_compile` passed for config and summarizer.
  - Local guard grep found no `livenode01`, no `__LATEST__`, and no stale active branch names.
  - Remote `bash -n` passed for staged sbatch files.
  - Remote `conda run -n racformerfix --no-capture-output python -m py_compile` passed for config and summarizer.
  - Remote guard grep passed.
  - Smoke job `1359` passed on `livenode03`:
    - `radar_occtime_bev_residual (128, 128) (3, 6) (1000000.0, 1.0)`
    - `state_keys 4`
    - `half_forward_zero_init True`
- Submitted full dependency chain on `livenode03`:
  - Train: `1360` `s0_occtimebev`.
  - Eval: `1361` `s0_occtimebev_eval`, `afterok:1360`.
  - Summary: `1362` `s0_occtimebev_summary`, `afterok:1361`.
- Concurrent branch:
  - Seed1 replication remains running on `livenode02` as `1356 -> 1357 -> 1358`.
- Decision:
  - Monitor `1360` for clean training start.
  - Keep the main paper candidate as `S0_occveltimebev` pending seed1 confirmation.

## S0 occupancy + time clean start - 2026-05-14 05:02 UTC

- Queue:
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `0:35`.
  - `1361` `s0_occtimebev_eval`: dependency-pending.
  - `1362` `s0_occtimebev_summary`: dependency-pending.
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `1:12:59`.
  - `1357`/`1358`: dependency-pending.
- Clean-start evidence for `S0_occtimebev`:
  - Training setup reached parameter count, optimizer creation, and pretrained checkpoint load.
  - The checkpoint missing/unexpected keys are the same expected pretrain mismatch pattern seen in other RaCFormer runs.
  - No immediate traceback or dependency failure observed.
- Decision:
  - Both allowed GPUs are now occupied again:
    - `livenode02`: seed1 replication of the passing branch.
    - `livenode03`: occupancy+time ablation.
  - Next useful checks are seed1 epoch/checkpoint progress and `S0_occtimebev` epoch-1 progress/stderr.

## Parallel lane poll - seed1 and occtime training healthy - 2026-05-14 05:14 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
  - No use of `livenode01`.
- Queue:
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `1:14:16`.
  - `1357`/`1358`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `1:52`.
  - `1361`/`1362`: dependency-pending.
- Seed1 replication:
  - Latest observed progress: epoch `3/12`, iteration `800/1000`.
  - Checkpoints exist through `epoch_2.pth`.
  - `slurm_1356.err`: `0` bytes.
  - No summary file yet.
- Occupancy+time ablation:
  - Latest observed progress: epoch `1/12`, iteration `50/1000`.
  - `slurm_1360.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - No intervention needed.
  - Both allowed GPUs are occupied; continue monitoring.

## Parallel lane poll - seed1 epoch 4 and occtime epoch 1 - 2026-05-14 05:30 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
  - No use of `livenode01`.
- Queue:
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `1:30:12`.
  - `1357`/`1358`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `17:48`.
  - `1361`/`1362`: dependency-pending.
- Seed1 replication:
  - Latest observed progress: epoch `4/12`, iteration `400/1000`.
  - Checkpoints exist through `epoch_3.pth`.
  - `slurm_1356.err`: `0` bytes.
  - No summary file yet.
- Occupancy+time ablation:
  - Latest observed progress: epoch `1/12`, iteration `650/1000`.
  - `slurm_1360.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - No intervention needed.
  - Continue monitoring both long-running training lanes.

## Parallel lane poll - seed1 epoch 5 and occtime epoch 2 - 2026-05-14 06:01 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
  - No use of `livenode01`.
- Queue:
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `2:01:11`.
  - `1357`/`1358`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `48:47`.
  - `1361`/`1362`: dependency-pending.
- Seed1 replication:
  - Latest observed progress: epoch `5/12`, iteration `600/1000`.
  - Checkpoints exist through `epoch_4.pth`.
  - `slurm_1356.err`: `0` bytes.
  - No summary file yet.
- Occupancy+time ablation:
  - Latest observed progress: epoch `2/12`, iteration `850/1000`.
  - Checkpoints exist through `epoch_1.pth`.
  - `slurm_1360.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - No intervention needed.
  - Continue monitoring both long-running training lanes.

## Parallel lane poll - seed1 epoch 8 and occtime epoch 5 - 2026-05-14 06:52 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
  - No use of `livenode01`.
- Queue:
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `3:05:42`.
  - `1357`/`1358`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `1:53:18`.
  - `1361`/`1362`: dependency-pending.
- Seed1 replication:
  - Latest observed progress: epoch `8/12`, iteration `50/1000`.
  - Checkpoints exist through `epoch_7.pth`.
  - `slurm_1356.err`: `0` bytes.
  - No summary file yet.
- Occupancy+time ablation:
  - Latest observed progress: epoch `5/12`, iteration `300/1000`.
  - Checkpoints exist through `epoch_4.pth`.
  - `slurm_1360.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - No intervention needed.
  - Both allowed GPUs are still occupied; continue monitoring until eval/summary jobs start.

## CPU-side paper/code audit - IMKD and ContextualFusion - 2026-05-14 06:54 UTC

- Reason:
  - Both allowed GPUs are occupied by active SLURM chains, so the only useful parallel work is CPU/read-only paper implementation inspection.
  - User asked not to adopt paper ideas without checking whether open implementations exist and reading the code.
- Web/source checks:
  - IMKD page/paper advertises `https://github.com/dfki-av/IMKD` as code/models for WACV 2026 intensity-aware multi-level KD.
  - ContextualFusion paper/source is `https://github.com/ssuralcmu/ContextualFusion.git`, focused on adverse operating conditions.
- IMKD remote audit:
  - Remote clone: `/srv/nfs/shared/gnmp/paper_impls/IMKD`
  - Git head: `de144db`.
  - Current repository contents are only `README.md`, `LICENSE`, and media/sample files; no model/config/training implementation is available.
  - README still says code/models will be publicly released.
  - Decision: keep IMKD as paper framing for intensity-aware distillation, but do not plan a code-level RaCFormer adaptation from this repo right now.
- ContextualFusion remote audit:
  - Remote clone: `/srv/nfs/shared/gnmp/paper_impls/ContextualFusion`
  - Git head: `eb2a132`.
  - Implementation is BEVFusion/TransFusion-style camera+LiDAR, not radar-camera.
  - Relevant code exists in:
    - `mmdet3d/models/gating/gating.py`
    - `mmdet3d/models/fusers/conv_3conditions_trainable.py`
    - `mmdet3d/models/fusers/conv_trainable_sigmoid_bounded.py`
  - Useful idea: condition-aware feature scaling before concatenation/fusion.
  - Weakness for RaCFormer/NB2: it depends on explicit scene context flags and LiDAR/camera channel layout; direct port would repeat the already-tested adaptive-gate family and is less aligned than the current radar occupancy/velocity/time branch.
- Decision:
  - Do not launch a new fallback from either implementation while `S0_occveltimebev_seed1` and `S0_occtimebev` are running.
  - If both active branches fail, prefer the previously ranked SGDet3D-style radar pre-encoder/centered-velocity normalization before another context-gate branch.

## CPU-side paper/code audit - CRT-Fusion and TransCAR clones - 2026-05-14 06:56 UTC

- Reason:
  - Earlier paper scan ranked CRT-Fusion and TransCAR as fallback ideas, but they had not been cloned under `paper_impls`.
  - User asked to clone/read open implementations before adopting paper ideas.
- New remote clones:
  - CRT-Fusion: `/srv/nfs/shared/gnmp/paper_impls/CRT-Fusion`, git head `2fa611a`, source `https://github.com/mjseong0414/CRT-Fusion`.
  - TransCAR: `/srv/nfs/shared/gnmp/paper_impls/TransCAR`, git head `0a618b9`, source `https://github.com/pangsu0613/TransCAR`.
- CRT-Fusion audit:
  - Relevant files:
    - `mmdet3d/models/detectors/crtfusion.py`
    - `mmdet3d/models/necks/view_transformer_crtfusion.py`
    - `mmdet3d/models/utils/radar_camera_gating.py`
    - `tools/radar_multi_sweeps.py`
    - `configs/crt-fusion/crtfusion-r50-fp16_phase1.py`
  - Implementation uses radar PointPillars, radar-camera attention in camera/PV view, motion estimation, BEV occupancy/segmentation loss, and optional history fusion.
  - The key low-risk idea for RaCFormer is not a full port; it is the paper's validated use of compensated velocity plus time/history to improve radar-camera robustness.
  - The full CRT-Fusion stack is too invasive for the current branch because it adds view-transformer changes, segmentation/velocity auxiliary heads, history state, and two-phase training.
  - Decision: keep as support for the current occupancy+velocity+time result and as a later temporal-carry branch if the seed replication fails.
- TransCAR audit:
  - Relevant files:
    - `projects/mmdet3d_plugin/models/dense_heads/detr3d_head.py`
    - `projects/mmdet3d_plugin/models/dense_heads/detr3d_head_backup.py`
    - `projects/mmdet3d_plugin/models/utils/detr3d_transformer.py`
  - Implementation encodes radar point position/features, masks radar tokens near predicted query centers/front/rear points, then applies repeated multi-head radar cross-attention to refine queries.
  - Useful idea: object-conditioned radar support around candidate boxes, especially for velocity refinement.
  - Weakness for RaCFormer: current implementation loads multisweep radar inside the head, uses hardcoded `.cuda()`, fixed token padding, and DETR3D-specific query/reference plumbing.
  - Decision: do not port now. If all BEV residual branches fail, a small query-local radar support score near RaCFormer proposals is more realistic than transplanting TransCAR attention.
- Overall decision:
  - No new job submitted and no RaCFormer source changed.
  - Active priority remains `S0_occveltimebev_seed1`; `S0_occtimebev` continues as an ablation of whether velocity is necessary.

## Parallel lane health poll - seed1 epoch 8 and occtime epoch 5 - 2026-05-14 06:57 UTC

- Queue:
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `3:10:45`.
  - `1357`/`1358`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `1:58:21`.
  - `1361`/`1362`: dependency-pending.
- Seed1 replication:
  - Latest observed progress: epoch `8/12`, iteration `250/1000`.
  - ETA in train log around `2:02:56`.
  - `slurm_1356.err`: `0` bytes.
- Occupancy+time ablation:
  - Latest observed progress: epoch `5/12`, iteration `500/1000`.
  - ETA in train log around `3:13:34`.
  - `slurm_1360.err`: `0` bytes.
- Decision:
  - No intervention needed.
  - Both allowed GPUs remain allocated; no idle `livenode02` capacity is available for another job.

## Parallel lane poll - seed1 epoch 8 and occtime epoch 5 near completion - 2026-05-14 07:08 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
  - No use of `livenode01`.
- Queue:
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `3:21:52`.
  - `1357`/`1358`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `2:09:28`.
  - `1361`/`1362`: dependency-pending.
- Seed1 replication:
  - Latest observed progress: epoch `8/12`, iteration `700/1000`.
  - ETA in train log around `1:51:18`.
  - Checkpoints exist through `epoch_7.pth`.
  - `slurm_1356.err`: `0` bytes.
  - No summary file yet.
- Occupancy+time ablation:
  - Latest observed progress: epoch `5/12`, iteration `950/1000`.
  - ETA in train log around `3:01:58`.
  - Checkpoints exist through `epoch_4.pth`.
  - `slurm_1360.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - No intervention needed.
  - Continue monitoring until the train jobs hand off to eval/summary.

## Parallel lane poll - seed1 epoch 8 and occtime epoch 5 complete - 2026-05-14 07:09 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Remote repo state:
  - Branch: `main`.
  - Dirty research state remains expected; no reset/revert performed.
- Queue:
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `3:23:11`.
  - `1357`/`1358`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `2:10:59`.
  - `1361`/`1362`: dependency-pending.
- Seed1 replication:
  - Latest observed progress: epoch `8/12`, iteration `750/1000`.
  - ETA in train log around `1:50:02`.
  - Checkpoints exist through `epoch_7.pth`.
  - `slurm_1356.err`: `0` bytes.
  - No summary file yet.
- Occupancy+time ablation:
  - Latest observed progress: epoch `5/12`, iteration `1000/1000`.
  - New checkpoint: `outputs/racformer_train2k_day_occtimebev_research/2026-05-14/01-59-04/epoch_5.pth`.
  - ETA in train log around `3:00:40`.
  - `slurm_1360.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - No intervention needed.
  - Both allowed GPUs remain allocated; keep waiting for train completion and dependent eval/summary jobs.

## Parallel lane poll - seed1 epoch 9 and occtime epoch 6 - 2026-05-14 07:25 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
  - No use of `livenode01`.
- Queue:
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `3:39:35`.
  - `1357`/`1358`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `2:27:11`.
  - `1361`/`1362`: dependency-pending.
- Seed1 replication:
  - Latest observed progress: epoch `9/12`, iteration `350/1000`.
  - ETA in train log around `1:34:29`.
  - `slurm_1356.err`: `0` bytes.
  - No summary file yet.
- Occupancy+time ablation:
  - Latest observed progress: epoch `6/12`, iteration `600/1000`.
  - ETA in train log around `2:45:17`.
  - Checkpoints exist at least through `epoch_5.pth`.
  - `slurm_1360.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - No intervention needed.
  - Continue monitoring; both allowed GPUs remain allocated and there is no idle `livenode02` capacity.

## Parallel lane poll - seed1 epoch 9 complete and occtime epoch 7 start - 2026-05-14 07:42 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
  - No use of `livenode01`.
- Queue:
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `3:55:56`.
  - `1357`/`1358`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `2:43:32`.
  - `1361`/`1362`: dependency-pending.
- Seed1 replication:
  - Latest observed progress: epoch `9/12`, iteration `1000/1000`.
  - New checkpoints:
    - `outputs/racformer_train2k_day_occveltimebev_seed1_research/2026-05-14/00-46-41/epoch_8.pth`
    - `outputs/racformer_train2k_day_occveltimebev_seed1_research/2026-05-14/00-46-41/epoch_9.pth`
  - ETA in train log around `1:17:37`.
  - `slurm_1356.err`: `0` bytes.
  - No summary file yet.
- Occupancy+time ablation:
  - Latest observed progress: epoch `7/12`, iteration `200/1000`.
  - New checkpoint:
    - `outputs/racformer_train2k_day_occtimebev_research/2026-05-14/01-59-04/epoch_6.pth`
  - ETA in train log around `2:29:48`.
  - `slurm_1360.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - No intervention needed.
  - Continue monitoring until the seed1 chain reaches eval/summary, because that is the current paper-candidate confirmation.

## Parallel lane poll - seed1 epoch 10 and occtime epoch 8 start - 2026-05-14 08:03 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
  - No use of `livenode01`.
- Queue:
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `4:16:58`.
  - `1357`/`1358`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `3:04:34`.
  - `1361`/`1362`: dependency-pending.
- Seed1 replication:
  - Latest observed progress: epoch `10/12`, iteration `800/1000`.
  - ETA in train log around `0:56:55`.
  - Checkpoints exist through `epoch_9.pth`.
  - `slurm_1356.err`: `0` bytes.
  - No summary file yet.
- Occupancy+time ablation:
  - Latest observed progress: epoch `8/12`, iteration `50/1000`.
  - New checkpoint:
    - `outputs/racformer_train2k_day_occtimebev_research/2026-05-14/01-59-04/epoch_7.pth`
  - ETA in train log around `2:07:54`.
  - `slurm_1360.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - No intervention needed.
  - Keep monitoring; seed1 remains the gating result for whether `S0_occveltimebev` becomes a credible paper-candidate result.

## Parallel lane poll - seed1 epoch 11 and occtime epoch 8 complete - 2026-05-14 08:29 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
  - No use of `livenode01`.
- Queue:
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `4:43:02`.
  - `1357`/`1358`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `3:30:38`.
  - `1361`/`1362`: dependency-pending.
- Seed1 replication:
  - Latest observed progress: epoch `11/12`, iteration `800/1000`.
  - New checkpoint:
    - `outputs/racformer_train2k_day_occveltimebev_seed1_research/2026-05-14/00-46-41/epoch_10.pth`
  - ETA in train log around `0:31:02`.
  - `slurm_1356.err`: `0` bytes.
  - No summary file yet.
- Occupancy+time ablation:
  - Latest observed progress: epoch `8/12`, iteration `1000/1000`.
  - New checkpoint:
    - `outputs/racformer_train2k_day_occtimebev_research/2026-05-14/01-59-04/epoch_8.pth`
  - ETA in train log around `1:43:20`.
  - `slurm_1360.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - No intervention needed.
  - Seed1 should be within roughly one training epoch of handoff to eval; next poll should watch for job `1357` starting.

## Parallel lane poll - seed1 final epoch and occtime epoch 9 - 2026-05-14 08:50 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Queue:
  - `1356` `s0_occveltime_s1`: RUNNING on `livenode02`, elapsed about `5:04:11`.
  - `1357`/`1358`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `3:51:47`.
  - `1361`/`1362`: dependency-pending.
- Seed1 replication:
  - Latest observed progress: epoch `12/12`, iteration `600/1000`.
  - New checkpoint:
    - `outputs/racformer_train2k_day_occveltimebev_seed1_research/2026-05-14/00-46-41/epoch_11.pth`
  - ETA in train log around `0:10:20`.
  - `slurm_1356.err`: `0` bytes.
  - No summary file yet.
- Occupancy+time ablation:
  - Latest observed progress: epoch `9/12`, iteration `850/1000`.
  - Checkpoints exist through `epoch_8.pth`.
  - ETA in train log around `1:21:21`.
  - `slurm_1360.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - No intervention needed.
  - Poll soon for seed1 train completion and eval job `1357` start.

## Seed1 train complete and condition eval running - 2026-05-14 09:03 UTC

- Host/cwd:
  - `cluster-live`
  - `/srv/nfs/shared/gnmp/RaCFormer`
- Queue:
  - `1357` `s0_occveltime_s1_eval`: RUNNING on `livenode02`, elapsed about `1:31`.
  - `1358` `s0_occveltime_s1_summary`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `4:04:45`.
  - `1361`/`1362`: dependency-pending.
- Seed1 replication:
  - Train job `1356` completed and handed off to eval `1357`.
  - New checkpoint:
    - `outputs/racformer_train2k_day_occveltimebev_seed1_research/2026-05-14/00-46-41/epoch_12.pth`
  - Train job final validation line in `slurm_1356.out`:
    - mAP `0.3195`
    - NDS `0.3538`
  - Condition eval `1357` is using `epoch_12.pth` and running full `6019`-sample inference.
  - Nonzero stderr bytes are from tqdm/progress/log output, not tracebacks.
  - No condition summary file yet.
- Occupancy+time ablation:
  - Latest observed progress: epoch `10/12`, iteration `350/1000`.
  - New checkpoint:
    - `outputs/racformer_train2k_day_occtimebev_research/2026-05-14/01-59-04/epoch_9.pth`
  - `slurm_1360.err`: `0` bytes.
  - No summary file yet.
- Decision:
  - Wait for seed1 eval `1357` to finish and summary `1358` to produce `summary_metrics.md`.
  - Do not interpret the train-job final mAP/NDS as the condition gate; the gate requires the condition summary.

## Seed1 condition eval progress - 2026-05-14 09:19 UTC

- Queue:
  - `1357` `s0_occveltime_s1_eval`: RUNNING on `livenode02`, elapsed about `17:21`.
  - `1358` `s0_occveltime_s1_summary`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `4:20:35`.
  - `1361`/`1362`: dependency-pending.
- Seed1 eval:
  - Inference progress observed around `4011/6019` samples.
  - Throughput around `4.0 task/s`.
  - ETA around `505` seconds.
  - `eval_slurm_1357.err` contains normal eval logging only; no traceback observed.
  - No condition summary file yet.
- Occupancy+time ablation:
  - Latest observed progress: epoch `10/12`, iteration `950/1000`.
  - No summary file yet.
- Decision:
  - Wait for `1357` completion and `1358` summary.
  - Keep `livenode02` dedicated to seed1 eval until it finishes.

## Seed1 condition eval near completion - 2026-05-14 09:31 UTC

- Queue:
  - `1357` `s0_occveltime_s1_eval`: RUNNING on `livenode02`, elapsed about `29:05`.
  - `1358` `s0_occveltime_s1_summary`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `4:32:19`.
  - `1361`/`1362`: dependency-pending.
- Seed1 eval:
  - Inference progress in stderr reached at least `5847/6019` before switching into result writing/evaluation output.
  - Eval stdout shows result writing to `research/night_gen_phase1/results/S0_occveltimebev_seed1/eval/submission_overall/pts_bbox/results_nusc.json`.
  - No condition summary file yet.
  - No traceback observed.
- Occupancy+time ablation:
  - Latest observed progress: epoch `11/12`, iteration `400/1000`.
  - New checkpoint:
    - `outputs/racformer_train2k_day_occtimebev_research/2026-05-14/01-59-04/epoch_10.pth`
  - No summary file yet.
- Decision:
  - Poll soon for eval `1357` completion and summary `1358`.

## Seed1 strict gate fail and seed2 replication submitted - 2026-05-14 09:42 UTC

- Seed1 final summary:
  - Summary file: `research/night_gen_phase1/results/S0_occveltimebev_seed1/summary_metrics.md`
  - day: `0.3041 / 0.3638`, delta `-1.12 pp / -1.07 pp`
  - night: `0.1594 / 0.2217`, delta `+1.06 pp / +0.66 pp`
  - rain: `0.2608 / 0.3546`, delta `-1.36 pp / -1.68 pp`
  - overall: `0.2937 / 0.3581`, delta `-1.03 pp / -1.17 pp`
  - Gate verdict: FAIL.
- Interpretation:
  - The night target replicated, and overall mAP stayed within the gate.
  - The strict day mAP preservation gate missed by about `0.12 pp` (`-1.12 pp` vs allowed `-1.00 pp`).
  - Do not claim the seed0 result as a strict replicated paper result.
  - Two-seed mean is potentially defensible but needs explicit framing and preferably a third seed because one seed individually fails.
- Seed2 action:
  - `livenode02` became idle after seed1 eval/summary completed.
  - Staged a config-only seed2 replication without touching `models/racformer.py`.
  - Local staged files:
    - `remote_patch_work/configs/racformer_train2k_day_occveltimebev_seed2_research.py`
    - `remote_patch_work/staged_occ_vel_time_seed2/`
  - Remote staged files:
    - `configs/racformer_train2k_day_occveltimebev_seed2_research.py`
    - `research/night_gen_phase1/staged_occ_vel_time_seed2/`
  - Validation:
    - Local `bash -n` passed.
    - Local `python -m py_compile` passed.
    - Local guard grep found no `livenode01`, no `__LATEST__`, and no stale seed1 stage/config names.
    - Remote `bash -n` passed.
    - Remote `conda run -n racformerfix --no-capture-output python -m py_compile` passed.
    - Remote guard grep passed.
  - Smoke job `1363` passed on `livenode02`:
    - `seed 2`
    - `radar_occveltime_bev_residual_seed2 (128, 128) (3, 4, 5, 6) (1000000.0, 20.0, 20.0, 1.0)`
    - `half_forward_zero_init True`
  - Submitted dependency chain on `livenode02`:
    - Train: `1364` `s0_occveltime_s2`
    - Eval: `1365` `s0_occveltime_s2_eval`, `afterok:1364`
    - Summary: `1366` `s0_occveltime_s2_summary`, `afterok:1365`
- Concurrent branch:
  - `1360` `s0_occtimebev` continues on `livenode03`.
- Decision:
  - Monitor seed2 train start and occtime completion.
  - Keep strict interpretation: current seed1 result is not sufficient to mark the objective complete.

## Seed2 clean train start - 2026-05-14 09:43 UTC

- Queue:
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `0:46`.
  - `1365`/`1366`: dependency-pending.
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `4:44:18`.
  - `1361`/`1362`: dependency-pending.
- Seed2 evidence:
  - Training started on `livenode02`.
  - Work dir: `outputs/racformer_train2k_day_occveltimebev_seed2_research/2026-05-14/06-42-38`
  - Runner reached hook registration and checkpoint setup.
  - `slurm_1364.err` has no content at this poll.
  - The early `no points within the predefined bev receptive field` warnings are the known dataset/radar warning pattern seen in prior runs.
- Occupancy+time ablation:
  - Latest observed progress: epoch `11/12`, iteration `850/1000`.
  - No summary file yet.
- Decision:
  - Seed2 training is healthy enough to leave running.
  - Next priority is `S0_occtimebev` completion on `livenode03`, then seed2 progress.

## Occtime final epoch and seed2 epoch 1 - 2026-05-14 10:10 UTC

- Queue:
  - `1360` `s0_occtimebev`: RUNNING on `livenode03`, elapsed about `5:12:00`.
  - `1361`/`1362`: dependency-pending.
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `28:28`.
  - `1365`/`1366`: dependency-pending.
- Node availability:
  - `livenode02`: allocated to seed2; not idle.
  - `livenode03`: allocated to occtime.
  - `livenode01`: intentionally unused because of the NVIDIA driver issue.
- Occupancy+time ablation:
  - Latest observed progress: epoch `12/12`, at least iteration `900/1000`.
  - `slurm_1360.err`: no actionable error observed.
  - No condition summary file yet.
- Seed2 replication:
  - Latest observed progress: epoch `1/12` completed in the previous poll.
  - `slurm_1364.err`: no actionable error observed at clean start.
  - No condition summary file yet.
- Decision:
  - Do not submit extra GPU work now; both allowed nodes are allocated.
  - Poll soon for `S0_occtimebev` handoff from train `1360` to eval `1361`.

## Occtime train complete, condition eval started - 2026-05-14 10:15 UTC

- Queue:
  - `1361` `s0_occtimebev_eval`: RUNNING on `livenode03`, elapsed about `0:29`.
  - `1362` `s0_occtimebev_summary`: dependency-pending.
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `32:30`.
  - `1365`/`1366`: dependency-pending.
- Occupancy+time train result:
  - Train job `1360` ended at `2026-05-14T07:14:13-03:00`.
  - Final checkpoint:
    - `outputs/racformer_train2k_day_occtimebev_research/2026-05-14/01-59-04/epoch_12.pth`
  - Train-job validation printed mAP `0.3342` and NDS `0.3669`.
  - Treat this as a training sanity signal only; the promotion gate still requires condition eval `1361` and summary `1362`.
  - `slurm_1360.err`: `0` bytes.
- Eval evidence:
  - Eval `1361` started with:
    - `STAGE=S0_occtimebev`
    - `WEIGHTS=outputs/racformer_train2k_day_occtimebev_research/2026-05-14/01-59-04/epoch_12.pth`
  - No condition summary file yet.
- Seed2:
  - Latest observed progress: epoch `2/12`, iteration `200/1000`.
  - No actionable errors observed.
- Decision:
  - Wait for `1361` full condition eval and `1362` summary before interpreting occtime.
  - No extra submission: both allowed GPU nodes remain allocated.

## Occtime condition eval running - 2026-05-14 10:19 UTC

- Queue:
  - `1361` `s0_occtimebev_eval`: RUNNING on `livenode03`, elapsed about `4:12`.
  - `1362` `s0_occtimebev_summary`: dependency-pending.
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `36:13`.
  - `1365`/`1366`: dependency-pending.
- Occtime eval:
  - `eval_by_condition` is running inference on `6019` samples.
  - `eval_slurm_1361.err` contains startup logging only so far; no traceback observed.
  - No `summary_metrics.md` yet.
- Seed2:
  - Latest observed progress: epoch `2/12`, iteration `350/1000`.
  - `slurm_1364.err`: `0` bytes.
- Parallelization note:
  - `livenode02` is not idle; it is actively training seed2.
  - Keep extra GPU work paused until an allowed node is free.

## Occtime condition eval still running - 2026-05-14 10:25 UTC

- Queue:
  - `1361` `s0_occtimebev_eval`: RUNNING on `livenode03`, elapsed about `10:00`.
  - `1362` `s0_occtimebev_summary`: dependency-pending.
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `42:01`.
  - `1365`/`1366`: dependency-pending.
- Occtime eval:
  - Still in full `6019`-sample inference/evaluation.
  - `eval_slurm_1361.err`: unchanged startup logging only; no traceback observed.
  - No summary file yet.
- Seed2:
  - Latest observed progress: epoch `2/12`, iteration `550/1000`.
  - `slurm_1364.err`: `0` bytes.
- Decision:
  - Continue waiting for condition summary.
  - No additional GPU job because `livenode02` and `livenode03` are both allocated.

## Lightweight paper-code scan while GPUs busy - 2026-05-14 10:28 UTC

- Reason:
  - Both allowed GPU nodes are allocated, so no new GPU experiment was launched.
  - Used waiting time for a small web scan only; no code cloned and no RaCFormer files changed.
- Leads checked:
  - RCDINO, `https://github.com/OlgaMatykina/RCDINO`
    - Paper page claims DINOv2 semantic features for radar-camera 3D detection with reported `56.4` NDS and `48.1` mAP.
    - GitHub repo is public, MIT licensed, one commit, and appears based on an RCTrans-style `mmdetection3d/projects` tree.
    - Potential adoption path: inspect whether DINOv2 features can be used as an offline/auxiliary camera feature branch. This is likely much more invasive than the current radar residual ablations, so only consider after current seed/ablation results.
  - RobuRCDet, `https://huggingface.co/papers/2502.13071`
    - Interesting idea: 3D Gaussian Expansion uses RCS/velocity priors plus weather-adaptive fusion.
    - No linked model/code was visible from the Hugging Face paper page; do not treat as open-implementation-ready until a repo is found.
  - D3PD, `https://github.com/no-Name128/D3PD` as mentioned in the article metadata/search result.
    - Idea is dynamic radar-camera feature enhancement plus distillation.
    - Code availability not audited yet in this pass; do not reuse until the repo is cloned/read.
- Decision:
  - Do not adopt paper ideas yet.
  - If occtime and seed2 do not give a clean result, first clone/read RCDINO or D3PD before proposing a RaCFormer adaptation.

## GPU activity sanity check - 2026-05-14 10:34 UTC

- Direct SSH from `cluster-live` to `livenode02`/`livenode03` was denied, so no node-shell inspection was possible that way.
- Used read-only `srun --jobid ... --overlap` checks inside the existing allocations:
  - `1361` on `livenode03`: GPU util about `82%`, memory about `4733 / 24564` MiB.
  - `1364` on `livenode02`: GPU util about `90%`, memory about `19846 / 24564` MiB.
- Interpretation:
  - Occtime eval is active despite sparse log progress.
  - Seed2 train is active.
  - No idle allowed GPU capacity is available.

## Occtime eval sparse-log health check - 2026-05-14 10:40 UTC

- Queue:
  - `1361` still RUNNING on `livenode03`, elapsed about `25:26`.
  - `1362` still dependency-pending.
  - `1364` still RUNNING on `livenode02`, elapsed about `57:27`.
- Occtime eval:
  - `summary_metrics.md`: not ready.
  - `eval_slurm_1361.err`: still startup logging only.
  - `eval_slurm_1361.out`: file size grew to about `466785` bytes, updated at `10:39` UTC.
  - Read-only overlap GPU check on `1361`: `livenode03`, about `83%` GPU util and `4733 / 24564` MiB.
  - Interpretation: eval is active despite sparse readable progress.
- Seed2:
  - Latest observed progress: epoch `3/12`, iteration `150/1000`.
  - Still no actionable error in `slurm_1364.err`.

## Occtime final summary - 2026-05-14 10:49 UTC

- Summary file:
  - `research/night_gen_phase1/results/S0_occtimebev/summary_metrics.md`
- Metrics:
  - day: `0.3175 / 0.3754`, delta `+0.22 pp / +0.08 pp`
  - night: `0.1436 / 0.2022`, delta `-0.52 pp / -1.29 pp`
  - rain: `0.2764 / 0.3694`, delta `+0.21 pp / -0.19 pp`
  - overall: `0.3049 / 0.3691`, delta `+0.09 pp / -0.07 pp`
  - Gate verdict: FAIL.
- Interpretation:
  - Time-only occupancy residual preserves day/overall, but hurts night relative to S0.
  - The successful seed0 `S0_occveltimebev` gain is not explained by sweep-time alone.
  - Velocity without time failed earlier, and time without velocity now fails; the useful signal appears to be the velocity+time combination.
  - Since seed1 narrowly missed the day gate, the next low-risk branch should reduce velocity strength rather than add a new paper-scale architecture.
- Current queue after occtime:
  - `livenode02`: still training `1364` `s0_occveltime_s2`.
  - `livenode03`: free after occtime eval/summary.
- Decision:
  - Keep waiting for seed2.
  - Consider using free `livenode03` for a config-only velocity-scale ablation if continuing without user interruption.

## V10 velocity-scale ablation submitted - 2026-05-14 10:57 UTC

- Rationale:
  - `S0_occveltimebev` seed0 passed, seed1 narrowly failed the day gate, velocity-only failed, and time-only failed.
  - Next conservative hypothesis: reduce vx/vy scaling from `20.0` to `10.0` while keeping occupancy + vx_comp + vy_comp + sweep-time.
  - Expected effect: preserve some night gain while reducing day/rain regression from over-strong velocity residuals.
- Local staged files:
  - `remote_patch_work/configs/racformer_train2k_day_occveltimebev_v10_research.py`
  - `remote_patch_work/staged_occ_vel_time_v10/`
- Remote staged files:
  - `configs/racformer_train2k_day_occveltimebev_v10_research.py`
  - `research/night_gen_phase1/staged_occ_vel_time_v10/`
- Config:
  - `rcs_index=3`
  - `rcs_scale=1000000.0`
  - `extra_indices=(4, 5, 6)`
  - `extra_scales=(10.0, 10.0, 1.0)`
  - RCS remains muted.
- Validation:
  - Local `bash -n` passed.
  - Local `python -m py_compile` passed.
  - Local guard grep found no `livenode01`, no stale seed stage names, no `__LATEST__`, and no executable stale `20.0, 20.0` scale.
  - Remote `bash -n` passed.
  - Remote `conda run -n racformerfix --no-capture-output python -m py_compile` passed.
  - Remote guard grep passed.
  - SLURM showed `livenode03` idle before submission and `livenode02` allocated to seed2.
- Smoke:
  - Smoke job `1367` passed on `livenode03`.
  - Output:
    - `radar_occveltime_bev_residual_v10 (128, 128) (3, 4, 5, 6) (1000000.0, 10.0, 10.0, 1.0)`
    - `state_keys 4`
    - `half_forward_zero_init True`
- Submitted dependency chain:
  - Train: `1368` `s0_occveltime_v10`, `livenode03`
  - Eval: `1369` `s0_occveltime_v10_eval`, `afterok:1368`
  - Summary: `1370` `s0_occveltime_v10_summary`, `afterok:1369`
- Clean start:
  - `1368` RUNNING on `livenode03`.
  - Work dir: `outputs/racformer_train2k_day_occveltimebev_v10_research/2026-05-14/07-56-03`
  - Seed: `0`
  - `slurm_1368.err`: empty at first poll.
  - Known `no points within the predefined bev receptive field` warnings appeared in stdout, matching prior runs.
- Concurrent seed2:
  - `1364` still RUNNING on `livenode02`.
  - Latest observed seed2 progress: epoch `3/12`, iteration `850/1000`.
- Decision:
  - Both allowed nodes are occupied.
  - Monitor seed2 and v10; do not submit additional GPU work.

## V10 and seed2 early health - 2026-05-14 11:01 UTC

- Queue:
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `1:18:29`.
  - `1365`/`1366`: dependency-pending.
  - `1368` `s0_occveltime_v10`: RUNNING on `livenode03`, elapsed about `5:06`.
  - `1369`/`1370`: dependency-pending.
- V10:
  - Training reached epoch `1/12`, iteration `150/1000`.
  - ETA about `5:07:07`.
  - `slurm_1368.err`: `0` bytes.
  - First work dir remains:
    - `outputs/racformer_train2k_day_occveltimebev_v10_research/2026-05-14/07-56-03`
- Seed2:
  - Training reached epoch `3/12`, iteration `950/1000`.
  - `slurm_1364.err`: `0` bytes.
  - Checkpoints observed:
    - `epoch_1.pth`
    - `epoch_2.pth`
- Decision:
  - Both allowed nodes are healthy and occupied.
  - No more parallel submissions until one chain finishes or a node becomes idle.

## Seed2 epoch 3 checkpoint and v10 early progress - 2026-05-14 11:02 UTC

- Queue:
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `1:19:46`.
  - `1365`/`1366`: dependency-pending.
  - `1368` `s0_occveltime_v10`: RUNNING on `livenode03`, elapsed about `6:23`.
  - `1369`/`1370`: dependency-pending.
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
  - `livenode01`: intentionally unused.
- Seed2:
  - Latest observed progress: epoch `3/12`, iteration `1000/1000`.
  - Saved checkpoint at epoch `3`.
  - Checkpoints observed: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`.
  - `slurm_1364.err`: `0` bytes.
  - No condition summary yet.
- V10:
  - Latest observed progress: epoch `1/12`, iteration `200/1000`.
  - `slurm_1368.err`: `0` bytes.
  - No checkpoint yet.
  - No condition summary yet.
- Decision:
  - Keep both GPU chains running.
  - Use only non-GPU sidecar work while both allowed nodes are allocated.

## Seed2 epoch 4 and v10 epoch 1 progress - 2026-05-14 11:13 UTC

- Queue:
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `1:30:45`.
  - `1365`/`1366`: dependency-pending.
  - `1368` `s0_occveltime_v10`: RUNNING on `livenode03`, elapsed about `17:22`.
  - `1369`/`1370`: dependency-pending.
- Seed2:
  - Latest observed progress: epoch `4/12`, iteration `450/1000`.
  - `slurm_1364.err`: `0` bytes.
  - Latest checkpoint remains `epoch_3.pth`.
  - No condition summary yet.
- V10:
  - Latest observed progress: epoch `1/12`, iteration `600/1000`.
  - `slurm_1368.err`: `0` bytes.
  - No checkpoint yet.
  - No condition summary yet.
- Sidecar work:
  - Spawned a read-only researcher subagent to audit RCDINO and D3PD code availability/adoptability.
  - No files or jobs are to be changed by that audit.
- Decision:
  - Continue monitoring both GPU chains.
  - Do not submit more jobs while both allowed nodes are allocated.

## Seed2 epoch 5 start and v10 epoch 2 start - 2026-05-14 11:28 UTC

- Queue:
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `1:46:39`.
  - `1365`/`1366`: dependency-pending.
  - `1368` `s0_occveltime_v10`: RUNNING on `livenode03`, elapsed about `33:16`.
  - `1369`/`1370`: dependency-pending.
- Node state:
  - `livenode02`: allocated.
  - `livenode03`: allocated.
- Seed2:
  - Latest observed progress: epoch `5/12`, iteration `50/1000`.
  - Saved checkpoint at epoch `4`.
  - Checkpoints observed: `epoch_1.pth` through `epoch_4.pth`.
  - `slurm_1364.err`: `0` bytes.
  - No condition summary yet.
- V10:
  - Latest observed progress: epoch `2/12`, iteration `250/1000`.
  - Saved checkpoint at epoch `1`.
  - `slurm_1368.err`: `0` bytes.
  - No condition summary yet.
- Decision:
  - Both allowed nodes remain healthy and allocated.
  - Continue monitoring; no more jobs until one chain finishes or fails.

## Seed2 epoch 5 late and v10 epoch 3 start - 2026-05-14 11:51 UTC

- Queue:
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `2:08:49`.
  - `1365`/`1366`: dependency-pending.
  - `1368` `s0_occveltime_v10`: RUNNING on `livenode03`, elapsed about `55:26`.
  - `1369`/`1370`: dependency-pending.
- Seed2:
  - Latest observed progress: epoch `5/12`, iteration `900/1000`.
  - Latest checkpoint remains `epoch_4.pth`.
  - `slurm_1364.err`: `0` bytes.
  - No condition summary yet.
- V10:
  - Latest observed progress: epoch `3/12`, iteration `100/1000`.
  - Saved checkpoint at epoch `2`.
  - Checkpoints observed: `epoch_1.pth`, `epoch_2.pth`.
  - `slurm_1368.err`: `0` bytes.
  - No condition summary yet.
- Decision:
  - Continue monitoring. Both allowed nodes remain occupied and healthy.

## Seed2 epoch 6 and v10 epoch 3 progress - 2026-05-14 12:09 UTC

- Queue:
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `2:27:20`.
  - `1365`/`1366`: dependency-pending.
  - `1368` `s0_occveltime_v10`: RUNNING on `livenode03`, elapsed about `1:13:57`.
  - `1369`/`1370`: dependency-pending.
- Seed2:
  - Latest observed progress: epoch `6/12`, iteration `600/1000`.
  - Saved checkpoint at epoch `5`.
  - Checkpoints observed: `epoch_2.pth` through `epoch_5.pth` in latest listing.
  - `slurm_1364.err`: `0` bytes.
  - No condition summary yet.
- V10:
  - Latest observed progress: epoch `3/12`, iteration `800/1000`.
  - Latest checkpoint remains `epoch_2.pth`.
  - `slurm_1368.err`: `0` bytes.
  - No condition summary yet.
- Decision:
  - Both allowed nodes remain occupied and healthy.
  - Continue monitoring. No new submissions.

## Seed2 epoch 7 start and v10 epoch 4 progress - 2026-05-14 12:23 UTC

- Queue:
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `2:40:42`.
  - `1365`/`1366`: dependency-pending.
  - `1368` `s0_occveltime_v10`: RUNNING on `livenode03`, elapsed about `1:27:19`.
  - `1369`/`1370`: dependency-pending.
- Seed2:
  - Latest observed progress: epoch `7/12`, iteration `100/1000`.
  - Saved checkpoint at epoch `6`.
  - Checkpoints observed: `epoch_3.pth` through `epoch_6.pth` in latest listing.
  - `slurm_1364.err`: `0` bytes.
  - No condition summary yet.
- V10:
  - Latest observed progress: epoch `4/12`, iteration `300/1000`.
  - Saved checkpoint at epoch `3`.
  - Checkpoints observed: `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`.
  - `slurm_1368.err`: `0` bytes.
  - No condition summary yet.
- Decision:
  - Both allowed nodes remain occupied and healthy.
  - Continue monitoring. No new submissions.

## Seed2 epoch 8 and v10 epoch 5 progress - 2026-05-14 12:54 UTC

- Queue:
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `3:12:12`.
  - `1365`/`1366`: dependency-pending.
  - `1368` `s0_occveltime_v10`: RUNNING on `livenode03`, elapsed about `1:58:49`.
  - `1369`/`1370`: dependency-pending.
- Seed2:
  - Latest observed progress: epoch `8/12`, iteration `350/1000`.
  - Saved checkpoint at epoch `7`.
  - Checkpoints observed: `epoch_4.pth` through `epoch_7.pth` in latest listing.
  - `slurm_1364.err`: `0` bytes.
  - No condition summary yet.
- V10:
  - Latest observed progress: epoch `5/12`, iteration `500/1000`.
  - Saved checkpoint at epoch `4`.
  - Checkpoints observed: `epoch_1.pth` through `epoch_4.pth`.
  - `slurm_1368.err`: `0` bytes.
  - No condition summary yet.
- Decision:
  - Continue monitoring. Both allowed nodes remain occupied and healthy.

## Seed2 epoch 9 and v10 epoch 6 progress - 2026-05-14 13:25 UTC

- Queue:
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `3:43:11`.
  - `1365`/`1366`: dependency-pending.
  - `1368` `s0_occveltime_v10`: RUNNING on `livenode03`, elapsed about `2:29:48`.
  - `1369`/`1370`: dependency-pending.
- Seed2:
  - Latest observed progress: epoch `9/12`, iteration `500/1000`.
  - Saved checkpoint at epoch `8`.
  - Checkpoints observed: `epoch_5.pth` through `epoch_8.pth` in latest listing.
  - `slurm_1364.err`: `0` bytes.
  - No condition summary yet.
- V10:
  - Latest observed progress: epoch `6/12`, iteration `700/1000`.
  - Saved checkpoint at epoch `5`.
  - Checkpoints observed: `epoch_2.pth` through `epoch_5.pth` in latest listing.
  - `slurm_1368.err`: `0` bytes.
  - No condition summary yet.
- Decision:
  - Continue monitoring. Both allowed nodes remain occupied and healthy.

## Seed2 epoch 10 and v10 epoch 7 late - 2026-05-14 13:56 UTC

- Queue:
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `4:14:30`.
  - `1365`/`1366`: dependency-pending.
  - `1368` `s0_occveltime_v10`: RUNNING on `livenode03`, elapsed about `3:01:07`.
  - `1369`/`1370`: dependency-pending.
- Seed2:
  - Latest observed progress: epoch `10/12`, iteration `700/1000`.
  - Saved checkpoint at epoch `9`.
  - Checkpoints observed: `epoch_6.pth` through `epoch_9.pth` in latest listing.
  - `slurm_1364.err`: `0` bytes.
  - No condition summary yet.
- V10:
  - Latest observed progress: epoch `7/12`, iteration `900/1000`.
  - Saved checkpoint at epoch `6`.
  - Checkpoints observed: `epoch_3.pth` through `epoch_6.pth` in latest listing.
  - `slurm_1368.err`: `0` bytes.
  - No condition summary yet.
- Decision:
  - Seed2 is close enough to training completion to poll more tightly for eval `1365` handoff.
  - Continue monitoring v10 on its normal cadence.

## Seed2 near final epoch and v10 epoch 8 late - 2026-05-14 14:24 UTC

- Queue:
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `4:42:36`.
  - `1365`/`1366`: dependency-pending.
  - `1368` `s0_occveltime_v10`: RUNNING on `livenode03`, elapsed about `3:29:13`.
  - `1369`/`1370`: dependency-pending.
- Seed2:
  - Latest observed progress: epoch `11/12`, iteration `800/1000`.
  - ETA about `0:30:58`.
  - Saved checkpoint at epoch `10`.
  - Checkpoints observed include `epoch_10.pth`.
  - `slurm_1364.err`: `0` bytes.
  - No condition summary yet.
- V10:
  - Latest observed progress: epoch `8/12`, at least iteration `950/1000`; output was truncated before the full epoch-8 end.
  - Saved checkpoint at epoch `6` in latest listing; likely checkpoint `7`/`8` will appear in the next poll.
  - `slurm_1368.err`: `0` bytes.
  - No condition summary yet.
- Decision:
  - Poll seed2 more tightly until train `1364` hands off to eval `1365`.

## Seed2 final epoch progress - 2026-05-14 14:40 UTC

- Queue:
  - `1364` `s0_occveltime_s2`: RUNNING on `livenode02`, elapsed about `4:58:27`.
  - `1365`/`1366`: dependency-pending.
  - `1368` `s0_occveltime_v10`: RUNNING on `livenode03`, elapsed about `3:45:04`.
  - `1369`/`1370`: dependency-pending.
- Seed2:
  - Latest observed progress: epoch `12/12`, iteration `400/1000`.
  - ETA about `0:15:29` at that log point.
  - Saved checkpoint at epoch `11`.
  - Checkpoints observed include `epoch_10.pth` and `epoch_11.pth`.
  - `slurm_1364.err`: `0` bytes.
  - Eval `1365` not started yet.
  - No condition summary yet.
- V10:
  - Latest observed progress: epoch `9/12`, around iteration `600/1000` before output truncation.
  - `slurm_1368.err`: `0` bytes.
  - No condition summary yet.
- Decision:
  - Poll soon for seed2 train completion and eval `1365` handoff.

## Seed2 train complete and condition eval started - 2026-05-14 14:58 UTC

- Queue:
  - `1365` `s0_occveltime_s2_eval`: RUNNING on `livenode02`, elapsed about `0:25`.
  - `1366` `s0_occveltime_s2_summary`: dependency-pending.
  - `1368` `s0_occveltime_v10`: RUNNING on `livenode03`, elapsed about `4:02:20`.
  - `1369`/`1370`: dependency-pending.
- Seed2 train:
  - Train job `1364` ended at `2026-05-14T11:57:31-03:00`.
  - Final train-job validation printed mAP `0.3096` and NDS `0.3548`.
  - Treat this as a training sanity signal only; the condition gate requires eval `1365` and summary `1366`.
  - Final checkpoint:
    - `outputs/racformer_train2k_day_occveltimebev_seed2_research/2026-05-14/06-42-38/epoch_12.pth`
  - `slurm_1364.err`: nonzero due normal progress/log text; no traceback observed.
- Seed2 eval:
  - Eval `1365` started with:
    - `STAGE=S0_occveltimebev_seed2`
    - `WEIGHTS=outputs/racformer_train2k_day_occveltimebev_seed2_research/2026-05-14/06-42-38/epoch_12.pth`
  - `eval_slurm_1365.err`: `0` bytes at first poll.
  - No condition summary yet.
- V10:
  - Latest observed progress: epoch `10/12`, iteration `250/1000`.
  - Still no actionable error.
- Decision:
  - Monitor seed2 eval closely until summary `1366` writes the strict gate result.

## Seed2 final summary - 2026-05-14 15:34 UTC

- Summary file:
  - `research/night_gen_phase1/results/S0_occveltimebev_seed2/summary_metrics.md`
- Metrics:
  - day: `0.2992 / 0.3602`, delta `-1.60 pp / -1.43 pp`
  - night: `0.1543 / 0.2058`, delta `+0.56 pp / -0.93 pp`
  - rain: `0.2546 / 0.3542`, delta `-1.97 pp / -1.71 pp`
  - overall: `0.2888 / 0.3553`, delta `-1.52 pp / -1.44 pp`
  - Gate verdict: FAIL.
- Evaluation evidence:
  - Eval `1365` used:
    - `outputs/racformer_train2k_day_occveltimebev_seed2_research/2026-05-14/06-42-38/epoch_12.pth`
  - Full condition eval wrote `eval_by_condition.json` and per-split metrics.
  - Summary job `1366` wrote `summary_metrics.md`; `summary_slurm_1366.err`: `0` bytes.
- Interpretation:
  - Seed2 does not replicate the seed0 strict pass.
  - It also fails more clearly than seed1: night mAP gain is below target, night NDS regresses too much, and day/overall mAP cross the preservation limits.
  - The original `S0_occveltimebev` is not reliable enough as a paper claim without additional stabilization or a cleaner variant.
  - V10 remains useful to finish because it tests the conservative hypothesis that weaker velocity scaling improves the day/night tradeoff.
- Current queue:
  - `1368` `s0_occveltime_v10`: RUNNING on `livenode03`.
  - `1369`/`1370`: dependency-pending.
  - `livenode02`: freed after seed2 summary, but do not submit a new GPU branch until v10 summary is known or a new hypothesis is justified.
- Decision:
  - Keep v10 running and wait for its strict gate summary.
  - Do not mark the objective complete.

## V10 final stretch and livenode02 idle - 2026-05-14 15:35 UTC

- Queue:
  - `1368` `s0_occveltime_v10`: RUNNING on `livenode03`, elapsed about `4:39:29`.
  - `1369`/`1370`: dependency-pending.
  - Stale failed-dependency jobs `1320`/`1321` still appear but are not active experiments.
- Node state:
  - `livenode02`: idle after seed2 summary completed.
  - `livenode03`: allocated to v10.
- V10:
  - Latest observed progress: epoch `11/12`, iteration `650/1000`.
  - ETA about `0:34:50` at that log point.
  - Latest checkpoint listing includes `epoch_10.pth`.
  - `slurm_1368.err`: `0` bytes.
  - No condition summary yet.
- Decision:
  - Do not submit another branch just to occupy `livenode02`.
  - Wait for v10 train/eval/summary because seed2 failed and v10 is now the relevant candidate.

## V10 epoch-10 parallel probe on idle livenode02 - 2026-05-14 15:40 UTC

- Reason:
  - User asked to check whether idle `livenode02` could be used.
  - V10 final training was still running on `livenode03`; an epoch-10 condition-eval probe can run independently and does not mutate model code or block final epoch-12 eval.
  - This is a probe only. The final decision still depends on the epoch-12 V10 strict summary.
- Staged files:
  - Local:
    - `remote_patch_work/staged_occ_vel_time_v10_epoch10_probe/run_s0_occveltimebev_v10_epoch10_probe_eval_livenode02.sbatch`
    - `remote_patch_work/staged_occ_vel_time_v10_epoch10_probe/run_s0_occveltimebev_v10_epoch10_probe_summary_livenode02.sbatch`
    - `remote_patch_work/staged_occ_vel_time_v10_epoch10_probe/summarize_s0_occveltimebev_v10_epoch10_probe.py`
  - Remote:
    - `research/night_gen_phase1/staged_occ_vel_time_v10_epoch10_probe/`
- Validation:
  - Local `bash -n` passed for both sbatch files.
  - Local `python -m py_compile` passed for the summary script.
  - Local guard grep found no `livenode01`, no stale `epoch_12`, no `__LATEST__`.
  - Remote `bash -n` passed for both sbatch files.
  - Remote `conda run -n racformerfix --no-capture-output python -m py_compile` passed for the summary script.
  - Remote guard grep found no `livenode01`, no stale `epoch_12`, no `__LATEST__`.
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_occveltimebev_v10_research/2026-05-14/07-56-03/epoch_10.pth`
- Submitted:
  - Eval job `1371`: `s0_occveltime_v10_e10_probe`, pinned to `livenode02`.
  - Summary job `1372`: `s0_occveltime_v10_e10_sum`, dependency `afterok:1371`, pinned to `livenode02`.
- Concurrent jobs at submission:
  - `1368` final V10 train still RUNNING on `livenode03`.
  - `1369`/`1370` final V10 eval/summary still dependency-pending.
- Decision:
  - Use `1371`/`1372` only as an early trend signal.
  - Do not treat the probe as publication evidence unless final epoch-12 V10 also passes.

## Paper-code audit while V10 runs - RCDINO, D3PD, RobuRCDet - 2026-05-14 15:45 UTC

- Reason:
  - User reminded that paper ideas should be checked against open implementations before adapting them to RaCFormer.
  - Current GPU jobs are already running; this was CPU/read-only audit plus cloning into `paper_impls`.
- Web/source checks:
  - RCDINO paper page and arXiv state implementation is available at `https://github.com/OlgaMatykina/RCDINO`.
  - D3PD Pattern Recognition page says code will be available at `https://github.com/no-Name128/D3PD`.
  - RobuRCDet search found the paper/OpenReview/summary pages, but no clear public implementation repo. Do not spend another cycle searching the same terms unless new evidence appears.
- Cloned/checked:
  - `RCDINO`: `/srv/nfs/shared/gnmp/paper_impls/RCDINO`, head `13e6ef6` (`Initial commit`), source `https://github.com/OlgaMatykina/RCDINO.git`.
  - `D3PD`: `/srv/nfs/shared/gnmp/paper_impls/D3PD`, head `e2f3a8e` (`mod`), source `https://github.com/no-Name128/D3PD.git`; repo already existed when checked.
- RCDINO implementation notes:
  - Main files:
    - `projects/mmdet3d_plugin/models/detectors/rcdetr_dinov2.py`
    - `projects/mmdet3d_plugin/models/backbones/dino.py`
    - `projects/configs/RCDINO/rcdetr_90e_256×704_dino.py`
  - Mechanism:
    - DINOv2 features are extracted at resized image scale, compressed to `latent_dim`, fused across scales, and added to the first image feature map.
    - The DINO contribution uses a learnable scalar `self.dino_weight = nn.Parameter(torch.tensor(0.05))`.
  - Applicability to RaCFormer:
    - Interesting as a conservative residual-camera-context idea.
    - Too invasive for the current NB2 branch because it adds a new pretrained DINOv2 dependency and modifies the image backbone path, not the radar BEV residual path.
    - Do not use until the radar-feature ablation path is exhausted or a separate camera-semantics branch is explicitly chosen.
- D3PD implementation notes:
  - Main files:
    - `projects/mmdet3d/models/detectors/d3pd_v3.py`
    - `projects/mmdet3d/models/necks/fusion.py`
    - `projects/mmdet3d/models/losses/distill_loss.py`
    - `projects/configs/d3pd/d3pd-r50_sf_radar-detfeatsdistill.py`
    - `projects/configs/d3pd/d3pd-r101_sfd_samd_dcrd_smfd.py`
  - Mechanism:
    - Builds a radar pillar branch (`radar_voxel_layer`, `PillarFeatureNet`, `PointPillarsScatter`, `SECOND`, `SECONDFPN`) and fuses radar/camera BEV features.
    - `SamplingWarpFusion` predicts 2D offsets from concatenated BEV features and samples camera BEV features with `grid_sample`.
    - Offset generators are zero-initialized, so the fusion starts as identity sampling.
    - Config uses radar dims `[0, 1, 2, 8, 9, 18]`, i.e. xyz, compensated velocity, and time-like sweep index without RCS. This supports our current RCS-muted velocity/time line.
    - Distillation stack includes sparse BEV feature distillation, sampling position/feature distillation, detection-result distillation, and self-learning mask-focused distillation.
  - Applicability to RaCFormer:
    - Full D3PD is too invasive for quick adoption: BEVDet/CenterPoint teacher-student pipeline, extra pretrained checkpoints, and distillation losses.
    - Useful near-term ideas are: zero-initialized spatial offset/fusion, radar velocity+time without RCS, and possibly lightweight BEV feature sampling. These should be separate branches only after V10 final gate is known.
- Tooling note:
  - `rg` was not installed on the cluster shell used for `paper_impls`; used targeted `find`/`grep` instead.

## V10 final summary and velocity-scale correction - 2026-05-14 18:17 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Queue now only shows stale failed-dependency jobs `1320`/`1321`; active V10 jobs are complete.
  - Final V10 summary exists at `research/night_gen_phase1/results/S0_occveltimebev_v10/summary_metrics.md`.
  - Epoch-10 probe summary exists at `research/night_gen_phase1/results/S0_occveltimebev_v10_epoch10_probe/summary_metrics.md`.
- V10 final metrics:
  - day: `0.3153 / 0.3753`, delta `+0.01 pp / +0.07 pp`.
  - night: `0.1325 / 0.1939`, delta `-1.63 pp / -2.12 pp`.
  - rain: `0.2768 / 0.3730`, delta `+0.25 pp / +0.17 pp`.
  - overall: `0.3037 / 0.3700`, delta `-0.03 pp / +0.02 pp`.
  - Gate verdict: FAIL.
- V10 epoch-10 probe metrics:
  - day: `0.2996 / 0.3631`, delta `-1.56 pp / -1.14 pp`.
  - night: `0.1236 / 0.1801`, delta `-2.51 pp / -3.50 pp`.
  - rain: `0.2654 / 0.3651`, delta `-0.89 pp / -0.62 pp`.
  - overall: `0.2891 / 0.3587`, delta `-1.49 pp / -1.11 pp`.
  - Gate verdict: FAIL.
- Scale-direction correction:
  - Code uses `torch.tanh(stat / scale)` in `RadarRCSBEVResidual.build_map`.
  - Therefore changing vx/vy scales from `20.0` to `10.0` strengthens/saturates the velocity channels; it is not a weaker velocity branch.
  - The previous V10 rationale should be interpreted as a stronger-normalized velocity ablation.
- Interpretation:
  - Stronger-normalized velocity preserves day/overall but destroys the night target.
  - The original `S0_occveltimebev` seed0 pass plus seed1/seed2 failures still leave the velocity+time idea interesting but unstable.
  - A cleaner config-only follow-up, if continuing, is a larger vx/vy scale such as `40.0` to truly weaken velocity magnitude while keeping the occupancy + velocity + time structure.
- Decision:
  - Do not promote V10.
  - Do not submit another GPU experiment until the next branch is explicitly approved.
  - If approved, stage `S0_occveltimebev_v40` as the conservative next branch before considering heavier D3PD/RCDINO-style architectural changes.

## LightDiff and similar low-light enhancement options - 2026-05-14 18:22 UTC

- Trigger:
  - User asked to look into `https://github.com/jinlong17/LightDiff` and similar options.
  - This was a web/source audit only; no remote code changed and no GPU jobs were submitted.
- LightDiff source evidence:
  - Repo: `https://github.com/jinlong17/LightDiff`, official CVPR 2024 implementation.
  - Paper: `https://openaccess.thecvf.com/content/CVPR2024/html/Li_Light_the_Night_A_Multi-Condition_Diffusion_Framework_for_Unpaired_Low-Light_CVPR_2024_paper.html`.
  - Paper mechanism: multi-condition controlled diffusion using low-light image, depth map, and text prompt, plus perception-specific reward modeling.
  - Paper reports nuScenes nighttime 3D vehicle AP gains of about `+4.2` and `+4.6` for BEVDepth and BEVStereo.
  - Repo testing path is not turnkey: `test.py` contains placeholder checkpoint/data paths and appears set up around `CAM_FRONT`, 512x512 inference, depth maps, prompts, and a released checkpoint.
  - Repo issues show checkpoint access problems: open issue `#15` says the README checkpoint URLs were unavailable as of `2025-09-10`; related open issues also ask how to get checkpoints.
- Applicability to RaCFormer/NB2:
  - Potentially more relevant than DriveGEN for test-time nighttime enhancement because it aims to brighten low-light images while preserving scene content, rather than generating synthetic object appearances.
  - Main mismatch: RaCFormer is 6-camera radar-camera 3D detection; LightDiff's public test script is front-camera-oriented. Applying only `CAM_FRONT` may give a weak or biased signal; applying all six cameras requires per-camera depth/prompts and a path override for nuScenes images.
  - Training LightDiff from scratch is not a near-term NB2 branch: it uses Stable Diffusion/ControlNet, BEVDepth setup, custom depth generation, and long training.
  - Best bounded use, if checkpoint access works, is an inference-only QC pilot on a small set of night frames, then possibly a night-split eval with generated images injected through a copied/symlinked nuScenes image tree.
- Similar options checked:
  - `ICCV_MAET` (`https://github.com/cuiziteng/ICCV_MAET`): useful source for physics-based low-light degradation; LightDiff already builds on this family. Cheap augmentation idea, but less directly perception-aligned for RaCFormer 3D.
  - `QuadPrior` (`https://github.com/daooshee/QuadPrior`): zero-reference CVPR 2024 low-light enhancement trained from normal-light images; easier as an off-the-shelf image enhancer, but not driving/perception-specific.
  - `Retinexformer` (`https://github.com/caiyuanhao1998/Retinexformer`): mature low-light enhancement toolbox with many benchmarks and high-resolution support; practical baseline for visual QC, but generic and not task-aligned.
  - `AllWeatherNet` (`https://github.com/Jumponthemoon/AllWeatherNet`): autonomous-driving adverse-weather/low-light enhancer with pretrained model instructions; broader weather focus, likely useful as a fast enhancement baseline.
  - `MonoWAD` (`https://github.com/VisualAIKHU/MonoWAD`): weather-adaptive diffusion for monocular 3D detection; useful conceptually, but detector/task stack is monocular and not a direct RaCFormer retrofit.
  - `DarkDriving` (`https://arxiv.org/abs/2603.18067`): new aligned day/night dataset and benchmark; useful framing/evaluation source, not an immediate RaCFormer implementation branch.
- Decision:
  - Do not replace the current radar BEV branch with a full LightDiff training effort.
  - If image-enhancement evidence is needed next, prioritize a small LightDiff checkpoint-access and QC probe:
    1. Verify checkpoint/test data links or locate a usable mirror.
    2. Run 6-12 representative night images, preferably across multiple cameras if feasible.
    3. Reject immediately if vehicles, lane geometry, or bounding-box-relevant silhouettes shift, as happened with DriveGEN.
    4. Only after QC, consider a controlled night-split RaCFormer eval with original day/rain images unchanged.
  - If LightDiff checkpoint access is blocked, try Retinexformer or AllWeatherNet as cheaper visual-enhancement baselines before heavier diffusion training.

## V40 velocity-scale branch staged and smoke-passed - 2026-05-14 18:27 UTC

- Rationale:
  - V10 failed, and code inspection showed V10 was actually a stronger velocity normalization because the map uses `tanh(stat / scale)`.
  - V40 is the conservative opposite ablation: keep the successful seed0 occupancy + velocity + time structure, but weaken vx/vy saturation with scales `40.0`.
- Local staged files:
  - `remote_patch_work/configs/racformer_train2k_day_occveltimebev_v40_research.py`.
  - `remote_patch_work/staged_occ_vel_time_v40/`.
- Remote staged files:
  - `configs/racformer_train2k_day_occveltimebev_v40_research.py`.
  - `research/night_gen_phase1/staged_occ_vel_time_v40/`.
- Validation:
  - Local `bash -n` passed for smoke/train/eval/summary sbatch files.
  - Local `python -m py_compile` passed for config and summarizer.
  - Local guard grep found no `livenode01`, no `__LATEST__`, no stale V10 stage names, and no seed1/seed2 leftovers.
  - Remote `bash -n` passed for smoke/train/eval/summary sbatch files.
  - Remote `conda run -n racformerfix --no-capture-output python -m py_compile` passed for config and summarizer.
  - Remote config parse asserted:
    - `output_shape=(128, 128)`.
    - `rcs_index=3`, `rcs_scale=1000000.0`.
    - `extra_indices=(4, 5, 6)`.
    - `extra_scales=(40.0, 40.0, 1.0)`.
  - Remote guard grep passed.
- Smoke job:
  - Job `1373`: `s0_occveltime_v40_smoke` on `livenode03`.
  - Output confirmed:
    - `radar_occveltime_bev_residual_v40 (128, 128) (3, 4, 5, 6) (1000000.0, 40.0, 40.0, 1.0)`.
    - `state_keys 4`.
    - `half_forward_zero_init True`.
- Decision:
  - V40 is ready for full train/eval/summary submission.
  - Use the same S0 publication gate after summary: night mAP >= `+1.0 pp`, day mAP >= `-1.0 pp`, overall mAP >= `-1.5 pp`, night NDS >= `-0.5 pp`.

## V40 full chain submitted and clean train start - 2026-05-14 18:29 UTC

- Submitted dependency chain on `livenode03`:
  - Train: `1374` `s0_occveltime_v40`.
  - Eval: `1375` `s0_occveltime_v40_eval`, dependency `afterok:1374`.
  - Summary: `1376` `s0_occveltime_v40_summary`, dependency `afterok:1375`.
- Train start evidence:
  - Job `1374` is RUNNING on `livenode03`.
  - Work dir: `outputs/racformer_train2k_day_occveltimebev_v40_research/2026-05-14/15-27-27`.
  - Config: `configs/racformer_train2k_day_occveltimebev_v40_research.py`.
  - GPU: RTX 4090, initial memory use `15367M`.
  - Early train log reached epoch `1/12`, iteration `50/1000`, loss `49.95`, ETA about `5:13:26`.
  - `slurm_1374.err`: `0` bytes.
- Queue note:
  - Only stale failed-dependency jobs `1320`/`1321` remain besides the active V40 chain.
  - `livenode02` is idle; do not submit another branch unless it is clearly useful and non-overlapping.
- Decision:
  - Continue monitoring V40.
  - Next useful transition is the epoch-1 checkpoint or any stderr/job failure.

## V40 early training health - 2026-05-14 18:31 UTC

- Queue:
  - `1374` `s0_occveltime_v40`: RUNNING on `livenode03`.
  - `1375`/`1376`: dependency-pending.
- Training progress:
  - Epoch `1/12`, iteration `100/1000`.
  - Loss moved from `49.95` at iter `50` to `39.73` at iter `100`.
  - GPU overlap check: RTX 4090 at `100%` util, about `19323 / 24564` MiB used.
  - `slurm_1374.err`: `0` bytes.
- Decision:
  - Training is active and healthy.
  - Keep `livenode02` idle unless a clearly bounded, non-overlapping check is chosen.
  - Next useful V40 check remains the epoch-1 checkpoint or any stderr/job failure.

## LightDiff checkpoint access check - 2026-05-14 18:32 UTC

- Reason:
  - While V40 train was running, checked whether the LightDiff inference/QC path is blocked at artifact access.
- Evidence:
  - Raw LightDiff README still points to:
    - Test data: `https://drive.google.com/drive/folders/1nG5j3h7b8ERXezzprt1a4dRUHS-TpMeW?usp=sharing`.
    - Checkpoint: `https://csuohio-my.sharepoint.com/:u:/g/personal/2819040_vikes_csuohio_edu/EYIrVBctW3ZIu_NqC93whaABhyJgLUz2eAnd53Aw2lYKCg?e=M1TaU2`.
  - `curl -I -L` on the Google Drive folder returned HTTP `200` HTML.
  - `curl -I -L` on the SharePoint checkpoint URL returned HTTP `404`.
- Interpretation:
  - LightDiff test data appears at least page-reachable, but the official checkpoint URL is currently dead from this environment.
  - This matches the open GitHub issue reporting unavailable checkpoint URLs.
- Decision:
  - Treat LightDiff as blocked for an immediate QC pilot unless a working checkpoint mirror is found.
  - If image enhancement is needed before such a mirror exists, prefer Retinexformer or AllWeatherNet as the next practical off-the-shelf baseline.

## Wayback checkpoint recovery check - 2026-05-14 18:41 UTC

- Trigger:
  - User asked whether implementations without checkpoints can be recovered via the Wayback Machine.
- LightDiff:
  - Checked Wayback availability for the official SharePoint checkpoint URL in the README:
    - Exact URL with `?e=M1TaU2`: no archived snapshots.
    - Same path without query string: no archived snapshots.
    - HTTP variant: no archived snapshots.
    - `?download=1` variant: no archived snapshots.
  - Checked Wayback availability for `raw.githubusercontent.com/jinlong17/LightDiff/main/README.md` and `test.py`: no archived snapshots.
  - GitHub repo page itself has a Wayback snapshot, but that only recovers README/source-page metadata, not the missing checkpoint binary.
  - Broad CDX wildcard queries against `web.archive.org/cdx` timed out with 504, so the reliable evidence is the exact Wayback availability API result: no recoverable snapshot for the checkpoint artifact URL.
- Similar options:
  - `AllWeatherNet` currently has a live Google Drive pretrained model page (`latest_net_G_A.pth`) and does not require Wayback recovery at this stage.
  - `QuadPrior` currently has a live Google Drive checkpoint/results folder page and Baidu fallback in the README; Wayback availability for the Google Drive folder returned no snapshots, but current access is not blocked.
  - `MonoWAD` currently has a live Google Drive pretrained-model page; Wayback availability for the file URL returned no snapshots, but current access is not blocked.
  - `Retinexformer`, `LightenDiffusion`, `AGLLDiff`, and `M2Retinexformer` advertise current pretrained-weight paths or model-download instructions; they are better treated as live-check candidates rather than Wayback-recovery candidates.
- Decision:
  - Do not spend GPU or integration time on LightDiff unless a non-Wayback mirror or author-provided checkpoint appears.
  - If an image-enhancement pilot is needed now, use a current-access baseline first: AllWeatherNet for driving/weather relevance, Retinexformer for maturity, or QuadPrior/AGLLDiff for diffusion-style LLIE with available artifacts.

## V40 mid-epoch health - 2026-05-14 18:45 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Branch/head: `main`, `869407e`.
  - Train job `1374` is still RUNNING on `livenode03`, elapsed `18:26` of `8:00:00`.
  - Eval `1375` and summary `1376` remain dependency-pending.
- Training progress:
  - Epoch `1/12`, iteration `650/1000`.
  - Loss trend from latest concise poll:
    - iter `300`: loss `29.76`.
    - iter `400`: loss `29.71`.
    - iter `500`: loss `28.46`.
    - iter `600`: loss `27.62`.
    - iter `650`: loss `26.33`.
  - `slurm_1374.err`: `0` bytes.
  - No `epoch_*.pth` checkpoint yet.
  - No V40 summary files yet.
- Decision:
  - V40 remains healthy but has not reached a decision point.
  - Next useful checkpoint is epoch-1 checkpoint creation, eval start, summary output, or any stderr/failure signal.

## V40 epoch-1 checkpoint reached - 2026-05-14 18:54 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Train job `1374` remains RUNNING on `livenode03`, elapsed `27:26` of `8:00:00`.
  - Eval `1375` and summary `1376` remain dependency-pending.
- Epoch-1 evidence:
  - Latest log reached `Epoch [1/12][1000/1000]`.
  - Log line: `Saving checkpoint at 1 epochs`.
  - Checkpoint exists:
    - `outputs/racformer_train2k_day_occveltimebev_v40_research/2026-05-14/15-27-27/epoch_1.pth`
    - size: `765772993` bytes.
  - `slurm_1374.err`: `0` bytes.
  - No V40 summary files yet.
- Interpretation:
  - V40 has passed the first epoch/checkpoint health marker.
  - This is still not a metric decision; final train/eval/summary must complete before comparing to the S0 gate.

## V40 epoch-2 start confirmed - 2026-05-14 18:55 UTC

- Remote state:
  - Train job `1374` is still RUNNING on `livenode03`, elapsed `28:36` of `8:00:00`.
  - Eval `1375` and summary `1376` remain dependency-pending.
- Evidence:
  - Latest log reached `Epoch [2/12][50/1000] loss: 24.33, loss_depth: 1.28`.
  - `slurm_1374.err`: `0` bytes.
  - Only checkpoint so far is `epoch_1.pth`.
- Decision:
  - V40 continues normally after the first checkpoint.
  - Continue low-frequency monitoring; no new branch submission while V40 is the active gate experiment.

## V40 final summary and decision - 2026-05-15 01:32 UTC

- Remote state:
  - Host/cwd: `cluster-live`, `/srv/nfs/shared/gnmp/RaCFormer`.
  - Branch/head: `main`, `869407e`.
  - V40 train/eval/summary jobs are no longer in `squeue`.
  - Final checkpoint exists:
    - `outputs/racformer_train2k_day_occveltimebev_v40_research/2026-05-14/15-27-27/epoch_12.pth`
    - size: `765773185` bytes.
  - Summary files exist:
    - `research/night_gen_phase1/results/S0_occveltimebev_v40/summary_metrics.md`
    - `research/night_gen_phase1/results/S0_occveltimebev_v40/summary_metrics.json`
    - `research/night_gen_phase1/results/S0_occveltimebev_v40/eval/eval_by_condition.json`
- V40 metrics:
  - day: `0.3121 / 0.3711`, delta `-0.32 pp / -0.35 pp`.
  - night: `0.1410 / 0.2006`, delta `-0.77 pp / -1.45 pp`.
  - rain: `0.2676 / 0.3681`, delta `-0.67 pp / -0.33 pp`.
  - overall: `0.3026 / 0.3669`, delta `-0.14 pp / -0.29 pp`.
  - Gate verdict: FAIL.
- Gate check:
  - day mAP >= `-1.0 pp`: PASS (`-0.32 pp`).
  - overall mAP >= `-1.5 pp`: PASS (`-0.14 pp`).
  - night mAP >= `+1.0 pp`: FAIL (`-0.77 pp`).
  - night NDS >= `-0.5 pp`: FAIL (`-1.45 pp`).
- Log/stderr notes:
  - Train stderr size was `1746` bytes and contains tqdm/progress-bar output only.
  - Eval stderr size was `4328` bytes and contains eval progress/info output only.
  - Summary stderr is empty.
- Interpretation:
  - V40 confirms the velocity-scale family is not the answer: weakening vx/vy saturation preserves day/overall better than V10, but still hurts the night target.
  - Do not promote V40.
  - The original `S0_occveltimebev` seed0 remains a single-seed positive result, but seed1/seed2/V10/V40 now argue against investing more in simple velocity-scale tuning.
- Decision:
  - Stop this V10/V40 velocity-scale line.
  - Next branch should not be another vx/vy scale-only ablation.
  - If continuing radar-BEV residual work, prefer a structurally different low-risk hypothesis such as occupancy+time only, occupancy-only replication, or a zero-initialized spatial alignment/fusion idea inspired by D3PD.
  - If continuing image-enhancement work, LightDiff remains blocked by checkpoint access; use a current-access baseline such as AllWeatherNet, Retinexformer, QuadPrior, or AGLLDiff only after a small visual QC pass.
