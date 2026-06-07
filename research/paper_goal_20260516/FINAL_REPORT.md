# FINAL REPORT

Verdict: `PASS-PAPER-NR`

Summary: Stage 1B completed the negative-results paper audit artifacts and the NR outline now contains prose sections, figure plan, and claims. The all-variant subset matrix is now exact CPU-only nuScenes subset re-evaluation for all 44 variants.

UTC: 20260517T211223Z

## Code / Config / Script Paths

- `research/paper_goal_20260516/audit_stage1b/stage1b_audit.py`
- `research/paper_goal_20260516/audit_stage1a/s0_subsets.json`
- `AUDIT_STAGE1B_20260517T211223Z.md`

## Git SHA / Status

```text
M configs/racformer_r50_nuimg_704x256_f8.py
 M loaders/pipelines/__init__.py
 M loaders/pipelines/transforms.py
 m mmcv
 M models/necks/view_transformer_racformer.py
 M models/racformer.py
 M models/racformer_head.py
 M models/racformer_transformer.py
?? 256
?? configs/racformer_eval_fullval_calib_yaw2_research.py
?? configs/racformer_eval_fullval_calib_yaw4_research.py
?? configs/racformer_eval_fullval_dualviewdistill_zero_research.py
?? configs/racformer_eval_fullval_research.py
?? configs/racformer_r50_nuimg_704x256_f8.py.bak.conditionfusion_20260512_212540
?? configs/racformer_train2k_day_calibnoise_research.py
?? configs/racformer_train2k_day_occtimebev_research.py
?? configs/racformer_train2k_day_occvelbev_research.py
?? configs/racformer_train2k_day_occveltimebev_research.py
?? configs/racformer_train2k_day_occveltimebev_seed1_research.py
?? configs/racformer_train2k_day_occveltimebev_seed2_research.py
?? configs/racformer_train2k_day_occveltimebev_v10_research.py
?? configs/racformer_train2k_day_occveltimebev_v40_research.py
?? configs/racformer_train2k_day_radarbevexp_research.py
?? configs/racformer_train2k_day_radarquery_research.py
?? configs/racformer_train2k_day_radarquery_seed20260502_research.py
?? configs/racformer_train2k_day_radarquery_topk90_research.py
?? configs/racformer_train2k_day_radarquery_topk90_seed20260502_research.py
?? configs/racformer_train2k_day_rcsbev_research.py
?? configs/racformer_train2k_day_rcsoccbev_research.py
?? configs/racformer_train2k_day_rcsvelbev_research.py
?? configs/racformer_train2k_day_rcsveltimebev_research.py
?? configs/racformer_train2k_day_research.py
?? configs/racformer_train2k_genaug_research.py
?? configs/racformer_train2k_genaug_seed20260425_ratio12p5_research.py
?? configs/racformer_train2k_genaug_seed20260425_ratio18p75_research.py
?? configs/racformer_train2k_genaug_seed20260425_ratio18p75_w025_research.py
?? configs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research.py
?? configs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_research.py
?? configs/racformer_train2k_genaug_seed20260425_ratio21p25_research.py
?? configs/racformer_train2k_genaug_seed20260425_research.py
?? configs/racformer_train2k_genaug_seed20260502_ratio18p75_research.py
?? configs/racformer_train2k_genaug_seed20260502_ratio18p75_w05_adaptfusion_research.py
?? configs/racformer_train2k_genaug_seed20260502_ratio18p75_w05_research.py
?? configs/racformer_train2k_mixed_conditionfusion_research.py
?? configs/racformer_train2k_mixed_contrelqfusion_no_cosine_research.py
?? configs/racformer_train2k_mixed_contrelqfusion_no_geometry_research.py
?? configs/racformer_train2k_mixed_contrelqfusion_research.py
?? configs/racformer_train2k_mixed_contrelqfusion_stats_only_research.py
?? configs/racformer_train2k_mixed_dualviewdistill_epoch6_research.py
?? configs/racformer_train2k_mixed_dualviewdistill_research.py
?? configs/racformer_train2k_mixed_dualviewdistill_smoke_research.py
?? configs/racformer_train2k_mixed_research.py
?? configs/racformer_train2k_simnight_research.py
?? loaders/pipelines/__init__.py.bak.phase1
?? loaders/pipelines/__init__.py.bak.t9simnight
?? loaders/pipelines/manifest_loading.py
?? models/dualview_distill.py
?? models/racformer.py.bak.kernelpersistent_20260513_0413
?? models/racformer.py.bak.radarbevexp_20260513_033229
?? models/racformer.py.bak.rcsbev_20260513_092940
?? models/racformer.py.bak.rcsvelbev_20260513_1536
?? models/racformer.py.bak.t9modelpatch
?? models/racformer_transformer.py.bak.conditionfusion_20260512_212540
?? nuscenes_infos_train_2k_day.pkl.bak.t9broken
?? nuscenes_infos_train_2k_mixed.pkl.bak.t9broken
?? nuscenes_infos_train_2k_mixed_oversampled.pkl.bak.t9broken
?? nuscenes_infos_val_day_matched.pkl.bak.t9broken
?? research/night_gen_phase1/
?? research/paper_goal_20260515/
?? research/paper_goal_20260516/
?? research/spec_validation/
?? research_artifacts_BranchD_S6_20260516T230808Z.tar.gz
?? research_artifacts_BranchD_S6_202605
```

## Training / Eval Commands

No training or GPU eval was submitted for Stage 1B. The audit consumed existing full-val prediction JSONs. The subset deliverable uses exact CPU-only nuScenes subset re-evaluation for every audited variant.

## Sub-Stage Progression

- Stage 1A: complete (`AUDIT_STAGE1A_20260517T031357Z.md`).
- Stage 1B: complete (`AUDIT_STAGE1B_20260517T211223Z.md`), variants audited: 44.
- Stage 2: Branch E BLOCKED-AT-STAGE-2 in prior diagnostic; NR track promoted.
- Stage 3A/3B/3C: no CRKD training submitted by this Stage 1B run.

## NR Paper Outline Status

`research/paper_goal_20260516/NEGATIVE_RESULTS_PAPER_OUTLINE.md` contains prose sections, figure plan, taxonomy claims, limitations, and conclusion.

## Failure Cases and Negative Branches

- Prior Corruption: Branch D radar-query replacement.
- Gate Collapse: Branch A/S5 decoder-side gate family.
- Geometry-Label Drift: NB2/DriveGEN night generation failures.
- Stochastic Gate Flipping: C-family seed instability.

## CLAIM_INVENTORY Reference

The Stage 1B audit artifacts provide the claim-to-evidence source for the NR paper. No Branch D archive file was modified in this Stage 1B pass.
