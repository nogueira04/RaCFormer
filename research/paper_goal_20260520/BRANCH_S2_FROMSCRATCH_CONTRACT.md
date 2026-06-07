# Branch S2 Contract — Day/Night Specialists From Scratch on Full Per-Condition Data

**Status**: continuation of Branch S; tests one specific caveat from `BRANCH_S_RESULT_20260522T103535Z.md` §"Engineering caveats" — whether the regression was caused by fine-tuning eroding canonical's cross-condition features OR by per-condition specialisation being structurally bounded by data sparsity.
**Authored**: 2026-05-22, post Branch S FAIL_GATE_MISS.
**Binding context**: Branch S contract (`BRANCH_S_STAGE3_CONTRACT.md`); Branch S result (`BRANCH_S_RESULT_20260522T103535Z.md`); v3 §9 brightness-gating clause (distinguished as in Branch S §preamble); clean-state audit (`CLEAN_STATE_AUDIT_20260521T202215Z.md`).

## Mechanism (one paragraph)

Train two RaCFormer specialists *from random initialisation* on the full per-condition training data — day specialist on all ~19,685 day samples, night specialist on all ~3,385 night samples — using canonical's default training schedule (12 epochs, default LR, no backbone LR reduction). Reuse Branch S's already-trained router (`router.pth`, SHA `9fb1a5dc8bd1511c817791623e3a51b5c39ddccc88260e754fc3844a270a8ce2`, 99.98% accuracy). At inference, route per Branch S; aggregate predictions. The hypothesis is narrower than original Branch S: not "specialists work in general," but specifically "specialists work when trained from scratch with full per-condition data, ruling out the fine-tuning-erosion mechanism documented in Branch S's failure mode."

## What Branch S2 tests that Branch S did not

Branch S's failure mode (specialists regressed on their own conditions) admits two structural interpretations:

- **H1 — Fine-tuning erosion**: Canonical's cross-condition features are useful within-condition; fine-tuning on a single-condition subset at LR 4e-5 for 6-12 epochs erodes those features faster than it specialises.
- **H2 — Data-sparsity binding**: Per-condition specialists fundamentally need more data than nuScenes provides per condition, and Branch S's failure was the early signal of this constraint rather than a fine-tuning artefact.

Branch S used 2,000 day samples + 640 night samples (small subsets, fine-tune init). Branch S2 uses ~19,685 day + ~3,385 night (full per-condition train data, from-scratch init). If Branch S2 PASSes the formal gate, H1 is supported and H2 is falsified. If Branch S2 fails and the night specialist undertrains (val night-mAP collapses early), H2 is supported. If both specialists regress at full data and from-scratch init, the diagnosis deepens to "per-condition specialisation is structurally bounded by missing cross-condition information, regardless of training approach."

## Baselines

- Baseline B (canonical): `checkpoints/racformer_r50_f8.pth`, SHA `0afe006...`, day mAP 0.5387, night mAP 0.3228, overall mAP 0.5297, ECE 0.0776. Same baseline as Branch S used for the formal gate.
- Branch S original specialists for cross-comparison: day_specialist_seed0.pth (SHA `3c7267...`), night_specialist_seed0.pth (SHA `809622...`). Read-only references — do NOT modify or replace.

## Compute budget

≤ 72 GPU-h hard session cap, dominated by:
- Step 2 (day from scratch): ≤ 24 GPU-h
- Step 3 (night from scratch): ≤ 8 GPU-h
- Step 4 (integrated eval): ≤ 1 GPU-h
- Step 5-6 (replication seed-20260502): ≤ 32 GPU-h
- Step 7 (ablation): ≤ 2 GPU-h
- Step 8 (twilight): ≤ 1 GPU-h
- Step 9 (final report): ≤ 0.1 GPU-h
- Margin: ≤ 4 GPU-h

Per-invocation soft cap: ≤ 14 GPU-h, ≤ 12 wall-clock hours. Multi-invocation expected; state file is the continuity mechanism.

## Master state file

Path: `research/paper_goal_20260520/branch_s2/BRANCH_S2_STATE.json`. Same schema as Branch S's BRANCH_S_STATE.json. Atomic writes. Records: current_step, current_seed, step_status, checkpoint paths + SHA256s, per-step metrics, cumulative GPU-h, halt_reasons_log.

## Step 0 — Branch-choice memo extension (mandatory)

Write `research/night_gen_phase1/reports/BRANCH_CHOICE_S2_<UTC>.md` citing:

1. Branch S §"Engineering caveats" specifically flagging that the original Branch S used 2,000 day + 640 night samples (small) with fine-tune from canonical, and that from-scratch + full-per-condition-data was NOT tested. Branch S2 is the structurally-motivated next experiment to discriminate H1 vs H2.
2. v3 §9 brightness-gating distinction: same as Branch S (two independently-trained models, learned classifier router, per-scene routing — not post-hoc score adjustment).
3. Same NR paper substrate-incompatibility framing as Branch S; Branch S2 is a finer-grained test of the same diagnosis.
4. Pre-registered formal gate (identical to Branch S §formal gate, since the baseline and routing setup are the same).
5. Pre-registered kill conditions (tighter than Branch S given the higher GPU commitment): see §Hard kill conditions below.

If any cannot be written truthfully → HALT.

## Step 1 — Router reuse (no GPU)

DO NOT retrain the router. Branch S's router achieved 99.98% accuracy on the full-val labelled gate; retraining is wasted budget. Steps:

1. Verify `research/paper_goal_20260520/branch_s/checkpoints/router.pth` exists at the expected SHA `9fb1a5dc8bd1511c817791623e3a51b5c39ddccc88260e754fc3844a270a8ce2`.
2. Symlink or path-reference it as the Branch S2 router. Do NOT copy or modify.
3. Re-verify router accuracy on full-val using the existing router_metrics.json — should be 99.98%. If different, BLOCKED with reason `router_drift`.
4. Record `step_status[1] = "complete"` with router_path = Branch S original.

## Step 2 — Day specialist from scratch (≤ 24 GPU-h)

1. Build day-only training pkl: filter `nuscenes_infos_train_sweep.pkl` (or whichever full train pkl Branch S's router training used) for `scene.description == "day"`. Expected count: ~19,685 day samples. Document count in `DAY_SPECIALIST_S2_SUBSET.md`.
2. Build new config `branch_s2_day_specialist_seed0.py`:
   - `_base_` = canonical RaCFormer training config (e.g., `configs/racformer_r50_nuimg_704x256_f8.py`).
   - `data.train.ann_file` = day-only pkl from step 1.
   - `data.train.max_samples` = full count (not capped).
   - `total_epochs` = 12 (canonical default, not the fine-tune 6).
   - Optimizer: default canonical LR (typically 2e-4 for AdamW), NO `lr_mult=0.1` for backbone or sampling_offset. From-scratch.
   - `load_from` = `null` (NO canonical init). Random init from default mmdet3d module initialisers.
   - `resume_from` = `null`.
3. Train via the canonical training launcher (e.g., `tools/dist_train.py` or `staged_*_replication/train_seeded.py` pattern), seed 0, on livenode02 or 03.
4. Eval at epoch 3, 6, 9, 12 on full val (NOT capped). Record per-epoch full-val day mAP, night mAP, overall mAP, NDS, ECE.
5. **Hard kill checks** (aggressive — see §Hard kill conditions):
   - After epoch 3: day mAP must be ≥ 0.40 (74% of Baseline B's 0.539). Below → KILL.
   - After epoch 6: day mAP must be ≥ 0.48 (89% of Baseline B's 0.539). Below → KILL.
6. Save `day_specialist_s2_seed0.pth` to `research/paper_goal_20260520/branch_s2/checkpoints/`. Compute SHA256.
7. Write `DAY_SPECIALIST_STEP2_S2_SUMMARY.json` with per-epoch metrics, final SHA, runtime, GPU-h.

## Step 3 — Night specialist from scratch (≤ 8 GPU-h)

This is the highest-risk step. 3,385 night samples from scratch is data-sparsity territory.

1. Build night-only training pkl: filter `nuscenes_infos_train_sweep.pkl` for `scene.description == "night"`. Expected count: ~3,385 night samples. Document in `NIGHT_SPECIALIST_S2_SUBSET.md`.
2. Build new config `branch_s2_night_specialist_seed0.py`: same pattern as Step 2's day config, but ann_file = night-only pkl. `total_epochs` = 12 (or up to 24 if compute permits — see step 5).
3. Train from scratch, seed 0, default LR, no backbone reduction.
4. Eval at epoch 3, 6, 9, 12 on full val.
5. **Hard kill checks** (most aggressive — data-sparsity guard):
   - After epoch 3: night mAP must be ≥ 0.15 (46% of Baseline B's 0.323). Below → KILL.
   - After epoch 6: night mAP must be ≥ 0.22 (68% of Baseline B's 0.323). Below → KILL.
   - After epoch 9: night mAP must be ≥ 0.28 (87% of Baseline B's 0.323). Below → KILL with reason `night_undertrained_data_sparsity_binding`.
6. If hit kill after epoch 6 or 9: this is itself the H2-supporting result. Write `BRANCH_S2_NIGHT_UNDERTRAINED_<UTC>.md` documenting the per-epoch trajectory and the data-sparsity verdict; record as a substantive finding for the NR paper.
7. If passes all kill checks: save `night_specialist_s2_seed0.pth`. Compute SHA256.
8. Write `NIGHT_SPECIALIST_STEP3_S2_SUMMARY.json` with per-epoch metrics.

## Step 4 — Integrated eval seed-0 (≤ 1 GPU-h)

Same protocol as Branch S Step 4:
- Route each val sample via Branch S router.
- Apply matching specialist.
- Aggregate full-val overall mAP, NDS, ECE; per-condition (day/night/rain) breakdown.
- Apply formal gate (same thresholds as Branch S).
- All ✓ → `step_status[4] = "pass"`, proceed to Step 5.
- Any ✗ → write `BRANCH_S2_FAILURE_MODE_<UTC>.md` with gate gaps + interpretation (H1 vs H2 framing), set TERMINAL, skip Steps 5-8.

## Step 5 — Seed-20260502 replication (≤ 32 GPU-h)

Only if Step 4 passes. Re-train both specialists from scratch with seed-20260502 via `staged_branch_s2_replication/train_seeded.py`. Same configs, same kill checks. Save checkpoints, compute SHAs.

## Step 6 — Integrated eval seed-20260502 (≤ 1 GPU-h)

Same as Step 4 with seed-20260502 specialists. Apply replication PASS predicates:
- Both seeds pass formal gate.
- Seed-to-seed night-mAP delta ≤ 1.0 pp.
- Seed-to-seed overall-mAP delta ≤ 0.7 pp.

All ✓ → proceed to Step 7. Any ✗ → `step_status[6] = "fail_replication"`, set TERMINAL.

## Step 7 — Ablation (≤ 2 GPU-h)

Force-all-day and force-all-night routing on seed-0 specialists. Same protocol as Branch S Step 7. Ablation PASS predicate: both force-all-day and force-all-night must regress by ≥ +1.5 pp absolute on the opposite condition vs integrated. If routing is decorative → `BRANCH_S2_FAIL_ABLATION_NULL`.

## Step 8 — Twilight subset analysis (≤ 1 GPU-h)

Same protocol as Branch S Step 8. Router confidence < 0.7 subset (was 4 samples in Branch S). Compute oracle-router-vs-actual gap on this subset.

## Step 9 — Final report

Write `research/paper_goal_20260520/branch_s2/BRANCH_S2_RESULT_<UTC>.md` with:

- Verdict: one of `BRANCH_S2_PASS` / `BRANCH_S2_FAIL_GATE_MISS` / `BRANCH_S2_FAIL_NIGHT_UNDERTRAINED` / `BRANCH_S2_FAIL_REPLICATION` / `BRANCH_S2_FAIL_ABLATION_NULL`.
- Per-step status table.
- Specialist checkpoint paths + SHA256s.
- Seed-0 and (if reached) seed-20260502 per-condition metrics.
- Per-epoch training trajectories for both specialists (the key NR paper evidence either way).
- H1 vs H2 verdict: explicit one-paragraph interpretation of what the result tells us about Branch S's failure mode.
- Paper implication: PASS → "NR paper restructures with from-scratch specialist routing as positive contribution; H2 falsified, fine-tune erosion is the binding mechanism." Various FAILs → "NR paper adds Branch S2 to the inventory with mechanism class (specific to fail mode); the trajectory data is a substantive figure regardless of verdict."

## Hard kill conditions

Any one → terminal failure:

- Step 2 day specialist day-mAP < 0.40 after epoch 3 or < 0.48 after epoch 6 (severe undertraining).
- Step 3 night specialist night-mAP < 0.15 after epoch 3, < 0.22 after epoch 6, or < 0.28 after epoch 9 (data-sparsity binding; H2 supported).
- Step 4 formal gate fails.
- Step 6 replication fails.
- Step 7 ablation shows null mechanism.
- Cumulative GPU > 72 hours.
- User STOP sentinel: `research/paper_goal_20260520/branch_s2/STOP_BRANCH_S2`.

## Do NOT touch

- `checkpoints/racformer_r50_f8.pth` (Baseline B; read-only).
- `outputs/racformer_train2k_day_research/.../epoch_12.pth` (Baseline A / Stage 1B S0; read-only).
- `research/paper_goal_20260520/branch_s/*` (Branch S original artifacts; read-only — Branch S router is symlinked/path-referenced but not modified).
- All v3/v4/v5 artifacts under `research/paper_goal_2026051{5,6,8}/*` and `research/paper_goal_20260520/{candidates,FINAL_REPORT_v5,CLEAN_STATE_AUDIT,NR_PAPER_OUTLINE_FINAL,BRANCH_S_*}.md`.
- `goal_v3.md`, `goal_v4.md`, `goal_v5.md`, original Branch S contract.

All Branch S2 outputs under `research/paper_goal_20260520/branch_s2/`.

## Environment

Conda env `racformerfix`. SSH MCP `cluster_live_tail` (fallback to `cluster_live` if DNS fails). Repo `/srv/nfs/shared/gnmp/RaCFormer`. SLURM partition `livecluster`, livenode02/03 (RTX 4090, 24 GB).

## Resumability

Same pattern as v5 / Branch S: every invocation reads BRANCH_S2_STATE.json, resumes from current_step, advances as far as soft cap permits, writes state, exits. Re-paste same /goal to continue.
