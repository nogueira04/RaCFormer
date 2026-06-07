# Clean State Audit 20260521T202215Z

## Executive Verdict

Verdict: CLEAN for the original Stage 1B S0 baseline. The clean-state replay reproduced the cached Stage 1B S0 metrics within tiny run-to-run/evaluator noise:

| Source | Checkpoint | mAP | NDS | ECE proxy | Prediction count |
|---|---|---:|---:|---:|---:|
| Cached audit_stage1b S0 | outputs/racformer_train2k_day_research/2026-04-25/20-24-58/epoch_12.pth | 0.30399059113285365 | 0.3697754272016486 | 0.11874962530039652 | 1289551 |
| Clean-state replay S0 | outputs/racformer_train2k_day_research/2026-04-25/20-24-58/epoch_12.pth | 0.3039686398905509 | 0.3697807993338371 | 0.11875051300911935 | 1289585 |
| v5/rctrans fresh baseline | checkpoints/racformer_r50_f8.pth | 0.5296299576513736 | 0.6062222358717055 | 0.07760817409995023 | not used here |

Primary cause of the apparent v5 jump is checkpoint identity drift, not a hidden Stage 1B evaluation contamination. Stage 1B S0 used the trained day checkpoint with SHA256 `6a12183fed06832fc5478de38cc8510c71db143d36871182fa70a77369e5a055`; v5/rctrans fresh baseline used canonical `checkpoints/racformer_r50_f8.pth` with SHA256 `0afe006954836de2ab059d417328fb46ed41efaab40386186168a0c38c129d66`.

No remediation was performed. No checkpoints, old v3/v4/v5 artifacts, model weights, datasets, or prior reports were moved, deleted, renamed, or overwritten. The only new repo artifact is this report.

## Contract And Environment

- Contract: `/home/gabriel/LIVE/research/paper_goal_20260520/CLEAN_STATE_AUDIT_CONTRACT.md`.
- Remote repo: `/srv/nfs/shared/gnmp/RaCFormer`.
- Requested SSH MCP target: `cluster_live_tail`; this failed DNS resolution as `LIVENODE01` (`getaddrinfo ENOTFOUND livenode01`). Audit execution used working target `cluster_live`, which reported host `cluster-live` for control commands and ran the eval job on `livenode02`.
- Conda environment: `racformerfix`.
- nuScenes devkit: `1.1.11`.
- Default repo HEAD before audit: `869407ec64918b7254684bbd8d51b3f50b076578`.
- GPU budget: <=4 GPU-hours; consumed approximately 0.561 GPU-hours for the successful full-val replay, plus a failed wrapper attempt that exited before Python evaluation.

## Step 0: Preflight State

- UTC snapshot: `20260521T191933Z`.
- `AUTONOMOUS_STATE_V5.json`: `current_phase='TERMINAL'`, `active_jobs=None`.
- `squeue -u gnmp`: no active jobs at audit start.
- Repo is a git repository.
- Initial branch/status: `## main...origin/main [ahead 18]`.
- Initial `git status --porcelain | wc -l`: `95`.
- Initial `git stash list`: empty.
- Pre-audit HEAD: `869407ec64918b7254684bbd8d51b3f50b076578`.
- Snapshot files:
  - `/tmp/clean_state_audit_20260521T191933Z/step0_git_status_short_branch.txt`
  - `/tmp/clean_state_audit_20260521T191933Z/step0_git_stash_list.txt`
  - `/tmp/clean_state_audit_20260521T191933Z/pre_audit_head.txt`

Initial dirty state included tracked modifications in:

```text
configs/racformer_r50_nuimg_704x256_f8.py
loaders/pipelines/__init__.py
loaders/pipelines/transforms.py
models/necks/view_transformer_racformer.py
models/racformer.py
models/racformer_head.py
models/racformer_transformer.py
mmcv/setup.py via dirty submodule
```

## Step 1: Audit SHA Recovery

Status: recovered by history inference, not embedded provenance.

- `research/paper_goal_20260516/audit_stage1b/manifest.json` did not contain a commit SHA.
- `AUDIT_STAGE1B_*.md` files did not expose an explicit commit SHA.
- `git log --all` and `git reflog` showed no May 2026 commits around Stage 1B/v5; the latest commit was:
  - `869407ec64918b7254684bbd8d51b3f50b076578 2026-03-25 experiment(009-010): DINOv3 inference-time integration - all strategies negative`
- Audit SHA used: `SHA_INFERRED_BY_HISTORY = 869407ec64918b7254684bbd8d51b3f50b076578`.

This is weaker than an embedded artifact SHA. The inference is acceptable for this audit because there were no later commits/reflog entries and the Stage 1B/v5 artifacts were produced on a dirty worktree layered over the same HEAD.

## Step 2: Source State Diff And Contamination Classification

`git diff --stat 869407ec..HEAD` was empty because the inferred audit SHA equals pre-audit HEAD. The relevant contamination risk was the dirty worktree before stashing.

Tracked dirty diff before stash:

```text
configs/racformer_r50_nuimg_704x256_f8.py  |   5 +-
loaders/pipelines/__init__.py              |  10 +-
loaders/pipelines/transforms.py            |  92 +++++++++++
models/necks/view_transformer_racformer.py |  21 ++-
models/racformer.py                        | 237 +++++++++++++++++++++++++++--
models/racformer_head.py                   | 125 +++++++++++++--
models/racformer_transformer.py            | 176 ++++++++++++++++++++-
7 files changed, 628 insertions(+), 38 deletions(-)
```

Classification:

- Shared infrastructure contamination present before stash: config, loaders, pipelines, view transformer, model, head, transformer, plus dirty `mmcv/setup.py` inside submodule. These could affect forward/evaluation if left active.
- Experimental scaffolding present as many untracked configs, sbatch scripts, research artifacts, and local backups.
- Residual after top-level stash: dirty `mmcv/setup.py` and untracked nested teacher-source directories remained because they were submodule/ignored-state artifacts. They were recorded but not remediated.

The clean replay stashed top-level tracked/untracked files, checked out the inferred SHA, and used read-only copies of the originally untracked eval wrapper/config from outside the repo.

## Step 3: Checkpoint Inventory And S0 Reconciliation

Full inventory file: `/srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/checkpoint_inventory.tsv`.

Inventory count: 159 checkpoint-like files under `checkpoints/`, `outputs/`, and `research/` plus TSV header.

Key reconciled checkpoints:

| Role | Path | SHA256 | Size bytes | mtime UTC |
|---|---|---|---:|---|
| v5/rctrans canonical fresh baseline | `checkpoints/racformer_r50_f8.pth` | `0afe006954836de2ab059d417328fb46ed41efaab40386186168a0c38c129d66` | 256290121 | 2025-10-19 14:35:28.000000000 +0000 |
| Stage 1B original S0 | `outputs/racformer_train2k_day_research/2026-04-25/20-24-58/epoch_12.pth` | `6a12183fed06832fc5478de38cc8510c71db143d36871182fa70a77369e5a055` | 764836881 | 2026-04-26 04:40:22.674061576 +0000 |

Conclusion: canonical v5 fresh baseline and original Stage 1B S0 are different checkpoints. The original S0 checkpoint was located, so the audit did not block on missing S0.

## Step 4: Evaluation Methodology Trace

Stage 1B/S0 pipeline:

```text
conda run -n racformerfix --no-capture-output python -u research/night_gen_phase1/eval_by_condition.py   --config configs/racformer_eval_fullval_research.py   --weights outputs/racformer_train2k_day_research/2026-04-25/20-24-58/epoch_12.pth   --out-dir <S0 eval dir>   --full-val
```

- Full validation split: `nuscenes_infos_val_sweep.pkl`, 6019 samples.
- nuScenes metric: official CVPR 2019 detection evaluation, 10 classes, centre-distance thresholds 0.5/1.0/2.0/4.0.
- Condition counts: day 4449, night 602, rain 968, day_matched 0.
- ECE proxy: `stage1b_audit.py`, detection score vs same-class 2m TP proxy after greedy matching. This is not official nuScenes ECE.

v5/rctrans Stage 3B pipeline:

- Used `research/paper_goal_20260518/tools/rctrans_calib_stage3b_full.py` style evidence.
- Used config family `configs/racformer_r50_nuimg_704x256_f8.py`.
- Fresh baseline checkpoint: `checkpoints/racformer_r50_f8.pth`.
- Candidate mAP: `0.5295144765337056`; candidate NDS: `0.6061747214918463`; candidate ECE proxy: `0.04738352834223519`.
- Fresh baseline mAP: `0.5296299576513736`; fresh baseline NDS: `0.6062222358717055`; fresh baseline ECE proxy: `0.07760817409995023`.

Primary methodological divergence: checkpoint path/SHA. Split size and official mAP/NDS evaluator family were consistent enough for this audit question.

## Step 5: Clean-State Stash

Before stashing, read-only copies were made to audit scratch because the Stage 1B wrapper and eval config were untracked and would be removed by `git stash -u`:

```text
/srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/research/night_gen_phase1/eval_by_condition.py
/srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/research/paper_goal_20260516/audit_stage1b/stage1b_audit.py
/srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/configs/racformer_eval_fullval_research.py
```

Stash command completed successfully:

```text
git stash push -u -m "clean_state_audit_20260521T191933Z"
```

Stash ref: `cd87e5fe03463d439f1c68f9d3f53658afaf1f18`.

Status after top-level stash:

```text
## main...origin/main [ahead 18]
 m mmcv
?? research/night_gen_phase1/teachers/bevfusion_src/
?? research/night_gen_phase1/teachers/transfusion_l/openpcdet_src/
```

Submodule status after stash:

```text
## HEAD (no branch)
 M setup.py
```

## Step 6: Clean-State Replay

Checked out inferred audit SHA:

```text
git checkout 869407ec64918b7254684bbd8d51b3f50b076578
```

Post-checkout status retained only the residual submodule/ignored teacher artifacts recorded above.

First SLURM attempt:

- Job: `1497`.
- Result: failed before Python eval because stdout/stderr paths were under node-local `/tmp` not available on `livenode02`.
- No evaluation metrics were produced by this attempt.

Successful SLURM attempt:

- Job: `1498`.
- Node: `livenode02`.
- Runtime captured during audit: 00:33:37, approximately 0.561 GPU-hours for one GPU.
- Exit code captured during audit: `0:0`.
- Output log: `/srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/clean_eval_slurm_1498.out`.
- Error log: `/srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/clean_eval_slurm_1498.err`.

Clean replay command:

```text
conda run -n racformerfix --no-capture-output python -u /srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/research/night_gen_phase1/eval_by_condition.py --config /srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/configs/racformer_eval_fullval_research.py --weights /srv/nfs/shared/gnmp/RaCFormer/outputs/racformer_train2k_day_research/2026-04-25/20-24-58/epoch_12.pth --out-dir /srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/results/S0/eval --full-val
```

ECE audit command:

```text
conda run -n racformerfix --no-capture-output python -u /srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/research/paper_goal_20260516/audit_stage1b/stage1b_audit.py --repo-root /srv/nfs/shared/gnmp/RaCFormer --results-root /srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/results --paper-root /srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/paper --val-pkl nuscenes_infos_val_sweep.pkl --dataroot /srv/nfs/shared/shared/nuscenes --baseline S0 --max-variants 1 --n-bootstrap 200 --skip-official-subsets
```

Clean-state outputs:

```text
/srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/results/S0/eval/eval_by_condition.json
/srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/results/S0/eval/submission_overall/pts_bbox/results_nusc.json
/srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/paper/audit_stage1b/calibration_reliability.json
```

## Step 7: Finding Classification

### Three-Way Metrics

| Comparison point | mAP | NDS | ECE proxy | Notes |
|---|---:|---:|---:|---|
| Cached audit_stage1b S0 | 0.30399059113285365 | 0.3697754272016486 | 0.11874962530039652 | Original Stage 1B S0 artifacts. |
| Clean-state replay S0 | 0.3039686398905509 | 0.3697807993338371 | 0.11875051300911935 | Same intended checkpoint, clean checkout, wrapper/config copied from scratch. |
| v5/rctrans fresh baseline | 0.5296299576513736 | 0.6062222358717055 | 0.07760817409995023 | Canonical checkpoint, not original Stage 1B S0. |
| v5/rctrans candidate | 0.5295144765337056 | 0.6061747214918463 | 0.04738352834223519 | Score calibration candidate on canonical/fresh baseline family. |

### Verdict By Claim Family

- Stage 1B cached S0 baseline: ROBUST for this audit. Clean-state replay reproduced mAP/NDS/ECE proxy closely using the intended S0 checkpoint.
- Stage 1B analyses that compare variants against the cached S0 prediction/eval pipeline: ROBUST with respect to the suspected v5 contamination, provided each variant used its recorded intended checkpoint. This audit replayed S0 only.
- v5/rctrans claims framed as `+0.2255 mAP vs S0`: SUSPECT/invalid as stated. The candidate mAP `0.5295144765337056` was compared against Stage 1B S0 `0.30399059113285365`, but the v5 fresh baseline on the same checkpoint family was already `0.5296299576513736`. Candidate delta vs fresh baseline is approximately `-0.0001154811176680`, not +0.2255.
- v5/rctrans ECE improvement: partly valid only within the canonical checkpoint family. Candidate ECE proxy `0.04738352834223519` improves over canonical fresh baseline ECE proxy `0.07760817409995023`, but the comparison should not be presented as using original Stage 1B S0 unless replayed on the trained day S0 checkpoint.
- v3/v4/v5 negative substrate-incompatibility results: likely ROBUST if they used the Stage 1B result JSON pipeline and trained S0 checkpoint. Claims that mixed canonical `checkpoints/racformer_r50_f8.pth` with original S0 naming are SUSPECT until replayed or relabeled.

### Recommendation

The paper can keep the Stage 1B S0 baseline and its negative/robustness audit foundation. The rctrans/v5 section needs an apples-to-apples cleanup before being treated as paper evidence:

1. Relabel canonical-checkpoint experiments as canonical full-checkpoint analyses, not original S0 comparisons, or replay them against `outputs/racformer_train2k_day_research/2026-04-25/20-24-58/epoch_12.pth`.
2. Record checkpoint path, SHA256, size, and mtime in every manifest and report.
3. Avoid generic `S0` labels without checkpoint identity.
4. Isolate dirty source experiments from shared eval infrastructure before future audit/paper runs.

No remediation was applied here.

## Step 8: Restoration

Restoration was completed before writing this report.

Commands/evidence:

```text
git checkout main
git stash pop stash@{0}
diff -u /tmp/clean_state_audit_20260521T191933Z/step0_git_status_short_branch.txt /tmp/clean_state_audit_20260521T191933Z/step8_restored_git_status_short_branch.txt
DIFF_EXIT=0
git rev-parse --abbrev-ref HEAD -> main
git rev-parse HEAD -> 869407ec64918b7254684bbd8d51b3f50b076578
git stash list -> empty
```

Final restored status before this report file was written exactly matched the Step 0 snapshot. After this report is written, the only expected additional worktree difference is this new untracked report file.

## Step 9: Final Audit Artifacts

Remote report:

```text
/srv/nfs/shared/gnmp/RaCFormer/research/paper_goal_20260520/CLEAN_STATE_AUDIT_20260521T202215Z.md
```

Local report mirror:

```text
/home/gabriel/LIVE/research/paper_goal_20260520/CLEAN_STATE_AUDIT_20260521T202215Z.md
```

Scratch/evidence directory:

```text
/srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z
```

Full checkpoint inventory is embedded below and also stored as TSV at:

```text
/srv/nfs/shared/gnmp/clean_state_audit_20260521T191933Z/checkpoint_inventory.tsv
```

## Appendix: Full Checkpoint Inventory

```tsv
sha256	size_bytes	mtime_utc	path
0afe006954836de2ab059d417328fb46ed41efaab40386186168a0c38c129d66	256290121	2025-10-19 14:35:28.000000000 +0000	checkpoints/racformer_r50_f8.pth
6c6cb9d008d810ee0b316f131cf280c8bf8fc2d43d583c9cb1fe1c8260d78979	764836945	2026-02-03 18:48:38.100673571 +0000	outputs/racformer_r50_nuimg_704x256_f8/2026-01-25/20-49-49/epoch_33.pth
cb867559df01b598a6c616212980142d63defbaa035eb29a09570bee33e33a5b	764836945	2026-02-04 00:52:32.125031591 +0000	outputs/racformer_r50_nuimg_704x256_f8/2026-01-25/20-49-49/epoch_34.pth
933d8a817df8bd642555557bd1f4455066afbdb14747a3087d044f6e2d3b776c	764836945	2026-02-04 07:35:59.099904075 +0000	outputs/racformer_r50_nuimg_704x256_f8/2026-01-25/20-49-49/epoch_35.pth
5852d1ca6f79a4af966d41c22bab97fc4252672b82ee0f0f5f73529d18091a00	764836945	2026-02-04 13:39:00.079419848 +0000	outputs/racformer_r50_nuimg_704x256_f8/2026-01-25/20-49-49/epoch_36.pth
a143ea1ff76c2e4d3d92a4e41c11e1d803dea34cea51998c1f6b1582418cadd2	764836753	2025-12-01 11:01:32.046582985 +0000	outputs/racformer_r50_nuimg_704x256_f8/old/2025-12-01/01-13-28/epoch_1.pth
b67d115170c7a991da85e7809dc1b45ea2d1f6d2ac07ec91a91c9707786d2324	764836753	2025-12-02 19:05:55.637223911 +0000	outputs/racformer_r50_nuimg_704x256_f8/old/2025-12-02/10-31-11/epoch_1.pth
b0f82fab597e6574adf76e4380c03d7a333c237226d1fc19d063e859cf594cc5	764836881	2025-12-03 00:41:08.566123171 +0000	outputs/racformer_r50_nuimg_704x256_f8/old/2025-12-02/10-31-11/epoch_2.pth
67c8ed7348ebebb8e7b36a8751f09266bfa6a7a09c1616eed72646b6bcb072ce	764836881	2025-12-03 06:23:50.172565198 +0000	outputs/racformer_r50_nuimg_704x256_f8/old/2025-12-02/10-31-11/epoch_3.pth
8e97cd5661990d706fe16955491105e139e33e22d38c47b34177542ca833eee8	764836881	2025-12-03 11:48:13.077592270 +0000	outputs/racformer_r50_nuimg_704x256_f8/old/2025-12-02/10-31-11/epoch_4.pth
56e55ff3c6536b402a7810d16ca21238ced076871a9ef2ed978edbb6591e7433	764836945	2025-12-22 06:01:00.689012684 +0000	outputs/racformer_r50_nuimg_704x256_f8_3cam_3rad/2025-12-13/10-19-58/epoch_33.pth
c46a3e825835b09d1ec839ae85b536161fcd51b84f878aa058e015e98e6a97cc	764836945	2025-12-22 12:15:35.307026639 +0000	outputs/racformer_r50_nuimg_704x256_f8_3cam_3rad/2025-12-13/10-19-58/epoch_34.pth
339942caff3e90034c48ff611e1d0d770fd014a65cb6e16a881cc1bb79a222b5	764836945	2025-12-22 18:39:43.328221422 +0000	outputs/racformer_r50_nuimg_704x256_f8_3cam_3rad/2025-12-13/10-19-58/epoch_35.pth
533fd66f9c9839e63c109817a68f5073b21d1d9831c7c6e230df6858b82a3fd4	764836945	2025-12-23 00:54:21.364186860 +0000	outputs/racformer_r50_nuimg_704x256_f8_3cam_3rad/2025-12-13/10-19-58/epoch_36.pth
a7e741d7f61b60a98f675a542c6f8c28f3d884b036f8c3e7c12aa4c1dec3a53d	764836945	2026-01-28 22:55:48.369465392 +0000	outputs/racformer_r50_nuimg_704x256_f8_dropout/2026-01-25/22-45-16/epoch_12.pth
437f725c740bc0520719e2efd37f978aca8103246a12f1a032e1c8e0f64254f4	764836945	2026-01-30 10:01:14.757515679 +0000	outputs/racformer_r50_nuimg_704x256_f8_dropout/2026-01-25/22-45-16/epoch_18.pth
a55c43ff6b890a31465bcf11c366c086a8f32a7e88007d718d25ca1562e93ac6	764836945	2026-01-31 20:43:13.900045408 +0000	outputs/racformer_r50_nuimg_704x256_f8_dropout/2026-01-25/22-45-16/epoch_24.pth
bbea3f18df4aabc84ce2fd7d7c58fa2966bbcbcf2e7f7d59a6ce18f437b3e35e	764836945	2026-02-02 07:21:49.773482697 +0000	outputs/racformer_r50_nuimg_704x256_f8_dropout/2026-01-25/22-45-16/epoch_30.pth
88a5cd89520b5020490220b07cd4915c1f868ab120489c1bab1b2b6d5f2438d7	764836945	2026-02-03 18:12:54.301123692 +0000	outputs/racformer_r50_nuimg_704x256_f8_dropout/2026-01-25/22-45-16/epoch_36.pth
dd5601c27e1723df1bb83978f48f48e85889a2ecb05dadc1897d7db1166e4ed5	764836753	2026-01-27 12:00:24.196158485 +0000	outputs/racformer_r50_nuimg_704x256_f8_dropout/2026-01-25/22-45-16/epoch_6.pth
69c626cad4111c145e173980243c83f1ee082265d859e73b5306bec2f7b4d3df	764836753	2026-01-26 16:59:55.699844039 +0000	outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-01-25/22-42-30/epoch_1.pth
5efff5e6876c1d936700171c921d7c3901a0c7ae395aaff62cc2f0e7a5ddc698	764836945	2026-01-27 08:00:25.814315469 +0000	outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-01-25/22-42-30/epoch_2.pth
d560fa157e7c251d616024914695f90ee6a035f885d66f263be88af9c4dad233	764836945	2026-01-27 23:19:41.915015630 +0000	outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-01-25/22-42-30/epoch_3.pth
4fec98888cfbcc559c9eb7c7730b2491c1336343767bf6588c7cc4ac4428a3dc	764836945	2026-01-30 09:37:31.006332480 +0000	outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-01-28/09-23-07/epoch_3.pth
049272c060f20d2f79b979db55307e9ace7fdbecc99e92747e7007ca48ed3fc7	764836945	2026-01-30 23:16:22.760360576 +0000	outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-01-28/09-23-07/epoch_4.pth
6a052cc59ade495d3965f1645a8a45760efddc548fc6291f26263f9b66d8d3bb	764836945	2026-01-31 14:46:12.764068677 +0000	outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-01-28/09-23-07/epoch_5.pth
fd25cbec09da916f68211b0e780d50aada6b5c637207fca4ad0982f18d77cb3b	764836945	2026-02-01 04:31:25.776912288 +0000	outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-01-28/09-23-07/epoch_6.pth
52bf910ed7aa28cb10bd37a77631d51085997297d352152e4f82482ba58cb61c	764841489	2026-03-07 08:28:54.984917283 +0000	outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-02-01/01-41-59/epoch_33.pth
b5702b4aa2b584e83d30f8090a11404d206d31c403198093580826a5b35ed91f	764841489	2026-03-07 14:27:09.342632956 +0000	outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-02-01/01-41-59/epoch_34.pth
f4992712533903e82f8edaa1c12093b92cc8d6103df50b066e0bc077d9b64c64	764841489	2026-03-07 20:51:26.203095211 +0000	outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-02-01/01-41-59/epoch_35.pth
fed8cbe7e69e54b04204678c6a9c4cb834e063890a2b588f851900f5c4d12381	764841489	2026-03-08 02:49:40.233922928 +0000	outputs/racformer_r50_nuimg_704x256_f8_nightaug/2026-02-01/01-41-59/epoch_36.pth
d6cf7e6e4a50d4cad36647678ec7ebcc9bdfe1792d9c65af8c5f771dee96946f	764836945	2026-05-15 06:47:52.561806350 +0000	outputs/racformer_train2k_day_calibnoise_research/2026-05-14/23-27-48/epoch_10.pth
ad15789ee049488a37b6e63a236f410793bd9d305014092ac8454f38d49cfb73	764836945	2026-05-15 07:13:52.660881435 +0000	outputs/racformer_train2k_day_calibnoise_research/2026-05-14/23-27-48/epoch_11.pth
e09ed33a53790d8fcaced138728af77f3aa13377b3c75245a59d40340a3149de	764836945	2026-05-15 07:39:53.397614491 +0000	outputs/racformer_train2k_day_calibnoise_research/2026-05-14/23-27-48/epoch_12.pth
6a9825970686470fbb010339cd751300066727ae0d7a6022e1b144820e3c7705	764836945	2026-05-15 06:21:50.745676726 +0000	outputs/racformer_train2k_day_calibnoise_research/2026-05-14/23-27-48/epoch_9.pth
3f90c4dccf927171203bf1ab0ae17188483abcc776dd2ad348003dad837bc702	765717825	2026-05-14 09:20:05.900330528 +0000	outputs/racformer_train2k_day_occtimebev_research/2026-05-14/01-59-04/epoch_10.pth
29904a59821a5af607c4c65080498733c0233db594971f61c76add8bce25a139	765717825	2026-05-14 09:46:12.404141412 +0000	outputs/racformer_train2k_day_occtimebev_research/2026-05-14/01-59-04/epoch_11.pth
b3f4ff023669b4304c05132b4bf97f597853991d30d2aea80f386c10e9c9e754	765717825	2026-05-14 10:12:20.206478134 +0000	outputs/racformer_train2k_day_occtimebev_research/2026-05-14/01-59-04/epoch_12.pth
0e5e5e4059c6e401151ccbaf1d311e193d1eabf0c0533b32cce1356b9f9d6c97	765717825	2026-05-14 08:53:57.301790593 +0000	outputs/racformer_train2k_day_occtimebev_research/2026-05-14/01-59-04/epoch_9.pth
06809ab8ee74a0f66bb645eb6954caca85cd2fb625f26ba74c1588a2424e4842	765745473	2026-05-13 21:25:51.920439063 +0000	outputs/racformer_train2k_day_occvelbev_research/2026-05-13/14-04-52/epoch_10.pth
79fc353e7bd18127d2d957ddde4031bf95545937816ccb9af2592f9220cddf29	765745473	2026-05-13 21:51:57.657424932 +0000	outputs/racformer_train2k_day_occvelbev_research/2026-05-13/14-04-52/epoch_11.pth
4dc97c2aff06e3caab715449b53264c7fd19fb0135714f13e7ff671c196e7420	765745473	2026-05-13 22:18:05.138340410 +0000	outputs/racformer_train2k_day_occvelbev_research/2026-05-13/14-04-52/epoch_12.pth
9a8d6fea98c30aabaff9febb14e6d59772559d732fd1131b564c32a039520aa3	765745473	2026-05-13 20:59:45.209591174 +0000	outputs/racformer_train2k_day_occvelbev_research/2026-05-13/14-04-52/epoch_9.pth
146097fe1c8ecffeda9d1f615198b74c4c27dc885ad865eb90a788c3027fcf3c	765773121	2026-05-14 02:00:34.080551963 +0000	outputs/racformer_train2k_day_occveltimebev_research/2026-05-13/18-39-42/epoch_10.pth
cec3bdb5586971581d03b30464a0db10890a88c77aae850738da47a60fed06c7	765773121	2026-05-14 02:26:40.029735602 +0000	outputs/racformer_train2k_day_occveltimebev_research/2026-05-13/18-39-42/epoch_11.pth
db5caeaa02407a274a7e8cc2412365f7a8ccafc20046a4221a4abfc5cafaf11e	765773121	2026-05-14 02:52:43.916049191 +0000	outputs/racformer_train2k_day_occveltimebev_research/2026-05-13/18-39-42/epoch_12.pth
7d705bf65c30e7261e80f9d73e0b612f8deedfa09d009f742cf0caadca5fdbe3	765773121	2026-05-14 01:34:27.497234046 +0000	outputs/racformer_train2k_day_occveltimebev_research/2026-05-13/18-39-42/epoch_9.pth
f9322a3e284cd26233b01437e3472980045b83af7745dd600e6ff42e98b4892e	765773185	2026-05-14 08:07:46.279806646 +0000	outputs/racformer_train2k_day_occveltimebev_seed1_research/2026-05-14/00-46-41/epoch_10.pth
4a6f13cc48de00fc794e1df7f2faf5bf143074b404c96fdf7170164539a2d9ce	765773185	2026-05-14 08:33:51.749101294 +0000	outputs/racformer_train2k_day_occveltimebev_seed1_research/2026-05-14/00-46-41/epoch_11.pth
4861addcaa8c3654d3f286fd07c95a44d8a252aa97f38b9a53876f165a09bba2	765773185	2026-05-14 08:59:59.139748713 +0000	outputs/racformer_train2k_day_occveltimebev_seed1_research/2026-05-14/00-46-41/epoch_12.pth
e4f3c66f6d9094e411f10e138ccc79fff218622f3032edb6cf56cb53a612f85e	765773185	2026-05-14 07:41:37.497995152 +0000	outputs/racformer_train2k_day_occveltimebev_seed1_research/2026-05-14/00-46-41/epoch_9.pth
cf787a8fcd33aa53b5c1a451d74d0734ea2c463050679cc57a8c4bd0439c66db	765773185	2026-05-14 14:03:29.738345788 +0000	outputs/racformer_train2k_day_occveltimebev_seed2_research/2026-05-14/06-42-38/epoch_10.pth
054d93aa3a0825b96f6d87d37df000462ea3bc988a099fbd81fbfa91fbe8c583	765773185	2026-05-14 14:29:34.844159878 +0000	outputs/racformer_train2k_day_occveltimebev_seed2_research/2026-05-14/06-42-38/epoch_11.pth
67228f7ee207bc1c0ea4dc41acb7bb30d68f532769f1f46539495bcfe1756967	765773185	2026-05-14 14:55:40.239250856 +0000	outputs/racformer_train2k_day_occveltimebev_seed2_research/2026-05-14/06-42-38/epoch_12.pth
6913843d05d2169ad349c14e98df344719bf9b964940ac0ca3e0522033b6a97c	765773185	2026-05-14 13:37:21.999936581 +0000	outputs/racformer_train2k_day_occveltimebev_seed2_research/2026-05-14/06-42-38/epoch_9.pth
e4569118afa2422512a90c3925f828dd517df542ace13ea99feab634849bdc78	765773121	2026-05-14 15:16:57.638056466 +0000	outputs/racformer_train2k_day_occveltimebev_v10_research/2026-05-14/07-56-03/epoch_10.pth
9f04a68fae755225de398d29650ce8531ac217109be4285fc3587ac4046d9341	765773185	2026-05-14 15:43:02.176058614 +0000	outputs/racformer_train2k_day_occveltimebev_v10_research/2026-05-14/07-56-03/epoch_11.pth
74fb775c311e9dde90d98856729b4d53e52c8a42a736c4b62da7e457fc1f47c6	765773185	2026-05-14 16:09:08.428077685 +0000	outputs/racformer_train2k_day_occveltimebev_v10_research/2026-05-14/07-56-03/epoch_12.pth
b56fda8292346958f4f319f64c6f514cf13b5cfeb4eb182546d9ede0acf9e141	765773121	2026-05-14 14:50:50.841007674 +0000	outputs/racformer_train2k_day_occveltimebev_v10_research/2026-05-14/07-56-03/epoch_9.pth
e3ca0613b4f590209062791fdc4e5b56f4b9d9347210be89f84e52d188c633bc	765773121	2026-05-14 22:48:17.696937360 +0000	outputs/racformer_train2k_day_occveltimebev_v40_research/2026-05-14/15-27-27/epoch_10.pth
f6e184206dcb8110eb5610c443f25e5d5982f307a0dfad2f200a9c11368d3056	765773185	2026-05-14 23:14:25.041855837 +0000	outputs/racformer_train2k_day_occveltimebev_v40_research/2026-05-14/15-27-27/epoch_11.pth
4f510d062285faf69929b012ae0a4cb632932cbde46c1972dfd70c49c9f56520	765773185	2026-05-14 23:40:31.395605814 +0000	outputs/racformer_train2k_day_occveltimebev_v40_research/2026-05-14/15-27-27/epoch_12.pth
8c7193fe89b0b46567dcc4340dc1eb1e4237cb4be50b13a93a3c0ea995ac3acc	765773121	2026-05-14 22:22:10.102489831 +0000	outputs/racformer_train2k_day_occveltimebev_v40_research/2026-05-14/15-27-27/epoch_9.pth
38fd078d93614f5d6e632d2180f5f6d27671c9ce1ffa19d9945c3171daf125d0	767203142	2026-05-13 07:57:05.401475991 +0000	outputs/racformer_train2k_day_radarbevexp_research/2026-05-13/00-35-54/epoch_10.pth
277f0662b5836e11073d6e2fdfefa3df309e1163ab4c7911fd44e3ee132e2bf6	767203142	2026-05-13 08:23:13.493090949 +0000	outputs/racformer_train2k_day_radarbevexp_research/2026-05-13/00-35-54/epoch_11.pth
5748d9f5379f62afebd72d2332a5f0464ea5c36fa77a52205aa24bea6606f5fb	767203142	2026-05-13 08:49:22.052700882 +0000	outputs/racformer_train2k_day_radarbevexp_research/2026-05-13/00-35-54/epoch_12.pth
d4f37edf21e4b64a2c8e4333e3305f80135315dcbd31b3c2fc9ed9bdba77b5b1	767203142	2026-05-13 07:30:57.288007760 +0000	outputs/racformer_train2k_day_radarbevexp_research/2026-05-13/00-35-54/epoch_9.pth
01b715e1e930d3a4d4d2af50814a57498b2e817fab98e04f1e73fce948396012	764836945	2026-05-16 00:08:05.159180751 +0000	outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_10.pth
93aa695e2a3d90de172c81cf6b1e022065b0851ac97f978c71650bbaa314a0d3	764836945	2026-05-16 00:34:04.610878377 +0000	outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_11.pth
baf47406ef7aa386fa2de54a2221db2a83de942cbe927706d8c493dbd0e4a8cd	764836945	2026-05-16 01:00:04.484026781 +0000	outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_12.pth
ff0c9d8f110643d5649b2a497d32a8b547617d2f41330d1cca9ac68fc1bfc26f	764836945	2026-05-15 23:42:04.118547601 +0000	outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_9.pth
150aca682a285a83311ef835d38759a480adc1214a0d84834b51544f6b81f20a	764836945	2026-05-16 00:52:34.679603978 +0000	outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_10.pth
cff0556e8b3db1e1304741b4261fa625912083ba0f70f39d72e80342f39e7a63	764836945	2026-05-16 01:18:35.650334488 +0000	outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_11.pth
d5b49718aa90239a6eccdacf11e205b8aa0518665257454b01d3da3f025bfc8b	764836945	2026-05-16 01:44:38.190532109 +0000	outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_12.pth
e107447b0152d4fb4dc3c53c3cbcc8e7af468ef0c564a83abe54b056d40d4437	764836945	2026-05-16 00:26:31.724161053 +0000	outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_9.pth
5a4c9526ba2d426e1e91e30b2ae5c5f14330b0c8b265bf1b37ad8b5ff732fc1d	765690177	2026-05-13 14:07:55.907854744 +0000	outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_10.pth
85b9a18b7daacb88a3038f8863a5ce270cb808a8ed90f515d8f6ff2723d25349	765690177	2026-05-13 14:33:59.020215492 +0000	outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_11.pth
c35b00287ae5aef298e5a6877d45f1d39525cc787023d3e1ff544e3e7b8c3316	765690177	2026-05-13 15:00:03.339960844 +0000	outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_12.pth
8dfec5ca014837703cb7bb9d08af5a8d6ced415522180e20d0abe3985ecbc361	765690177	2026-05-13 13:41:51.949524747 +0000	outputs/racformer_train2k_day_rcsbev_research/2026-05-13/06-47-25/epoch_9.pth
fb2841dc7d6d77543d0d2b585c4d5dc3e8ec2688d2957e98a003db9ccc82f448	765690177	2026-05-13 14:18:25.915985334 +0000	outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_10.pth
d1ae9471a7766d580633356d11accd639031100cc9db4dc69512bbc73181e70f	765690177	2026-05-13 14:44:33.640331063 +0000	outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_11.pth
7266d1ec7faeea1fe335a3c1c9009a8c84a6e4ff9861f8a83b17578e25ed4ebd	765690177	2026-05-13 15:10:42.312903301 +0000	outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_12.pth
4b791f728d323b6b9659add21cfaa9443e4f5ad560491b214e38105e66c5a8c9	765690177	2026-05-13 13:52:16.270652229 +0000	outputs/racformer_train2k_day_rcsoccbev_research/2026-05-13/06-57-22/epoch_9.pth
fe07abc68b1e3498783d3960f12f1625755b2846fa9c70a8c6f0cadf40e7852b	765745473	2026-05-13 20:08:20.636144165 +0000	outputs/racformer_train2k_day_rcsvelbev_research/2026-05-13/12-47-35/epoch_10.pth
7d84dbfd3e6e636629159e892cb7d424cbd47666af8436b3ddb500b6c2d330f7	765745473	2026-05-13 20:34:24.875778273 +0000	outputs/racformer_train2k_day_rcsvelbev_research/2026-05-13/12-47-35/epoch_11.pth
a2704b8bf344a1adb401172f01e37160b4cd3df8ff40697d2e4cdb8f078fbdb4	765745473	2026-05-13 21:00:29.306453809 +0000	outputs/racformer_train2k_day_rcsvelbev_research/2026-05-13/12-47-35/epoch_12.pth
021a3172bf331254c2822d33d57d908e3006cf0752c41b6fee3bbaef060bf27e	765745473	2026-05-13 19:42:15.663790438 +0000	outputs/racformer_train2k_day_rcsvelbev_research/2026-05-13/12-47-35/epoch_9.pth
92aa1b5396d8fa8e1f97047eee444635699ad628a42a047f205e434617b0006e	765773121	2026-05-14 03:23:13.900030426 +0000	outputs/racformer_train2k_day_rcsveltimebev_research/2026-05-13/20-02-10/epoch_10.pth
51545808e2a27886cba17b022c4677cc76adb9f3a7cc1e342acdeebdb380d931	765773121	2026-05-14 03:49:20.249968630 +0000	outputs/racformer_train2k_day_rcsveltimebev_research/2026-05-13/20-02-10/epoch_11.pth
0e61af381ac749e952359ec70d5a85cf3cf8c100d29ef6d9361c01088839cec7	765773121	2026-05-14 04:15:26.379658027 +0000	outputs/racformer_train2k_day_rcsveltimebev_research/2026-05-13/20-02-10/epoch_12.pth
67951889c1e58ea1bcb910f221da3f2298856870e34d97695f9516b15ab845ac	765773121	2026-05-14 02:57:05.432268168 +0000	outputs/racformer_train2k_day_rcsveltimebev_research/2026-05-13/20-02-10/epoch_9.pth
1c10dd5ef57423c13c0f5f06e1adef09376536053bb00baf87fff70ab3cfe8eb	764836881	2026-04-26 03:48:15.643452121 +0000	outputs/racformer_train2k_day_research/2026-04-25/20-24-58/epoch_10.pth
53917586a6dcb378d4feb8ca05a975608146add83a41ee062f3e64cdf2c911e4	764836881	2026-04-26 04:14:19.576908216 +0000	outputs/racformer_train2k_day_research/2026-04-25/20-24-58/epoch_11.pth
6a12183fed06832fc5478de38cc8510c71db143d36871182fa70a77369e5a055	764836881	2026-04-26 04:40:22.674061576 +0000	outputs/racformer_train2k_day_research/2026-04-25/20-24-58/epoch_12.pth
a02d0f3798f967daa676509e5f6db3d98fc25ef17dd59ac39c39707d55c605d1	764836881	2026-04-26 03:22:11.373224181 +0000	outputs/racformer_train2k_day_research/2026-04-25/20-24-58/epoch_9.pth
e855d39d34dd8b91234e31488398f974440113c14698ddd76eb6befd801f9ede	764836945	2026-05-02 07:55:45.828628058 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio12p5_research/2026-05-02/00-35-52/epoch_10.pth
e3ce30eaa8e0248504232feba06bc3bd1e31b24c59129053a8c80df84c0d6612	764836945	2026-05-02 08:21:45.709521534 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio12p5_research/2026-05-02/00-35-52/epoch_11.pth
6f2b9dd38d9e45792aa3e8b86b581279405cf0fa3db59e6adb1a91f2facc16d1	764836945	2026-05-02 08:47:42.632110597 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio12p5_research/2026-05-02/00-35-52/epoch_12.pth
788bb0bb72f8df329238aabee6abad38e602dd5f30bca7af33f9b33ea02a0056	764836945	2026-05-02 07:29:45.362681700 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio12p5_research/2026-05-02/00-35-52/epoch_9.pth
82b840aa2da702a79a0d01ee982ae0843a62c43ee7b491b5b3feecd916fc4a7f	764836945	2026-05-04 03:17:13.462386103 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_research/2026-05-03/19-56-54/epoch_10.pth
5012061e83fc8493181f848a030217aa74c7bb17f87f69bb68cdc78853183134	764836945	2026-05-04 03:43:15.563023911 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_research/2026-05-03/19-56-54/epoch_11.pth
1e6b89f9ec85689d1591a4a9f93e29b3220fc6c547dad9f4aa932160b03f5b7f	764836945	2026-05-04 04:09:18.206638065 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_research/2026-05-03/19-56-54/epoch_12.pth
5af7ee7e16c08881c6b057badd56d82b4bdaa2c870d9ff8cbe365592902b91ec	764836945	2026-05-04 02:51:10.155498735 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_research/2026-05-03/19-56-54/epoch_9.pth
bd1ddde340744f07c36ee6bedfbab6c2ed53d29e37aac982b8bdc169540af88f	764836945	2026-05-12 07:45:45.398574176 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w025_research/2026-05-12/00-25-43/epoch_10.pth
647c2aedab6d1eb15ed9e0823a40599197a1ce1ff529844112708fdfd7c55da2	764836945	2026-05-12 08:11:45.008095799 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w025_research/2026-05-12/00-25-43/epoch_11.pth
71dd476ca072f7c9d25a42911fc14d052c21f592e7947362376c6b4e6065e8f0	764836945	2026-05-12 08:37:48.215111544 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w025_research/2026-05-12/00-25-43/epoch_12.pth
b9c32dce0e2c5e57f076bea1cab6d11185b2e97f9f0fadc5fb7dc17d33af19a0	764836945	2026-05-12 07:19:41.879334989 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w025_research/2026-05-12/00-25-43/epoch_9.pth
c8276e2da2a2a7530f33a1b82c5046aa7659acdceab89996776ccc66522d3e38	767214145	2026-05-12 13:44:10.689655547 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research/2026-05-12/06-23-27/epoch_10.pth
3c5229e4e2ddf7c9c02cf0197d1f48f5c6081f739bb68861b20ac7e3218a04a0	767214145	2026-05-12 14:10:15.564329096 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research/2026-05-12/06-23-27/epoch_11.pth
08d9b9cdfe61420c461b528f32d550b48af5fc50e3e1dd7c46ba05abfbd8fe36	767214145	2026-05-12 14:36:22.168148633 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research/2026-05-12/06-23-27/epoch_12.pth
dcabb7c2ccae751dc561dca866be09e1a9cae29b8545a19f1a4cb1974f744628	767214145	2026-05-12 13:18:04.344956835 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research/2026-05-12/06-23-27/epoch_9.pth
546afd6acf35e5ed679bca08936c54d567bfe64ff59d7c63d50ab3ae3522f2b8	764836945	2026-05-12 06:43:05.779289116 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_research/2026-05-11/23-22-47/epoch_10.pth
1dfe19baecfd022ec40df86cad72e5195bd9d71d4cff8d55ac9d17629e3f2de8	764836945	2026-05-12 07:09:09.099030878 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_research/2026-05-11/23-22-47/epoch_11.pth
28241b5e724c505b92d5bd2ae06fd7b111d2eb390e1f490dee46339a9dc44fa1	764836945	2026-05-12 07:35:13.115450483 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_research/2026-05-11/23-22-47/epoch_12.pth
25e5825c3bd5972a0b2ab451cb5d5f7f0e52fe578f7123d830240c2cdb21b58a	764836945	2026-05-12 06:17:00.432944704 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_research/2026-05-11/23-22-47/epoch_9.pth
144b92b65cc8cd0e8e242b344392ffdc96e4150564aa96459d09e523f821d9de	764836945	2026-05-04 03:17:27.547349134 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio21p25_research/2026-05-03/19-57-20/epoch_10.pth
995f60b8ee1b5761d173d087343e0941d5888f5d222f54cbd4a1011423cc514c	764836945	2026-05-04 03:43:29.296982285 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio21p25_research/2026-05-03/19-57-20/epoch_11.pth
af11d2594ae9800ebc78322194e45a88ebf69a726c22a9ae615a85c933c988bf	764836945	2026-05-04 04:09:30.073605805 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio21p25_research/2026-05-03/19-57-20/epoch_12.pth
4c7079b1b6f40daa366210e24607ab903c72d1ef8f2c9ea0a115f71db8a7bf60	764836945	2026-05-04 02:51:26.011457993 +0000	outputs/racformer_train2k_genaug_seed20260425_ratio21p25_research/2026-05-03/19-57-20/epoch_9.pth
7adc9dcf9bbb338460d2642b817f35baa5a44d4c5763163f6717610dce0b50ae	764836945	2026-05-01 20:17:09.596019362 +0000	outputs/racformer_train2k_genaug_seed20260425_research/2026-05-01/12-57-20/epoch_10.pth
16ef7fbcc3fc630c8993c975fc18971ada3f5261659c9a86963ae8c8952ca164	764836945	2026-05-01 20:43:09.034636008 +0000	outputs/racformer_train2k_genaug_seed20260425_research/2026-05-01/12-57-20/epoch_11.pth
6223c0eed3140bf79d5ee6e399b0686dfb73455a72dfdd2081d5f0b5a97bec16	764836945	2026-05-01 21:09:07.718638536 +0000	outputs/racformer_train2k_genaug_seed20260425_research/2026-05-01/12-57-20/epoch_12.pth
14091ac9f2858d50e43aa57beb6ee1a39c140292cc2bc6da692d989e2e264d8b	764836945	2026-05-01 19:51:08.547174027 +0000	outputs/racformer_train2k_genaug_seed20260425_research/2026-05-01/12-57-20/epoch_9.pth
3a6e8dfac2bf4dc5caf6d5a6d2cafd518f7c35748076b0b40e9a74bde5fce4f1	764836945	2026-05-07 21:56:20.757827491 +0000	outputs/racformer_train2k_genaug_seed20260502_ratio18p75_research/2026-05-07/14-36-18/epoch_10.pth
6ef72d51f08d7c1a27a13c933f350524cada461d7db94bd615ab46f202c977a9	764836945	2026-05-07 22:22:20.693326314 +0000	outputs/racformer_train2k_genaug_seed20260502_ratio18p75_research/2026-05-07/14-36-18/epoch_11.pth
d49687b8a3298fd487ad0e4466cad95cc934b9bb4af5192edc26d9ed86c38c57	764836945	2026-05-07 22:48:21.479270179 +0000	outputs/racformer_train2k_genaug_seed20260502_ratio18p75_research/2026-05-07/14-36-18/epoch_12.pth
1f532f20933649f146110d6d788d55673ccc133f2491b4a4fb17ccf96a8ee458	764836945	2026-05-07 21:30:19.654721807 +0000	outputs/racformer_train2k_genaug_seed20260502_ratio18p75_research/2026-05-07/14-36-18/epoch_9.pth
52272522b881eee551a007bc9bf22255e5f6008732fddad3d165e302df3195f4	767214145	2026-05-12 19:41:42.977122499 +0000	outputs/racformer_train2k_genaug_seed20260502_ratio18p75_w05_adaptfusion_research/2026-05-12/12-21-12/epoch_10.pth
e056bd9b447c1af000dc3cdfa52d3c745aad55c9d86c15574699061a6cc26c96	767214145	2026-05-12 20:07:45.853664397 +0000	outputs/racformer_train2k_genaug_seed20260502_ratio18p75_w05_adaptfusion_research/2026-05-12/12-21-12/epoch_11.pth
07481f5923a92ae504e2495be0b610a794a2b1a8d131de1a67931162804cb2cf	767214145	2026-05-12 20:33:51.776153314 +0000	outputs/racformer_train2k_genaug_seed20260502_ratio18p75_w05_adaptfusion_research/2026-05-12/12-21-12/epoch_12.pth
b4a847205215dea2578abead77c315142f9f6d9dfccddac4cb95c0bc4a4e93e8	767214145	2026-05-12 19:15:39.352475328 +0000	outputs/racformer_train2k_genaug_seed20260502_ratio18p75_w05_adaptfusion_research/2026-05-12/12-21-12/epoch_9.pth
84135445ed7ce2c826f5ccc0ceb521029308b2d0e8873d1d0c26dc9f35ad1e41	768011197	2026-05-13 01:47:05.220346737 +0000	outputs/racformer_train2k_mixed_conditionfusion_research/2026-05-12/18-29-20/epoch_10.pth
3740d4ff786e7dd0c1511c2498ed1e58a06b6b0cf5f2af4ddaf22948ad153b61	768011197	2026-05-13 02:12:54.543529257 +0000	outputs/racformer_train2k_mixed_conditionfusion_research/2026-05-12/18-29-20/epoch_11.pth
edab59e7271d133aac4492801da38b5a37d86611e512a959dbb9b55e8b34d896	768011197	2026-05-13 02:38:45.347170413 +0000	outputs/racformer_train2k_mixed_conditionfusion_research/2026-05-12/18-29-20/epoch_12.pth
80235d364cba5043c9a24873451addadab7e7c6eeb00db20611465292432c1d6	768011197	2026-05-13 01:21:17.773078737 +0000	outputs/racformer_train2k_mixed_conditionfusion_research/2026-05-12/18-29-20/epoch_9.pth
0ca4b008f79781f74bca2806c3dcf11f20458904f315510b2ad37bd04fa2fe5a	764865729	2026-05-15 17:47:25.274493588 +0000	outputs/racformer_train2k_mixed_contrelqfusion_research/2026-05-15/10-30-04/epoch_10.pth
cb567077567dba904538ed4a43b03d24bb6fffcf59002b51510ddbb9e551d9d8	764865729	2026-05-15 18:13:10.262971455 +0000	outputs/racformer_train2k_mixed_contrelqfusion_research/2026-05-15/10-30-04/epoch_11.pth
c79744ba118858cffb01c1e434e5b9eaec517d5124de50fd5822e5b7eb8256ff	764865729	2026-05-15 18:38:56.938328336 +0000	outputs/racformer_train2k_mixed_contrelqfusion_research/2026-05-15/10-30-04/epoch_12.pth
366d61b1b562f88353146fa4910467d33166d02fe84bd711ddcb15014a4f9ea9	764865729	2026-05-15 17:21:40.102173833 +0000	outputs/racformer_train2k_mixed_contrelqfusion_research/2026-05-15/10-30-04/epoch_9.pth
0d9053af9812f20a4b33e09bc43be98354d16cd4141b9043c90eccbb4bad8f46	1376965264	2026-05-18 15:09:28.152323541 +0000	outputs/racformer_train2k_mixed_dualviewdistill_epoch6_research/2026-05-17/17-02-26/epoch_10.pth
8181bcf6bb139a1491a567a55c43f3db2984f405a8c6c3954218e7f0cfd33c94	1376965264	2026-05-18 15:37:55.208285440 +0000	outputs/racformer_train2k_mixed_dualviewdistill_epoch6_research/2026-05-17/17-02-26/epoch_11.pth
e4263ff8b1ee5ef9efe77ccb8c7588260e046721054641ed57f2066e81b1a607	1376965264	2026-05-18 16:06:21.158764671 +0000	outputs/racformer_train2k_mixed_dualviewdistill_epoch6_research/2026-05-17/17-02-26/epoch_12.pth
475bb199cead8df37d2403c275221ae3f9a44db2d7c819fd494767169351d267	768080271	2026-05-18 16:06:41.486716152 +0000	outputs/racformer_train2k_mixed_dualviewdistill_epoch6_research/2026-05-17/17-02-26/epoch_12_inference.pth
82d17b25f7f751ff9c3633f92eabaf578efe6b3dee6d02e12e96ec947eb6cc1f	1376960656	2026-05-17 21:29:04.944180040 +0000	outputs/racformer_train2k_mixed_dualviewdistill_epoch6_research/2026-05-17/17-02-26/epoch_3.pth
8e0f2d1d4f40b778c6684042d15e81ef749131a9196d7061a6358d24baf0da6c	1376965264	2026-05-17 22:27:44.146108249 +0000	outputs/racformer_train2k_mixed_dualviewdistill_epoch6_research/2026-05-17/17-02-26/epoch_4.pth
2cc41b9986274c73cd1fa7c11714c1ee4fee7d99a4c337bd0a7fe9a586946e77	1376965264	2026-05-17 22:56:15.264723652 +0000	outputs/racformer_train2k_mixed_dualviewdistill_epoch6_research/2026-05-17/17-02-26/epoch_5.pth
d8446698c43fb3aad85935e7db0e55dc1c1469f25b4cd651706686ef5c4b77bd	1376965264	2026-05-17 23:24:49.347017389 +0000	outputs/racformer_train2k_mixed_dualviewdistill_epoch6_research/2026-05-17/17-02-26/epoch_6.pth
94c05e2efeadb64792a63a73e5acf05b89f2f1c592a55ff04a52dcfc2a06aa1f	768078717	2026-05-18 11:36:33.407827410 +0000	outputs/racformer_train2k_mixed_dualviewdistill_epoch6_research/2026-05-17/17-02-26/epoch_6_inference.pth
b827eb12c46a3a236cb9255463261aff2ec386162aaa902d59fbf04f81c0e821	1376965264	2026-05-18 13:44:05.637561804 +0000	outputs/racformer_train2k_mixed_dualviewdistill_epoch6_research/2026-05-17/17-02-26/epoch_7.pth
17055c3d8681d52bf1ed8b539fd1f7fd34809762969453eaa816b2128fcd9a94	1376965264	2026-05-18 14:12:30.099332366 +0000	outputs/racformer_train2k_mixed_dualviewdistill_epoch6_research/2026-05-17/17-02-26/epoch_8.pth
74b7b390fa121ca4d529b08ddd632d6e395c17a33f47810ff54cac247fd060db	1376965264	2026-05-18 14:40:57.564072281 +0000	outputs/racformer_train2k_mixed_dualviewdistill_epoch6_research/2026-05-17/17-02-26/epoch_9.pth
c6636d9098569724a7fe05329b919dec33536e3204454d4460012fc62f2328b1	764836881	2026-04-26 09:00:32.318326597 +0000	outputs/racformer_train2k_mixed_research/2026-04-26/01-42-45/epoch_10.pth
388c9e769cc9f60cbefa1615fda53084c79572a229d52205f8c1638e103b89ca	764836945	2026-04-26 09:26:11.929113614 +0000	outputs/racformer_train2k_mixed_research/2026-04-26/01-42-45/epoch_11.pth
2648d699a47b9b0b8f35e8c77bd0481c5d69ba281b51509894d89425661ed94c	764836945	2026-04-26 09:51:51.817104733 +0000	outputs/racformer_train2k_mixed_research/2026-04-26/01-42-45/epoch_12.pth
cb8b923a9cc4d35ecd09cfefd72eb38d799f59fc68f2f02f130b34c112218727	764836881	2026-04-26 08:34:52.974195191 +0000	outputs/racformer_train2k_mixed_research/2026-04-26/01-42-45/epoch_9.pth
bf0b36aeda2fde39fc40ab0d0d475e73c7cd10826c746a475f5b9fc312f8c513	764836945	2026-04-27 09:09:56.254124859 +0000	outputs/racformer_train2k_simnight_research/2026-04-27/01-49-03/epoch_10.pth
70a81df7f2c90486d26ad0fcdf90374b4d7b83b58309b528ce3ebaa73a5d76bf	764836945	2026-04-27 09:36:05.034844118 +0000	outputs/racformer_train2k_simnight_research/2026-04-27/01-49-03/epoch_11.pth
e11f564c78aa0bca16e99bf95b3010b4922e3a1b336aadcdb7fd5ce967de1bab	764836945	2026-04-27 10:02:13.924109337 +0000	outputs/racformer_train2k_simnight_research/2026-04-27/01-49-03/epoch_12.pth
34d105ee77c108631613faa39b63fc87c0c7972a0c6c197ea1fa3245b726600e	764836945	2026-04-27 08:43:51.856483565 +0000	outputs/racformer_train2k_simnight_research/2026-04-27/01-49-03/epoch_9.pth
```
