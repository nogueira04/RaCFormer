# Verdict: FAIL - Branch D / S6 Radar-Guided Query Initialization

Last updated: 2026-05-16 22:57 UTC

Branch D cannot support a publishable claim under `goal_v2.md`. Both active S6 variants completed the required seed-0 train2k and full-val evaluation path, and both failed the pre-registered primary gate. The secondary long-range/radar-rich path is also closed because the day mAP preservation predicate fails for both variants.

## Method And Claim

Method under test: sparse object-centric radar tokens via radar-guided decoder query initialization.

Rejected claim: current-frame radar points can seed a subset of decoder queries and improve night, long-range, or radar-rich detection without harming day/overall performance.

## Code, Config, Scripts, And Git State

- Remote repo: `/srv/nfs/shared/gnmp/RaCFormer`
- Git SHA: `869407e`
- Git status at entry check: dirty live research tree with Branch D model/config/research artifacts present.
- Git status after this endpoint: no code, checkpoint, eval JSON, output directory, or generated-result artifact was deleted or modified by this endpoint. Only report/tracker files were written.
- Model code: `models/racformer_head.py`, `models/racformer.py`
- Configs:
  - `configs/racformer_train2k_day_radarquery_research.py`
  - `configs/racformer_train2k_day_radarquery_topk90_research.py`
- Smoke scripts:
  - `research/night_gen_phase1/smoke_radarquery.py`
  - `research/night_gen_phase1/smoke_radarquery_topk90.py`
- SLURM wrappers:
  - `research/night_gen_phase1/staged_radarquery/`
  - `research/night_gen_phase1/staged_radarquery_topk90/`
- Subset helper: `research/paper_goal_20260515/eval_radarquery_subsets.py`
- Conda env: `racformerfix`
- Partition: `livecluster`

## Jobs And Evidence

`S6_radarquery` (`N=180`):

- Smoke: job `1396`, PASS, `research/night_gen_phase1/results/S6_radarquery/smoke_slurm_1396.out`
- Train: job `1397`, final checkpoint `outputs/racformer_train2k_day_radarquery_research/2026-05-15/16-48-03/epoch_12.pth`
- Eval: job `1398`, `research/night_gen_phase1/results/S6_radarquery/eval/eval_by_condition.json`
- Summary: job `1399`, `research/night_gen_phase1/results/S6_radarquery/summary_metrics.md`
- Subset diagnostics: job `1404`, `research/night_gen_phase1/results/S6_radarquery/subset_eval/subset_metrics.md`

`S6_radarquery_topk90` (`N=90`):

- Smoke: job `1400`, PASS, `research/night_gen_phase1/results/S6_radarquery_topk90/smoke_slurm_1400.out`
- Train: job `1401`, final checkpoint `outputs/racformer_train2k_day_radarquery_topk90_research/2026-05-15/17-32-19/epoch_12.pth`
- Eval: job `1402`, `research/night_gen_phase1/results/S6_radarquery_topk90/eval/eval_by_condition.json`
- Summary: job `1403`, `research/night_gen_phase1/results/S6_radarquery_topk90/summary_metrics.md`
- Subset diagnostics: job `1405`, `research/night_gen_phase1/results/S6_radarquery_topk90/subset_eval/subset_metrics.md`

## Primary Gate

Baseline S0 gate thresholds:

- Night mAP >= `0.1588`
- Day mAP >= `0.3053`
- Overall mAP >= `0.2890`
- Night NDS >= `0.2101`

| Variant | Split | mAP | NDS | mAP delta vs S0 | NDS delta vs S0 |
|---|---|---:|---:|---:|---:|
| `S6_radarquery` | day | 0.2554 | 0.3312 | -5.99 pp | -4.34 pp |
| `S6_radarquery` | night | 0.0958 | 0.1675 | -5.30 pp | -4.76 pp |
| `S6_radarquery` | rain | 0.2343 | 0.3416 | -4.00 pp | -2.98 pp |
| `S6_radarquery` | overall | 0.2477 | 0.3285 | -5.63 pp | -4.13 pp |
| `S6_radarquery_topk90` | day | 0.2977 | 0.3586 | -1.76 pp | -1.60 pp |
| `S6_radarquery_topk90` | night | 0.1230 | 0.1901 | -2.58 pp | -2.50 pp |
| `S6_radarquery_topk90` | rain | 0.2554 | 0.3550 | -1.89 pp | -1.63 pp |
| `S6_radarquery_topk90` | overall | 0.2863 | 0.3549 | -1.77 pp | -1.49 pp |

Primary gate verdict:

- `S6_radarquery`: FAIL. Misses night mAP, day mAP, overall mAP, and night NDS.
- `S6_radarquery_topk90`: FAIL. Misses night mAP, day mAP, overall mAP, and night NDS.

## Secondary Subset Path

The secondary path requires all of:

- long-range mAP delta >= +2.0 pp,
- radar-rich subset mAP delta >= +1.5 pp,
- day mAP delta >= -1.0 pp.

The day predicate fails for both variants, so the secondary path cannot pass regardless of subset deltas:

- `S6_radarquery`: day mAP delta is -5.99 pp.
- `S6_radarquery_topk90`: day mAP delta is -1.76 pp.

Subset diagnostics were still ingested as failure-mode evidence:

| Variant | all samples mAP/NDS | radar-rich top quartile mAP/NDS | object far >=30m mAP/NDS | object far >=40m mAP/NDS |
|---|---:|---:|---:|---:|
| `S6_radarquery` | 0.2477 / 0.3285 | 0.2745 / 0.3614 | 0.0615 / 0.2052 | 0.0275 / 0.1128 |
| `S6_radarquery_topk90` | 0.2863 / 0.3549 | 0.3119 / 0.3797 | 0.0788 / 0.2138 | 0.0365 / 0.1207 |

## Ablation And Replication Status

No `random_query_init` mechanism-isolation run was submitted, and no seed-`20260502` replication was submitted. This is intentional and required by `goal_v2.md`: both `N=180` and `N=90` fail the primary and secondary gates, so completion probes are forbidden. The only top-k ablation used for the D verdict is the already-active `N=90` variant.

Staged seed wrappers remain unsubmitted:

- `research/night_gen_phase1/staged_radarquery_replication/train_seeded.py`
- `research/night_gen_phase1/staged_radarquery_replication/run_s6_radarquery_seed20260502_livenode02.sbatch`
- `research/night_gen_phase1/staged_radarquery_replication/run_s6_radarquery_topk90_seed20260502_livenode03.sbatch`

## Failure Mode

The causal failure-mode memo is:

`research/night_gen_phase1/reports/D_failure_mode_20260516T225745Z.md`

Summary: direct radar-derived query replacement disrupts the baseline decoder query prior. The dose response from `N=90` to `N=180` suggests the mechanism becomes more harmful as more learned anchor slots are overwritten. The current radar points are too sparse, noisy, and class-agnostic to act as direct query-box initializers under this training budget.

## Claim Inventory

The claim inventory has a Branch D final row citing the S6 summary, eval, subset, failure-mode, branch-choice, and final-report artifacts:

`research/paper_goal_20260515/CLAIM_INVENTORY.md`

## Branch-Choice Memo

Required branch-choice memo:

`research/night_gen_phase1/reports/BRANCH_CHOICE_20260516T225745Z.md`

This memo does not activate a new branch. It proposes only a user-gated, zero-GPU Branch C diagnostic memo as the next possible direction because Branch C is the only standby branch in `goal_v2.md`; all other branches are blocked by missing artifacts, failed substrates, or missing implementations.

## Final Verdict

FAIL. Branch D / S6 radar-guided query initialization does not support a publishable claim. The next legal action is to halt for user review of the failure-mode and branch-choice memos; no new GPU submission is legal from this state.
