# GATE-B manifest — camera-removal cells G1–G4

**Scope.** Camera removal only, all six views. One-camera and three-camera variants are out of
scope here. Written at the implement stage; no cell has been run. The orchestrator reviews this
file before any cell is authorised.

**Sources read.** All line numbers are at the execution commit unless noted.

| Source | Where | What it settled |
|---|---|---|
| Frozen model | `models/racformer.py` | the two methods the eval path funnels through, and where normalization sits relative to them |
| Frozen transformer | `models/racformer_transformer.py:19-35` | the dropout configs are not instantiable (below) |
| Phase-0 harness | `research/paper_goal_20260520/tools/phase0_sensor_baseline.py:71-95` | G2's exact mechanism |
| Dropout config | `configs/racformer_r50_nuimg_704x256_f8_dropout_cam10.py:10-21` | what the tracked "dropout config" lineage actually targets |
| Eval config | `configs/racformer_eval_fullval_research.py`, base `configs/racformer_r50_nuimg_704x256_f8.py` | `num_frames = 8` (base `:30`), `final_dim = (256, 704)` (base `:47`), ResNet-50 (base `:69`) |
| Frozen driver | `research/night_gen_phase1/eval_by_condition.py:219-343` | it has a clean `main()` the runner can call in-process |
| Paper | arXiv 2412.12725v2, §4.4 "Robustness" + Table 9 | the published removal semantics, and the ablation-setup default |

---

## 0. Two findings that precede the cell design

**(a) The tracked dropout configs define no removal mechanism and cannot be built.**
`configs/racformer_r50_nuimg_704x256_f8_dropout_cam10.py:14-21` sets
`model.pts_bbox_head.transformer.modality_dropout_prob = 0.1` and `..._mode = 'camera'`.
`RaCFormerTransformer.__init__` (`models/racformer_transformer.py:19-35`) accepts neither key and
has no `**kwargs`; nothing under `models/` or `loaders/` references `modality_dropout` at all.
Building this config raises on the unexpected keyword. Verified statically at the execution commit
by comparing the resolved config keys against `inspect.signature`:

```
config transformer keys: [..., modality_dropout_mode, modality_dropout_prob, ...]
ctor params:            [bev_depth_num, code_size, d_region_list, embed_dims, img_depth_num,
                         init_cfg, num_cams, num_classes, num_frames, num_layers, num_levels,
                         num_points, num_points_bev, num_ray, pc_range, spatial_shapes]
KEYS NOT ACCEPTED BY CTOR: ['modality_dropout_prob', 'modality_dropout_mode']
has_var_kw: False
```

Two consequences. It is a *training-time* dead config, so it contributes no eval-time removal
semantics to copy. And the level it aims at is the **transformer's inputs** — i.e. features, not
pixels — so the "dropout config" lineage is evidence for a *network-units* reading of removal, not
an input-level one. That is what pins G3 below.

**(b) The paper's removal semantics are one sentence.** §4.4, "Robustness": *"To evaluate the
robustness gains from radar-camera fusion in our system, we test its performance under sensor
failure scenarios, as detailed in Tab. 9. We systematically exclude either image or radar inputs,
recording the AP for detecting cars."* Table 9's caption adds only: *"Analysis of robustness using
Car class AP. 'All' denotes that the single modality is entirely off."* There is no further text.

The paper therefore does **not** state: the tensor or pipeline stage at which images are excluded;
the substituted value; whether history frames are excluded along with the current frame; whether
camera-derived metadata is altered; or whether the model was retrained or finetuned for the test.
This is under-specification, not impossibility — see §3.

---

## 1. Pipeline geometry (shared by all four cells)

The eval path is `forward_test` (`models/racformer.py:446`) → `simple_test` (`:465`) →
`simple_test_offline` (`:468`) → `extract_feat` (`:470`, defined `:179`). `simple_test_online`
(`:479`) is never reached.

Inside `extract_feat`, on the eval branch (`self.training` is False, so `:299-342` runs):

| Line | What happens to the image tensor |
|---|---|
| `:179` | entry; `img` is `(B, NT, C, H, W)`, **raw sensor units** — the pipeline loads with `to_float32=False`, so these are uint8-valued BGR pixels in a float container |
| `:185-189` | `B, NT, C, H, W = img.size()`; `N = self.num_cams`; `T = NT // N`; flatten to `(B*NT, C, H, W)`; `.float()` |
| `:208-209` | BGR → RGB |
| `:211-212` | `img = (img - mean) / std` — **the normalization** |
| `:222` | `pad_multiple` to size-divisor 32 |
| `:300` | `extract_img_feat(img)` — the **only** consumer of `img` on the eval branch; backbone → FPN → LSS feats |
| `:334` | `img_lss_view_transformer(img_lss_feats, radar_depth[:, :, i], radar_rcs[:, :, i], ...)` per frame `i` — the camera-BEV stack, which also consumes the radar-derived PV depth/RCS maps |
| `:348` | returns `(img_feats_reshaped, all_bev_feats, radar_bev_feats, all_depths[0])`; `img_feats_reshaped[lvl]` is `(B, NT, C, H, W)` (`:344-346`), `all_bev_feats` is `(B, T, C, H, W)` (`:341`) |

`img` has exactly one consumer, at `:300`. That is what makes G3 and G4 clean: intervening before
`:211` and intervening at `:300` differ by the normalization affine and by nothing else.

`num_cams = 6` — the ctor default at `models/racformer.py:45`; neither the base config nor the eval
config overrides it. `num_frames = 8` — `configs/racformer_r50_nuimg_704x256_f8.py:30`, echoed into
`model.pts_bbox_head.transformer.num_frames` and into the test pipeline's
`LoadMultiViewImageFromMultiSweeps(sweeps_num=num_frames-1)` (`:225`), i.e. 1 current frame +
7 history frames. So `NT = 6 × 8 = 48` view-frames per sample and, over the 6,019-sample full val,
**288,912 view-frames** per cell.

---

## 2. The four cells

All four run through one committed script, `research/robust_study/tools/gate_b_removal.py`, with
`--removal {none,phase0,input,table9}`. It installs wrappers on two `RaCFormer` methods and then
calls the frozen driver's `main()` in-process. Both wrappers are installed in every mode, including
`none`, so the counters and probe digests are produced by identical code in every cell. No frozen
file is modified and no config is added: all four cells load
`configs/racformer_eval_fullval_research.py` and `checkpoints/racformer_r50_f8.pth` unchanged.

| | G1 `none` | G2 `phase0` | G3 `input` | G4 `table9` |
|---|---|---|---|---|
| **Mechanism** | wrappers installed, no tensor touched | zero the image-branch **outputs** of `extract_feat` | zero the image tensor at `extract_img_feat` entry | zero the image tensor at `extract_feat` entry |
| **Intervention point (frozen code)** | `models/racformer.py:179` + `:107` (wrapped, pass-through) | after `models/racformer.py:348` returns | `models/racformer.py:107` (before `:111` backbone) | `models/racformer.py:179` (before `:211-212`) |
| **Runner line** | `gate_b_removal.py` `patched_extract_feat` / `patched_extract_img_feat`, both no-op branches | `patched_extract_feat`, `removal == "phase0"` branch | `patched_extract_img_feat`, `removal == "input"` branch | `patched_extract_feat`, `removal == "table9"` branch |
| **Tensor** | none | `img_feats` (all 4 FPN levels, each `(B, NT, C, H, W)`) and `bev_feats` `(B, T, C, H, W)` | `img` `(B*NT, C, H, W)` | `img` `(B, NT, C, H, W)` |
| **Stage vs normalization** | n/a | **post**-normalization, post-backbone, post-view-transform | **post**-normalization, pre-backbone | **pre**-normalization (raw pixels) |
| **Value seen by the backbone** | unchanged | unchanged (masking is downstream of it) | exactly 0 in network units | 0 pixels → `-mean/std` after `:211-212`, i.e. a black frame |
| **Camera scope** | — | all 6 views | all 6 views | all 6 views |
| **Frame scope** | — | all 8 (current + 7 history) | all 8 | all 8 |
| **Lineage** | control | `phase0_sensor_baseline.py:87-89`, verbatim | dropout-config level (finding (a)) | published sentence, sensor-literal reading |
| **Config diff vs the clean cell** | none | none | none | none |

**Metadata handling — identical in all four cells, nothing is altered.** `lidar2img`,
`intrinsics`, `img_shape`, `ori_shape`, `pad_shape`, `img_timestamp` and `filename` are left
untouched. This is deliberate and it is a modelling choice worth stating: the camera *extrinsics
and intrinsics* are still needed after the camera is off, because the radar-derived PV depth and
RCS maps are projected through them (`models/racformer.py:334`) and the LSS view transformer
consumes them. Removing the metadata would remove part of the radar path, not the camera path.

**What survives camera removal in G3 and G4, and why G2 may disagree.** With black or zeroed
images, the camera-BEV branch still *executes*: `img_lss_view_transformer` at `:334` receives
zeroed image features together with the untouched radar depth/RCS maps and produces a non-trivial
BEV feature. G2 zeroes that stack outright. So G2 removes strictly more than G3/G4 do — it removes
the radar-informed part of the camera-BEV branch as well. Whether that difference is worth
mAP points is exactly the question GATE-B was built to answer; it is not a defect in either cell.

**Why G3 and G4 are split on the normalization axis.** The paper leaves the stage open (finding
(b)), and the two readings of "the camera has no signal" that the code admits are: zero in the
network's units (post-normalization — what an in-network modality dropout does, and what the
tracked configs aim at, finding (a)), and zero in sensor units (pre-normalization — a black frame,
the literal reading of "exclude the image *input*" and of "the modality is entirely off" as a
statement about a *sensor*). G3 takes the first, G4 the second. They are numerically distinct: a
post-normalization zero is the image whose pixels all equal the per-channel training mean, a
mid-grey frame; a pre-normalization zero is black.

*Rejected alternative, recorded so the choice is auditable:* making G4 a second feature-level mask
on the grounds that "entirely off" means the camera branch contributes nothing. Rejected because it
would make G4 a duplicate of G2 under a different name, which destroys the `agree(G3, G4)`
predicate — that predicate is only informative if G3 and G4 are two independent implementations of
the *same* intended removal. **This is the one interpretive decision in this manifest and the
orchestrator can overrule it; nothing else here is a judgment call.**

---

## 3. Table-9 compatibility matrix

Ours = `checkpoints/racformer_r50_f8.pth` under `configs/racformer_eval_fullval_research.py`.
Theirs = the setup behind arXiv 2412.12725v2 Table 9.

| Axis | Ours | Table 9 | Match |
|---|---|---|---|
| Backbone | ResNet-50 (base config `:69`) | ResNet-50 (§4.4: *"Unless specified, we perform ablation studies using single-frame inputs with an image resolution of 256×704 and a ResNet-50 backbone."*) | **yes** |
| Resolution | 256×704 (`ida_aug_conf['final_dim']`, base config `:47`) | 256×704 (same sentence) | **yes** |
| Temporal frames | **8** (`num_frames = 8`, base config `:30`; test pipeline `:225`) | **single-frame** — Table 9 is inside §4.4 and does not override the section default | **NO** |
| Radar sweeps | `Loadnuradarpoints(num_sweeps=5)` + `LoadradarpointsFromMultiSweeps(sweeps_num=7, num_aggr_sweeps=5)` (base config `:227-228`) | not stated for the ablation setup | **unstated** |
| Epochs | **unrecorded** — the checkpoint carries only `state_dict`, no `meta` block, so no epoch/config is recoverable from the artifact (md5 `d8b3a3141f704df7ac164676a14369b9`); the repo's base config says `total_epochs = 36` (`:307`) but did not demonstrably produce this file | 24 (§4.2: *"Unless specifically indicated, training is conducted for a standard 24 epochs for all models."*) | **unknown** |
| Removal semantics | zeroing, stage pinned per cell (§2) | one sentence, stage unspecified (§0(b)) | **under-specified on their side** |
| Reported quantity | 10-class mAP, NDS, per-class AP, 5 TP errors, per-class TP/FP/FN at 2.0 m at two operating points | Car-class AP only | **ours is a superset** |
| Sweep/ladder count | 1 point (all 6 views off) | 4 points (0 / 1 / 3 / all views) | **ours is a subset** — 1-cam and 3-cam are out of GATE-B scope |

**Anchor check on the clean run.** Our clean full-val reference
(`/srv/nfs/shared/gnmp/robust_study_runs/eval_oracle/clean_20260803T021120Z/submission_overall/pts_bbox/metrics_summary.json`)
gives mAP 54.1727 pp, NDS 61.4434 pp, **Car AP 77.9978 pp**. The paper's *main* validation result
for R50 at 256×704 is 54.1 mAP / 61.3 NDS (§4.3), and §4.3 states *"We default to an 8-frame
sequence with 0.5-second intervals"* — so our checkpoint reproduces the **main** 8-frame table.
Table 9's zero-drop Car AP is **71.5**. Our clean Car AP is 6.50 pp above it. That gap is consistent
with Table 9 being the single-frame ablation setup while our checkpoint is the 8-frame one, and it
is independent evidence for the frame-count mismatch above.

**Verdict: `setup_comparable = false`** — on the temporal-frame axis, corroborated by the 6.50 pp
zero-drop Car-AP gap, with the epoch axis unknown. Per the execution spec this annotates G4 as
***not comparable to published Table 9***; the published ≈27.2 Car-AP figure must not be compared
against our G4 number, and this does not by itself fire Branch C. No weakened variant of G4 was
constructed.

**Verdict: `semantics_implemented = true`, with a caveat.** The published sentence is implementable
at face value — "exclude the image inputs" is exactly what G4 does — so this is *not* the
`UNIMPLEMENTABLE` state. The caveat is that the sentence does not uniquely determine the stage, so
G4 implements one of several readings consistent with the text (§2), and G3 implements another. The
evidence for both the verdict and the caveat is the full removal-semantics text quoted in §0(b);
there is no further text in the paper to draw on.

---

## 4. Runtime intervention attestation

Written by the runner to `<run-dir>/intervention_attestation.json`; a FAIL exits non-zero, so the
job records `validity=INVALID` in `provenance.json` and never writes `_COMPLETE`. The cell body
re-reads the verdict from the artifact rather than trusting the exit code alone.

1. **Branch-hit count** — `extract_feat` wrapper invocations, required `== 6019`, and
   cross-checked against the driver's own `n_total` from `eval_by_condition.json`.
2. **Coverage** — `covered_view_frames` required `== n_samples × num_cams × num_frames = 288,912`,
   computed from the config, not hardcoded. `num_frames` comes from the config; `num_cams` comes
   from the ctor default (the config does not set it) and the runtime `self.num_cams` is re-checked
   on every call, as is `NT // num_cams == num_frames`. Partial coverage is INVALID.
   `altered_view_frames` must equal the same figure for G2–G4 and must be exactly **0** for G1.
3. **Paired probe digests** — for a fixed 3-sample probe set (the first three dataloader samples;
   `shuffle=False`, `batch_size=1`, so these are dataset indices 0–2, and each probe records the
   first camera filename so the sample is identifiable), sha256 of the pre- and post-intervention
   tensor at that cell's own intervention stage. Required bit-identical for G1, required different
   for G2–G4, and the post tensor must have `abs_max == 0`.

---

## 5. Per-cell emission

`research/robust_study/jobs/g_crosscheck.sbatch` — one parameterised job for all eight runs — runs
the already-committed `research/robust_study/tools/devkit_crosscheck.py` under the isolated devkit
venv, against `research/robust_study/frozen/maxf1_thresholds_clean.json`. **`devkit_crosscheck.py`
was not modified**: it already emits 10-class mAP, NDS, per-class AP, the five TP errors, and
per-class TP/FP/FN/GT at centre-distance 2.0 m at both pre-registered operating points — (i) no
score cut and (ii) the frozen per-class max-F1 threshold — plus `thresholds_source` and
`near_zero_recall_at_op_i`. Same committed code and same thresholds for every cell, so
comparability holds by construction.

Each run directory additionally carries `thresholds_source.json` (repo-relative path, absolute
path, sha256, and the execution commit the file is pinned at). It is a sidecar because
`write_provenance.py` is frozen tooling with no field for it and no deliverable required changing
it.

`g_crosscheck.sbatch` passes `--allow-unverifiable-rows` for the same single row as
`e2_crosscheck.sbatch`, and refuses to score a cell whose attestation is not PASS.

---

## 6. Files

| Path | Role |
|---|---|
| `research/robust_study/tools/gate_b_removal.py` | the common runner; the only new code that touches the model |
| `research/robust_study/jobs/_gate_b_common.sh` | GATE-B helpers on top of the frozen `_job_common.sh` |
| `research/robust_study/jobs/_gate_b_cell_body.sh` | the shared cell body; the eight templates differ only in `$CELL`, `$REMOVAL` and the job name |
| `research/robust_study/jobs/g{1_none,2_phase0,3_input,4_table9}[_repeat].sbatch` | the eight cells |
| `research/robust_study/jobs/g_crosscheck.sbatch` | per-cell emission |

Eight templates, not four: the decision procedure resolves every predicate by the pair rule, so
each cell is measured twice. **The GPU-hour budget in the execution spec (4 cells ≈ 1.7 GPU-h) does
not cover this.** Measured wall time of the three completed clean full-val cells — run-directory
UTC stamp to `_FINALIZED` — is 40m56s, 30m34s and 30m33s (n=3, median 30m34s, range 30m33s–40m56s;
the outlier is the first, cold-cache run). Eight cells at the median is ≈4.1 GPU-h, and ≈5.5 GPU-h
if every cell behaves like the cold one. Flagged, not resolved: the reconciliation is the
orchestrator's.
