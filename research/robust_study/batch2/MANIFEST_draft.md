# Batch-2 cell MANIFEST — **DRAFT**

**Status: DRAFT.** Not a gate artifact. The orchestrator reviews and promotes this into the real
`MANIFEST.md`; nothing here is certified and no cell here has been signed off. Drafted
2026-08-03 during the Batch-2 prep phase, alongside the implementation of families (a), (b),
(c), (d2).

Discipline follows the G-prep rules the GATE-B manifest was written under (`exec-oracle.md`,
Phase G, G-prep item 1): every cell states its exact mechanism as file+line, the tensor and
pipeline stage it acts on, its sensor scope, its frame scope, how metadata is handled, its
config file, its seeds / common-random-number keying, and the runtime attestation the cell needs
before its numbers may be read.

Line anchors below were read from the frozen checkout at HEAD `356621c` on 2026-08-03. Where the
planning docs carry stale cites, the corrected anchor is used and the stale one is named.

## Global rules inherited by every cell

- Canonical checkpoint `checkpoints/racformer_r50_f8.pth`, P2 pipeline, full val (6,019 samples),
  after the Evaluator Oracle passes (`fault-families.md`, Cross-family rule 6).
- Base config for every cell: `configs/racformer_eval_fullval_research.py` (which inherits
  `configs/racformer_r50_nuimg_704x256_f8.py`). Cell fragments never restate the pipeline; they
  swap named entries or add a scope block, so a cell cannot drift from the clean path by
  transcription error (`research/robust_study/corruptions/cell_config.py:29-51`).
- Out-dirs live OUTSIDE the asserted checkout, under
  `/srv/nfs/shared/gnmp/robust_study_runs/<phase>/<cell>_<UTCstamp>/`; `provenance.json` per run;
  `_COMPLETE` only after eval + end-state assertions (`exec-oracle.md`, run-directory discipline).
- Stochastic families: 3 seeds, common random numbers across severities within the family
  (`fault-families.md`, Cross-family rules 1-2).
- Mini-screen (95 s/cell) before any full cell, for every new corruption implementation
  (Cross-family rule 3). No cell in this draft has had its mini-screen yet.

## Radar pipeline anchors (corrected)

The eval pipeline's two radar loader entries are
`configs/racformer_r50_nuimg_704x256_f8.py:227` (`Loadnuradarpoints`, frame-t radar, element 0 of
the stack) and `:228` (`LoadradarpointsFromMultiSweeps`, the remaining sweep elements). The
`:212`/`:213` entries cited in older drafts are the TRAIN pipeline and are not used by any cell
here. Both entries must be swapped together or the fault is partial — the trap named in
`fault-families.md` (d1) and (d2) spec item 1, enforced at
`research/robust_study/corruptions/cell_config.py:48-50`.

The frozen driver `research/night_gen_phase1/eval_by_condition.py` ignores `custom_imports`
(`:237`, `:243-244`), so every radar cell config self-registers its classes at parse time by
importing the corruption module from `cell_config` (`cell_config.py:23`). Job cwd must be the
repo root: `loaders/nuscenes_dataset.py:21` resolves `data/nuscenes/` relatively.

---

# (a) Camera removal — severity ladder

Ladder (`plan.md` §16.4): frontal-camera drop < worst-sector single-camera drop < all-6 drop.
The all-6 endpoint is the EXISTING GATE-B `G4` + `g4_repeat` pair. **It is not re-run.**

**A finding that constrains the whole family.** `extract_feat` receives images on a flat
view-frame axis `NT = num_frames * num_cams`, frame-major
(`models/racformer.py:313` slices `lidar2img[i*num_cams:(i+1)*num_cams]` per frame i). The camera
order *within* a frame is not constant across frames:

| frames | order | source |
|---|---|---|
| frame 0 | `CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT` | pkl `info['cams']` dict order, `loaders/nuscenes_dataset.py:207-213` → `:234` |
| frames 1..T-1 | `CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT` | hardcoded `cam_types`, `loaders/pipelines/loading.py:632-636` → `:672` |

A fixed-index slice (`img[:, cam_idx::num_cams]`) would therefore black CAM_FRONT in frame 0 and
CAM_FRONT_LEFT in frames 1..7 — a different fault from the registered one, silently. Scene-start
samples are the exception: with no previous sweeps the loader copies frame 0 T times
(`loading.py:640-646`), so the two orders coincide there, which is why the discrepancy is
invisible on val sample 0. Both regimes are represented in the probe set below.
**Consequence:** the (a) mechanism resolves each view-frame's channel from
`img_metas['filename']`, never from an index.

## Cell A1 — CAM_FRONT removed

| field | value |
|---|---|
| **Mechanism** | `research/robust_study/corruptions/cam_subset_removal.py:install_subset_intervention` (`:326-369`) wraps `RaCFormer.extract_feat`. The substituted values come from `research/robust_study/tools/gate_b_removal.py:_zero_image_arg` (`:246-252`) — the same function the frozen `--removal table9` branch calls at `gate_b_removal.py:295` — applied at `cam_subset_removal.py:137-155` (the `_zero_image_arg` call is `:144`). No copy of the G4 semantics exists. |
| **Tensor + stage** | RAW image tensor (uint8, sensor units) at the entry of `RaCFormer.extract_feat`, `models/racformer.py:179` — BEFORE the frozen normalization at `models/racformer.py:202-212`. Identical stage to G4; a black frame, not a zero activation. |
| **Camera scope** | `CAM_FRONT` only. 1 of 6 cameras. Channel resolved per view-frame from `img_metas['filename']` (`cam_subset_removal.py:97-124`); a filename that does not parse raises rather than guessing. |
| **Frame scope** | All `num_frames = 8` frames — current + 7 history frames — same as G4. Verified: 8/8 frames blacked per sample, both camera-order regimes. |
| **Metadata** | Untouched. The wrapper reads `img_metas['filename']` and writes nothing. `lidar2img`, `intrinsics`, `img_timestamp`, `img_shape` all unchanged; probe-verified by digest. Deliberate: the fault is "this camera produced a black frame", not "this camera is absent from the rig", so the geometry the model uses to project it must stay. |
| **Config** | `research/robust_study/configs/a_removal_front.py` (md5 `79beec201ed820f395917dd392ba9ba5`). Changes NO pipeline entry — probe-verified identical to the frozen clean pipeline. |
| **Seeds / CRN** | None. Deterministic (`fault-families.md` (a), Severity). |
| **Runner** | `research/robust_study/tools/a_removal_subset.py`. The fault is installed by the runner, not the config: running the frozen driver on the config alone yields a CLEAN eval. Guard: the runner refuses a config with no `cam_removal` block (`a_removal_subset.py:78-82`); the reverse mistake is caught by a run with no `intervention_attestation.json`. |
| **Attestation** | `cam_subset_removal.SubsetAttestation` (`:158-323`), subclassing `gate_b_removal.Attestation` (`:85-243`) so the counters and digests are produced by GATE-B's own code. Writes `<out-dir>/intervention_attestation.json`; FAIL exits 3 → job records `validity=INVALID`. Asserts: (i) branch hits == n_samples, cross-checked against the driver's `n_total`; (ii) covered view-frames == n_samples × 6 × 8 AND altered == n_samples × 1 × 8; (iii) each of the 6 channels seen exactly n_samples × 8 times, so a loader layout change cannot silently degrade the cell; (iv) per-camera paired pre/post digests on a 3-sample probe set — target camera changed and exactly zero, other 5 bit-identical. It also records the observed frame camera-order signatures, so the ordering finding above is re-derived at run time rather than trusted. |

## Cell A2 — CAM_BACK removed (worst sector)

Identical to A1 in every field except:

- **Camera scope:** `CAM_BACK`.
- **Config:** `research/robust_study/configs/a_removal_back.py` (md5 `113c0b989208de5fb6a15f12ace27124`).
- **Selection provenance:** worst-sector rule registered before any model run
  (`fault-families.md`, "Worst-sector camera"): among the 5 non-frontal cameras, most GT box
  centers inside the view frustum over val, from annotations + calibrations alone. Computed
  2026-08-03 on livenode03: CAM_BACK 59,570 vs CAM_FRONT_LEFT 27,613 (next); no tie-break needed.
  **⚠ PENDING Aug-7 sign-off.** If sign-off selects a different channel, only `cameras` in the
  config changes.

## (a) endpoint reference — the all-6 drop (NO RE-RUN)

The third ladder rung is the existing GATE-B `G4` cell (`--removal table9`) plus its `g4_repeat`,
run by `research/robust_study/tools/gate_b_removal.py` against
`configs/racformer_eval_fullval_research.py`. Mechanism, stage and units are the ones A1/A2
restrict; frame scope is the same 8 frames; camera scope is all 6. Its artifacts and its
`intervention_attestation.json` are cited from the GATE-B run directory. **No Batch-2 GPU time is
spent on it.** Comparability of G4 to the paper's Table 9 is governed by the G-prep compatibility
matrix (`exec-oracle.md`, G-prep item 3) and is unaffected by A1/A2.

### Implementation-identity argument for A1/A2 vs G4

The claim "A1/A2 apply G4's semantics to a subset" is not asserted, it is constructed and then
checked. Constructed: the substituted pixels are produced by G4's own `_zero_image_arg`, called
from `cam_subset_removal.apply_subset` (`:137-155`); the stage is G4's stage; the attestation is
G4's `Attestation` subclassed, not reimplemented. Checked: the probe below installs the REAL
`gate_b_removal.install_intervention("table9", …)` and the subset wrapper over a common capture
stub and compares per-camera digests — target camera bit-identical to G4's output, other five
bit-identical to clean.

### Probe of record (no model, no inference)

`/srv/nfs/shared/gnmp/robust_study_runs/batch2_prep/a_subset_probe_20260803T210021Z/probe.json`,
script `/srv/nfs/shared/gnmp/robust_study_runs/batch2_prep/scripts/probe_a_subset.py`.
3 val samples (0 = scene-start, history frames copy frame 0; 1 and 2 = ordinary, history frames
use the other camera order). Verdict **PASS**, 0 failures. Per cell: `altered_view_frames` 24 of
144 covered view-frames = 1 camera × 8 frames × 3 samples; `channel_seen` 24 for all six
channels; frame order signatures 10 pkl-order + 14 sweep-order = 24 frames. Determinism across
re-instantiation, equality across both `img` layouts, and metadata-unchanged all hold. This
probe is a mechanism check, NOT a mini-screen — the 95 s/cell mini-screen is still owed.

---

# (b) Radar point dropout — p ∈ {25, 50, 75} % × 3 seeds (9 cells)

| field | value |
|---|---|
| **Mechanism** | `research/robust_study/corruptions/radar_noise.py`. `_DropoutMixin.corrupt` (`:189-195`) draws `u ~ U(0,1)` per point and keeps `u >= p`, slicing `points[:, keep]` and `times[:, keep]` at `:195`. Installed by `_RadarCorruptionMixin._install` (`:143-148`), which wraps `self.get_nu_radar` — the single function both radar loaders funnel through — and is stateless: no per-call counter, so draws do not depend on call order, on which pipeline class issued the call, or on DataLoader worker sharding (`:133-139`). |
| **Tensor + stage** | The radar point matrix returned by `get_nu_radar` (`loaders/nuscenes_dataset.py:358`), i.e. AFTER the devkit's multisweep accumulation and ego/time compensation and BEFORE anything downstream consumes it. Points are removed, not zeroed. The PV depth/RCS maps derive from `radar_points` downstream (`loaders/pipelines/loading.py:523-606`) and inherit the dropout automatically — correct for a true sensor-dropout fault. |
| **Radar scope** | All radar channels in the sweep stack, uniformly. Not target-region (that is RobuRCDet's γ=1 variant, not adopted). |
| **Frame scope** | Every sweep element: both `Loadnuradarpoints` (frame-t radar, element 0) and `LoadradarpointsFromMultiSweeps` (the rest). Cells swap both entries together. |
| **Metadata** | Untouched. `radar_tokens` are passed through unchanged (`:167`); only `points` and `times` are re-sliced, and consistently with each other. |
| **Configs** | `research/robust_study/configs/radar_dropout_p{25,50,75}_s{0,1,2}.py` (9 files), generated by `configs/_generate_radar_cells.py`. Classes `RadarDropoutLoadnuradarpoints` (`radar_noise.py:285`) and `RadarDropoutLoadradarpointsFromMultiSweeps` (`:293`). |
| **Seeds / CRN** | Seeds 0,1,2. Counter-based: the RNG is `SHA-256("radar_dropout:{seed}:{identity}")[:8]` seeding PCG64 (`derive_rng`, `:126-131`), where `identity` is `sweep_identity` (`:98-124`) — the resolved starting `sample_data` token per radar channel, mirroring `nuscenes_dataset.py:383-400` and `:470`. Common random numbers across p-levels hold by construction: same key ⇒ same `u` vector ⇒ the p=25 kept set is a superset of p=50's, which is a superset of p=75's. **This nesting is a claim the mini-screen should verify, not assume.** |
| **Attestation** | **IMPLEMENTED 2026-08-03.** Collected by `research/robust_study/corruptions/attest.py` and judged by `research/robust_study/tools/radar_cell_runner.py`, which writes `<out-dir>/intervention_attestation.json` in the same shape as the (a)/GATE-B reports and exits 3 on FAIL so the job records `validity=INVALID`. **Evidence is written from inside the DataLoader workers**, because the frozen driver builds its loader with `workers_per_gpu=4` (`eval_by_condition.py:266`, value at `configs/racformer_r50_nuimg_704x256_f8.py:251`) and an in-process counter in the parent would observe zero applications — indistinguishable from the corruption never running. Each worker appends one JSON line per application to `<out-dir>/attestation_evidence/worker_<pid>_<uid>.jsonl` (`attest.py:139-197`) and the runner sums every line of every file (`attest.py:218-269`). Shared coverage checks (`attest.py:293-337`): evidence exists at all; applications > 0; distinct samples == n_samples; total applications == n_samples × applications-per-sample; every individual sample has exactly that count; worker-recorded parameters match the config. Emitted at `radar_noise.py:165-166` → `_DropoutMixin._attest` (`:197-215`); judged by `attest.DropoutAttestation` (`attest.py:369-414`). Family checks: pooled realised drop fraction `1 − Σn_after/Σn_before` against nominal p, inside a band of max(5·√(p(1−p)/N), 0.002) — a binomial 5σ, so the band scales with the run and cannot fire by luck (registered here, not an external budget); plus counts of empty-point-cloud short-circuits (`radar_noise.py:178-179`), any call that returned MORE points than it was given, and any points/times column-count mismatch. Applications per sample = 8 (1 `Loadnuradarpoints` + 7 sweeps). **Construction-without-invocation is caught by the first two checks**: a class that is built but never invoked writes no evidence file, and the runner reports "the corruption class was constructed but never invoked" rather than a clean-looking result. Applications-per-sample is pinned per family at `radar_cell_runner.py:42-48` and asserted, never inferred. |

---

# (c) Radar Doppler/RCS noise — σ ∈ {1, 3, 5} × 3 seeds (9 cells)

| field | value |
|---|---|
| **Mechanism** | `_DopplerRcsNoiseMixin.corrupt` (`radar_noise.py:235-245`): `z ~ N(0,1)^{3×n}`, `delta = σ·z`, added to rows `ROW_RCS = 5`, `ROW_VX_COMP = 8`, `ROW_VY_COMP = 9` (`:80-82`) at `:242-244`. Positions untouched — this is RobuRCDet's "non-positional disturbance". Same stateless `get_nu_radar` wrap as (b). |
| **Tensor + stage** | Same point matrix and stage as (b). Injection is on RAW fields — raw RCS in dBsm and compensated velocity in m/s — which is the registered stage (`fault-families.md` (c), resolved 2026-08-02: RobuRCDet adds noise before train-set normalization, `nusc_det_dataset_rnoise.py:372-380`, normalization at `:393-395`). |
| **Radar scope** | All radar channels, all points. |
| **Frame scope** | Every sweep element; both loader entries swapped together. |
| **Metadata** | Untouched; `radar_tokens` and `times` pass through, `points` is cloned before mutation (`:241`) so no aliased tensor is written in place. |
| **Configs** | `research/robust_study/configs/radar_noise_sig{1,3,5}_s{0,1,2}.py` (9 files). Classes `RadarNoiseLoadnuradarpoints` (`:301`) and `RadarNoiseLoadradarpointsFromMultiSweeps` (`:309`). |
| **Seeds / CRN** | Seeds 0,1,2; family key `radar_doppler_rcs_noise`; same `derive_rng` + `sweep_identity` scheme as (b). Severity scales the SAME standard-normal draws (`delta = σ·z`), so common random numbers across σ-levels hold by construction. |
| **Severity provenance** | σ ∈ {3,5} are RobuRCDet Table 2's published eval levels; σ = 1 is their Figure 7(a) level. Nothing above 5 is published — `_install_noise` hard-refuses σ > 5 (`:226-230`), so an invented cell cannot be run by accident. §IV must state that the protocol follows the publication, the released corruption code being non-runnable (`fault-families.md` (c) caveat). |
| **Attestation** | **IMPLEMENTED 2026-08-03.** Collected by `research/robust_study/corruptions/attest.py` and judged by `research/robust_study/tools/radar_cell_runner.py`, which writes `<out-dir>/intervention_attestation.json` in the same shape as the (a)/GATE-B reports and exits 3 on FAIL so the job records `validity=INVALID`. **Evidence is written from inside the DataLoader workers**, because the frozen driver builds its loader with `workers_per_gpu=4` (`eval_by_condition.py:266`, value at `configs/racformer_r50_nuimg_704x256_f8.py:251`) and an in-process counter in the parent would observe zero applications — indistinguishable from the corruption never running. Each worker appends one JSON line per application to `<out-dir>/attestation_evidence/worker_<pid>_<uid>.jsonl` (`attest.py:139-197`) and the runner sums every line of every file (`attest.py:218-269`). Shared coverage checks (`attest.py:293-337`): evidence exists at all; applications > 0; distinct samples == n_samples; total applications == n_samples × applications-per-sample; every individual sample has exactly that count; worker-recorded parameters match the config. Emitted at `radar_noise.py:165-166` → `_DopplerRcsNoiseMixin._attest` (`:250-285`); judged by `attest.NoiseAttestation` (`attest.py:416-493`). Family checks: the delta is measured as `points_after − points_before` element-wise in float64, i.e. what LANDED in the tensor, not what was drawn — so the dtype demotion at `radar_noise.py:240` (`.to(points.dtype)`) is caught from both sides, by a shrunken realised SD and by a recorded non-floating `points_dtype`. Pooled realised SD per row against σ inside max(5/√(2n), 0.02); realised mean against 0 inside 5σ/√n; positions (rows x,y,z) and every row outside {rcs, vx_comp, vy_comp} asserted bit-identical per call. Applications per sample = 8. **Construction-without-invocation is caught by the first two checks**: a class that is built but never invoked writes no evidence file, and the runner reports "the corruption class was constructed but never invoked" rather than a clean-looking result. Applications-per-sample is pinned per family at `radar_cell_runner.py:42-48` and asserted, never inferred. |

---

# (d2) Extrinsic miscalibration — {medium, severe} × 3 seeds (6 cells)

| field | value |
|---|---|
| **Mechanism** | `research/robust_study/corruptions/misalign.py`. `ExtrinsicPerturber.perturb` (`:116-129`) returns a perturbed COPY of a radar `calibrated_sensor` record: `R' = R · exp([θ·a]×)` in the sensor frame (`:124`), `t' = t + σ_t·z_t` (`:125`). Delivered by wrapping `renusc.get` (`_install_hook`, `:140-184`), so the devkit's own sensor→ego mapping consumes the perturbed record; the injection adds no filtering of its own (spec item 7). |
| **Tensor + stage** | Not a tensor — the calibration record, loader-side, before point mapping (spec item 8). The radar points and the PV depth/RCS maps inherit the geometry change downstream automatically. |
| **Radar scope** | Independent draw per radar sensor channel; constant within a scene (quasi-static drift). Camera calibrations untouched. |
| **Frame scope** | All sweep elements AND the separately-loaded frame-t radar — both loader entries swapped (`D2MiscalibLoadnuradarpoints` `:237`, `D2MiscalibLoadradarpointsFromMultiSweeps` `:259`), each entering the context manager `perturbed_extrinsics` (`:186`) around the frozen `__call__` (`:250`, `:277`). |
| **Metadata** | The original record is never mutated: `perturb` builds `out = dict(record)` and replaces two keys (`:126-128`). Everything else in the record, and every non-radar record, passes through the hook unchanged. |
| **Configs** | `research/robust_study/configs/d2_extrinsic_{medium,severe}_seed{0,1,2}.py` (6 files). Levels from `D2_LEVELS` (`:44-47`): medium (σ_r, σ_t) = (0.06, 0.006), severe = (0.10, 0.010) — Dong et al. A.1 levels 3 and 5, kept as published. |
| **Seeds / CRN** | Seeds 0,1,2, hard-validated at `:228-229`. Key = `SHA-256("{seed}:{scene_token}:{channel}")[:8]` seeding `default_rng`; axis a, `z_rot`, `z_t` drawn once and cached (`draws`, `:100-114`). Severity scales the same draws (`θ = σ_r·z_rot`, `Δt = σ_t·z_t`), so CRN across the 2 levels holds by construction. No global RNG state. |
| **Registered deviations to disclose in §IV** | Physically-proper SO(3) instead of Dong's released 9-iid-entry non-orthogonal matrix; σ_r is the SD of the rotation ANGLE, not of a 3-vector (spec item 3); Dong's release swaps the rot/trans lists vs the paper text and multiplies by 2 — not reproduced. |
| **Companion artifact (no inference)** | Per-level median [IQR] + P95 reprojection displacement and excluded-pair count, by `research/robust_study/tools/reproj_error_d2.py`, under the both-valid z ≥ 0.1 m rule (spec item 6). Load-bearing: it is what makes the severity levels physically interpretable given the SD-interpretation choice. |
| **GATE-C row type** | Stochastic: 3-seed mean NDS, u(level) = max−min per-seed range. 2 levels meets the ≥2-corrupted-levels minimum exactly — disclosed (spec item 9). |
| **Attestation** | **IMPLEMENTED 2026-08-03.** Collected by `research/robust_study/corruptions/attest.py` and judged by `research/robust_study/tools/radar_cell_runner.py`, which writes `<out-dir>/intervention_attestation.json` in the same shape as the (a)/GATE-B reports and exits 3 on FAIL so the job records `validity=INVALID`. **Evidence is written from inside the DataLoader workers**, because the frozen driver builds its loader with `workers_per_gpu=4` (`eval_by_condition.py:266`, value at `configs/racformer_r50_nuimg_704x256_f8.py:251`) and an in-process counter in the parent would observe zero applications — indistinguishable from the corruption never running. Each worker appends one JSON line per application to `<out-dir>/attestation_evidence/worker_<pid>_<uid>.jsonl` (`attest.py:139-197`) and the runner sums every line of every file (`attest.py:218-269`). Shared coverage checks (`attest.py:293-337`): evidence exists at all; applications > 0; distinct samples == n_samples; total applications == n_samples × applications-per-sample; every individual sample has exactly that count; worker-recorded parameters match the config. Every `calibrated_sensor` lookup is classified in the hook itself (`misalign.py:161-180`) as perturbed / passed-through-in-context / passed-through-outside-context, per channel; the application record is emitted by `_D2Mixin._attest_application` (`misalign.py:206-217`) from a `finally` AFTER the context closes (`:255`, `:279`) so a sample's lookups ride on its own evidence line. Judged by `attest.MisalignAttestation` (`attest.py:495-583`). Family checks: every radar channel perturbed and never passed through inside a context; **LIDAR_TOP perturbed == 0 and pass-through rate == 1.0**, and its absence is itself a failure ("unverified, not confirmed"); no non-radar channel ever perturbed; distinct (scene, channel) perturbation pairs == scenes × 5 radar channels. Applications per sample = 2 (one per loader entry). **Construction-without-invocation is caught by the first two checks**: a class that is built but never invoked writes no evidence file, and the runner reports "the corruption class was constructed but never invoked" rather than a clean-looking result. Applications-per-sample is pinned per family at `radar_cell_runner.py:42-48` and asserted, never inferred. |

---

# (d1) Radar↔camera async — 3 cells (offset 1, 2, 3)

**Injection point re-registered 2026-08-03** (`fault-families.md` (d1), "AMENDED 2026-08-03").
The originally registered mechanism — shifting `choices` in `LoadradarpointsFromMultiSweeps`
while passing the unshifted `sample_idx` — was implemented and **measured to be a no-op**:
offsets 1/2/3 were bit-identical to one another, the frame-t element never moved (0/600 samples),
and the 109/4800 stack elements that did change received NEWER radar. Root cause: a
`results['sweeps']['prev']` entry resolves through a SAMPLE record on the ≈0.5 s keyframe grid
(`loaders/nuscenes_dataset.py:388-395` → `:470`) and never touches the sweep grid. That mechanism
has been REMOVED from the code, not kept behind a flag.

| field | value |
|---|---|
| **Mechanism** | `research/robust_study/corruptions/misalign.py`. `_SweepShiftedRadarPointCloud.from_file_multisweep` (`:378-400`, class at `:369`) subclasses the frozen `RadarPointCloud_v2` and walks `current_sd_rec['prev']` k steps along the true sweep chain before delegating to the frozen parent via `super()` (`:399-400`). Only the aggregation START moves — `sample_rec['data'][chan]`, read once at `loaders/nuscenes_dataset.py:470` — while the loop's own `prev` stride at `:503` is untouched, so the stack keeps its length and spacing and slides k sweeps into the past. Delivered by the context manager `sweep_offset` (`misalign.py:404-421`), which swaps the module global `loaders.nuscenes_dataset.RadarPointCloud_v2`; `get_nu_radar` resolves that name at call time (`:399`, `:404`), which is what routes BOTH radar loader paths through the shifted aggregation. The name and the offset are restored in a `finally`, so the module is a pass-through outside a cell. |
| **Tensor + stage** | Not a tensor — the `sample_data` token that seeds the devkit's multisweep accumulation, loader-side, before any point is read. The radar point matrix, the `Δt` 7th point feature (`loading.py:806/:858/:892`) and the PV depth/RCS maps (`loading.py:523-606`) all inherit the shift downstream automatically. |
| **Radar scope** | All 5 radar channels, walked independently (each has its own `prev` chain). Camera stream untouched — that asymmetry IS the fault. |
| **Frame scope** | Every element of `radar_points`: all 7 sweep-stack elements AND the separately-loaded frame-t element (`D1AsyncLoadnuradarpoints` `:461`, `D1AsyncLoadradarpointsFromMultiSweeps` `:483`). Both config entries are swapped together — the same partial-injection trap as (b)/(c)/(d2), and the one the superseded mechanism fell into. |
| **Metadata** | `sample_rec` and its `data` dict are shallow-copied before substitution (`:395-398`), so the in-memory nuScenes tables and the caller's records are never mutated. `ref_sample_rec` is deliberately NOT touched, so `ref_from_car`, `car_from_global` and `ref_time` (`nuscenes_dataset.py:456-463`) stay anchored on keyframe t — the devkit therefore compensates stale sweeps against the pose and clock of t, which is the async artifact itself. `sample_idx` is passed through unshifted. |
| **Configs** | `research/robust_study/configs/d1_async_offset{1,2,3}.py` (3 files). Classes `D1AsyncLoadnuradarpoints` (`misalign.py:461`) and `D1AsyncLoadradarpointsFromMultiSweeps` (`:483`), parameter `offset`. |
| **Severity ladder** | k ∈ {1,2,3} steps on the physical sweep grid ≈ {77, 154, 231} ms (nuScenes radar ≈ 13 Hz). This is the registered {83, 167, 250} ms camera-frame ladder realised on the grid the data actually has; **the re-basing is a disclosed deviation for §IV**. Measured per-step lag is ≈ 74.5 ms, slightly under the 77 ms nominal. |
| **Seeds / CRN** | None. Deterministic, no seeds (`fault-families.md` (d1)). Determinism is probe-verified by rebuilding the pipeline from scratch and re-running: bit-identical radar tensors, 15/15 sample×offset cases. |
| **Clamp rule** | If the `prev` chain ends before k steps — the sensor stream start at the beginning of a scene — the walk stops at the oldest sweep available rather than failing or wrapping. Counted in `D1_CLAMP_STATS` and in the evidence. Measured: 1856/12000 walks clamp (15.5%) over a 300-sample scan, and `steps_taken` scales exactly ×2 and ×3 with k — **every measured clamp took zero steps**. Partial clamps (chain ends after 1..k−1 steps) are permitted by the registered rule and are recorded in the evidence sums rather than hard-failed (re-ruled 2026-08-03 post-review; a deterministic false-FAIL would kill the row on every rerun). Those elements are precisely the ones the frozen loader had ALREADY collapsed onto the scene's oldest sweep in the CLEAN stack (the `min(idx, len(prev)-1)` clamp at `loading.py:885` plus the out-of-order keyframe entry appended at `nuscenes_dataset.py:165`); no offset can make them older because no older sweep exists. |
| **GATE-C row type** | Deterministic (plan.md §16.4 step 1): aggregate = mean(cell, repeat); u(level) = max(anchor tolerance, \|cell − repeat\|). 3 corrupted levels, comfortably above the ≥2 minimum. |
| **Attestation** | **IMPLEMENTED 2026-08-03.** Collected by `research/robust_study/corruptions/attest.py` and judged by `research/robust_study/tools/radar_cell_runner.py`, which writes `<out-dir>/intervention_attestation.json` and exits 3 on FAIL so the job records `validity=INVALID`. **Evidence is written from inside the DataLoader workers**, because the frozen driver builds its loader with `workers_per_gpu=4` (`eval_by_condition.py:266`, value at `configs/racformer_r50_nuimg_704x256_f8.py:251`) and an in-process counter in the parent would observe zero applications. Each walk records its realised lag — measured from the `sample_data` records the walk already fetched, as clean-start timestamp minus shifted-start timestamp — via `_record_walk` (`misalign.py:327-366`), held with `SINK.add` so a 40-walk stack costs 1 evidence line, not 40 (`attest.py:128-137`). The application record is emitted by `_D1Mixin._attest_application` (`misalign.py:426-451`) AFTER the frozen `__call__` returns, unlike (d2)'s before-the-context record: the evidence that decides the verdict is the realised per-walk lag, which only exists once the walks have happened, and a cell whose hook silently stopped shifting still emits its application and then fails on `d1_walks == 0`. Judged by `attest.D1AsyncAttestation` (`attest.py:581-663`). Family checks: walks > 0; **zero walks may produce NEWER radar** — the exact failure of the superseded mechanism, so it is an assertion rather than an assumption; not every walk clamped; partial-clamp step/lag sums recorded in evidence (hard-fail removed 2026-08-03 — the registered clamp rule permits partial walks); `steps_taken ≤ steps_requested`; mean lag per sweep step inside the physical band [0.050, 0.110] s and mean realised lag consistent with k. Applications per sample = 2, pinned at `radar_cell_runner.py:42-50` (family map at `:57-62`) and asserted, never inferred. |

### Probe of record (no model, no inference)

**Mechanism probe** — `research/robust_study/tools/d1_probe.py`, log
`robust_study_runs/batch2_prep/d1_amended_probe_20260803T212656Z/d1_amended.out` (job 1792),
5 val samples × offsets 1/2/3, timing read from the loaded tensors rather than re-derived index
arithmetic (for one element, `min(Δt)` is the lag of its newest constituent sweep, i.e. exactly
what the injection moves). Results: per-element `Δdt_min` = +0.068..0.097 s at k=1, +0.141..0.172
at k=2, +0.215..0.246 at k=3; **frame-t element shifted in 15/15 cases**; elements identical
between offset levels 0/8 on every unclamped sample; camera filenames AND pixels bit-identical in
15/15; determinism on re-instantiation 15/15. Population scan, 300 samples × 8 elements = 2400
slots: **OLDER 2032, unchanged 368, NEWER 0**, identical at k=1,2,3 — against the superseded
mechanism's 0 older / 109 newer on the same kind of scan. The 368 unchanged are the clamped tails.

**Attestation probe** — `robust_study_runs/batch2_prep/scripts/probe_d1_attest.py`, log
`robust_study_runs/batch2_prep/d1_attest_probe_20260803T215058Z/probe.out` (job 1797).
Cells `d1_async_offset1` and `d1_async_offset3`, 4 scenarios each, **8/8 expectations met**:
HEALTHY PASS (120/120 walks unclamped, per-step lag 0.0745 s in band, k=3 lag 0.2234 s ≈ 3× k=1's
0.0745 s); **BROKEN FAIL** — the class is constructed exactly as the config constructs it but
`misalign.sweep_offset` is replaced by a no-op context manager, so the module global is never
swapped and the frozen aggregation runs: 0 walks recorded, caught by name; WORKERS PASS through a
real `DataLoader(num_workers=2)`, 2 evidence files aggregating in the parent; **NEWER FAIL** — the
walk is inverted to step along `next`, reproducing the superseded mechanism's fresher-data
failure, caught on the direction evidence with all coverage counts healthy. Mid-scene val indices
(20–22) are used on purpose: indices 0–2 sit at a scene start where ~88% of walks clamp, which
would leave the healthy path resting on a handful of unclamped walks.

---

# Cell count and open items

| family | cells | status |
|---|---|---|
| (a) A1, A2 | 2 | implemented, mechanism-probed PASS; A2's camera pending Aug-7 sign-off; mini-screen owed |
| (a) all-6 endpoint | 0 | existing GATE-B G4 + g4_repeat, referenced, never re-run |
| (b) dropout | 9 | implemented + runtime attestation (probe PASS); mini-screen owed |
| (c) noise | 9 | implemented + runtime attestation (probe PASS); mini-screen owed |
| (d2) extrinsic | 6 | implemented + runtime attestation (probe PASS); mini-screen owed |
| (d1) async | 3 | implemented on the re-registered injection point + runtime attestation (probe PASS 8/8, incl. 2 deliberate failures); mini-screen owed |
| **total new cells** | **29** | none run |

1. ~~The (b)/(c)/(d2) attestation gap~~ **CLOSED 2026-08-03.** All three families now write
   per-application evidence from inside the DataLoader workers, judged by
   `research/robust_study/tools/radar_cell_runner.py`. Verified on 3 val samples per family with
   no model: healthy cells PASS with realised magnitudes matching the config; a cell whose class
   is constructed but never invoked FAILS on missing evidence; worker-process aggregation
   reproduces the single-process numbers exactly; a forced dtype demotion FAILS on both the
   realised-SD band and the recorded non-float dtype. Probe of record:
   `/srv/nfs/shared/gnmp/robust_study_runs/batch2_prep/radar_attest_probe_20260803T212854Z/probe.json`
   (re-run after (d1) was added to `attest.py`, still 10/10:
   `.../bc_d2_regression_20260803T215426Z/probe.out`).
   **(d1) closed the same gap on 2026-08-03** with its own family class and probe — see its
   section above; probe of record
   `/srv/nfs/shared/gnmp/robust_study_runs/batch2_prep/d1_attest_probe_20260803T215058Z/probe.out`.
2. Mini-screens (95 s/cell) owed for all 29 cells before any full cell; what a screen rejects is
   recorded, not silently dropped (Cross-family rule 3).
3. A2's camera identity is pending the Aug-7 sign-off of the worst-sector computation.
4. New code means a new `exec_commit` before any full cell runs (`exec-oracle.md`,
   execution-commit rule). None of the files named here is committed yet.
5. OccNuScenes license question for (b) is moot as implemented — the mechanism is our own
   10-line masker, not their script.
