# Adaptive Fusion Gate Branch

This is the RobuRCDet-inspired fallback branch if loss-weighted NB2 replay does not meet the gate.

## Rationale

RobuRCDet motivates adaptive radar-camera fusion under poor camera signal. The lowest-risk RaCFormer hook is decoder-level fusion, where image ray, radar BEV, and LSS BEV query features are already aligned as `[B, Q, 256]`.

## Staged Artifacts

- Patch: `research/night_gen_phase1/staged_adaptfusion/adaptive_fusion_gate.patch`
- Config: `research/night_gen_phase1/staged_adaptfusion/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research.py`
- Job script: `research/night_gen_phase1/staged_adaptfusion/run_t11_adaptfusion_seed20260425_ratio18p75_w05.sbatch`
- Eval script: `research/night_gen_phase1/staged_adaptfusion/run_t11_adaptfusion_seed20260425_ratio18p75_w05_eval.sbatch`
- Summary script: `research/night_gen_phase1/staged_adaptfusion/run_t11_adaptfusion_seed20260425_ratio18p75_w05_summary.sbatch`
- Summarizer: `research/night_gen_phase1/staged_adaptfusion/summarize_adaptfusion.py`

The patch passed:

```bash
cd /srv/nfs/shared/gnmp/RaCFormer
git apply --check research/night_gen_phase1/staged_adaptfusion/adaptive_fusion_gate.patch
```

## Apply After Current Eval

Only apply this after the active loss-weight chains finish and their summaries are recorded: `1245 -> 1265 -> 1266`, `1261 -> 1262 -> 1263`, then compare job `1267`. This keeps the S3 w05/w025 evals isolated from code churn.

```bash
cd /srv/nfs/shared/gnmp/RaCFormer
git apply research/night_gen_phase1/staged_adaptfusion/adaptive_fusion_gate.patch
cp research/night_gen_phase1/staged_adaptfusion/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research.py configs/
cp research/night_gen_phase1/staged_adaptfusion/run_t11_adaptfusion_seed20260425_ratio18p75_w05.sbatch .
cp research/night_gen_phase1/staged_adaptfusion/run_t11_adaptfusion_seed20260425_ratio18p75_w05_eval.sbatch .
cp research/night_gen_phase1/staged_adaptfusion/run_t11_adaptfusion_seed20260425_ratio18p75_w05_summary.sbatch .
conda run -n racformerfix --no-capture-output python -m py_compile models/racformer_transformer.py
bash -n run_t11_adaptfusion_seed20260425_ratio18p75_w05.sbatch
bash -n run_t11_adaptfusion_seed20260425_ratio18p75_w05_eval.sbatch
bash -n run_t11_adaptfusion_seed20260425_ratio18p75_w05_summary.sbatch
conda run -n racformerfix --no-capture-output python -m py_compile research/night_gen_phase1/staged_adaptfusion/summarize_adaptfusion.py
conda run -n racformerfix --no-capture-output python - <<'PY'
from pathlib import Path
from mmcv import Config
cfg = Config.fromfile('configs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research.py')
expected_cfg_dir = 'outputs/' + Path('configs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research.py').stem
manifest = next(step for step in cfg.train_pipeline if step.get('type') == 'LoadMultiViewImageFromManifest')
collect = next(step for step in cfg.train_pipeline if step.get('type') == 'Collect3D')
assert cfg.model['pts_bbox_head']['transformer'].get('adaptive_fusion_gate') is True
assert manifest.get('generated_sample_weight') == 0.5
assert 'generated_sample_weight' in collect.get('meta_keys', [])
assert expected_cfg_dir == 'outputs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research'
assert cfg.get('total_epochs') == 12
print('adaptive fusion config OK')
PY
```

## Smoke Before Training

Run the existing config parser/smoke pattern before a full train job. The config intentionally uses a `./...w05_research.py` base, so parse it only after copying it into `configs/`; parsing it directly from `staged_adaptfusion/` will not resolve the base chain. At minimum, verify that the config resolves:

```bash
cd /srv/nfs/shared/gnmp/RaCFormer
conda run -n racformerfix --no-capture-output python - <<'PY'
from mmcv import Config
cfg = Config.fromfile("configs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_adaptfusion_research.py")
print(cfg.model["pts_bbox_head"]["transformer"]["adaptive_fusion_gate"])
PY
```

## Train/Eval

```bash
sbatch run_t11_adaptfusion_seed20260425_ratio18p75_w05.sbatch
```

After train finishes:

```bash
eval_job=$(sbatch --parsable run_t11_adaptfusion_seed20260425_ratio18p75_w05_eval.sbatch)
sbatch --dependency=afterok:${eval_job} run_t11_adaptfusion_seed20260425_ratio18p75_w05_summary.sbatch
```

## Gate

Compare against the just-finished S3 w05 run, not only against S0. The staged summarizer reports both:

- Night mAP improves or is retained.
- Day mAP drop is no worse than 1.0 pp against S3 w05.
- Overall mAP drop is no worse than 1.5 pp against S3 w05.
- If metrics are close, inspect adaptive gate dumps using `DUMP.enabled` on a small val subset.
