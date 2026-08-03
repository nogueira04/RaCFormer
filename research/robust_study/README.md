# robust_study — evaluator oracle tooling

Everything here exists to make one claim defensible: that the numbers the evaluation wrapper
reports for a submission are the numbers the official nuScenes devkit computes for that same
submission. Nothing here trains, and nothing here runs model inference except the eval driver the
cells invoke.

## Layout

| Path | What it is |
|---|---|
| `tools/devkit_crosscheck.py` | Independent devkit cross-check: official `DetectionEval`, staged box counts, per-class TP/FP/FN at `dist_th_tp`, and the mechanical tolerance table. Exit status is the verdict. |
| `tools/run_devkit_crosscheck.sh` | The only correct way to launch it (isolated interpreter, isolated venv). |
| `tools/assert_val_token_set.py` | Post-run assertion that a submission covers exactly the official val split. |
| `tools/write_provenance.py` | Writes `provenance.json` for a cell, on success and on abort alike. |
| `tools/constraints-devkit-venv.txt` | `pip freeze` of the cross-check venv. |
| `jobs/_job_common.sh` | Shared run discipline: SHA + clean-tree assertions, run-dir refusal, markers, seeds, provenance. |
| `jobs/_clean_eval_body.sh` | The clean-eval body shared by the reference cell and its two anchor repeats. |
| `jobs/e0_smoke.sbatch` | Environment smoke cell (imports, CUDA device, versions, devkit paths). |
| `jobs/e1_clean_fullval.sbatch` | Clean full-val reference cell. |
| `jobs/e1_anchor_repeat_{a,b}.sbatch` | The two repeats whose spread defines anchor equivalence. |
| `jobs/e2_crosscheck.sbatch` | Independent cross-check of one completed cell (CPU only). |
| `frozen/` | Values computed once and then pinned by committing them. See `frozen/README.md`. |
| `env_snapshots/` | Environment snapshots (the directory is not called `env/` because the repo `.gitignore` swallows that name). |

## Run discipline

Every cell:

1. creates a fresh run directory under `/srv/nfs/shared/gnmp/robust_study_runs/<phase>/<cell>_<UTC>`
   and refuses to start if it already exists. Run directories live **outside** the checkout, so
   producing output can never be what breaks the clean-tree assertion;
2. asserts at **start and end** that `HEAD` equals the injected execution commit and that
   `git status --porcelain` is empty. A start failure aborts; an **end** failure records
   `validity=INVALID` — it does not quietly pass;
3. always writes `_FINALIZED` (from an EXIT trap, with the exit status) and writes `_COMPLETE`
   **only** when `validity=VALID`. **A run directory without `_COMPLETE` is not a result.**
4. exports and records its seeds, and writes `provenance.json` on every path including aborts.

The execution commit is injected, not hardcoded: these scripts are part of the very commit whose
SHA they assert. Put it in `/srv/nfs/shared/gnmp/robust_study_runs/exec_commit.txt` or export
`EXEC_COMMIT`.

## The cross-check venv

`~/venvs/nusc-devkit-check` holds a separately installed `nuscenes-devkit==1.1.11`. The
cross-check runs under `python -I` with `PYTHONNOUSERSITE=1` and asserts that `nuscenes` resolves
inside that venv, so it can never end up validating the evaluation environment against itself.

The cluster has no DNS, so the venv was populated from a locally built wheel set rather than from
PyPI directly; the devkit version is the pinned one either way, and `tools/constraints-devkit-venv.txt`
records the exact closure. To rebuild it, replay that file with
`pip install --no-index --find-links <wheel dir> -r tools/constraints-devkit-venv.txt`.

## Reading a result

```
python3 -m json.tool <run_dir>/crosscheck.json | less   # verdict + every comparison row
cat <run_dir>/provenance.json                           # what ran, from what tree, with what seeds
ls <run_dir>/_COMPLETE                                  # absent => do not read the numbers
```
