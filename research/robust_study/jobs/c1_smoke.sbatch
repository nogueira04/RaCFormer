#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --job-name=rs_c1_smoke
#SBATCH --output=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.out
#SBATCH --error=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.err
#
# Unit-level smoke for the C1 radar-removal arm. PREP job, not a study cell.
#
# Deliberately does NOT source _job_common.sh. That helper asserts a clean checkout at start and
# end and writes a provenance.json for a measured cell; this job is candidate unit-testing on a
# working tree that necessarily carries the un-committed candidate itself, and it produces no
# metric. Running it through the cell harness would either abort on the dirty tree or mint a
# provenance record for something that is not a result.
#
# Clean input only: one val sample, no corrupted data, no detection metrics, no submission.

set -euo pipefail
source "$HOME/.bashrc" 2>/dev/null || true

REPO=/srv/nfs/shared/gnmp/RaCFormer
RUN_ROOT=/srv/nfs/shared/gnmp/robust_study_runs
ENV_NAME=racformerfix
CONFIG=$REPO/configs/racformer_eval_fullval_research.py
WEIGHTS=$REPO/checkpoints/racformer_r50_f8.pth
OUT_DIR=$RUN_ROOT/prep/c1_smoke_${SLURM_JOB_ID:-manual}

mkdir -p "$OUT_DIR" "$RUN_ROOT/slurm_logs"

echo "[job] node=$(hostname) job_id=${SLURM_JOB_ID:-manual}"
echo "[job] repo_head=$(git -C "$REPO" rev-parse HEAD)"
echo "[job] out_dir=$OUT_DIR"
nvidia-smi || echo "[job] nvidia-smi unavailable"

conda run -n "$ENV_NAME" --no-capture-output python -u \
  "$REPO/research/robust_study/mitigation/smoke_c1_radar_removal.py" \
  --repo "$REPO" \
  --config "$CONFIG" \
  --weights "$WEIGHTS" \
  --out-dir "$OUT_DIR"

echo "[job] C1 smoke complete: $OUT_DIR/c1_smoke.json"
