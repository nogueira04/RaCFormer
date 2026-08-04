#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --job-name=rs_b2_mini
#SBATCH --output=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.out
#SBATCH --error=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.err
#
# Batch-2 mini-screens (fault-families.md, cross-family rule 3). PREP job, not a study cell.
#
# Deliberately does NOT source _job_common.sh, for the same reason c1_smoke.sbatch does not:
# that helper asserts a clean checkout at the pinned exec_commit and mints provenance for a
# measured cell; this job screens implementations at 300 samples and produces no result any
# gate may read. HEAD and dirtiness are still recorded in screen_summary.json.
#
# Sharding across nodes: override the node at submit time and pass the shard through the
# environment --
#   sbatch -w livenode01 --export=ALL,CELLS=radar_dropout_p25_s0,radar_dropout_p25_s1 \
#     research/robust_study/jobs/batch2_mini_screen.sbatch
# CELLS defaults to "all" (29 cells). Every shard runs its own clean baseline (SKIP_CLEAN=1
# to opt out when another shard on the SAME node already ran one): the baseline costs one
# 300-sample eval and anchors the shard's metric deltas to the node that produced them.

set -euo pipefail
source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh

REPO=/srv/nfs/shared/gnmp/RaCFormer
RUN_ROOT=/srv/nfs/shared/gnmp/robust_study_runs
CELLS="${CELLS:-all}"
SAMPLES="${SAMPLES:-300}"
STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="$RUN_ROOT/batch2_prep/mini_screens_${STAMP}_$(hostname)"

mkdir -p "$RUN_ROOT/slurm_logs"
echo "[job] node=$(hostname) job_id=${SLURM_JOB_ID:-manual} cells=$CELLS samples=$SAMPLES"
echo "[job] repo_head=$(git -C "$REPO" rev-parse HEAD)"
echo "[job] out_root=$OUT_ROOT"
nvidia-smi || echo "[job] nvidia-smi unavailable"

conda run -n racformerfix --no-capture-output python -u \
  "$REPO/research/robust_study/tools/mini_screen.py" \
  --repo "$REPO" \
  --out-root "$OUT_ROOT" \
  --samples "$SAMPLES" \
  --cells "$CELLS" \
  ${SKIP_CLEAN:+--skip-clean}

echo "[job] mini-screen shard complete: $OUT_ROOT/screen_summary.json"
