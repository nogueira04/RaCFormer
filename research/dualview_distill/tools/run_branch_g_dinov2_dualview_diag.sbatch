#!/bin/bash
#SBATCH --job-name=g_dino_diag
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:55:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/teachers/diagnostics/G_DINO_DIAG_%j.out
#SBATCH --error=research/night_gen_phase1/teachers/diagnostics/G_DINO_DIAG_%j.err

set -euo pipefail

echo "HOST $(hostname)"
echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
UTC="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="research/night_gen_phase1/teachers/diagnostics/G_DINO_DUALVIEW_${UTC}.json"

conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/teachers/diagnostics/branch_g_dinov2_dualview_diag.py \
  --repo-root . \
  --teacher-dir research/night_gen_phase1/teachers/dinov2_vitl14 \
  --val-pkl nuscenes_infos_val_sweep.pkl \
  --max-scenes 10 \
  --coverage-frames 50 \
  --output "${OUT}"

echo "JSON ${OUT}"
echo "END $(date -u +%Y-%m-%dT%H:%M:%SZ)"
