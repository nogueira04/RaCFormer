#!/bin/bash
#SBATCH --job-name=g_stage3a_smoke
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/teachers/diagnostics/G_STAGE3A_SMOKE_%j.out
#SBATCH --error=research/night_gen_phase1/teachers/diagnostics/G_STAGE3A_SMOKE_%j.err

set -euo pipefail

echo "HOST $(hostname)"
echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/teachers/diagnostics/branch_g_stage3a_dualview_smoke.py \
  --config configs/racformer_train2k_mixed_dualviewdistill_smoke_research.py \
  --checkpoint checkpoints/racformer_r50_f8.pth \
  --report-dir research/paper_goal_20260516 \
  --json-dir research/night_gen_phase1/teachers/diagnostics

echo "END $(date -u +%Y-%m-%dT%H:%M:%SZ)"
