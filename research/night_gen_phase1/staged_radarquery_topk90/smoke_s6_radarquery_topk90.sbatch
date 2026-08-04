#!/bin/bash
#SBATCH --job-name=s6_radarq90_smoke
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S6_radarquery_topk90/smoke_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S6_radarquery_topk90/smoke_slurm_%j.err

set -euo pipefail

mkdir -p research/night_gen_phase1/results/S6_radarquery_topk90

echo "HOST $(hostname)"
echo "START $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/smoke_radarquery_topk90.py

echo "END $(date -Is)"
