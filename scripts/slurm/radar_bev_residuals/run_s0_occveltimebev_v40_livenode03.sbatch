#!/bin/bash
#SBATCH --job-name=s0_occveltime_v40
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S0_occveltimebev_v40/slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S0_occveltimebev_v40/slurm_%j.err

set -euo pipefail

mkdir -p research/night_gen_phase1/results/S0_occveltimebev_v40

echo "HOST $(hostname)"
echo "START $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

conda run -n racformerfix --no-capture-output python -u train.py \
  --config configs/racformer_train2k_day_occveltimebev_v40_research.py

echo "END $(date -Is)"
