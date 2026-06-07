#!/bin/bash
#SBATCH --job-name=s5_contrel_smoke
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S5_contrelqfusion/smoke_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S5_contrelqfusion/smoke_slurm_%j.err

set -euo pipefail

mkdir -p research/night_gen_phase1/results/S5_contrelqfusion

echo "HOST $(hostname)"
echo "START $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/smoke_contrelqfusion.py

echo "END $(date -Is)"
