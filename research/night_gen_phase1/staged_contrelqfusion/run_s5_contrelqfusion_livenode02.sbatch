#!/bin/bash
#SBATCH --job-name=s5_contrel
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S5_contrelqfusion/slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S5_contrelqfusion/slurm_%j.err

set -euo pipefail

mkdir -p research/night_gen_phase1/results/S5_contrelqfusion

echo "HOST $(hostname)"
echo "START $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

export NCCL_SOCKET_IFNAME=^lo,docker,cali
export NCCL_IB_DISABLE=1
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

conda run -n racformerfix --no-capture-output python -u train.py \
  --config configs/racformer_train2k_mixed_contrelqfusion_research.py

echo "END $(date -Is)"
