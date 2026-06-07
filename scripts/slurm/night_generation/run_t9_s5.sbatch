#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S5/slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S5/slurm_%j.err
#SBATCH --job-name=t9_s5_mixed

set -euo pipefail
mkdir -p research/night_gen_phase1/results/S5

export NCCL_SOCKET_IFNAME=^lo,docker,cali
export NCCL_IB_DISABLE=1

conda run -n racformerfix --no-capture-output python -u train.py \
  --config configs/racformer_train2k_mixed_research.py
