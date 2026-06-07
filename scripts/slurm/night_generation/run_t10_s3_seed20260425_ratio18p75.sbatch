#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S3_seed20260425_ratio18p75/slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S3_seed20260425_ratio18p75/slurm_%j.err
#SBATCH --job-name=t10_s3_r18p75_seed20260425

set -euo pipefail
mkdir -p research/night_gen_phase1/results/S3_seed20260425_ratio18p75

export NCCL_SOCKET_IFNAME=^lo,docker,cali
export NCCL_IB_DISABLE=1

conda run -n racformerfix --no-capture-output python -u train.py \
  --config configs/racformer_train2k_genaug_seed20260425_ratio18p75_research.py
