#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/full_camdrop20_epoch36_condition/eval_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/full_camdrop20_epoch36_condition/eval_slurm_%j.err
#SBATCH --job-name=full_camdrop20_ep36_cond

set -euo pipefail

STAGE="full_camdrop20_epoch36_condition"
WEIGHTS="outputs/racformer_r50_nuimg_704x256_f8_dropout/2026-01-25/22-45-16/epoch_36.pth"
OUT="research/night_gen_phase1/results/${STAGE}/eval"

mkdir -p "${OUT}"

export NCCL_SOCKET_IFNAME=^lo,docker,cali
export NCCL_IB_DISABLE=1

echo "STAGE=${STAGE}"
echo "WEIGHTS=${WEIGHTS}"
conda run -n racformerfix --no-capture-output python -u research/night_gen_phase1/eval_by_condition.py \
  --config configs/racformer_eval_fullval_research.py \
  --weights "${WEIGHTS}" \
  --out-dir "${OUT}" \
  --full-val
