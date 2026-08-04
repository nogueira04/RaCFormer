#!/bin/bash
#SBATCH --job-name=s0_occveltime_v10_e10_probe
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S0_occveltimebev_v10_epoch10_probe/eval_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S0_occveltimebev_v10_epoch10_probe/eval_slurm_%j.err

set -euo pipefail

STAGE="S0_occveltimebev_v10_epoch10_probe"
WEIGHTS="outputs/racformer_train2k_day_occveltimebev_v10_research/2026-05-14/07-56-03/epoch_10.pth"
OUT="research/night_gen_phase1/results/${STAGE}/eval"

test -f "${WEIGHTS}" || { echo "missing weights: ${WEIGHTS}"; exit 2; }
mkdir -p "${OUT}"

export NCCL_SOCKET_IFNAME=^lo,docker,cali
export NCCL_IB_DISABLE=1

echo "STAGE=${STAGE}"
echo "WEIGHTS=${WEIGHTS}"
echo "NODELIST=${SLURM_NODELIST:-unknown}"
conda run -n racformerfix --no-capture-output python -u research/night_gen_phase1/eval_by_condition.py \
  --config configs/racformer_train2k_day_occveltimebev_v10_research.py \
  --weights "${WEIGHTS}" \
  --out-dir "${OUT}" \
  --full-val
