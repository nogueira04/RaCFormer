#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S3_seed20260502_ratio18p75_w05_adaptfusion/eval_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S3_seed20260502_ratio18p75_w05_adaptfusion/eval_slurm_%j.err
#SBATCH --job-name=S3_seed20260502_ratio18p75_w05_adaptfusion

set -euo pipefail

STAGE="S3_seed20260502_ratio18p75_w05_adaptfusion"
CFG_DIR="outputs/racformer_train2k_genaug_seed20260502_ratio18p75_w05_adaptfusion_research"
OUT="research/night_gen_phase1/results/${STAGE}/eval"

latest=$(ls -dt "${CFG_DIR}"/*/* 2>/dev/null | head -1)
if [ -z "${latest}" ]; then
  echo "no work_dir under ${CFG_DIR}"
  exit 2
fi
WEIGHTS="${latest}/epoch_12.pth"
test -f "${WEIGHTS}" || { echo "missing weights: ${WEIGHTS}"; exit 2; }

mkdir -p "${OUT}"

export NCCL_SOCKET_IFNAME=^lo,docker,cali
export NCCL_IB_DISABLE=1

echo "STAGE=${STAGE}"
echo "WEIGHTS=${WEIGHTS}"
conda run -n racformerfix --no-capture-output python -u research/night_gen_phase1/eval_by_condition.py \
  --config configs/racformer_train2k_genaug_seed20260502_ratio18p75_w05_adaptfusion_research.py \
  --weights "${WEIGHTS}" \
  --out-dir "${OUT}" \
  --full-val
