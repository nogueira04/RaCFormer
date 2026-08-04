#!/bin/bash
#SBATCH --job-name=s5_contrel_gate
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S5_contrelqfusion/gate_stats_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S5_contrelqfusion/gate_stats_slurm_%j.err

set -euo pipefail

STAGE="S5_contrelqfusion"
CFG_DIR="outputs/racformer_train2k_mixed_contrelqfusion_research"
OUT="research/night_gen_phase1/results/${STAGE}/gate_stats"

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
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

echo "STAGE=${STAGE}"
echo "WEIGHTS=${WEIGHTS}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/staged_contrelqfusion/inspect_contrelqfusion_gates.py \
  --config configs/racformer_train2k_mixed_contrelqfusion_research.py \
  --weights "${WEIGHTS}" \
  --out-dir "${OUT}" \
  --max-samples 96
