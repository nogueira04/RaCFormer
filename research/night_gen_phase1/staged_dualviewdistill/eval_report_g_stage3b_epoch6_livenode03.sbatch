#!/bin/bash
#SBATCH --job-name=g_stage3b_e6eval
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/G_stage3b_dualviewdistill_epoch6/eval_report_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/G_stage3b_dualviewdistill_epoch6/eval_report_slurm_%j.err

set -euo pipefail

STAGE="G_stage3b_dualviewdistill_epoch6"
TRAIN_CONFIG="configs/racformer_train2k_mixed_dualviewdistill_epoch6_research.py"
EVAL_CONFIG="configs/racformer_eval_fullval_dualviewdistill_zero_research.py"
RUN_DIR="outputs/racformer_train2k_mixed_dualviewdistill_epoch6_research/2026-05-17/17-02-26"
RESULT_DIR="research/night_gen_phase1/results/${STAGE}"

TRAIN_LOG="${RUN_DIR}/train.log"
WEIGHTS="${RUN_DIR}/epoch_6.pth"
INFERENCE_WEIGHTS="${RUN_DIR}/epoch_6_inference.pth"
NEUTRALITY_JSON="${RESULT_DIR}/inference_neutrality_epoch6.json"
EVAL_DIR="${RESULT_DIR}/eval_epoch6"
MONITOR_JSONL="${RESULT_DIR}/pathology_monitor.jsonl"

mkdir -p "${RESULT_DIR}" "${EVAL_DIR}"
test -f "${TRAIN_LOG}" || { echo "missing train log: ${TRAIN_LOG}"; exit 2; }
test -f "${WEIGHTS}" || { echo "missing checkpoint: ${WEIGHTS}"; exit 2; }

echo "HOST $(hostname)"
echo "START $(date -Is)"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "EVAL_CONFIG=${EVAL_CONFIG}"
echo "RUN_DIR=${RUN_DIR}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

export NCCL_SOCKET_IFNAME=^lo,docker,cali
export NCCL_IB_DISABLE=1
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/staged_dualviewdistill/strip_dualview_teacher_checkpoint.py \
  --input "${WEIGHTS}" \
  --output "${INFERENCE_WEIGHTS}"

conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/staged_dualviewdistill/check_dualview_inference_neutrality.py \
  --config "${TRAIN_CONFIG}" \
  --checkpoint "${WEIGHTS}" \
  --out-json "${NEUTRALITY_JSON}"

conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/eval_by_condition.py \
  --config "${EVAL_CONFIG}" \
  --weights "${INFERENCE_WEIGHTS}" \
  --out-dir "${EVAL_DIR}" \
  --full-val

UTC=$(date -u +%Y%m%dT%H%M%SZ)
REPORT="research/paper_goal_20260516/STAGE3B_G_EPOCH6_${UTC}.md"
TRAJECTORY_JSON="${RESULT_DIR}/epoch6_report_${UTC}.json"
GIT_SHA=$(git rev-parse HEAD)

conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/staged_dualviewdistill/report_stage3b_epoch6.py \
  --stage "${STAGE}" \
  --train-log "${TRAIN_LOG}" \
  --eval-dir "${EVAL_DIR}" \
  --out-md "${REPORT}" \
  --trajectory-json "${TRAJECTORY_JSON}" \
  --run-dir "${RUN_DIR}" \
  --weights "${WEIGHTS}" \
  --inference-weights "${INFERENCE_WEIGHTS}" \
  --neutrality-json "${NEUTRALITY_JSON}" \
  --monitor-jsonl "${MONITOR_JSONL}" \
  --job-id "${SLURM_JOB_ID:-unknown}" \
  --host "$(hostname)" \
  --config "${TRAIN_CONFIG}" \
  --eval-config "${EVAL_CONFIG}" \
  --git-sha "${GIT_SHA}"

echo "REPORT=${REPORT}"
echo "END $(date -Is)"
