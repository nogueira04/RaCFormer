#!/bin/bash
#SBATCH --job-name=g_stage3b_e12
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/G_stage3b_dualviewdistill_epoch12/slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/G_stage3b_dualviewdistill_epoch12/slurm_%j.err

set -euo pipefail

STAGE="G_stage3b_dualviewdistill_epoch12"
TRAIN_CONFIG="configs/racformer_train2k_mixed_dualviewdistill_research.py"
EVAL_CONFIG="configs/racformer_eval_fullval_dualviewdistill_zero_research.py"
RUN_ROOT="outputs/racformer_train2k_mixed_dualviewdistill_epoch6_research"
RUN_DIR="${RUN_ROOT}/2026-05-17/17-02-26"
RESUME_FROM="${RUN_DIR}/epoch_6.pth"
WEIGHTS="${RUN_DIR}/epoch_12.pth"
INFERENCE_WEIGHTS="${RUN_DIR}/epoch_12_inference.pth"
RESULT_DIR="research/night_gen_phase1/results/${STAGE}"
MONITOR_JSONL="${RESULT_DIR}/pathology_monitor_epoch12_${SLURM_JOB_ID:-unknown}.jsonl"

mkdir -p "${RESULT_DIR}"
test -f "${RESUME_FROM}" || { echo "missing resume checkpoint: ${RESUME_FROM}"; exit 2; }

echo "HOST $(hostname)"
echo "START $(date -Is)"
echo "STAGE=${STAGE}"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "EVAL_CONFIG=${EVAL_CONFIG}"
echo "RUN_DIR=${RUN_DIR}"
echo "RESUME_FROM=${RESUME_FROM}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

export NCCL_SOCKET_IFNAME=^lo,docker,cali
export NCCL_IB_DISABLE=1
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

if [ ! -f "${WEIGHTS}" ]; then
  setsid conda run -n racformerfix --no-capture-output python -u train.py \
    --config "${TRAIN_CONFIG}" \
    --override resume_from="${RESUME_FROM}" checkpoint_config.max_keep_ckpts=-1 &
  TRAIN_PID=$!

  conda run -n racformerfix --no-capture-output python -u \
    research/night_gen_phase1/staged_dualviewdistill/monitor_stage3b_log.py \
    --run-root "${RUN_ROOT}" \
    --pid "${TRAIN_PID}" \
    --poll-seconds 60 \
    --log-copy "${MONITOR_JSONL}" &
  MONITOR_PID=$!

  set +e
  wait "${TRAIN_PID}"
  TRAIN_RC=$?
  set -e

  if kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi

  if [ "${TRAIN_RC}" -ne 0 ]; then
    echo "Training failed with exit code ${TRAIN_RC}"
    exit "${TRAIN_RC}"
  fi
else
  echo "epoch_12 checkpoint already exists; skipping training resume"
fi

TRAIN_LOG="${RUN_DIR}/train.log"
test -f "${TRAIN_LOG}" || { echo "missing train log: ${TRAIN_LOG}"; exit 2; }
test -f "${WEIGHTS}" || { echo "missing epoch-12 checkpoint: ${WEIGHTS}"; exit 2; }

conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/staged_dualviewdistill/strip_dualview_teacher_checkpoint.py \
  --input "${WEIGHTS}" \
  --output "${INFERENCE_WEIGHTS}"

NEUTRALITY_JSON="${RESULT_DIR}/inference_neutrality_epoch12.json"
conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/staged_dualviewdistill/check_dualview_inference_neutrality.py \
  --config "${TRAIN_CONFIG}" \
  --checkpoint "${WEIGHTS}" \
  --out-json "${NEUTRALITY_JSON}"

EVAL_DIR="${RESULT_DIR}/eval_epoch12"
mkdir -p "${EVAL_DIR}"
conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/eval_by_condition.py \
  --config "${EVAL_CONFIG}" \
  --weights "${INFERENCE_WEIGHTS}" \
  --out-dir "${EVAL_DIR}" \
  --full-val

UTC=$(date -u +%Y%m%dT%H%M%SZ)
REPORT="research/paper_goal_20260516/G_RESULT_STAGE3B_${UTC}.md"
RESULT_JSON="${RESULT_DIR}/stage3b_result_${UTC}.json"
FAILURE_MD="research/paper_goal_20260516/G_FAILURE_MODE_${UTC}.md"
GIT_SHA=$(git rev-parse HEAD)

conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/staged_dualviewdistill/report_stage3b_result.py \
  --stage "${STAGE}" \
  --train-log "${TRAIN_LOG}" \
  --eval-dir "${EVAL_DIR}" \
  --out-md "${REPORT}" \
  --result-json "${RESULT_JSON}" \
  --failure-md "${FAILURE_MD}" \
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
