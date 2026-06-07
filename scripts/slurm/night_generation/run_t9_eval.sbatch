#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/%x/eval_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/%x/eval_slurm_%j.err

# Inputs (passed via --export):
#   STAGE  - one of S0 / S5 / S1
#   WEIGHTS - absolute path to epoch_12.pth

set -euo pipefail
: "${STAGE:?STAGE env var required}"
: "${WEIGHTS:?WEIGHTS env var required}"

# Resolve __LATEST__ in WEIGHTS to the most recent work_dir for this stage's
# config. Useful when chaining a training job + its eval before knowing the
# train's timestamped work_dir.
if [[ "$WEIGHTS" == *__LATEST__* ]]; then
  case "$STAGE" in
    S0) cfg_dir="outputs/racformer_train2k_day_research" ;;
    S5) cfg_dir="outputs/racformer_train2k_mixed_research" ;;
    S1) cfg_dir="outputs/racformer_train2k_simnight_research" ;;
    *)  echo "unknown STAGE for __LATEST__ resolve: $STAGE"; exit 2 ;;
  esac
  latest=$(ls -dt $cfg_dir/*/* 2>/dev/null | head -1)
  if [ -z "$latest" ]; then echo "no work_dir under $cfg_dir"; exit 2; fi
  WEIGHTS="${WEIGHTS//__LATEST__/${latest#outputs/}}"
  WEIGHTS="${WEIGHTS#$cfg_dir/*/*/}"  # in case of double prefix
  WEIGHTS="$latest/$(basename "$WEIGHTS")"
  echo "resolved __LATEST__ -> $WEIGHTS"
fi

test -f "$WEIGHTS" || { echo "missing weights: $WEIGHTS"; exit 2; }

OUT="research/night_gen_phase1/results/${STAGE}/eval"
mkdir -p "$OUT"

export NCCL_SOCKET_IFNAME=^lo,docker,cali
export NCCL_IB_DISABLE=1

conda run -n racformerfix --no-capture-output python -u research/night_gen_phase1/eval_by_condition.py \
  --config configs/racformer_eval_fullval_research.py \
  --weights "$WEIGHTS" \
  --out-dir "$OUT" \
  --full-val
