#!/bin/bash
#SBATCH --job-name=s6_radarquery_subset
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S6_radarquery/subset_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S6_radarquery/subset_slurm_%j.err

set -euo pipefail

STAGE="S6_radarquery"
PRED_JSON="research/night_gen_phase1/results/${STAGE}/eval/submission_overall/pts_bbox/results_nusc.json"
OUT="research/night_gen_phase1/results/${STAGE}/subset_eval"

test -f "${PRED_JSON}" || { echo "missing prediction json: ${PRED_JSON}"; exit 2; }
mkdir -p "${OUT}"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

echo "STAGE=${STAGE}"
echo "PRED_JSON=${PRED_JSON}"
echo "OUT=${OUT}"

conda run -n racformerfix --no-capture-output python -u \
  research/paper_goal_20260515/eval_radarquery_subsets.py \
  --pred-json "${PRED_JSON}" \
  --out-dir "${OUT}"
