#!/bin/bash
#SBATCH --job-name=s6_radarq90_s20260502_sum
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S6_radarquery_topk90_seed20260502/summary_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S6_radarquery_topk90_seed20260502/summary_slurm_%j.err

set -euo pipefail

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

conda run -n racformerfix --no-capture-output python -u \
  research/paper_goal_20260515/summarize_condition_stage.py \
  --stage S6_radarquery_topk90_seed20260502 \
  --baseline-stage S0 \
  --title "S6 radar-guided query initialization topk90 seed20260502 summary" \
  --note "Seed-20260502 replication of the S6 radar-query topk90 ablation. Uses the same current-frame radar query initialization and RCS plus compensated speed ranking as the seed-0 topk90 run." \
  --gate-mode s0
