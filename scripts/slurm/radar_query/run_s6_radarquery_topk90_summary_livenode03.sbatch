#!/bin/bash
#SBATCH --job-name=s6_radarq90_summary
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S6_radarquery_topk90/summary_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S6_radarquery_topk90/summary_slurm_%j.err

set -euo pipefail

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

conda run -n racformerfix --no-capture-output python -u \
  research/paper_goal_20260515/summarize_condition_stage.py \
  --stage S6_radarquery_topk90 \
  --baseline-stage S0 \
  --title "S6 radar-guided query initialization topk90 summary" \
  --note "Train2k-day ablation of the S6 radar-query screen that initializes the first 90 decoder queries from top-scored current-frame radar points, using RCS plus compensated speed as the ranking signal. Publication gate compares full-val day/night/rain/overall metrics against S0." \
  --gate-mode s0
