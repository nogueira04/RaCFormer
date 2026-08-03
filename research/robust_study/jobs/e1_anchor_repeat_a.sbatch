#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --time=06:00:00
#SBATCH --job-name=rs_e1_repeat_a
#SBATCH --output=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.out
#SBATCH --error=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.err
#
# Anchor repeat a: a byte-identical rerun of the clean reference cell.
# Its only purpose is to measure run-to-run spread; that spread, per metric, becomes the
# anchor-equivalence tolerance used everywhere downstream.

set -euo pipefail
source "$HOME/.bashrc" 2>/dev/null || true
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_job_common.sh

CELL="clean_repeat_a"
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_clean_eval_body.sh
