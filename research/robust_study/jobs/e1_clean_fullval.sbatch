#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --time=06:00:00
#SBATCH --job-name=rs_e1_clean
#SBATCH --output=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.out
#SBATCH --error=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.err
#
# Reference cell: one clean full-val evaluation of the canonical checkpoint. This is the artifact
# the independent devkit cross-check reads, and the first of the three clean repeats whose spread
# defines the anchor-equivalence tolerances.

set -euo pipefail
source "$HOME/.bashrc" 2>/dev/null || true
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_job_common.sh

CELL="clean"
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_clean_eval_body.sh
