#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --time=06:00:00
#SBATCH --job-name=rs_g2_phase0
#SBATCH --output=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.out
#SBATCH --error=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.err
#
# G2 -- feature-level camera mask (phase 0 lineage).
#
# Replicates research/paper_goal_20260520/tools/phase0_sensor_baseline.py:87-89: extract_feat
# runs normally and its image-branch outputs -- every FPN level and the camera-BEV stack --
# are zeroed afterwards. This is the implementation behind the historical mAP ~ 1.29e-6
# collapse observation that GATE-B exists to re-test.

set -euo pipefail
source "$HOME/.bashrc" 2>/dev/null || true
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_job_common.sh
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_gate_b_common.sh

CELL="g2_phase0"
REMOVAL="phase0"
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_gate_b_cell_body.sh
