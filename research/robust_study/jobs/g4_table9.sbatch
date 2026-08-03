#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --time=06:00:00
#SBATCH --job-name=rs_g4_table9
#SBATCH --output=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.out
#SBATCH --error=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.err
#
# G4 -- pre-normalization input zeroing (published-semantics reading).
#
# Zeroes the raw image tensor at the entry of extract_feat (models/racformer.py:179): zero in
# sensor units, i.e. a black frame, which the frozen normalization at models/racformer.py:211-212
# maps to -mean/std. This is the sensor-literal reading of the published sentence 'we
# systematically exclude either image or radar inputs'. See gate_b/MANIFEST.md for the
# compatibility matrix and for why this cell is annotated not comparable to published Table 9.

set -euo pipefail
source "$HOME/.bashrc" 2>/dev/null || true
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_job_common.sh
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_gate_b_common.sh

CELL="g4_table9"
REMOVAL="table9"
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_gate_b_cell_body.sh
