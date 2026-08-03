#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --time=06:00:00
#SBATCH --job-name=rs_g3_input_repeat
#SBATCH --output=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.out
#SBATCH --error=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.err
#
# G3 -- post-normalization input zeroing.
#
# This is the REPEAT of the cell above it. The decision procedure resolves every predicate by the
# pair rule -- each cell is measured twice, and a pair that straddles a cutoff resolves UNKNOWN
# rather than being rounded to TRUE or FALSE -- so the repeat is not optional and is not a
# spare. It is byte-identical to the original except for $CELL and the SLURM job name.
#
# Zeroes the normalized image tensor at the entry of extract_img_feat (models/racformer.py:107):
# zero in the network's own units. This is the in-network 'modality dropout' reading, the
# lineage the tracked dropout configs point at.

set -euo pipefail
source "$HOME/.bashrc" 2>/dev/null || true
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_job_common.sh
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_gate_b_common.sh

CELL="g3_input_repeat"
REMOVAL="input"
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_gate_b_cell_body.sh
