#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --time=06:00:00
#SBATCH --job-name=rs_g1_none_repeat
#SBATCH --output=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.out
#SBATCH --error=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.err
#
# G1 -- no-op control.
#
# This is the REPEAT of the cell above it. The decision procedure resolves every predicate by the
# pair rule -- each cell is measured twice, and a pair that straddles a cutoff resolves UNKNOWN
# rather than being rounded to TRUE or FALSE -- so the repeat is not optional and is not a
# spare. It is byte-identical to the original except for $CELL and the SLURM job name.
#
# Installs the same instrumentation as every other cell and touches no tensor. Its purpose is
# to prove the runner is transparent: the submission must reproduce the clean E1 reference,
# and the probe digests must be bit-identical pre and post. A difference here means the
# instrumented runner is broken, and GATE-B halts before any conclusion is drawn from G2-G4.

set -euo pipefail
source "$HOME/.bashrc" 2>/dev/null || true
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_job_common.sh
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_gate_b_common.sh

CELL="g1_none_repeat"
REMOVAL="none"
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_gate_b_cell_body.sh
