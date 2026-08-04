#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --time=01:15:00
#SBATCH --mem=64G
#SBATCH --job-name=rs_b2_cell
#SBATCH --output=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.out
#SBATCH --error=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.err
#
# One Batch-2 corruption cell under the full exec-oracle discipline. ONE template for all 29
# cells -- the cell is injected at submit time, so the code path is identical by construction
# (the GATE-B rationale) and the exact cell is pinned by the config file the provenance hashes:
#
#   sbatch --export=ALL,CELL=radar_dropout_p25_s0 --job-name=rs_b2_radar_dropout_p25_s0 \
#     research/robust_study/jobs/batch2_cell.sbatch
#
# Override the node with `-w livenodeNN` at submit time only after that node has passed a
# same-day sanity probe (account + GPU + env + data); slurmd "idle" alone is not evidence a
# node can run anything (the livenode04 trap).
#
# Time limit: measured full-val cell wall time is 25-31 min (n=3, median 30m34s); 75 min
# covers the slowest observed plus model load without masking a wedge.

set -euo pipefail
source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh

JOBS=/srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs
source "$JOBS/_job_common.sh"
source "$JOBS/_gate_b_common.sh"
source "$JOBS/_batch2_cell_body.sh"
