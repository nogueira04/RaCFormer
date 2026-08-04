#!/bin/bash
#SBATCH --job-name=radar_bc_probe
#SBATCH --nodelist=livenode03
#SBATCH --partition=livecluster
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/srv/nfs/shared/gnmp/robust_study_runs/batch2_prep/radar_bc_probe_20260803T200852Z.log
set -euo pipefail
source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh
cd /srv/nfs/shared/gnmp/RaCFormer
git rev-parse HEAD
conda run -n racformerfix --no-capture-output python -u research/robust_study/probes/probe_radar_bc.py --out-dir /srv/nfs/shared/gnmp/robust_study_runs/batch2_prep/radar_bc_probe_20260803T200852Z
