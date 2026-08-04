#!/usr/bin/env bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --time=00:30:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/reports/t10_seed20260502_ratio18p75_validate/slurm_%j.out
#SBATCH --error=research/night_gen_phase1/reports/t10_seed20260502_ratio18p75_validate/slurm_%j.err
#SBATCH --job-name=t10_val_seed20260502_ratio18p75

set -euo pipefail
mkdir -p research/night_gen_phase1/reports/t10_seed20260502_ratio18p75_validate

conda run -n racformerfix --no-capture-output python -u research/night_gen_phase1/validate_phase1.py \
  --generated-manifest research/night_gen_phase1/manifests/phase1_t10_seed20260502_ratio18p75_generated.json \
  --inputs-root /srv/nfs/shared/shared/nuscenes/ \
  --gt-boxes-pkl /srv/nfs/shared/gnmp/RaCFormer/nuscenes_infos_train_sweep.pkl \
  --reports-dir research/night_gen_phase1/reports/t10_seed20260502_ratio18p75_validate \
  --abs-delta-threshold 0.15 \
  --lum-kl-threshold 0.5
