#!/usr/bin/env bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --time=00:30:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer/research/night_gen_phase1
#SBATCH --output=reports/t8_validate/slurm_%j.out
#SBATCH --error=reports/t8_validate/slurm_%j.err
#SBATCH --job-name=t8_validate

set -euo pipefail
mkdir -p reports/t8_validate

conda run -n racformerfix --no-capture-output python -u validate_phase1.py \
  --generated-manifest manifests/phase1_t8_generated.json \
  --inputs-root /srv/nfs/shared/shared/nuscenes/ \
  --gt-boxes-pkl /srv/nfs/shared/gnmp/RaCFormer/nuscenes_infos_train_sweep.pkl \
  --reports-dir reports/t8_validate \
  --abs-delta-threshold 0.15 \
  --lum-kl-threshold 0.5
