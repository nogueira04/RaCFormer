#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --time=01:00:00
#SBATCH --job-name=racf_t5_post
#SBATCH --output=/srv/nfs/shared/gnmp/RaCFormer/research/night_gen_phase1/reports/eval_by_condition_post/slurm_%j.out
#SBATCH --error=/srv/nfs/shared/gnmp/RaCFormer/research/night_gen_phase1/reports/eval_by_condition_post/slurm_%j.err

set -euo pipefail
source "$HOME/.bashrc" 2>/dev/null || true

conda run -n racformerfix --no-capture-output python -u research/night_gen_phase1/eval_per_condition_post.py \
  --pred research/night_gen_phase1/reports/eval_by_condition/submission_overall/pts_bbox/results_nusc.json \
  --val-pkl /srv/nfs/shared/gnmp/RaCFormer/nuscenes_infos_val_sweep.pkl \
  --reference-summary research/night_gen_phase1/reports/eval_by_condition/submission_overall/pts_bbox/metrics_summary.json \
  --out-dir research/night_gen_phase1/reports/eval_by_condition_post \
  --dataroot data/nuscenes/ \
  --version v1.0-trainval \
  --tolerance 0.001
