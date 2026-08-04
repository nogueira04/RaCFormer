#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --time=06:00:00
#SBATCH --job-name=racf_t5_eval
#SBATCH --output=/srv/nfs/shared/gnmp/RaCFormer/research/night_gen_phase1/reports/eval_by_condition/slurm_%j.out
#SBATCH --error=/srv/nfs/shared/gnmp/RaCFormer/research/night_gen_phase1/reports/eval_by_condition/slurm_%j.err

set -euo pipefail
source "$HOME/.bashrc" 2>/dev/null || true

conda run -n racformerfix --no-capture-output python -u research/night_gen_phase1/eval_by_condition.py \
  --config configs/racformer_eval_fullval_research.py \
  --weights checkpoints/racformer_r50_f8.pth \
  --out-dir research/night_gen_phase1/reports/eval_by_condition \
  --full-val
