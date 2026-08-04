#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/full_3cam3rad_epoch36_compare_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/full_3cam3rad_epoch36_compare_slurm_%j.err
#SBATCH --job-name=full_3cam3rad_compare

set -euo pipefail

conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/staged_existing_3cam3rad_eval/summarize_full_3cam3rad_eval.py
