#!/bin/bash
#SBATCH --job-name=s0_rcsoccbev_summary
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S0_rcsoccbev/summary_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S0_rcsoccbev/summary_slurm_%j.err

set -euo pipefail

conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/staged_rcs_occ_residual/summarize_s0_rcsoccbev.py
