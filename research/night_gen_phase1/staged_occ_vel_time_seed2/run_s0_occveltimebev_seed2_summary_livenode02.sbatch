#!/bin/bash
#SBATCH --job-name=s0_occveltime_s2_summary
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S0_occveltimebev_seed2/summary_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S0_occveltimebev_seed2/summary_slurm_%j.err

set -euo pipefail

conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/staged_occ_vel_time_seed2/summarize_s0_occveltimebev_seed2.py
