#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S3_seed20260425_ratio18p75_w05/summary_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S3_seed20260425_ratio18p75_w05/summary_slurm_%j.err
#SBATCH --job-name=w05_summary

set -euo pipefail

conda run -n racformerfix --no-capture-output python -u research/night_gen_phase1/summarize_w05.py
