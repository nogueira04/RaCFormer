#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/loss_weight_compare_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/loss_weight_compare_slurm_%j.err
#SBATCH --job-name=loss_weight_compare

set -euo pipefail

conda run -n racformerfix --no-capture-output python -u research/night_gen_phase1/compare_loss_weight_sweep.py
cat research/night_gen_phase1/results/loss_weight_sweep_summary.md
