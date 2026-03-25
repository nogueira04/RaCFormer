#!/bin/bash
#SBATCH --job-name=rac_screen_004
#SBATCH --nodelist=livenode03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/screen_004.out
#SBATCH --error=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/screen_004.err

source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh
conda activate racformerfix
cd /srv/nfs/shared/gnmp/RaCFormer

python research/experiments/exp_004_radar_clocs.py
