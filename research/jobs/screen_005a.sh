#!/bin/bash
#SBATCH --job-name=rac_screen_005a
#SBATCH --nodelist=livenode03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/screen_005a.out
#SBATCH --error=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/screen_005a.err

source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh
conda activate racformerfix
cd /srv/nfs/shared/gnmp/RaCFormer

python val.py --config configs/racformer_mini_exp005a_ensemble3.py --weights checkpoints/racformer_r50_f8.pth
