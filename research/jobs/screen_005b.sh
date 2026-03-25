#!/bin/bash
#SBATCH --job-name=rac_screen_005b
#SBATCH --nodelist=livenode03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/screen_005b.out
#SBATCH --error=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/screen_005b.err

source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh
conda activate racformerfix
cd /srv/nfs/shared/gnmp/RaCFormer

python val.py --config configs/racformer_mini_exp005b_ensemble6.py --weights checkpoints/racformer_r50_f8.pth
