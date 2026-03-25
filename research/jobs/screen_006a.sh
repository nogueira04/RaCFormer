#!/bin/bash
#SBATCH --job-name=rac_screen_006a
#SBATCH --nodelist=livenode03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/screen_006a.out
#SBATCH --error=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/screen_006a.err

source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh
conda activate racformerfix
cd /srv/nfs/shared/gnmp/RaCFormer

python val.py --config configs/racformer_mini_exp006a_boxavg3.py --weights checkpoints/racformer_r50_f8.pth
