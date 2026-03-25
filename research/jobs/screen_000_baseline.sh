#!/bin/bash
#SBATCH --job-name=rac_screen_000
#SBATCH --nodelist=livenode03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/screen_000_baseline.out
#SBATCH --error=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/screen_000_baseline.err

source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh
conda activate racformerfix
cd /srv/nfs/shared/gnmp/RaCFormer

echo "START: $(date)"
echo "NODE: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python val.py --config configs/racformer_mini_research.py --weights checkpoints/racformer_r50_f8.pth

echo "END: $(date)"
