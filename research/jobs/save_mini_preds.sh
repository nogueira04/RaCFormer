#!/bin/bash
#SBATCH --job-name=rac_save_preds
#SBATCH --nodelist=livenode03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/save_mini_preds.out
#SBATCH --error=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/save_mini_preds.err

source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh
conda activate racformerfix
cd /srv/nfs/shared/gnmp/RaCFormer

mkdir -p research/outputs/mini_preds

python val.py \
    --config configs/racformer_mini_research.py \
    --weights checkpoints/racformer_r50_f8.pth \
    --output_dir research/outputs/mini_preds

echo "=== PREDICTIONS SAVED ==="
ls -la research/outputs/mini_preds/
