#!/bin/bash
#SBATCH --job-name=rac_exp010
#SBATCH --nodelist=livenode03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/exp_010.out
#SBATCH --error=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/exp_010.err

cd /srv/nfs/shared/gnmp/RaCFormer
source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh

echo "=== STAGE 1: Extract DINOv3 scores (dino_extract) ==="
conda activate dino_extract
export HF_TOKEN=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
python research/experiments/exp_010_dino_scores.py

echo ""
echo "=== STAGE 2: Strategy sweep (racformerfix) ==="
conda activate racformerfix
python research/experiments/exp_010_sweep.py

echo "=== DONE ==="
