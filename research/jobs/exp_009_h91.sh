#!/bin/bash
#SBATCH --job-name=rac_exp009_h91
#SBATCH --nodelist=livenode03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/exp_009_h91.out
#SBATCH --error=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/exp_009_h91.err

cd /srv/nfs/shared/gnmp/RaCFormer
: "${HF_TOKEN:?set HF_TOKEN in the environment before running (removed from repo)}"

echo "=== STAGE 1: DINOv3 reclassification (dino_extract env) ==="
source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh
conda activate dino_extract

python research/experiments/exp_009_dino_classify.py \
    --reclass_threshold 0.15 \
    --min_dino_sim 0.3

echo ""
echo "=== STAGE 2: Re-evaluation (racformerfix env) ==="
conda activate racformerfix

python research/experiments/reeval_predictions.py \
    --predictions research/outputs/exp_009/modified_predictions.pkl \
    --config configs/racformer_mini_research.py

echo "=== DONE ==="
