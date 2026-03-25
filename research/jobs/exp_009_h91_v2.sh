#!/bin/bash
#SBATCH --job-name=rac_exp009_v2
#SBATCH --nodelist=livenode03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/exp_009_h91_v2.out
#SBATCH --error=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/exp_009_h91_v2.err

cd /srv/nfs/shared/gnmp/RaCFormer
source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh

echo "=== STAGE 0: Convert predictions (racformerfix) ==="
conda activate racformerfix
python research/experiments/convert_predictions.py \
    --input research/outputs/mini_preds/predictions.pkl \
    --output research/outputs/mini_preds/predictions_simple.pkl

echo ""
echo "=== STAGE 1: DINOv3 reclassification (dino_extract) ==="
conda activate dino_extract
export HF_TOKEN=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

python research/experiments/exp_009_dino_classify.py \
    --predictions research/outputs/mini_preds/predictions_simple.pkl \
    --reclass_threshold 0.15 \
    --min_dino_sim 0.3

echo ""
echo "=== STAGE 2: Re-evaluation (racformerfix) ==="
conda activate racformerfix
python research/experiments/reeval_predictions.py \
    --predictions research/outputs/exp_009/modified_predictions.pkl \
    --config configs/racformer_mini_research.py

echo "=== DONE ==="
