#!/bin/bash
#SBATCH --job-name=rac_reeval_009
#SBATCH --nodelist=livenode03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/reeval_009.out
#SBATCH --error=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/reeval_009.err

source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh
conda activate racformerfix
cd /srv/nfs/shared/gnmp/RaCFormer

python research/experiments/reeval_from_torch.py \
    --predictions research/outputs/exp_009/modified_predictions.pt \
    --config configs/racformer_mini_research.py
