#!/bin/bash
#SBATCH --job-name=dino_prototypes
#SBATCH --nodelist=livenode03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/compute_prototypes.out
#SBATCH --error=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/compute_prototypes.err

source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh
conda activate dino_extract
cd /srv/nfs/shared/gnmp/RaCFormer

export HF_TOKEN=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

python research/experiments/compute_prototypes.py
