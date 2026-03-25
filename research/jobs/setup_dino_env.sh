#!/bin/bash
#SBATCH --job-name=dino_env_setup
#SBATCH --nodelist=livenode03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/setup_dino_env.out
#SBATCH --error=/srv/nfs/shared/gnmp/RaCFormer/research/outputs/setup_dino_env.err

echo "=== Installing PyTorch ==="
/srv/nfs/shared/gnmp/miniconda3/envs/dino_extract/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing transformers and deps ==="
/srv/nfs/shared/gnmp/miniconda3/envs/dino_extract/bin/pip install transformers pillow

echo "=== Verifying ==="
/srv/nfs/shared/gnmp/miniconda3/envs/dino_extract/bin/python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
from transformers import AutoModel
print('Transformers OK')
from PIL import Image
print('Pillow OK')
print('=== ALL DONE ===')
"
