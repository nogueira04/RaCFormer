#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --time=00:20:00
#SBATCH --job-name=rs_e0_smoke
#SBATCH --output=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.out
#SBATCH --error=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.err
#
# Smoke cell: imports, CUDA visibility, versions, devkit module paths. No model is built and no
# inference is run — this cell exists to prove the environment before any GPU-hours are spent.

set -euo pipefail
source "$HOME/.bashrc" 2>/dev/null || true
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_job_common.sh

rs_init "eval_oracle" "e0_smoke"

RS_EXACT_COMMAND="conda run -n racformerfix $CONDA_RUN_FLAGS python -u -c '<smoke imports>'"

echo "=== nvidia-smi ==="
nvidia-smi || echo "[job] nvidia-smi unavailable"

echo "=== evaluation env: racformerfix ==="
conda run -n "$ENV_NAME" $CONDA_RUN_FLAGS python -u -c "
import sys, os
print('python           :', sys.version.split()[0])
print('executable       :', sys.executable)
print('PYTHONHASHSEED   :', os.environ.get('PYTHONHASHSEED'))
import torch
print('torch            :', torch.__version__)
print('torch.version.cuda:', torch.version.cuda)
print('cudnn            :', torch.backends.cudnn.version())
print('cuda_available   :', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA required; this study never runs inference on CPU.'
print('device_count     :', torch.cuda.device_count())
print('current_device   :', torch.cuda.current_device())
print('device_name      :', torch.cuda.get_device_name(0))
print('capability       :', torch.cuda.get_device_capability(0))
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
import mmcv, mmdet, mmdet3d, numpy
print('mmcv             :', mmcv.__version__)
print('mmdet            :', mmdet.__version__)
print('mmdet3d          :', mmdet3d.__version__)
print('numpy            :', numpy.__version__)
import nuscenes
from nuscenes.eval.detection.config import config_factory
import importlib.metadata as md
print('nuscenes_devkit  :', md.version('nuscenes-devkit'))
print('devkit __file__  :', nuscenes.__file__)
cfg = config_factory('detection_cvpr_2019')
print('eval dist_th_tp  :', cfg.dist_th_tp)
print('eval dist_ths    :', cfg.dist_ths)
print('max_boxes/sample :', cfg.max_boxes_per_sample)
from nuscenes.utils.splits import val as val_scenes
print('val scenes       :', len(val_scenes))
sys.path.insert(0, '$REPO')
import models, loaders
print('repo modules     : models+loaders imported OK')
"

echo "=== cross-check env: isolated devkit venv ==="
DEVKIT_VENV="${DEVKIT_VENV:-$HOME/venvs/nusc-devkit-check}"
env -u PYTHONPATH PYTHONNOUSERSITE=1 "$DEVKIT_VENV/bin/python" -I -u -c "
import sys, os, importlib.metadata as md
import nuscenes
print('python           :', sys.version.split()[0])
print('executable       :', sys.executable)
print('isolated         :', bool(sys.flags.isolated))
print('no_user_site     :', bool(sys.flags.no_user_site))
print('PYTHONPATH       :', os.environ.get('PYTHONPATH'))
print('nuscenes_devkit  :', md.version('nuscenes-devkit'))
print('devkit __file__  :', nuscenes.__file__)
from nuscenes.eval.detection.evaluate import DetectionEval
print('DetectionEval    : importable')
"

echo "=== crosscheck script self-check ==="
"$TOOLS/run_devkit_crosscheck.sh" --help > /dev/null && echo "devkit_crosscheck.py --help OK"

rs_finalize "smoke checks passed (imports + CUDA device + versions + devkit paths)"
echo "[job] E0 smoke complete."
