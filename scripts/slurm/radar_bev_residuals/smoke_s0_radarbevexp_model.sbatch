#!/bin/bash
#SBATCH --job-name=s0_radarbevexp_smoke
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S0_radarbevexp/smoke_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S0_radarbevexp/smoke_slurm_%j.err

set -euo pipefail

mkdir -p research/night_gen_phase1/results/S0_radarbevexp

echo "HOST $(hostname)"
echo "START $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

conda run -n racformerfix --no-capture-output python - <<'PY'
from mmcv import Config
from mmdet3d.models import build_model
import models  # noqa: F401
import loaders  # noqa: F401

cfg = Config.fromfile("configs/racformer_train2k_day_radarbevexp_research.py")
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
model.init_weights()
assert getattr(model, "radar_bev_expansion", None) is not None
expansion = model.radar_bev_expansion
assert tuple(expansion.kernel_sizes) == (3, 5, 7)
assert float(expansion.residual_proj.weight.abs().sum()) == 0.0
assert float(expansion.residual_proj.bias.abs().sum()) == 0.0
assert not any(key.startswith("radar_bev_expansion.kernel_") for key in model.state_dict())
print("config", "configs/racformer_train2k_day_radarbevexp_research.py")
print("radar_bev_expansion", expansion.kernel_sizes)
print("zero_init_residual", True)
print("nonpersistent_gaussian_kernels", True)
PY

echo "END $(date -Is)"
