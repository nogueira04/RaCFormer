#!/bin/bash
#SBATCH --job-name=s0_occveltime_s2_smoke
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S0_occveltimebev_seed2/smoke_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S0_occveltimebev_seed2/smoke_slurm_%j.err

set -euo pipefail

mkdir -p research/night_gen_phase1/results/S0_occveltimebev_seed2

conda run -n racformerfix --no-capture-output python - <<'PY'
import torch
from mmcv import Config
from mmdet3d.models import build_model
import models  # noqa: F401
import loaders  # noqa: F401

cfg = Config.fromfile("configs/racformer_train2k_day_occveltimebev_seed2_research.py")
assert int(cfg.random_seed) == 2
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
assert getattr(model, "radar_rcs_bev_residual", None) is not None
branch = model.radar_rcs_bev_residual
assert branch.output_shape == (128, 128)
assert branch.rcs_index == 3
assert branch.rcs_scale == 1000000.0
assert branch.extra_indices == (4, 5, 6)
assert branch.stat_indices == (3, 4, 5, 6)
assert branch.stat_scales == (1000000.0, 20.0, 20.0, 1.0)
assert branch.residual[0].in_channels == 5
model.init_weights()
assert torch.count_nonzero(branch.residual[-1].weight).item() == 0
assert torch.count_nonzero(branch.residual[-1].bias).item() == 0
assert torch.cuda.is_available()
print("seed", int(cfg.random_seed))
print("radar_occveltime_bev_residual_seed2", branch.output_shape, branch.stat_indices, branch.stat_scales)
print("half_forward_zero_init", True)
PY
