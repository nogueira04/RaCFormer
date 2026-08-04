#!/bin/bash
#SBATCH --job-name=s0_occtimebev_smoke
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S0_occtimebev/smoke_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S0_occtimebev/smoke_slurm_%j.err

set -euo pipefail

mkdir -p research/night_gen_phase1/results/S0_occtimebev

conda run -n racformerfix --no-capture-output python - <<'PY'
import torch
from mmcv import Config
from mmdet3d.models import build_model
import models  # noqa: F401
import loaders  # noqa: F401

cfg = Config.fromfile("configs/racformer_train2k_day_occtimebev_research.py")
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
assert getattr(model, "radar_rcs_bev_residual", None) is not None
branch = model.radar_rcs_bev_residual
assert branch.output_shape == (128, 128)
assert branch.rcs_index == 3
assert branch.rcs_scale == 1000000.0
assert branch.extra_indices == (6,)
assert branch.stat_indices == (3, 6)
assert branch.stat_scales == (1000000.0, 1.0)
assert branch.residual[0].in_channels == 3
model.init_weights()
assert torch.count_nonzero(branch.residual[-1].weight).item() == 0
assert torch.count_nonzero(branch.residual[-1].bias).item() == 0
state_keys = [key for key in model.state_dict() if key.startswith("radar_rcs_bev_residual.")]
assert state_keys
assert torch.cuda.is_available()
device = torch.device("cuda")
branch = branch.to(device)
radar_bev = torch.zeros(2, model.pts_bbox_head.embed_dims, 128, 128, dtype=torch.float16, device=device)
voxels = torch.zeros(2, 10, 7, dtype=torch.float32, device=device)
voxels[:, :, 3] = torch.tensor([[12.0] * 10, [-8.0] * 10], device=device)
voxels[:, :, 6] = torch.tensor([[0.25] * 10, [0.75] * 10], device=device)
num_points = torch.tensor([10, 10], dtype=torch.int32, device=device)
coors = torch.tensor([[0, 0, 10, 12], [1, 0, 20, 22]], dtype=torch.int32, device=device)
with torch.no_grad():
    out = branch(radar_bev, voxels, num_points, coors, batch_size=2)
assert out.dtype == radar_bev.dtype
assert out.shape == radar_bev.shape
assert torch.equal(out, radar_bev)
print("radar_occtime_bev_residual", branch.output_shape, branch.stat_indices, branch.stat_scales)
print("state_keys", len(state_keys))
print("half_forward_zero_init", True)
PY
