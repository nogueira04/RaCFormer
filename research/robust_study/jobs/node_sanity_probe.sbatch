#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --job-name=rs_node_probe
#SBATCH --output=/srv/nfs/shared/gnmp/robust_study_runs/batch2_prep/node_probes/%x_%N_%j.out
#SBATCH --error=/srv/nfs/shared/gnmp/robust_study_runs/batch2_prep/node_probes/%x_%N_%j.err
#
# Per-node sanity probe before sharding mini-screens: slurmd "idle" is not evidence a node can
# run anything (livenode04 registered idle for days while every job died in ~1 s). Submit with
# `-w livenodeNN`. PASS = the last line prints PROBE_OK.

set -euo pipefail
echo "[probe] node=$(hostname) user=$(whoami) date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[probe] uptime: $(uptime)"
nvidia-smi -L
source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh
cd /srv/nfs/shared/gnmp/RaCFormer
echo "[probe] repo_head=$(git rev-parse HEAD)"
test -d data/nuscenes && echo "[probe] data/nuscenes reachable"
test -f checkpoints/racformer_r50_f8.pth && echo "[probe] weights reachable"
TMPF="/srv/nfs/shared/gnmp/robust_study_runs/batch2_prep/node_probes/write_test_$(hostname)"
date -u +%s > "$TMPF" && rm "$TMPF" && echo "[probe] NFS run tree writable"
conda run -n racformerfix --no-capture-output python -u -c "
import torch, mmcv, mmdet3d
print('[probe] torch', torch.__version__, 'cuda_available', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA not available on this node'
print('[probe] device:', torch.cuda.get_device_name(0))
x = (torch.randn(512, 512, device='cuda') @ torch.randn(512, 512, device='cuda')).sum().item()
print('[probe] cuda matmul ok:', x == x)
"
echo "PROBE_OK $(hostname)"
