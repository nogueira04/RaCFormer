#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=research/night_gen_phase1/results/S5_conditionfusion/smoke_slurm_%j.out
#SBATCH --error=research/night_gen_phase1/results/S5_conditionfusion/smoke_slurm_%j.err
#SBATCH --job-name=s5_condition_smoke

set -euo pipefail

conda run -n racformerfix --no-capture-output python -u - <<'PY'
from mmcv import Config
from mmdet3d.datasets import build_dataset
import loaders  # noqa: F401
import models  # noqa: F401

cfg = Config.fromfile('configs/racformer_train2k_mixed_conditionfusion_research.py')
dataset = build_dataset(cfg.data.train)
print('dataset_len', len(dataset))
print('first_info_condition', dataset.data_infos[0].get('scene_condition'))
sample = dataset[0]
meta = sample['img_metas'].data
print('meta_has_scene_condition', 'scene_condition' in meta)
print('meta_scene_condition', meta.get('scene_condition'))
print('image_shape', tuple(sample['img'].data.shape))
PY
