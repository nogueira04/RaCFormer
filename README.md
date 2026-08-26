<div align="center">
<h1>RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion (CVPR 2025)</h1>

Xiaomeng Chu, Jiajun Deng, Guoliang You, Yifan Duan, Houqiang Li, Yanyong Zhang

<a href="https://arxiv.org/abs/2412.12725"><img src="https://img.shields.io/badge/arXiv-2412.12725-b31b1b" alt="arXiv"></a>
<a href="https://drive.google.com/file/d/10Ky3lQWC2MLkQCpY81Jz5yxd4xWF8tAq/view?usp=sharing" target="_blank"><img src="https://img.shields.io/badge/Checkpoint-Orange" alt="checkpoint"></a>
</div>

```bibtex
@inproceedings{chu2025racformer,
  title={RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion},
  author={Chu, Xiaomeng and Deng, Jiajun and You, Guoliang and Duan, Yifan and Li, Houqiang and Zhang, Yanyong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={17081--17091},
  year={2025}
}
```

## Overview

This repository is an official implementation of [RaCFormer](https://openaccess.thecvf.com/content/CVPR2025/html/Chu_RaCFormer_Towards_High-Quality_3D_Object_Detection_via_Query-based_Radar-Camera_Fusion_CVPR_2025_paper.html),
an innovative query-based 3D object detection method through cross-perspective radar-camera fusion.

<div style="text-align: center;">
    <img src="arch.jpg" alt="Dialogue_Teaser" width=100% >
</div>

**This checkout is the SEGCOM/LIVE research fork.** It carries additional configs, evaluators and
experiment tooling on top of upstream; the commands below are the ones that actually work on the
`livecluster` SLURM cluster. `CLAUDE.md` is the research-state entry point (baseline numbers,
verified/rejected hypotheses, code locations); `research-notes/racformer/` holds the experiment
ledger.

## Cluster layout

| Thing | Where |
|---|---|
| Repo (shared NFS checkout) | `/srv/nfs/shared/gnmp/RaCFormer` |
| Conda installation | `/srv/nfs/shared/gnmp/miniconda3` |
| Environment name | `racformerfix` |
| nuScenes | `data/nuscenes` -> `/mnt/nfs/shared/nuscenes/` |
| SLURM partition | `livecluster` (`livenode01`–`livenode04`) |
| GPU per node | 1× RTX 4090, 24 GB |

Because each node has exactly **one** GPU, upstream's `torchrun --nproc_per_node 8` does not apply
here — use `--nproc_per_node 1`, or just run `python train.py` directly.

## Environment

The environment is captured in `environment.yml` (Python 3.8.20, PyTorch 2.0.0 + CUDA 11.8,
mmdet 2.28.2, mmdet3d 1.0.0rc6, mmsegmentation 0.30.0, flash-attn 2.6.3).

### 1. Create the conda environment

```bash
source /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh

# Recreate under the name recorded in the file (racformerfix):
conda env create -f environment.yml

# ...or under a different name / prefix, e.g. to test a change without touching the shared env:
conda env create -f environment.yml -n racformer-dev
```

`environment.yml` ends with a `prefix:` line pointing at the shared miniconda; conda ignores it when
`-n`/`-p` is given, so pass one explicitly if you do not want the shared location.

### 2. Install the in-repo MMCV (not captured by environment.yml)

`mmcv-full==1.6.0` is installed **editable from the vendored `mmcv/` source tree**, so it does not
appear in `environment.yml`'s pip section. It has to be built by hand:

```bash
conda activate racformerfix
cd mmcv
MMCV_WITH_OPS=1 pip install -e .
cd ..
```

If the build fails at link time with an invalid `-Wl,--sysroot=/` argument, apply the recorded
workaround from the `chore/mmcv-build-linker-workaround` branch
(`third_party_patches/mmcv_setup_sysroot_linker_workaround.patch`) inside the `mmcv/` checkout and
rebuild.

### 3. Compile the RaCFormer CUDA extensions

```bash
cd models/csrc
python setup.py build_ext --inplace
cd ../..
```

`models/csrc/setup.py` already filters the same bad sysroot linker flag, so this step needs no patch.

### 4. Verify

```bash
conda run -n racformerfix python -c "
import torch, mmcv, mmdet, mmdet3d
import mmcv._ext, models.csrc.wrapper
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print('mmcv', mmcv.__version__, mmcv.__file__)
"
```

Expected: `2.0.0 11.8 True` and mmcv resolving to `<repo>/mmcv/mmcv/__init__.py`.

### Optional: faster JPEG decoding

Upstream recommends turbojpeg + pillow-simd. **Neither is installed here** — `loaders/pipelines/loading.py`
wraps `mmcv.use_backend('turbojpeg')` in a `try/except ImportError` and falls back to `cv2`, so the
pipeline runs correctly without them. Installing them only changes data-loading throughput:

```bash
sudo apt-get install -y libturbojpeg
pip install pyturbojpeg
pip uninstall pillow && pip install pillow-simd==9.0.0.post1
```

## Prepare dataset

On the cluster this is already done — `data/nuscenes` is a symlink to the shared copy and the
`nuscenes_infos_*_sweep.pkl` files sit at the repo root.

From scratch:

1. Download nuScenes from [nuscenes.org](https://www.nuscenes.org/nuscenes) into `data/nuscenes`.
2. Download the generated info files from [Google Drive](https://drive.google.com/drive/folders/1Tec0I7tgJKF-w1_vVAScJ0wPek2YT28u?usp=sharing),
   or generate them with `python tools/gen_sweep_info.py`.

```
data/nuscenes
├── maps
├── nuscenes_infos_test_sweep.pkl
├── nuscenes_infos_train_sweep.pkl
├── samples
├── sweeps
├── v1.0-test
└── v1.0-trainval
```

## Training

The ImageNet/nuImages-pretrained ResNet-50 is already in place:

```
pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth
```

`train.py` takes `--config` (required) and an optional `--override k=v ...`. Output goes to the
`work_dir` declared inside the config.

### Interactive (one node, one GPU)

```bash
conda run -n racformerfix --no-capture-output python -u train.py \
  --config configs/racformer_r50_nuimg_704x256_f8.py
```

### Under SLURM (the normal path)

Copy `slurm_train.sh` or one of the templates in `scripts/slurm/night_generation/` and edit the
config + output paths. The pattern every current job uses:

```bash
#!/bin/bash
#SBATCH --job-name=racformer_myexp
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
mkdir -p slurm_logs
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

conda run -n racformerfix --no-capture-output python -u train.py \
  --config configs/racformer_train2k_day_research.py
```

```bash
sbatch scripts/slurm/<your_job>.sbatch
squeue -u "$USER"
tail -f slurm_logs/<jobname>_<jobid>.out
```

`--no-capture-output` matters: without it `conda run` buffers all stdout until the process exits, so
a multi-hour job shows nothing until it is over.

Pick an idle node with `sinfo` before pinning `--nodelist`.

### Multi-node

`slurm_train.sh` keeps the multi-node NCCL setup (`MASTER_ADDR` from `scontrol`, `NCCL_IB_DISABLE=1`,
`NCCL_SOCKET_IFNAME=^lo,docker,cali`) and launches with `srun python3 train.py`. Raise `--nodes` and
`--ntasks-per-node` there; each task still gets one GPU.

## Evaluation

Canonical checkpoint: `checkpoints/racformer_r50_f8.pth`.

### Standard validation

```bash
conda run -n racformerfix python -u val.py \
  --config configs/racformer_r50_nuimg_704x256_f8.py \
  --weights checkpoints/racformer_r50_f8.pth \
  --output_dir outputs/eval/baseline
```

`val.py` also accepts `--batch_size`, `--distance_bins`, `--max_vis_samples`, and the
`--local_rank`/`--world_size` pair used under `torchrun`. Multi-GPU on one node is not useful here
(one GPU per node), so run it single-process.

### Day / night / rain breakdown

```bash
conda run -n racformerfix python -u tools/eval_night_gap.py \
  --config configs/racformer_r50_nuimg_704x256_f8.py \
  --weights checkpoints/racformer_r50_f8.pth \
  --conditions all,day,night,rain \
  --output_dir outputs/night_eval/baseline
```

### Condition-split evaluator (used by the robustness study)

This is the driver the robustness cells invoke; it writes both a nuScenes submission JSON and
`eval_by_condition.json`.

```bash
conda run -n racformerfix --no-capture-output python -u \
  research/night_gen_phase1/eval_by_condition.py \
  --config configs/racformer_eval_fullval_research.py \
  --weights checkpoints/racformer_r50_f8.pth \
  --full-val \
  --out-dir outputs/eval/fullval_run
```

Pass `--full-val` explicitly rather than relying on a config to bake it in.

## Robustness study

`research/robust_study/` holds the sensor-failure evaluation campaign: corruption operators, cell
job scripts, frozen thresholds, and an independent nuScenes-devkit cross-check that validates the
evaluation wrapper against the official `DetectionEval`. Read `research/robust_study/README.md`
before running anything there — the cells enforce a clean-tree / pinned-commit discipline and a run
directory without a `_COMPLETE` marker is not a result.

## Analysis tools

| Script | Purpose |
|---|---|
| `tools/eval_night_gap.py` | Full-val inference, then per-condition (day/night/rain) nuScenes evaluation |
| `tools/analyze_error_types.py` | FP/FN decomposition comparing baseline vs zeroed-image predictions |
| `tools/analyze_feature_quality.py` | Whether image features degrade at night while radar stays stable |
| `tools/analyze_modality_contribution.py` | Image/radar/LSS contribution to the fused representation |
| `tools/visualize_night_augmentation.py` | Side-by-side original vs night-augmented images |
| `tools/gen_sweep_info.py` | Regenerate the nuScenes sweep info pkls |
| `tools/visual.py`, `tools/render.py` | nuScenes-devkit-derived detection rendering |
| `visualize_results.py`, `create_video.py` | Result visualisation and videos |

## Configs

`configs/racformer_r50_nuimg_704x256_f8.py` is the upstream baseline. Everything else is a fork
config:

- `*_dropout*.py` — modality-dropout variants
- `*_nightaug.py`, `*_simnight_research.py` — image-level night simulation
- `*_genaug_*_research.py` — generative night augmentation (seed / ratio / loss-weight sweeps)
- `*_dualviewdistill_*_research.py` — Branch G distillation
- `*_contrelqfusion_*.py`, `*_conditionfusion_*.py` — condition/reliability-aware fusion
- `racformer_eval_fullval*_research.py` — evaluation-only configs for the robustness study
- `*_3cam_3rad*.py` — reduced 3-camera / 3-radar sensor sets

## Acknowledgements

Many thanks to these excellent open-source projects:

* 3D Detection: [SparseBEV](https://github.com/MCG-NJU/SparseBEV), [PETR v2](https://github.com/megvii-research/PETR), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
* Codebase: [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
