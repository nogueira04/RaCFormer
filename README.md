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

The setup and run commands below are written for the LIVE `livecluster` SLURM cluster.

## Cluster layout

| Thing | Value |
|---|---|
| SLURM partition | `livecluster` (`livenode01`–`livenode04`) |
| GPU per node | 1× RTX 4090, 24 GB |
| CPUs / RAM per node | 16 CPUs, 128 GB (SLURM sees the full node) |
| nuScenes | `/mnt/nfs/shared/nuscenes` (read-only, shared) |
| Your home | NFS-mounted and visible from **every** node |

Two consequences worth knowing before you start:

- Each node has exactly **one** GPU, so upstream's `torchrun --nproc_per_node 8` does not apply.
  Use `--nproc_per_node 1`, or just run `python train.py` directly.
- SLURM on this cluster has **no GPU GRES configured** (`Gres=(null)`), so do *not* pass
  `--gres=gpu:1` — the job will never be scheduled. Request the node with `--cpus-per-task=16`
  instead, which is one whole node. Keep it to one training job per node; two memory-heavy jobs on
  the same node can drive it into uninterruptible sleep.
- Your home is on shared NFS, so clone once and every node sees the same checkout. There is no need
  to copy anything per-node.

## Setup

Everything below happens inside your own home directory. Nothing requires root or another user's
files.

### 1. Clone

```bash
cd "$HOME"
git clone https://github.com/LIVEUFPE/RaCFormer.git
cd RaCFormer
```

### 2. Install miniconda

The cluster has no system-wide conda and no environment modules, so install your own:

```bash
curl -fsSLo "$HOME/miniconda.sh" \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash "$HOME/miniconda.sh" -b -p "$HOME/miniconda3"
rm "$HOME/miniconda.sh"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda config --set auto_activate_base false
```

Add the `source` line to your `~/.bashrc` so batch jobs can find conda too.

### 3. Create the environment

`environment.yml` pins the whole stack: Python 3.8.20, PyTorch 2.0.0 + CUDA 11.8, mmdet 2.28.2,
mmdet3d 1.0.0rc6, mmsegmentation 0.30.0, flash-attn 2.6.3.

```bash
conda env create -f environment.yml     # creates the env named 'racformerfix'
conda activate racformerfix
```

The file ends with a `prefix:` line left over from where it was exported; conda ignores it whenever
`-n` or `-p` is given, and creating by name puts the env under your own `~/miniconda3/envs`. To use
a different name or location:

```bash
conda env create -f environment.yml -n my-racformer
conda env create -f environment.yml -p "$HOME/envs/racformer"
```

### 4. Build the in-repo MMCV

`mmcv-full==1.6.0` is installed **editable from the vendored `mmcv/` source tree**, which is why it
does not appear in `environment.yml`'s pip section. It has to be built by hand, once per
environment:

```bash
cd mmcv
MMCV_WITH_OPS=1 pip install -e .
cd ..
```

If the build fails at link time on an invalid `-Wl,--sysroot=/` argument, apply the recorded
workaround from the `chore/mmcv-build-linker-workaround` branch
(`third_party_patches/mmcv_setup_sysroot_linker_workaround.patch`) inside the `mmcv/` checkout and
rebuild.

### 5. Compile the CUDA extensions

```bash
cd models/csrc
python setup.py build_ext --inplace
cd ../..
```

`models/csrc/setup.py` already filters the same bad sysroot linker flag, so this step needs no
patch.

### 6. Verify

```bash
python -c "
import torch, mmcv, mmdet, mmdet3d
import mmcv._ext, models.csrc.wrapper
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print('mmcv', mmcv.__version__, mmcv.__file__)
"
```

Expected: `2.0.0 11.8 True`, and mmcv resolving to `<your checkout>/mmcv/mmcv/__init__.py` rather
than to site-packages. If it resolves to site-packages, step 4 did not take.

### Optional: faster JPEG decoding

Upstream recommends turbojpeg and pillow-simd. Neither is in `environment.yml`, and the code does
not need them — `loaders/pipelines/loading.py` wraps `mmcv.use_backend('turbojpeg')` in a
`try/except ImportError` and falls back to `cv2`. Installing them only changes data-loading
throughput:

```bash
pip install pyturbojpeg               # needs libturbojpeg present on the node
pip uninstall pillow && pip install pillow-simd==9.0.0.post1
```

## Prepare dataset

nuScenes already lives on the cluster, read-only. Point the repo at it instead of copying:

```bash
mkdir -p data
ln -s /mnt/nfs/shared/nuscenes data/nuscenes
```

You also need the generated info files. Download them from
[Google Drive](https://drive.google.com/drive/folders/1Tec0I7tgJKF-w1_vVAScJ0wPek2YT28u?usp=sharing),
or build them yourself:

```bash
python tools/gen_sweep_info.py
```

They belong at the repo root. Expected layout:

```
RaCFormer/
├── data/nuscenes -> /mnt/nfs/shared/nuscenes
│   ├── maps
│   ├── samples
│   ├── sweeps
│   ├── v1.0-test
│   └── v1.0-trainval
├── nuscenes_infos_train_sweep.pkl
└── nuscenes_infos_val_sweep.pkl
```

`nuscenes_infos_train_sweep.pkl` is ~1.4 GB and the val one ~290 MB. Home directories share one NFS
export, so if someone on the cluster already has them, symlink rather than copy.

## Checkpoints

```bash
mkdir -p pretrain checkpoints
```

Training needs the nuImages-pretrained ResNet-50 in `pretrain/`:

```bash
curl -fsSLO --output-dir pretrain \
  https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth
```

Evaluation needs the trained RaCFormer weights in `checkpoints/racformer_r50_f8.pth`, from
[Google Drive](https://drive.google.com/file/d/10Ky3lQWC2MLkQCpY81Jz5yxd4xWF8tAq/view?usp=sharing).

## Training

`train.py` takes `--config` (required) and an optional `--override k=v ...`. Output goes to the
`work_dir` declared inside the config.

### Interactive (one node, one GPU)

```bash
conda run -n racformerfix --no-capture-output python -u train.py \
  --config configs/racformer_r50_nuimg_704x256_f8.py
```

### Under SLURM (the normal path)

```bash
#!/bin/bash
#SBATCH --job-name=racformer_train
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/$USER/RaCFormer
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
mkdir -p slurm_logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

conda run -n racformerfix --no-capture-output python -u train.py \
  --config configs/racformer_r50_nuimg_704x256_f8.py
```

```bash
sinfo                              # pick an idle node before pinning --nodelist
sbatch my_train.sbatch
squeue -u "$USER"
tail -f slurm_logs/<jobname>_<jobid>.out
```

`--no-capture-output` matters: without it `conda run` buffers all stdout until the process exits, so
a multi-hour job shows nothing until it is over. `--chdir` must be an absolute path — expand
`$USER` yourself, SLURM does not.

### Multi-node

`slurm_train.sh` shows the multi-node NCCL setup: `MASTER_ADDR` from `scontrol`, plus
`NCCL_IB_DISABLE=1` (the nodes have no InfiniBand) and `NCCL_SOCKET_IFNAME=^lo,docker,cali` to keep
NCCL off the loopback and container interfaces. It launches with `srun python3 train.py`. Raise
`--nodes` and `--ntasks-per-node` there; each task still gets one GPU.

## Evaluation

### Standard validation

```bash
conda run -n racformerfix python -u val.py \
  --config configs/racformer_r50_nuimg_704x256_f8.py \
  --weights checkpoints/racformer_r50_f8.pth \
  --output_dir outputs/eval/baseline
```

`val.py` also accepts `--batch_size`, `--distance_bins`, `--max_vis_samples`, and the
`--local_rank`/`--world_size` pair used under `torchrun`. Multi-GPU on one node is not useful here,
so run it single-process.

### Day / night / rain breakdown

Runs inference once over the full val set, then evaluates each condition subset separately under the
nuScenes protocol:

```bash
conda run -n racformerfix python -u tools/eval_night_gap.py \
  --config configs/racformer_r50_nuimg_704x256_f8.py \
  --weights checkpoints/racformer_r50_f8.pth \
  --conditions all,day,night,rain \
  --output_dir outputs/night_eval/baseline
```

## Analysis tools

| Script | Purpose |
|---|---|
| `tools/eval_night_gap.py` | Full-val inference, then per-condition (day/night/rain) nuScenes evaluation |
| `tools/analyze_error_types.py` | FP/FN decomposition comparing baseline vs zeroed-image predictions |
| `tools/analyze_feature_quality.py` | Whether image features degrade at night while radar stays stable |
| `tools/analyze_modality_contribution.py` | Image/radar/LSS contribution to the fused representation |
| `tools/visualize_night_augmentation.py` | Side-by-side original vs night-augmented images |
| `tools/gen_sweep_info.py` | Generate the nuScenes sweep info pkls |
| `tools/visual.py`, `tools/render.py` | nuScenes-devkit-derived detection rendering |
| `visualize_results.py`, `create_video.py` | Result visualisation and videos |

## Acknowledgements

Many thanks to these excellent open-source projects:

* 3D Detection: [SparseBEV](https://github.com/MCG-NJU/SparseBEV), [PETR v2](https://github.com/megvii-research/PETR), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
* Codebase: [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
