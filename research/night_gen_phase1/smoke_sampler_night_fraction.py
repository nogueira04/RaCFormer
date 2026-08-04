"""
Compute-node smoke — confirm S5's empirical sampler-yielded night fraction lands in
[0.30, 0.35] over one full epoch on `racformer_train2k_mixed_research.py`.

Walks the sampler's index stream without decoding any images, so it's seconds.

Usage:
    srun -p livecluster --nodelist=livenode02 \
         --chdir=/srv/nfs/shared/gnmp/RaCFormer \
         conda run -n racformerfix python research/night_gen_phase1/smoke_sampler_night_fraction.py
"""

import os
import sys
from collections import Counter

# Make `loaders` importable when this script runs from <repo>/research/night_gen_phase1/.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mmcv import Config
from mmdet3d.datasets import build_dataset

from loaders.builder import build_dataloader


CFG_PATH = "configs/racformer_train2k_mixed_research.py"


def main():
    cfg = Config.fromfile(CFG_PATH)
    ds = build_dataset(cfg.data.train)
    print(f"[sampler] len(ds)={len(ds)}")
    if len(ds) != 2000:
        raise RuntimeError(f"expected 2000-entry S5 pkl, got len(ds)={len(ds)}")

    dl = build_dataloader(
        ds,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=True,
        seed=0,
    )

    indices = list(iter(dl.sampler))
    print(f"[sampler] sampler emitted {len(indices)} indices")
    if len(indices) != 2000:
        print(
            f"[sampler] WARNING: sampler emitted {len(indices)} indices, expected 2000",
            file=sys.stderr,
        )

    cnt = Counter(ds.data_infos[i]["scene_condition"] for i in indices)
    nf = cnt["night"] / sum(cnt.values())
    print(f"[sampler] counts={dict(cnt)} empirical_night_fraction={nf:.4f}")

    if not (0.30 <= nf <= 0.35):
        raise RuntimeError(
            f"empirical night fraction {nf:.4f} outside target band [0.30, 0.35]"
        )
    print("[sampler] PASS")


if __name__ == "__main__":
    main()
