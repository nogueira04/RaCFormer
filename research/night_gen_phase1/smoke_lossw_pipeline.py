import importlib
import json
import os
import sys

from mmcv import Config
from mmdet3d.datasets import build_dataset


def main():
    sys.path.insert(0, os.getcwd())
    importlib.import_module("models")
    importlib.import_module("loaders")

    cfg = Config.fromfile("configs/racformer_train2k_genaug_seed20260425_ratio18p75_w05_research.py")
    with open("research/night_gen_phase1/manifests/phase1_t10_seed20260425_ratio18p75_manifest.json") as fh:
        manifest = json.load(fh)
    manifest_tokens = {e["sample_token"] for e in manifest["entries"] if e.get("status", "ok") == "ok"}

    cfg.data.train.max_samples = 2000
    dataset = build_dataset(cfg.data.train)
    # Avoid the first scene boundary; collect_sweeps indexes previous frames.
    gen_idx = next(i for i, info in enumerate(dataset.data_infos) if i > 80 and info["token"] in manifest_tokens)
    day_idx = next(i for i, info in enumerate(dataset.data_infos) if i > 80 and info["token"] not in manifest_tokens)
    for name, idx in [("generated", gen_idx), ("day", day_idx)]:
        item = dataset[idx]
        meta = item["img_metas"].data
        print(
            name,
            idx,
            meta.get("sample_idx"),
            meta.get("generated_sample_weight"),
            meta.get("filename", [""])[0],
        )


if __name__ == "__main__":
    main()
