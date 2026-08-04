"""
Compute-node smoke — build_dataset for the 4 Phase 1 train configs and one targeted
ds[idx] for the genaug config that MUST produce 6 manifest hits / 0 misses.

Runs free, no GPU. Needs data/nuscenes/ to resolve, so use srun on livenode02 or
livenode03 with --chdir=/srv/nfs/shared/gnmp/RaCFormer.

Usage:
    srun -p livecluster --nodelist=livenode02 \
         --chdir=/srv/nfs/shared/gnmp/RaCFormer \
         conda run -n racformerfix python research/night_gen_phase1/smoke_build_datasets.py \
             --fake-manifest research/night_gen_phase1/manifests/phase1_genaug_FAKE_smoke.json
"""

import argparse
import json
import os
import sys
import traceback

# Make `loaders` importable: this script sits at <repo>/research/night_gen_phase1/, so
# the repo root is two directories up. Adding it to sys.path matches how train.py runs
# (sys.path[0] = repo root because that's where the script lives).
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mmcv import Config


CONFIGS = {
    "day": "configs/racformer_train2k_day_research.py",
    "mixed": "configs/racformer_train2k_mixed_research.py",
    "simnight": "configs/racformer_train2k_simnight_research.py",
    "genaug": "configs/racformer_train2k_genaug_research.py",
}


def _pipeline_class_names(pipeline_cfg):
    return [step["type"] for step in pipeline_cfg]


def _build_dataset(cfg_data_train):
    # Lazy import — touches loaders/__init__.py which instantiates NuScenes(...) at
    # module load time and registers CustomNuScenesDataset_radar + manifest_loading
    # pipelines into the mmdet/mmdet3d registries. Requires data/nuscenes/ to resolve.
    import loaders  # noqa: F401  -- registers the custom dataset and most pipelines

    # SimulateNight lives in loaders/pipelines/night_augmentation.py and is NOT included
    # in loaders/pipelines/__init__.py — explicit import to register it.
    import loaders.pipelines.night_augmentation  # noqa: F401
    from mmdet3d.datasets import build_dataset

    return build_dataset(cfg_data_train)


def smoke_one(name, cfg_path, fake_manifest_path):
    print(f"\n[smoke] === {name} ({cfg_path}) ===")
    cfg = Config.fromfile(cfg_path)

    pipeline = cfg.data.train.get("pipeline")
    if pipeline is None:
        # Some configs inherit pipeline from base via data.train; resolve via the runtime.
        pipeline_classes = ["(inherited from base)"]
    else:
        pipeline_classes = _pipeline_class_names(pipeline)
    print(f"[smoke] pipeline classes (from cfg): {pipeline_classes}")

    if name == "genaug":
        if pipeline is None:
            raise RuntimeError("genaug config must declare its own pipeline")
        if pipeline[0]["type"] != "LoadMultiViewImageFromManifest":
            raise RuntimeError(
                f"genaug pipeline[0] must be LoadMultiViewImageFromManifest, "
                f"got {pipeline[0]['type']}"
            )
        # In-memory override only; on-disk config is untouched.
        cfg.data.train.pipeline[0]["manifest_path"] = fake_manifest_path
        print(
            f"[smoke] overrode genaug pipeline[0].manifest_path = {fake_manifest_path}"
        )

    if name == "simnight":
        flat = pipeline_classes if pipeline is not None else []
        if "SimulateNight" not in flat:
            raise RuntimeError(
                f"simnight pipeline must contain SimulateNight, got: {flat}"
            )

    ds = _build_dataset(cfg.data.train)
    n = len(ds)
    print(f"[smoke] len(ds)={n}")
    if n != 2000:
        raise RuntimeError(f"{name}: len(ds) expected 2000, got {n}")

    if name == "genaug":
        with open(fake_manifest_path) as f:
            manifest = json.load(f)
        tokens = {e["sample_token"] for e in manifest["entries"]}
        if len(tokens) != 1:
            raise RuntimeError(
                f"fake manifest must have exactly 1 token, got {len(tokens)}"
            )
        fake_token = next(iter(tokens))
        idx = next(
            (i for i, info in enumerate(ds.data_infos) if info["token"] == fake_token),
            None,
        )
        if idx is None:
            raise RuntimeError(
                f"fake-manifest token {fake_token} not found in genaug ds.data_infos"
            )
        print(f"[smoke] genaug ds[idx={idx}] for fake token {fake_token}")

        loader = ds.pipeline.transforms[0]
        loader_cls = type(loader).__name__
        if loader_cls != "LoadMultiViewImageFromManifest":
            raise RuntimeError(
                f"first pipeline transform must be LoadMultiViewImageFromManifest, "
                f"got {loader_cls}"
            )
        hits0, miss0 = loader._n_hits, loader._n_misses
        item = ds[idx]
        hits_delta = loader._n_hits - hits0
        miss_delta = loader._n_misses - miss0
        print(
            f"[smoke] hits/misses delta for fake-token sample: {hits_delta}/{miss_delta} "
            f"(loader._n_hits cumulative={loader._n_hits}, "
            f"loader._n_misses cumulative={loader._n_misses})"
        )
        if hits_delta != 6:
            raise RuntimeError(
                f"genaug fake-token sample expected 6 manifest hits, got {hits_delta}"
            )
        if miss_delta != 0:
            raise RuntimeError(
                f"genaug fake-token sample expected 0 misses, got {miss_delta}"
            )

        item_keys = sorted(item.keys()) if hasattr(item, "keys") else []
        print(f"[smoke] genaug ds[idx] keys: {item_keys}")
        for required in ("img", "img_metas", "gt_bboxes_3d", "gt_labels_3d"):
            if required not in item_keys:
                raise RuntimeError(
                    f"genaug ds[idx] missing required key {required!r}; got {item_keys}"
                )
        print("[smoke] genaug ds[idx] keys OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake-manifest", required=True)
    args = ap.parse_args()

    failures = []
    for name, path in CONFIGS.items():
        try:
            smoke_one(name, path, args.fake_manifest)
        except Exception as e:  # noqa: BLE001
            print(f"[smoke] FAILED {name}: {e}", file=sys.stderr)
            traceback.print_exc()
            failures.append((name, str(e)))

    if failures:
        print(f"\n[smoke] FAILED ({len(failures)}): {failures}", file=sys.stderr)
        sys.exit(2)
    print("\n[smoke] ALL 4 PASS")


if __name__ == "__main__":
    main()
