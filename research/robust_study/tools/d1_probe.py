"""(d1) mini-screen probe for the AMENDED sweep-chain injection.

Checks required by fault-families.md (d1), "AMENDED 2026-08-03":
  (i)   strictly OLDER sweep timestamps at EVERY element vs clean, per-element delta ~ k*77 ms
  (ii)  DISTINCT data per offset level (the failure mode of the superseded mechanism)
  (iii) the frame-t element is shifted too (partial-injection trap)
  (iv)  camera path bit-identical
  (v)   determinism across re-instantiation of the pipeline
  (vi)  the dt 7th point feature grows accordingly

Timing evidence comes from the loaded tensors themselves, not from re-derived index
arithmetic: dt is written per point by the devkit as `ref_time - sweep_time`
(nuscenes_dataset.py:492) and survives as the 7th column of `radar_points`
(loading.py:809 selects [0,1,2,5,8,9,18]). For one stack element, min(dt) is the lag of its
NEWEST constituent sweep, i.e. exactly the aggregation start this injection moves. So
min(dt) shifted by k*sweep_period is a direct measurement of the fault.

Annotations, calibrations, radar .pcd and (for check iv) camera JPEGs only. No model.
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

REPO_ROOT = "/srv/nfs/shared/gnmp/RaCFormer"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import importlib                                            # noqa: E402
importlib.import_module("models")
importlib.import_module("loaders")

from mmcv import Config                                     # noqa: E402
from mmdet3d.datasets import build_dataset                  # noqa: E402
from mmdet3d.datasets.pipelines import Compose              # noqa: E402

from research.robust_study.corruptions import misalign      # noqa: E402

DT_FEATURE = 6
RADAR_TYPES = ("Loadnuradarpoints", "LoadradarpointsFromMultiSweeps")
D1_TYPES = {"Loadnuradarpoints": "D1AsyncLoadnuradarpoints",
            "LoadradarpointsFromMultiSweeps": "D1AsyncLoadradarpointsFromMultiSweeps"}


def build_steps(config_path, offset, with_cameras):
    """Fresh pipeline objects each call -- that is what makes check (v) meaningful.

    Compose is used rather than a registry lookup because the camera loaders live in the
    mmdet3d registry while the radar loaders are registered against mmdet's; Compose
    resolves both.
    """
    cfg = Config.fromfile(config_path)
    steps = []
    for step in cfg.test_pipeline:
        step = dict(step)
        if step["type"] in RADAR_TYPES:
            if offset:
                step["type"] = D1_TYPES[step["type"]]
                step["offset"] = offset
            steps.append(step)
        elif with_cameras and step["type"].startswith("LoadMultiViewImage"):
            steps.append(step)
    return Compose(steps)


def run(dataset, index, offset, config_path, with_cameras=False):
    data = dataset.get_data_info(index)
    return build_steps(config_path, offset, with_cameras)(data)


def element_stats(points_list):
    out = []
    for p in points_list:
        dt = p.tensor[:, DT_FEATURE]
        out.append(dict(n=int(p.tensor.shape[0]),
                        dt_min=round(float(dt.min()), 4),
                        dt_mean=round(float(dt.mean()), 4),
                        dt_max=round(float(dt.max()), 4)))
    return out


def identical(a_list, b_list):
    out = []
    for a, b in zip(a_list, b_list):
        out.append(a.tensor.shape == b.tensor.shape and bool((a.tensor == b.tensor).all()))
    return out


def order_regime(stats):
    """Label the clean stack by its ORDER REGIME, the distinction that decides whether an
    element can move at all.

    Stack elements are spaced ~0.5 s apart while the scene has history. Near a scene start
    the frozen loader runs out of history and repeats the oldest entry it has (the
    `min(idx, len(prev) - 1)` clamp at loading.py:885 and the out-of-order keyframe entry
    appended at nuscenes_dataset.py:165), producing a tail of elements that already sit on
    the oldest available sweep. Those tail elements cannot be made older by any offset --
    there is no older sweep -- so they are the ones this injection legitimately cannot move.
    """
    mins = [s["dt_min"] for s in stats]
    degenerate = sum(1 for a, b in zip(mins, mins[1:]) if abs(b - a) < 0.05)
    return "full" if degenerate == 0 else "clamped_tail_%d" % degenerate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/racformer_eval_fullval_research.py")
    parser.add_argument("--indices", default="100,1000,3000,2000,4500")
    parser.add_argument("--offsets", default="1,2,3")
    parser.add_argument("--scan-samples", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.data.val.test_mode = True
    dataset = build_dataset(cfg.data.val)
    indices = [int(x) for x in args.indices.split(",") if x]
    offsets = [int(x) for x in args.offsets.split(",") if x]
    period = misalign.D1_SWEEP_PERIOD_S

    report = dict(tool="research/robust_study/tools/d1_probe.py",
                  config=args.config, sweep_period_s=period, samples=[], scan=None)

    for index in indices:
        token = dataset.data_infos[index]["token"]
        clean_cam = run(dataset, index, 0, args.config, with_cameras=True)
        clean = clean_cam["radar_points"]
        clean_stats = element_stats(clean)
        regime = order_regime(clean_stats)
        entry = dict(val_index=index, sample_token=token, order_regime=regime,
                     clean=clean_stats, offsets={})
        print("\n=== val index %d  token %s  clean stack regime: %s ===" % (
            index, token[:8], regime))
        print("clean    dt_min per element = %s" % [s["dt_min"] for s in clean_stats])
        print("clean    dt_mean per element= %s" % [s["dt_mean"] for s in clean_stats])

        per_offset_points = {}
        for offset in offsets:
            misalign.reset_clamp_stats()
            shifted_cam = run(dataset, index, offset, args.config, with_cameras=True)
            clamp = dict(misalign.D1_CLAMP_STATS)
            shifted = shifted_cam["radar_points"]
            per_offset_points[offset] = shifted
            stats = element_stats(shifted)
            d_min = [round(s["dt_min"] - c["dt_min"], 4) for s, c in zip(stats, clean_stats)]
            d_mean = [round(s["dt_mean"] - c["dt_mean"], 4) for s, c in zip(stats, clean_stats)]
            expected = round(offset * period, 4)
            older = [d > 1e-6 for d in d_min]

            # (iv) camera path bit-identical
            cam_same_names = clean_cam["img_filename"] == shifted_cam["img_filename"]
            cam_imgs = clean_cam.get("img")
            cam_same_pixels = None
            if cam_imgs is not None:
                cam_same_pixels = (len(cam_imgs) == len(shifted_cam["img"]) and all(
                    np.array_equal(np.asarray(a), np.asarray(b))
                    for a, b in zip(cam_imgs, shifted_cam["img"])))

            # (v) determinism: rebuild the pipeline from scratch and re-run
            repeat = run(dataset, index, offset, args.config, with_cameras=False)["radar_points"]
            deterministic = all(identical(shifted, repeat))

            entry["offsets"][str(offset)] = dict(
                elements=stats,
                delta_dt_min_s=d_min, delta_dt_mean_s=d_mean,
                expected_delta_s=expected,
                all_elements_older=all(older),
                n_elements_older=int(sum(older)),
                element0_shifted=not identical(clean, shifted)[0],
                element_identical_to_clean=identical(clean, shifted),
                camera_filenames_identical=cam_same_names,
                camera_pixels_identical=cam_same_pixels,
                deterministic_on_reinstantiation=deterministic,
                clamp_stats=clamp,
            )
            print("offset=%d expected +%.4f s | delta dt_min = %s" % (offset, expected, d_min))
            print("offset=%d all elements OLDER: %s (%d/%d) | element0 shifted: %s" % (
                offset, all(older), sum(older), len(older), not identical(clean, shifted)[0]))
            print("offset=%d dt_mean delta      = %s" % (offset, d_mean))
            print("offset=%d cameras identical: names=%s pixels=%s | deterministic: %s" % (
                offset, cam_same_names, cam_same_pixels, deterministic))
            print("offset=%d clamps: %s" % (offset, clamp))

        # (ii) distinct data per offset level
        pairs = {}
        for i, a in enumerate(offsets):
            for b in offsets[i + 1:]:
                same = identical(per_offset_points[a], per_offset_points[b])
                pairs["%d_vs_%d" % (a, b)] = dict(
                    n_elements_identical=int(sum(same)), any_identical=any(same))
                print("distinct check offset %d vs %d: elements identical = %d/8" % (
                    a, b, sum(same)))
        entry["pairwise_level_overlap"] = pairs
        report["samples"].append(entry)

    if args.scan_samples > 0:
        n = min(args.scan_samples, len(dataset))
        step = max(1, len(dataset) // n)
        scan_indices = list(range(0, len(dataset), step))[:n]
        scan = {}
        for offset in offsets:
            misalign.reset_clamp_stats()
            older = unchanged = newer = 0
            regimes = collections.Counter()
            for index in scan_indices:
                c = element_stats(run(dataset, index, 0, args.config)["radar_points"])
                s = element_stats(run(dataset, index, offset, args.config)["radar_points"])
                regimes[order_regime(c)] += 1
                for cs, ss in zip(c, s):
                    d = ss["dt_min"] - cs["dt_min"]
                    if d > 1e-6:
                        older += 1
                    elif d < -1e-6:
                        newer += 1
                    else:
                        unchanged += 1
            scan[str(offset)] = dict(n_samples=len(scan_indices),
                                     elements_older=older, elements_unchanged=unchanged,
                                     elements_newer=newer, clean_regimes=dict(regimes),
                                     clamp_stats=dict(misalign.D1_CLAMP_STATS))
            print("\n[scan] offset=%d over %d samples: OLDER=%d unchanged=%d newer=%d" % (
                offset, len(scan_indices), older, unchanged, newer))
            print("[scan] offset=%d clean regimes=%s clamps=%s" % (
                offset, dict(regimes), misalign.D1_CLAMP_STATS))
        report["scan"] = scan

    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print("\n[d1_probe] wrote %s" % args.out)


if __name__ == "__main__":
    main()
