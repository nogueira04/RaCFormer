"""Induced reprojection error of the (d2) extrinsic miscalibration.

fault-families.md, "(d2) implementation spec", item 6:

    Point set: the frame-t radar points of every val sample, projected into all 6 cameras
    via the UNPERTURBED camera calibrations, comparing pixel positions under clean vs
    perturbed radar->ego. Pool over all val samples x 3 seeds; a point-camera pair enters
    the pool only if it projects with camera-frame depth z >= 0.1 m AND inside the image
    bounds under BOTH clean and perturbed transforms (both-valid rule). Statistic: median
    [IQR] pixel displacement per level, plus P95 and the excluded-pair count.

Annotations and calibrations only. No images are read, no model is built, no inference is
run. Nothing is written inside the checkout.

Two z conventions are reported because the spec does not pin one and they answer different
questions:
  * model    -- the lidar-frame z of every radar point is forced to 0 before projection,
                which is what the frozen loader does at loaders/pipelines/loading.py:807 and
                therefore what RadarPointToMultiViewDepth (:523) actually rasterises. This
                is the displacement the network sees.
  * geometric -- the z returned by the devkit is kept. This is the displacement of the
                physical radar->ego transform, unaffected by the loader flattening.
Both come from the same point set and the same both-valid rule, evaluated per convention.

Pixel units are the native camera resolution (1600x900). The network is fed a 704x256 crop
of a 0.38-0.55 resized image (ida_aug_conf, configs/racformer_r50_nuimg_704x256_f8.py:45-52),
so a displacement in network-input pixels is smaller by that resize factor.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

REPO_ROOT = "/srv/nfs/shared/gnmp/RaCFormer"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from mmcv import Config                                          # noqa: E402
from mmdet3d.datasets import build_dataset                       # noqa: E402

from loaders.nuscenes_dataset import get_nu_radar                # noqa: E402
from research.robust_study.corruptions.misalign import (         # noqa: E402
    D2_LEVELS, perturbed_extrinsics)

# Histogram grid for the pooled displacement distribution: 0.01 px resolution up to 2000 px.
# Exact counts, O(1) memory, and a resolution two orders of magnitude finer than any
# difference the study will report.
BIN_PX = 0.01
N_BINS = 200000
Z_CONVENTIONS = ("model", "geometric")

DATASET = None
LEVELS = None
SEEDS = None


def load_dataset(config_path):
    cfg = Config.fromfile(config_path)
    cfg.data.val.test_mode = True
    return build_dataset(cfg.data.val)


def frame_t_points(sample_token):
    """Element 0 of `radar_points`: the frame-t radar aggregate.

    Mirrors the frozen call at loaders/pipelines/loading.py:805 with the arguments the test
    pipeline passes at configs/racformer_r50_nuimg_704x256_f8.py:227 (num_sweeps=5,
    filter=True, radar_types=None). Returns an (N, 3) array in the LIDAR_TOP frame.
    """
    points, _tokens, _times = get_nu_radar(sample_token, True, 5, filter=True, radar_types=None)
    return points[:3, :].numpy().T.astype(np.float64)


def project(points_xyz, lidar2img):
    """Return (uv, depth) for one camera. uv is (N, 2), depth is (N,)."""
    cam = points_xyz @ lidar2img[:3, :3].T + lidar2img[:3, 3]
    depth = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = cam[:, :2] / depth[:, None]
    return uv, depth


def valid_mask(uv, depth, width, height):
    return ((depth >= 0.1)
            & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
            & (uv[:, 0] >= 0.0) & (uv[:, 0] < width)
            & (uv[:, 1] >= 0.0) & (uv[:, 1] < height))


def empty_accumulator():
    acc = {}
    for level in LEVELS:
        for conv in Z_CONVENTIONS:
            acc[(level, conv)] = dict(
                hist=np.zeros(N_BINS + 1, dtype=np.int64),
                total=0, included=0,
                excl_clean_only=0, excl_pert_only=0, excl_both=0,
                max_px=0.0,
            )
    return acc


def merge(into, other):
    for key, src in other.items():
        dst = into[key]
        dst["hist"] += src["hist"]
        for field in ("total", "included", "excl_clean_only", "excl_pert_only", "excl_both"):
            dst[field] += src[field]
        dst["max_px"] = max(dst["max_px"], src["max_px"])
    return into


def accumulate(acc, key, uv_c, d_c, uv_p, d_p, width, height):
    slot = acc[key]
    ok_c = valid_mask(uv_c, d_c, width, height)
    ok_p = valid_mask(uv_p, d_p, width, height)
    both = ok_c & ok_p
    slot["total"] += ok_c.size
    slot["included"] += int(both.sum())
    slot["excl_clean_only"] += int((ok_c & ~ok_p).sum())
    slot["excl_pert_only"] += int((~ok_c & ok_p).sum())
    slot["excl_both"] += int((~ok_c & ~ok_p).sum())
    if not both.any():
        return
    disp = np.linalg.norm(uv_p[both] - uv_c[both], axis=1)
    slot["max_px"] = max(slot["max_px"], float(disp.max()))
    idx = np.minimum((disp / BIN_PX).astype(np.int64), N_BINS)
    slot["hist"] += np.bincount(idx, minlength=N_BINS + 1)


def process_index(index):
    ds = DATASET
    info = ds.data_infos[index]
    token = info["token"]
    data = ds.get_data_info(index)
    lidar2img = [np.asarray(m, dtype=np.float64) for m in data["lidar2img"]]
    shapes = camera_shapes(info)

    clean = frame_t_points(token)
    variants = {"model": clean.copy(), "geometric": clean}
    variants["model"][:, 2] = 0.0  # loaders/pipelines/loading.py:807

    acc = empty_accumulator()
    for level in LEVELS:
        sigma = D2_LEVELS[level]
        for seed in SEEDS:
            with perturbed_extrinsics(token, seed, sigma["sigma_rot"], sigma["sigma_trans"]):
                pert = frame_t_points(token)
            if pert.shape != clean.shape:
                raise RuntimeError(
                    "point count changed under perturbation for {} ({} vs {}); the "
                    "injection must not add or drop points".format(
                        token, pert.shape, clean.shape))
            pert_variants = {"model": pert.copy(), "geometric": pert}
            pert_variants["model"][:, 2] = 0.0
            for conv in Z_CONVENTIONS:
                for cam_i, matrix in enumerate(lidar2img):
                    width, height = shapes[cam_i]
                    uv_c, d_c = project(variants[conv], matrix)
                    uv_p, d_p = project(pert_variants[conv], matrix)
                    accumulate(acc, (level, conv), uv_c, d_c, uv_p, d_p, width, height)
    return acc


def camera_shapes(info):
    """(width, height) per camera, in the same order get_data_info emits lidar2img."""
    shapes = []
    for _chan, cam in info["cams"].items():
        width = cam.get("width") or 0
        height = cam.get("height") or 0
        if not width or not height:
            width, height = 1600, 900  # every nuScenes camera; asserted in the smoke run
        shapes.append((width, height))
    return shapes


def quantile_from_hist(hist, q):
    total = hist.sum()
    if total == 0:
        return float("nan")
    target = q * total
    cumulative = np.cumsum(hist)
    idx = int(np.searchsorted(cumulative, target, side="left"))
    return idx * BIN_PX + BIN_PX / 2.0


def summarise(acc):
    out = {}
    for (level, conv), slot in acc.items():
        hist = slot["hist"]
        out["{}/{}".format(level, conv)] = dict(
            level=level,
            z_convention=conv,
            sigma_rot=D2_LEVELS[level]["sigma_rot"],
            sigma_trans=D2_LEVELS[level]["sigma_trans"],
            candidate_pairs=int(slot["total"]),
            included_pairs=int(slot["included"]),
            excluded_pairs=int(slot["total"] - slot["included"]),
            excluded_clean_valid_only=int(slot["excl_clean_only"]),
            excluded_perturbed_valid_only=int(slot["excl_pert_only"]),
            excluded_invalid_under_both=int(slot["excl_both"]),
            median_px=quantile_from_hist(hist, 0.50),
            q1_px=quantile_from_hist(hist, 0.25),
            q3_px=quantile_from_hist(hist, 0.75),
            p95_px=quantile_from_hist(hist, 0.95),
            max_px=slot["max_px"],
            overflow_pairs=int(hist[-1]),
        )
    return out


def main():
    global DATASET, LEVELS, SEEDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        default="configs/racformer_eval_fullval_research.py",
                        help="config supplying the val split; its pipeline is NOT used")
    parser.add_argument("--levels", default="medium,severe")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--scene-limit", type=int, default=0,
                        help="if > 0, restrict to the first N scenes (smoke run)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True, help="output json path, outside the checkout")
    args = parser.parse_args()

    LEVELS = [s for s in args.levels.split(",") if s]
    SEEDS = [int(s) for s in args.seeds.split(",") if s]
    for level in LEVELS:
        if level not in D2_LEVELS:
            raise SystemExit("unknown level {!r}".format(level))

    started = time.time()
    DATASET = load_dataset(args.config)
    import loaders.nuscenes_dataset as nd
    indices = list(range(len(DATASET)))
    scene_of = [nd.renusc.get("sample", DATASET.data_infos[i]["token"])["scene_token"]
                for i in indices]
    if args.scene_limit > 0:
        seen = []
        keep = []
        for i, scene in zip(indices, scene_of):
            if scene not in seen:
                if len(seen) >= args.scene_limit:
                    continue
                seen.append(scene)
            keep.append(i)
        indices = keep
        n_scenes_total = len(seen)
    else:
        n_scenes_total = len(set(scene_of))
    print("[reproj] dataset={} samples={} scenes={} levels={} seeds={} workers={}".format(
        type(DATASET).__name__, len(indices), n_scenes_total, LEVELS, SEEDS, args.workers),
        flush=True)

    acc = empty_accumulator()
    done = 0
    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers) as pool:
            for partial in pool.imap_unordered(process_index, indices, chunksize=4):
                merge(acc, partial)
                done += 1
                if done % 200 == 0:
                    print("[reproj] {}/{} samples, {:.1f}s".format(
                        done, len(indices), time.time() - started), flush=True)
    else:
        for i in indices:
            merge(acc, process_index(i))
            done += 1
            if done % 50 == 0:
                print("[reproj] {}/{} samples, {:.1f}s".format(
                    done, len(indices), time.time() - started), flush=True)

    summary = summarise(acc)
    payload = dict(
        tool="research/robust_study/tools/reproj_error_d2.py",
        config=args.config,
        n_samples=len(indices),
        n_scenes=n_scenes_total,
        seeds=SEEDS,
        levels=LEVELS,
        bin_px=BIN_PX,
        pixel_units="native camera resolution (1600x900)",
        elapsed_s=round(time.time() - started, 1),
        results=summary,
    )
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print("\n[reproj] pooled over {} samples x {} seeds".format(len(indices), len(SEEDS)))
    header = "{:<10} {:<10} {:>10} {:>10} {:>10} {:>10} {:>14} {:>14}".format(
        "level", "z-conv", "median", "Q1", "Q3", "P95", "included", "excluded")
    print(header)
    print("-" * len(header))
    for key in sorted(summary):
        row = summary[key]
        print("{:<10} {:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>14d} {:>14d}".format(
            row["level"], row["z_convention"], row["median_px"], row["q1_px"], row["q3_px"],
            row["p95_px"], row["included_pairs"], row["excluded_pairs"]))
    print("\n[reproj] wrote {}".format(args.out))


if __name__ == "__main__":
    main()
