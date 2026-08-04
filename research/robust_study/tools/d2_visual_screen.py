"""(d2) visual mini-screen: projected radar points on camera images, clean vs perturbed.

fault-families.md (d2) "Mini-screen check": confirm the implementation bends geometry
rather than breaking timestamps or indexing. For each requested val sample this renders one
PNG per camera with the clean projection in one colour and the perturbed projection in
another, plus a per-point connector, and prints the point-count and displacement summary
that a broken indexing bug would violate.

Reads camera JPEGs for plotting only. No model is built and no inference is run.
"""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = "/srv/nfs/shared/gnmp/RaCFormer"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from mmcv import Config                                          # noqa: E402
from mmdet3d.datasets import build_dataset                       # noqa: E402

import loaders.nuscenes_dataset as nd                            # noqa: E402
from loaders.nuscenes_dataset import get_nu_radar                # noqa: E402
from research.robust_study.corruptions.misalign import (         # noqa: E402
    D2_LEVELS, perturbed_extrinsics)


def frame_t_points(sample_token):
    points, _tokens, _times = get_nu_radar(sample_token, True, 5, filter=True, radar_types=None)
    return points[:3, :].numpy().T.astype(np.float64)


def project(points_xyz, lidar2img):
    cam = points_xyz @ lidar2img[:3, :3].T + lidar2img[:3, 3]
    depth = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = cam[:, :2] / depth[:, None]
    return uv, depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/racformer_eval_fullval_research.py")
    parser.add_argument("--indices", default="100,1000,3000")
    parser.add_argument("--level", default="severe", choices=sorted(D2_LEVELS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sigma = D2_LEVELS[args.level]

    cfg = Config.fromfile(args.config)
    cfg.data.val.test_mode = True
    dataset = build_dataset(cfg.data.val)

    summary = []
    for index in [int(x) for x in args.indices.split(",") if x]:
        info = dataset.data_infos[index]
        token = info["token"]
        scene = nd.renusc.get("sample", token)["scene_token"]
        data = dataset.get_data_info(index)
        matrices = [np.asarray(m, dtype=np.float64) for m in data["lidar2img"]]
        channels = list(info["cams"].keys())

        clean = frame_t_points(token)
        clean[:, 2] = 0.0  # loaders/pipelines/loading.py:807, what the model rasterises
        with perturbed_extrinsics(token, args.seed, sigma["sigma_rot"], sigma["sigma_trans"]):
            pert = frame_t_points(token)
        pert[:, 2] = 0.0

        record = dict(val_index=index, sample_token=token, scene_token=scene,
                      level=args.level, seed=args.seed,
                      n_points_clean=int(clean.shape[0]), n_points_perturbed=int(pert.shape[0]),
                      cameras={})
        print("\n=== index {} token {} scene {} ===".format(index, token[:8], scene[:8]))
        print("points clean={} perturbed={} (a count change would mean indexing broke, "
              "not geometry)".format(clean.shape[0], pert.shape[0]))

        for cam_i, channel in enumerate(channels):
            uv_c, d_c = project(clean, matrices[cam_i])
            uv_p, d_p = project(pert, matrices[cam_i])
            ok = ((d_c >= 0.1) & (d_p >= 0.1)
                  & (uv_c[:, 0] >= 0) & (uv_c[:, 0] < 1600)
                  & (uv_c[:, 1] >= 0) & (uv_c[:, 1] < 900)
                  & (uv_p[:, 0] >= 0) & (uv_p[:, 0] < 1600)
                  & (uv_p[:, 1] >= 0) & (uv_p[:, 1] < 900))
            n_ok = int(ok.sum())
            disp = (np.linalg.norm(uv_p[ok] - uv_c[ok], axis=1) if n_ok
                    else np.zeros(0))
            record["cameras"][channel] = dict(
                n_visible_both=n_ok,
                median_disp_px=float(np.median(disp)) if n_ok else None,
                max_disp_px=float(disp.max()) if n_ok else None,
            )
            print("  {:<16} visible_both={:<5d} median_disp={:>8}px".format(
                channel, n_ok,
                "n/a" if not n_ok else round(float(np.median(disp)), 2)))

            # Same file the pipeline loads: get_data_info emits img_filename in the same
            # order as lidar2img (loaders/nuscenes_dataset.py:321-344).
            image = plt.imread(data["img_filename"][cam_i])
            fig, axis = plt.subplots(figsize=(16, 9), dpi=80)
            axis.imshow(image)
            if n_ok:
                for a, b in zip(uv_c[ok], uv_p[ok]):
                    axis.plot([a[0], b[0]], [a[1], b[1]], "-", color="yellow",
                              linewidth=0.8, alpha=0.9)
                axis.scatter(uv_c[ok, 0], uv_c[ok, 1], s=26, c="lime", marker="o",
                             label="clean", zorder=3)
                axis.scatter(uv_p[ok, 0], uv_p[ok, 1], s=26, c="red", marker="x",
                             label="perturbed ({}, seed {})".format(args.level, args.seed),
                             zorder=4)
                axis.legend(loc="upper right", fontsize=11)
            axis.set_xlim(0, 1600)
            axis.set_ylim(900, 0)
            axis.set_title("{}  {}  sample {}  scene {}  n_both={}".format(
                channel, args.level, token[:8], scene[:8], n_ok), fontsize=12)
            axis.axis("off")
            name = "d2_{}_seed{}_idx{}_{}.png".format(args.level, args.seed, index, channel)
            fig.savefig(os.path.join(args.out_dir, name), bbox_inches="tight")
            plt.close(fig)
            record["cameras"][channel]["png"] = os.path.join(args.out_dir, name)
        summary.append(record)

    out = os.path.join(args.out_dir, "d2_visual_screen.json")
    with open(out, "w") as handle:
        json.dump(dict(tool="research/robust_study/tools/d2_visual_screen.py",
                       level=args.level, seed=args.seed, samples=summary),
                  handle, indent=2, sort_keys=True)
    print("\n[d2_visual_screen] wrote {}".format(out))


if __name__ == "__main__":
    main()
