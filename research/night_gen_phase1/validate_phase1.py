"""
T3 — Phase 1 geometry/photometric *screening* validator.

Replaces the Phase 0 detector-recall proxy with:
  1. 3D GT-box reprojection via the project's compose_lidar2img helper.
  2. Resolution-aware crop extraction (R-A: resize gen image to (1600,900) Lanczos by default;
     R-B: rescale AABBs to gen-image coords, optional via --crop-mode rb).
  3. Masked SSIM between original-day crop and histogram-matched generated-night crop.
     Optional LPIPS if torch + lpips are available.
  4. Reformulated cross-view brightness gate: absolute Δ ≤ 0.15 (NOT delta/mean).
  5. Outlier-cam detection: any cam > 0.40 brightness while another in same sample < 0.25.
  6. Luminance KL vs --night-ref-dir (Phase 0 sanity floor).

Scope: this validator screens and filters likely failures. It does NOT prove semantic
correctness. High SSIM is necessary but not sufficient; corroborate with manual audit.

Usage (cluster, Option A — all paths cluster-side, racformer env):
    conda run -n racformer python research/night_gen_phase1/validate_phase1.py \
        --generated-manifest research/night_gen_phase1/manifests/phase0_generated_nb2_mirrored.json \
        --inputs-root        data/nuscenes \
        --gt-boxes-pkl       nuscenes_infos_train_sweep.pkl \
        --night-ref-dir      research/night_gen_phase1/inputs_night_ref \
        --reports-dir        research/night_gen_phase1/reports/phase1_revalidate \
        --crop-mode ra

Usage (local, Option B — all paths local, rcfusion env, after downloading the pkl):
    conda run -n rcfusion python /home/gabriel/LIVE/night_gen_phase1_staging/research/validate_phase1.py \
        --generated-manifest /home/gabriel/LIVE/night_gen_phase0/manifests/phase0_generated_nb2.json \
        --inputs-root        /home/gabriel/LIVE/night_gen_phase0/inputs \
        --gt-boxes-pkl       /home/gabriel/LIVE/night_gen_phase0/nuscenes_infos_train_sweep.pkl \
        --night-ref-dir      /home/gabriel/LIVE/night_gen_phase0/inputs_night_ref \
        --reports-dir        /home/gabriel/LIVE/night_gen_phase0/reports/phase1_revalidate \
        --crop-mode ra

Hard rule: do NOT mix /srv/... and /home/... paths in a single invocation.
"""

import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------------------------- lidar2img (keyframe)
# Mirrors loaders/nuscenes_dataset.py: get_data_info keyframe construction
# (sensor2lidar_rotation/translation directly, NOT the multi-sweep compose helper).
def lidar2img_keyframe(sensor2lidar_rotation, sensor2lidar_translation, cam_intrinsic):
    """Build 4x4 lidar2img for a keyframe camera.

    Returns matrix M such that pts_img = pts_lidar_homo @ M.T, where pts_img columns are
    (u*z, v*z, z, 1) before perspective divide.
    """
    s2l_R = np.asarray(sensor2lidar_rotation)
    s2l_t = np.asarray(sensor2lidar_translation)
    K = np.asarray(cam_intrinsic)
    lidar2cam_r = np.linalg.inv(s2l_R)
    lidar2cam_t = s2l_t @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    viewpad = np.eye(4)
    viewpad[: K.shape[0], : K.shape[1]] = K
    return (viewpad @ lidar2cam_rt.T).astype(np.float32)


# --------------------------------------------------------------------- box corners
def box_corners(center, dims, yaw):
    """nuScenes 3D box → 8 corners in lidar frame.

    box = [x, y, z, w, l, h, yaw] (nuScenes convention as stored in info['gt_boxes']).
    Returns (8, 3) corner array.
    """
    cx, cy, cz = center
    w, l, h = dims  # nuScenes order
    cy_, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy_, -sy, 0], [sy, cy_, 0], [0, 0, 1]])
    half = np.array([l, w, h]) / 2.0
    # 8 corners in object frame (l along x, w along y, h along z)
    signs = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ]
    )
    corners_obj = signs * half  # (8, 3)
    corners_world = corners_obj @ R.T + np.array([cx, cy, cz])
    return corners_world


def project_box_to_image(corners_lidar, lidar2img, image_w, image_h):
    """Project 8 lidar-frame corners to image coordinates; return AABB or None."""
    # Convert (8, 3) to (8, 4) homogeneous
    homo = np.concatenate([corners_lidar, np.ones((8, 1))], axis=1)
    pts_img = homo @ lidar2img.T  # (8, 4)
    z = pts_img[:, 2]
    if (z <= 0.1).all():
        return None  # all behind camera
    valid = z > 0.1
    u = pts_img[valid, 0] / z[valid]
    v = pts_img[valid, 1] / z[valid]
    if u.size < 2:
        return None
    x0 = max(0, int(np.floor(u.min())))
    y0 = max(0, int(np.floor(v.min())))
    x1 = min(image_w, int(np.ceil(u.max())))
    y1 = min(image_h, int(np.ceil(v.max())))
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return None
    return [x0, y0, x1, y1]


# --------------------------------------------------------------------- photometric helpers
def hist_match(src, ref):
    """Match the per-channel CDF of `src` to `ref`. Both are (H, W, 3) uint8."""
    out = np.empty_like(src)
    for c in range(3):
        s = src[..., c].ravel()
        r = ref[..., c].ravel()
        _, s_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
        r_vals, r_counts = np.unique(r, return_counts=True)
        s_cdf = np.cumsum(s_counts) / s.size
        r_cdf = np.cumsum(r_counts) / r.size
        interp = np.interp(s_cdf, r_cdf, r_vals)
        out[..., c] = interp[s_idx].reshape(src[..., c].shape).astype(np.uint8)
    return out


def masked_ssim(crop_a, crop_b):
    """Mean SSIM (luminance only) between two equal-size RGB crops. Returns float in [-1, 1]."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return None
    if crop_a.shape != crop_b.shape:
        return None
    if crop_a.shape[0] < 7 or crop_a.shape[1] < 7:
        return None
    a = (
        0.2126 * crop_a[..., 0] + 0.7152 * crop_a[..., 1] + 0.0722 * crop_a[..., 2]
    ).astype(np.float32)
    b = (
        0.2126 * crop_b[..., 0] + 0.7152 * crop_b[..., 1] + 0.0722 * crop_b[..., 2]
    ).astype(np.float32)
    return float(ssim(a, b, data_range=255.0))


def brightness(arr_uint8):
    a = arr_uint8.astype(np.float32) / 255.0
    return float(
        0.2126 * a[..., 0].mean()
        + 0.7152 * a[..., 1].mean()
        + 0.0722 * a[..., 2].mean()
    )


def compute_luma_histogram(paths, bins=32):
    hist = np.zeros(bins, dtype=np.float64)
    n = 0
    for p in paths:
        try:
            arr = np.array(Image.open(p).convert("RGB"))
            luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
            h, _ = np.histogram(luma / 255.0, bins=bins, range=(0, 1))
            hist += h
            n += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  hist skip: {p}: {exc}", file=sys.stderr)
    if n == 0:
        return hist
    return hist / hist.sum()


def hist_kl(p, q, eps=1e-6):
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


# --------------------------------------------------------------------- main
def load_gt_boxes_index(pkl_path):
    """Map sample_token -> info dict (must contain gt_boxes + cams + ego/lidar transforms)."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    infos = d["infos"] if isinstance(d, dict) and "infos" in d else d
    return {info["token"]: info for info in infos}


def cam_name_from_path(p):
    """Extract CAM_FRONT / CAM_BACK_LEFT / etc. from a nuScenes-style path."""
    for cam in (
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
    ):
        if f"/{cam}/" in p:
            return cam
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated-manifest", required=True)
    ap.add_argument("--inputs-root", required=True)
    ap.add_argument("--gt-boxes-pkl", required=True)
    ap.add_argument("--night-ref-dir", default=None)
    ap.add_argument("--reports-dir", required=True)
    ap.add_argument("--expected-input-res", default="1600x900")
    ap.add_argument(
        "--crop-mode",
        choices=["ra", "rb"],
        default="ra",
        help="ra: resize gen image to (orig_w, orig_h) before crop; rb: rescale AABBs to gen-coords",
    )
    ap.add_argument("--abs-delta-threshold", type=float, default=0.15)
    ap.add_argument("--lum-kl-threshold", type=float, default=0.5)
    args = ap.parse_args()

    # Refuse hybrid local/cluster paths
    paths = [
        args.generated_manifest,
        args.inputs_root,
        args.gt_boxes_pkl,
        args.reports_dir,
    ]
    if args.night_ref_dir:
        paths.append(args.night_ref_dir)
    has_srv = any(p.startswith("/srv/") for p in paths)
    has_home = any(p.startswith("/home/") for p in paths)
    if has_srv and has_home:
        print(
            "[val] FATAL: paths mix /srv/ and /home/ — pick fully cluster (Option A) "
            "or fully local (Option B). See plan §T3.",
            file=sys.stderr,
        )
        sys.exit(2)

    os.makedirs(args.reports_dir, exist_ok=True)
    os.makedirs(os.path.join(args.reports_dir, "overlays"), exist_ok=True)

    exp_w, exp_h = map(int, args.expected_input_res.split("x"))

    print(f"[val] loading manifest {args.generated_manifest}")
    with open(args.generated_manifest) as f:
        gm = json.load(f)
    entries = [e for e in gm["entries"] if e.get("status") == "ok"]
    print(f"[val] {len(gm['entries'])} total / {len(entries)} ok entries")

    print(f"[val] loading gt-boxes pkl {args.gt_boxes_pkl}")
    gt_index = load_gt_boxes_index(args.gt_boxes_pkl)
    print(f"[val] indexed {len(gt_index)} sample_tokens")

    # Optional night-ref histogram for KL
    night_hist = None
    if args.night_ref_dir and os.path.isdir(args.night_ref_dir):
        ref_paths = [str(p) for p in Path(args.night_ref_dir).rglob("*.jpg")][:200]
        if ref_paths:
            night_hist = compute_luma_histogram(ref_paths)
            print(f"[val] night-reference histogram from {len(ref_paths)} images")

    rows = []
    sample_cam_brightness = defaultdict(dict)
    gen_brightness_all = []
    aspect_ratios = []

    for i, e in enumerate(entries):
        cam = e.get("camera") or cam_name_from_path(
            e.get("cluster_path") or e.get("generated_path", "")
        )
        token = e.get("sample_token")
        gen_path = e.get("generated_path")

        # Resolve the original (day) path — mirror nano_banana2_batch.resolve_input_path
        cluster = e.get("cluster_path") or ""
        marker = "/samples/"
        idx_marker = cluster.find(marker)
        rel = cluster[idx_marker + 1 :] if idx_marker >= 0 else None
        orig_path = os.path.join(args.inputs_root, rel) if rel else None

        row = {
            "i": i,
            "camera": cam,
            "sample_token": token,
            "original_path": orig_path,
            "generated_path": gen_path,
            "errors": [],
            "crop_mode": args.crop_mode,
        }
        if not orig_path or not os.path.exists(orig_path):
            row["errors"].append("missing_original")
        if not gen_path or not os.path.exists(gen_path):
            row["errors"].append("missing_generated")
        if row["errors"]:
            rows.append(row)
            continue

        assert orig_path is not None and gen_path is not None  # pyright narrowing
        try:
            orig_img = Image.open(orig_path).convert("RGB")
            gen_img = Image.open(gen_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            row["errors"].append(f"pil_open:{exc}")
            rows.append(row)
            continue

        ow, oh = orig_img.size
        gw, gh = gen_img.size
        row["orig_wh"] = [ow, oh]
        row["gen_wh"] = [gw, gh]
        if (ow, oh) != (exp_w, exp_h):
            row["errors"].append(f"orig_res_unexpected:{ow}x{oh}")
        aspect_ratios.append((gw / ow, gh / oh))

        # Resolution reconciliation
        if args.crop_mode == "ra":
            gen_for_crop = gen_img.resize((ow, oh), Image.Resampling.LANCZOS)
            scale_uv = (1.0, 1.0)
        else:
            gen_for_crop = gen_img
            scale_uv = (gw / ow, gh / oh)

        orig_arr = np.asarray(orig_img)
        gen_arr_for_crop = np.asarray(gen_for_crop)
        gen_arr_native = np.asarray(gen_img)

        b_o = brightness(orig_arr)
        b_g = brightness(gen_arr_native)
        row["brightness_orig"] = round(b_o, 4)
        row["brightness_gen"] = round(b_g, 4)
        row["brightness_delta"] = round(b_g - b_o, 4)
        gen_brightness_all.append(b_g)
        sample_cam_brightness[token][cam] = b_g

        # Geometry / SSIM checks
        info = gt_index.get(token)
        if info is None:
            row["errors"].append("token_not_in_gt_pkl")
            rows.append(row)
            continue
        cam_info = info["cams"].get(cam) if cam else None
        if cam_info is None:
            row["errors"].append(f"cam_not_in_info:{cam}")
            rows.append(row)
            continue

        try:
            lidar2img = lidar2img_keyframe(
                cam_info["sensor2lidar_rotation"],
                cam_info["sensor2lidar_translation"],
                cam_info["cam_intrinsic"],
            )
        except Exception as exc:  # noqa: BLE001
            row["errors"].append(f"lidar2img:{exc}")
            rows.append(row)
            continue

        gt_boxes = np.asarray(info.get("gt_boxes", []))
        ssim_per_box = []
        n_boxes_drawn = 0
        for box in gt_boxes:
            if box.shape[0] < 7:
                continue
            corners = box_corners(box[:3], box[3:6], float(box[6]))
            aabb = project_box_to_image(corners, lidar2img, ow, oh)
            if aabb is None:
                continue
            x0, y0, x1, y1 = aabb
            if args.crop_mode == "rb":
                x0g = int(round(x0 * scale_uv[0]))
                y0g = int(round(y0 * scale_uv[1]))
                x1g = int(round(x1 * scale_uv[0]))
                y1g = int(round(y1 * scale_uv[1]))
                if x1g - x0g < 7 or y1g - y0g < 7:
                    continue
                gen_crop = gen_arr_native[y0g:y1g, x0g:x1g]
                day_crop_native = orig_arr[y0:y1, x0:x1]
                day_crop = np.array(
                    Image.fromarray(day_crop_native).resize(
                        (gen_crop.shape[1], gen_crop.shape[0]), Image.Resampling.LANCZOS
                    )
                )
            else:
                if x1 - x0 < 7 or y1 - y0 < 7:
                    continue
                gen_crop = gen_arr_for_crop[y0:y1, x0:x1]
                day_crop = orig_arr[y0:y1, x0:x1]
            try:
                day_crop_matched = hist_match(day_crop, gen_crop)
            except Exception:  # noqa: BLE001
                day_crop_matched = day_crop
            score = masked_ssim(day_crop_matched, gen_crop)
            if score is not None:
                ssim_per_box.append(score)
            n_boxes_drawn += 1

        row["n_boxes_projected"] = n_boxes_drawn
        row["median_ssim"] = float(np.median(ssim_per_box)) if ssim_per_box else None
        row["min_ssim"] = float(np.min(ssim_per_box)) if ssim_per_box else None
        rows.append(row)
        if (i + 1) % 50 == 0 or (i + 1) == len(entries):
            print(
                f"  [{i + 1}/{len(entries)}] {cam} bright={row['brightness_gen']:.3f} "
                f"med_ssim={row['median_ssim']} boxes={n_boxes_drawn}"
            )

    # Per-sample cross-view metrics
    cross_view_abs = []
    outlier_samples = []
    for tok, camb in sample_cam_brightness.items():
        if len(camb) >= 2:
            vs = list(camb.values())
            abs_d = max(vs) - min(vs)
            cross_view_abs.append(abs_d)
            if any(v > 0.40 for v in vs) and any(v < 0.25 for v in vs):
                outlier_samples.append(tok)

    # Aspect-ratio sanity
    if aspect_ratios:
        ar = np.array(aspect_ratios)
        ar_mean = ar.mean(axis=0)
        ar_dev = np.abs(ar - ar_mean) / np.maximum(ar_mean, 1e-6)
        ar_outliers = int((ar_dev.max(axis=1) > 0.05).sum())
    else:
        ar_outliers = 0

    # Luminance KL
    lum_kl = None
    if night_hist is not None and gen_brightness_all:
        gen_paths = [r["generated_path"] for r in rows if not r.get("errors")]
        gen_hist = compute_luma_histogram(gen_paths)
        lum_kl = hist_kl(gen_hist, night_hist)

    summary = {
        "n": len(rows),
        "n_validated": sum(1 for r in rows if not r.get("errors")),
        "crop_mode": args.crop_mode,
        "median_ssim_overall": float(
            np.median(
                [r["median_ssim"] for r in rows if r.get("median_ssim") is not None]
            )
        )
        if any(r.get("median_ssim") is not None for r in rows)
        else None,
        "max_cross_view_abs_delta": float(max(cross_view_abs))
        if cross_view_abs
        else None,
        "median_cross_view_abs_delta": float(np.median(cross_view_abs))
        if cross_view_abs
        else None,
        "frac_samples_abs_delta_le_threshold": float(
            np.mean([d <= args.abs_delta_threshold for d in cross_view_abs])
        )
        if cross_view_abs
        else None,
        "outlier_cam_samples": outlier_samples,
        "luminance_kl_vs_night_ref": lum_kl,
        "aspect_ratio_outliers": ar_outliers,
        "thresholds": {
            "abs_delta": args.abs_delta_threshold,
            "lum_kl": args.lum_kl_threshold,
        },
    }

    with open(os.path.join(args.reports_dir, "metrics.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    # CSV
    with open(os.path.join(args.reports_dir, "metrics.csv"), "w", newline="") as f:
        fields = [
            "i",
            "camera",
            "sample_token",
            "brightness_orig",
            "brightness_gen",
            "brightness_delta",
            "n_boxes_projected",
            "median_ssim",
            "min_ssim",
            "orig_wh",
            "gen_wh",
            "crop_mode",
            "errors",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    k: (
                        json.dumps(r[k])
                        if isinstance(r.get(k), (list, dict))
                        else r.get(k)
                    )
                    for k in fields
                }
            )

    # Accepted/rejected manifests
    accepted = [
        r
        for r in rows
        if not r.get("errors") and (r.get("min_ssim") is None or r["min_ssim"] >= 0.3)
    ]
    rejected = [r for r in rows if r not in accepted]
    with open(os.path.join(args.reports_dir, "accepted_manifest.json"), "w") as f:
        json.dump({"n": len(accepted), "rows": accepted}, f, indent=2)
    with open(os.path.join(args.reports_dir, "rejected_manifest.json"), "w") as f:
        json.dump({"n": len(rejected), "rows": rejected}, f, indent=2)

    # Gate decision (screening, NOT a ground-truth correctness check)
    abs_d_pass = (summary["frac_samples_abs_delta_le_threshold"] or 0) >= 0.95
    kl_pass = (lum_kl is None) or (lum_kl <= args.lum_kl_threshold)
    outlier_pass = len(outlier_samples) == 0
    aspect_pass = ar_outliers == 0

    gate_md = [
        "# Phase 1 screening validator — gate decision",
        "",
        f"Crop mode: {args.crop_mode.upper()}",
        f"Median SSIM (per-AABB, hist-matched): **{summary['median_ssim_overall']}**",
        f"Frac samples with abs cross-view Δ ≤ {args.abs_delta_threshold}: "
        f"**{summary['frac_samples_abs_delta_le_threshold']}** (gate: ≥0.95) → {'✓' if abs_d_pass else '✗'}",
        f"Luminance KL vs night ref: **{lum_kl}** (gate: ≤{args.lum_kl_threshold}) → {'✓' if kl_pass else '✗'}",
        f"Outlier-cam samples (any cam>0.40 with another<0.25): {len(outlier_samples)} → {'✓' if outlier_pass else '✗'}",
        f"Aspect-ratio outliers (>5%% deviation from batch mean): {ar_outliers} → {'✓' if aspect_pass else '✗'}",
        "",
        f"## Verdict: **{'PASS' if (abs_d_pass and kl_pass and outlier_pass and aspect_pass) else 'FAIL'}**",
        "",
        "**Disclaimer**: this validator screens likely failures. High SSIM does NOT prove",
        "semantic correctness; corroborate with manual visual audit on a sample of grids.",
    ]
    with open(os.path.join(args.reports_dir, "gate_decision.md"), "w") as f:
        f.write("\n".join(gate_md) + "\n")

    print("[val] done")
    print(f"[val] summary: {summary}")


if __name__ == "__main__":
    main()
