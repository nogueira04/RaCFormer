"""Pre-registration unit test for mitigation candidate C2.

Synthetic-offset recovery on TRAIN-split scenes with CLEAN extrinsics. A known (d2)-shaped
offset is injected into the radar->ego extrinsic of one (scene, sensor); the C2 optimiser
then tries to undo it from images alone, and we measure how much of the offset is left.

Boundaries this script enforces by construction:
  * scenes come from ``nuscenes.utils.splits`` train only; the val split is never touched;
  * no detection metric is computed and no model is imported;
  * every output goes to a directory outside the checkout.

Offsets follow fault-families.md "(d2) implementation spec" items 2-5:
  rotation  dR = expm([theta * a]x) applied in the sensor frame (R -> R @ dR),
            a uniform on S^2, theta ~ N(0, sigma_r);
  translation  t -> t + dt, dt three iid N(0, sigma_t) in the ego frame;
  draws come from an RNG seeded by the first 8 bytes of SHA-256("<s>:<scene>:<channel>"),
  and severity scales the SAME draws, so the two levels share common random numbers.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_recalib as c2

SCALES = {
    "severe": (0.10, 0.010),   # Dong level 5
    "medium": (0.06, 0.006),   # Dong level 3
}


def draw_offset(seed_index, scene_token, channel):
    key = "{}:{}:{}".format(seed_index, scene_token, channel)
    h = hashlib.sha256(key.encode("utf-8")).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(h, "big"))
    a = rng.standard_normal(3)
    a = a / np.linalg.norm(a)
    z_rot = float(rng.standard_normal())
    z_t = rng.standard_normal(3)
    return a, z_rot, z_t


def select_scenes(nusc, n_scenes, offset):
    from nuscenes.utils.splits import create_splits_scenes

    train = set(create_splits_scenes()["train"])
    scenes = sorted([s for s in nusc.scene if s["name"] in train], key=lambda s: s["name"])
    idx = np.unique(np.linspace(0, len(scenes) - 1, n_scenes + offset).astype(int))[offset:]
    return [scenes[i] for i in idx]


def run_scene(args_tuple):
    nusc, scene, sensors, scales, seed_index, search, scfg, verbose = args_tuple
    import cv2

    cv2.setNumThreads(2)
    rows = []
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    t_sc = time.time()
    cam = c2.SceneCameraCache(nusc, scene["token"], cfg=scfg)
    log("[scene {}] gradient cache built in {:.1f}s, {:.0f} MB, {} samples".format(
        scene["name"], cam.build_seconds, cam.memory_mb(), cam.n_samples))

    for ch in sensors:
        sp = c2.SensorPointCache(nusc, cam, ch, cfg=scfg)
        for sname in scales:
            sigma_r, sigma_t = SCALES[sname]
            a, z_rot, z_t = draw_offset(seed_index, scene["token"], ch)
            theta = sigma_r * z_rot
            rvec_true = theta * a
            dt_true = sigma_t * z_t

            R_p, t_p = sp.perturbed_extrinsic(rvec_true, dt_true)
            sp.set_extrinsic(R_p, t_p)

            fine = len(scfg.levels) - 1
            s_pert, n_pert = sp.score(np.zeros(3), np.zeros(3), fine)
            s_clean, n_clean = sp.score(-rvec_true, -dt_true, fine)

            res = c2.optimise(sp, search=search, log=log)

            # Is the CLEAN extrinsic even a local maximum of the score? Probe the 12
            # axis-aligned neighbours of truth at a small step. This separates "the search
            # failed" from "the objective does not peak at the right answer".
            s_nb = []
            for k in range(6):
                for sg in (1.0, -1.0):
                    rr, dd = -np.array(rvec_true), -np.array(dt_true)
                    if k < 3:
                        rr = rr + sg * 0.01 * np.eye(3)[k]
                    else:
                        dd = dd + sg * 0.005 * np.eye(3)[k - 3]
                    s_nb.append(sp.score(rr, dd, fine)[0])
            s_best_val, n_best = sp.score(res["rvec"], res["dt"], fine)

            dR_true = c2.rodrigues(rvec_true)
            dR_est = c2.rodrigues(res["rvec"])
            resid_deg = c2.rotation_angle_deg(dR_true @ dR_est)
            base_deg = c2.rotation_angle_deg(dR_true)
            resid_mm = float(np.linalg.norm(dt_true + np.array(res["dt"])) * 1000.0)
            base_mm = float(np.linalg.norm(dt_true) * 1000.0)

            row = dict(res)
            row.update({
                "scale": sname,
                "sigma_r": sigma_r,
                "sigma_t": sigma_t,
                "seed_index": seed_index,
                "theta_true_rad": theta,
                "axis_true": a.tolist(),
                "rvec_true": rvec_true.tolist(),
                "dt_true": dt_true.tolist(),
                "resid_angle_deg": resid_deg,
                "baseline_angle_deg": base_deg,
                "resid_trans_mm": resid_mm,
                "baseline_trans_mm": base_mm,
                "score_clean": s_clean,
                "score_perturbed": s_pert,
                "n_valid_pairs_clean": n_clean,
                "n_valid_pairs_perturbed": n_pert,
                "clean_scores_higher": bool(s_clean > s_pert),
                "best_beats_clean": bool(res["score_best"] > s_clean),
                "clean_is_local_max": bool(s_clean >= max(s_nb)),
                "score_neighbour_max": float(max(s_nb)),
                "n_valid_pairs_best": int(n_best),
                "score_ratio_best_over_clean": float(res["score_best"] / s_clean) if s_clean > 0 else None,
                "scene_build_seconds": cam.build_seconds,
            })
            rows.append(row)
            log("  [{} {} {}] resid {:.3f} deg (was {:.3f}) / {:.2f} mm (was {:.2f}) "
                "| S clean {:.4f} pert {:.4f} best {:.4f} | {} pts | {:.1f}s{}".format(
                    scene["name"], ch, sname, resid_deg, base_deg, resid_mm, base_mm,
                    s_clean, s_pert, res["score_best"], res["n_points_total"], res["seconds"],
                    " FALLBACK:" + res["fallback_reason"] if res["fallback"] else ""))
        sp.set_extrinsic(sp.R_clean, sp.t_clean)
        del sp
    del cam
    return rows, time.time() - t_sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-scenes", type=int, default=2)
    ap.add_argument("--scene-offset", type=int, default=0)
    ap.add_argument("--seed-index", type=int, default=0)
    ap.add_argument("--sensors", default=",".join(c2.RADAR_CHANNELS))
    ap.add_argument("--scales", default="severe,medium")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--coarse-rot-span", type=float, default=0.30)
    ap.add_argument("--coarse-rot-steps", type=int, default=13)
    ap.add_argument("--mid-rot-span", type=float, default=0.05)
    ap.add_argument("--mid-trans-span", type=float, default=0.03)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--out-root", default="/srv/nfs/shared/gnmp/robust_study_runs/c2_unittest")
    ap.add_argument("--dataroot", default="data/nuscenes/")
    ap.add_argument("--version", default="v1.0-trainval")
    a = ap.parse_args()

    os.chdir("/srv/nfs/shared/gnmp/RaCFormer")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(a.out_root, "{}_{}".format(a.tag, stamp))
    if os.path.exists(out_dir):
        raise SystemExit("refusing to reuse existing run dir: " + out_dir)
    os.makedirs(out_dir)
    print("[c2] out_dir=" + out_dir, flush=True)

    from nuscenes.nuscenes import NuScenes

    t0 = time.time()
    nusc = NuScenes(version=a.version, dataroot=a.dataroot, verbose=False)
    print("[c2] devkit loaded in {:.1f}s".format(time.time() - t0), flush=True)

    scenes = select_scenes(nusc, a.num_scenes, a.scene_offset)
    sensors = [x for x in a.sensors.split(",") if x]
    scales = [x for x in a.scales.split(",") if x]
    search = c2.SearchConfig(timeout_s=a.timeout, coarse_rot_span=a.coarse_rot_span,
                             coarse_rot_steps=a.coarse_rot_steps, mid_rot_span=a.mid_rot_span,
                             mid_trans_span=a.mid_trans_span)
    scfg = c2.ScoreConfig()

    meta = {
        "utc": stamp,
        "argv": sys.argv,
        "args": vars(a),
        "split": "train",
        "scenes": [{"name": s["name"], "token": s["token"], "nbr_samples": s["nbr_samples"]}
                   for s in scenes],
        "sensors": sensors,
        "scales": {k: SCALES[k] for k in scales},
        "score_config": {"levels": scfg.levels, "min_depth": scfg.min_depth,
                         "grad_clip": scfg.grad_clip, "min_distance": scfg.min_distance},
        "search_config": vars(search),
        "note": "train split only; no detection metric; no model forward pass",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[c2] {} scenes: {}".format(len(scenes), [s["name"] for s in scenes]), flush=True)

    tasks = [(nusc, s, sensors, scales, a.seed_index, search, scfg, True) for s in scenes]
    rows = []
    t_run = time.time()
    if a.workers <= 1:
        for t in tasks:
            r, dt = run_scene(t)
            rows.extend(r)
            print("[c2] scene done in {:.1f}s".format(dt), flush=True)
    else:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(a.workers) as pool:
            for r, dt in pool.imap_unordered(run_scene, tasks):
                rows.extend(r)
                print("[c2] scene done in {:.1f}s".format(dt), flush=True)
    total = time.time() - t_run

    with open(os.path.join(out_dir, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = {"total_seconds": total, "n_rows": len(rows), "by_scale": {}}
    for sname in scales:
        sub = [r for r in rows if r["scale"] == sname]
        if not sub:
            continue

        def q(key):
            v = np.array([r[key] for r in sub], dtype=float)
            return {"median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
                    "q3": float(np.percentile(v, 75)), "min": float(v.min()),
                    "max": float(v.max()), "n": int(v.size)}

        summary["by_scale"][sname] = {
            "resid_angle_deg": q("resid_angle_deg"),
            "baseline_angle_deg": q("baseline_angle_deg"),
            "resid_trans_mm": q("resid_trans_mm"),
            "baseline_trans_mm": q("baseline_trans_mm"),
            "seconds_per_pair": q("seconds"),
            "n_points_total": q("n_points_total"),
            "points_per_frame_median": q("points_per_frame_median"),
            "fallback_count": int(sum(1 for r in sub if r["fallback"])),
            "fallback_rate": float(np.mean([r["fallback"] for r in sub])),
            "clean_scores_higher_frac": float(np.mean([r["clean_scores_higher"] for r in sub])),
            "clean_is_local_max_frac": float(np.mean([r["clean_is_local_max"] for r in sub])),
            "score_ratio_best_over_clean": q("score_ratio_best_over_clean"),
            "n_valid_pairs_identity": q("n_valid_pairs_identity"),
            "best_beats_clean_frac": float(np.mean([r["best_beats_clean"] for r in sub])),
            "improved_angle_frac": float(np.mean(
                [r["resid_angle_deg"] < r["baseline_angle_deg"] for r in sub])),
            "fallback_reasons": sorted(set(r["fallback_reason"] for r in sub if r["fallback"])),
        }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("[c2] wrote " + out_dir, flush=True)


if __name__ == "__main__":
    main()
