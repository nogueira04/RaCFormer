"""Pre-registration unit test for the C2 REDESIGN. Same battery, same scenes, same offsets.

Scene selection and the synthetic offset draws are IMPORTED from test_c2_recovery so the
numbers are directly comparable with the failed original run
(robust_study_runs/c2_unittest/arm1_span030_20260803T200233Z).

Boundaries, unchanged: TRAIN split only, clean extrinsics plus synthetic offsets, no
detection metric, no model import, no forward pass, outputs outside the checkout.

Variants
  DOP    raw Doppler yaw estimate.
  DOP-B  DOP plus a per-sensor bias constant and a soft threshold, both fitted on a set of
         clean TRAIN scenes DISJOINT from the 20 evaluation scenes and with no injected
         fault. The disjointness is asserted at run time.

Both estimate the YAW component only; roll, pitch and translation are left untouched by
construction (see c2_doppler for the observability argument), so the residual is bounded
below by ``yaw_only_floor_deg``, the angle of the fault after its best possible yaw-only
correction. That floor is reported alongside every residual.

Local-optimality columns
  clean_local_min_yaw   -- is the Doppler cost a local minimum at the clean extrinsic along
                           the OBSERVABLE (yaw) direction, probed at +/-0.01 rad? Direct
                           counterpart of the original clean_is_local_max (0/100).
  psi_cost_min_off_deg  -- signed distance from the clean extrinsic to the actual minimiser
                           of the Doppler cost. This is the informative version of the same
                           question: the boolean above only says whether that distance is
                           under 0.01 rad.
  clean_local_min_rot3  -- the same boolean along all three rotation axes. Expected to be
                           low, because roll and pitch are flat directions of this cost. It
                           is reported so the restriction is visible rather than hidden.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_recalib as c2
import c2_doppler as dop
from test_c2_recovery import SCALES, draw_offset, select_scenes


def rot_z(p):
    return np.array([[np.cos(p), -np.sin(p), 0.0], [np.sin(p), np.cos(p), 0.0], [0.0, 0.0, 1.0]])


def best_yaw_only(dR_true, span=0.5):
    """argmin_psi angle(dR_true @ Rz(psi)): the best achievable yaw-only correction, and the
    residual angle it leaves. The estimator cannot beat this."""
    g = np.arange(-span, span + 1e-12, 2.5e-4)
    ang = [c2.rotation_angle_deg(dR_true @ rot_z(p)) for p in g]
    i = int(np.argmin(ang))
    return float(g[i]), float(ang[i])


def pick_bias_scenes(nusc, n, eval_tokens):
    """Train scenes for the offline bias fit, disjoint from the evaluation scenes."""
    from nuscenes.utils.splits import create_splits_scenes

    train = set(create_splits_scenes()["train"])
    scl = sorted([s for s in nusc.scene if s["name"] in train], key=lambda s: s["name"])
    cand = [s for s in scl if s["token"] not in eval_tokens]
    idx = np.unique(np.linspace(0, len(cand) - 1, n).astype(int))
    out = [cand[i] for i in idx]
    assert not (set(s["token"] for s in out) & set(eval_tokens)), "bias set overlaps eval set"
    return out


def run_scene(args_tuple):
    (nusc, scene, sensors, scales, seed_index, cfg, bias, lam, sfloor, method,
     verbose) = args_tuple
    est = dop.estimate_yaw_kellner if method == "kellner" else dop.estimate_yaw
    rows = []
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    t_sc = time.time()
    for ch in sensors:
        t_b = time.time()
        cache = dop.SensorDopplerCache(nusc, scene["token"], ch, cfg=cfg)
        build_s = time.time() - t_b
        for sname in scales:
            sigma_r, sigma_t = SCALES[sname]
            a, z_rot, z_t = draw_offset(seed_index, scene["token"], ch)
            theta = sigma_r * z_rot
            rvec_true = theta * a
            dt_true = sigma_t * z_t
            dR_true = c2.rodrigues(rvec_true)
            base_deg = c2.rotation_angle_deg(dR_true)
            psi_opt, floor_deg = best_yaw_only(dR_true)

            lmin_yaw, lmin_yaw5, lmin_rot3, cost_off = False, False, False, float("nan")
            if cache.n_frames == 0 or cache.n_static == 0:
                res = {"channel": ch, "scene_name": scene["name"], "scene_token": scene["token"],
                       "n_frames": cache.n_frames, "n_static_points": cache.n_static,
                       "fallback": True, "fallback_reason": "no usable frames or static returns",
                       "psi": 0.0, "seconds": 0.0}
                dop.apply_corrections(res)
            else:
                R_p, t_p = cache.perturbed_extrinsic(rvec_true, dt_true)
                cache.set_extrinsic(R_p, t_p)
                res = est(cache, cfg=cfg, log=log)
                sc_r = res.get("robust_scale", None)
                # Local optimality of the objective at the clean extrinsic, and the actual
                # distance from the clean extrinsic to the cost minimiser.
                c_cl = cache.cost_rvec_or_psi(psi_opt, sc_r)
                lmin_yaw = bool(c_cl <= min(cache.cost_rvec_or_psi(psi_opt + s * 0.01, sc_r)
                                            for s in (1.0, -1.0)))
                # second radius, matched to the fault scale rather than to the original
                # probe: Levinson and Thrun note the perturbation size should equal the
                # granularity of miscalibration one wants to detect (sigma_r = 0.10 rad here).
                lmin_yaw5 = bool(c_cl <= min(cache.cost_rvec_or_psi(psi_opt + s * 0.05, sc_r)
                                             for s in (1.0, -1.0)))
                gg = np.arange(psi_opt - 0.06, psi_opt + 0.06 + 1e-12, 5e-4)
                cc = [cache.cost_rvec_or_psi(float(p), sc_r) for p in gg]
                cost_off = float(np.degrees(gg[int(np.argmin(cc))] - psi_opt))
                base = -np.array(rvec_true)
                c3 = dop.cost_rvec(cache, base, sc_r)
                nb = [dop.cost_rvec(cache, base + s * 0.01 * np.eye(3)[k], sc_r)
                      for k in range(3) for s in (1.0, -1.0)]
                lmin_rot3 = bool(c3 <= min(nb))
                dop.apply_corrections(
                    res, bias=bias.get(ch, 0.0),
                    shrink_lambda=(lam.get(ch, 0.0) if isinstance(lam, dict) else lam),
                    sigma_floor=sfloor)

            psi_hat = float(res["psi"])
            resid_deg = c2.rotation_angle_deg(dR_true @ rot_z(psi_hat))
            resid_mm = float(np.linalg.norm(dt_true) * 1000.0)   # translation untouched

            row = dict(res)
            row.update({
                "variant": "DOP-B" if (bias or lam) else "DOP",
                "scale": sname, "sigma_r": sigma_r, "sigma_t": sigma_t,
                "seed_index": seed_index,
                "theta_true_rad": theta, "axis_true": a.tolist(),
                "rvec_true": rvec_true.tolist(), "dt_true": dt_true.tolist(),
                "resid_angle_deg": resid_deg, "baseline_angle_deg": base_deg,
                "yaw_only_floor_deg": floor_deg,
                "headroom_deg": base_deg - floor_deg,
                "psi_opt_rad": psi_opt, "psi_hat_rad": psi_hat,
                "psi_err_deg": float(np.degrees(abs(psi_hat - psi_opt))),
                "psi_raw_err_deg": float(np.degrees(abs(res.get("psi_raw", 0.0) - psi_opt)))
                                   if not res["fallback"] else float("nan"),
                "psi_cost_min_off_deg": cost_off,
                "resid_trans_mm": resid_mm, "baseline_trans_mm": resid_mm,
                "clean_local_min_yaw": lmin_yaw, "clean_local_min_yaw_r05": lmin_yaw5,
                "clean_local_min_rot3": lmin_rot3,
                "cache_build_seconds": build_s,
            })
            rows.append(row)
            log("  [{} {} {}] resid {:.3f} (was {:.3f}, floor {:.3f}) | psi {:+.4f} vs opt "
                "{:+.4f} err {:.2f} deg | costmin_off {:+.2f} deg | lmin={} | {} pts{}".format(
                    scene["name"], ch, sname, resid_deg, base_deg, floor_deg, psi_hat, psi_opt,
                    np.degrees(abs(psi_hat - psi_opt)), cost_off, lmin_yaw, cache.n_static,
                    " FALLBACK:" + res["fallback_reason"] if res["fallback"] else ""))
        del cache
    return rows, time.time() - t_sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-scenes", type=int, default=2)
    ap.add_argument("--scene-offset", type=int, default=0)
    ap.add_argument("--seed-index", type=int, default=0)
    ap.add_argument("--sensors", default=",".join(c2.RADAR_CHANNELS))
    ap.add_argument("--scales", default="severe,medium")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-sigma-psi", type=float, default=0.02)
    ap.add_argument("--min-speed", type=float, default=1.0)
    ap.add_argument("--bias-scenes", type=int, default=0,
                    help="0 disables the offline bias/shrink calibration (variant DOP)")
    ap.add_argument("--shrink-lambda", type=float, default=0.0)
    ap.add_argument("--shrink-mode", default="const", choices=["const", "scatter"],
                    help="scatter: per-sensor lambda = the estimator noise scale measured "
                         "on the held-out clean scenes")
    ap.add_argument("--method", default="global", choices=["global", "kellner"])
    ap.add_argument("--use-sweeps", action="store_true")
    ap.add_argument("--sigma-floor", type=float, default=0.010)
    ap.add_argument("--tag", default="dop_pilot")
    ap.add_argument("--out-root", default="/srv/nfs/shared/gnmp/robust_study_runs/c2_redesign")
    ap.add_argument("--dataroot", default="data/nuscenes/")
    ap.add_argument("--version", default="v1.0-trainval")
    a = ap.parse_args()

    os.chdir("/srv/nfs/shared/gnmp/RaCFormer")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(a.out_root, "{}_{}".format(a.tag, stamp))
    if os.path.exists(out_dir):
        raise SystemExit("refusing to reuse existing run dir: " + out_dir)
    os.makedirs(out_dir)
    print("[c2v2] out_dir=" + out_dir, flush=True)

    from nuscenes.nuscenes import NuScenes

    t0 = time.time()
    nusc = NuScenes(version=a.version, dataroot=a.dataroot, verbose=False)
    print("[c2v2] devkit loaded in {:.1f}s".format(time.time() - t0), flush=True)

    scenes = select_scenes(nusc, a.num_scenes, a.scene_offset)
    sensors = [x for x in a.sensors.split(",") if x]
    scales = [x for x in a.scales.split(",") if x]
    cfg = dop.DopplerConfig(max_sigma_psi=a.max_sigma_psi, min_speed=a.min_speed,
                            use_sweeps=a.use_sweeps)
    print("[c2v2] {} scenes: {}".format(len(scenes), [s["name"] for s in scenes]), flush=True)

    calib = {"bias": {}, "scatter": {}, "scenes": []}
    if a.bias_scenes > 0:
        ev = set(s["token"] for s in scenes)
        bscenes = pick_bias_scenes(nusc, a.bias_scenes, ev)
        print("[c2v2] bias-fit scenes ({}, disjoint): {}".format(
            len(bscenes), [s["name"] for s in bscenes]), flush=True)
        t_b = time.time()
        calib = dop.calibrate_bias(
            nusc, bscenes, sensors, cfg=cfg, log=lambda *x: print(*x, flush=True),
            est=dop.estimate_yaw_kellner if a.method == "kellner" else dop.estimate_yaw)
        print("[c2v2] bias fitted in {:.1f}s".format(time.time() - t_b), flush=True)

    meta = {"utc": stamp, "argv": sys.argv, "args": vars(a),
            "variant": ("DOP-K" if a.method == "kellner" else "DOP")
                       + ("-B" if a.bias_scenes > 0 else "")
                       + ("-S" if a.use_sweeps else ""), "split": "train",
            "scenes": [{"name": s["name"], "token": s["token"]} for s in scenes],
            "sensors": sensors, "scales": {k: SCALES[k] for k in scales},
            "doppler_config": vars(cfg),
            "calibration": {"bias": calib["bias"], "scatter": calib["scatter"],
                            "scenes": calib["scenes"], "shrink_lambda": a.shrink_lambda,
                            "shrink_mode": a.shrink_mode,
                            "sigma_floor": a.sigma_floor},
            "note": "train split only; no detection metric; no model forward pass; offsets "
                    "imported from test_c2_recovery.draw_offset (identical draws); bias-fit "
                    "scenes are disjoint from evaluation scenes and carry no injected fault"}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    lam = a.shrink_lambda
    if a.shrink_mode == "scatter":
        # per-sensor soft threshold: the estimator noise scale measured on the disjoint
        # clean scenes. Nothing from the evaluation set or the injected offsets enters.
        lam = {c: (calib["scatter"].get(c) if np.isfinite(calib["scatter"].get(c, np.nan))
                   else a.shrink_lambda) for c in sensors}
        print("[c2v2] per-sensor lambda (deg): " + " ".join(
            "{}={:.3f}".format(k.replace("RADAR_", ""), np.degrees(v))
            for k, v in lam.items()), flush=True)
        meta["calibration"]["shrink_lambda_per_sensor"] = lam
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    tasks = [(nusc, s, sensors, scales, a.seed_index, cfg, calib["bias"], lam,
              a.sigma_floor, a.method, True) for s in scenes]
    rows = []
    t_run = time.time()
    if a.workers <= 1:
        for t in tasks:
            r, dt = run_scene(t)
            rows.extend(r)
    else:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(a.workers) as pool:
            for r, dt in pool.imap_unordered(run_scene, tasks):
                rows.extend(r)
    total = time.time() - t_run

    with open(os.path.join(out_dir, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = {"total_seconds": total, "n_rows": len(rows), "method": a.method,
               "use_sweeps": a.use_sweeps, "by_scale": {}}
    for sname in scales:
        sub = [r for r in rows if r["scale"] == sname]
        if not sub:
            continue

        def q(key):
            v = np.array([r.get(key, np.nan) for r in sub], dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return None
            return {"median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
                    "q3": float(np.percentile(v, 75)), "min": float(v.min()),
                    "max": float(v.max()), "n": int(v.size)}

        nf = [r for r in sub if not r["fallback"]]
        summary["by_scale"][sname] = {
            "resid_angle_deg": q("resid_angle_deg"),
            "baseline_angle_deg": q("baseline_angle_deg"),
            "yaw_only_floor_deg": q("yaw_only_floor_deg"),
            "headroom_deg": q("headroom_deg"),
            "psi_err_deg": q("psi_err_deg"),
            "psi_raw_err_deg": q("psi_raw_err_deg"),
            "psi_cost_min_off_deg": q("psi_cost_min_off_deg"),
            "resid_trans_mm": q("resid_trans_mm"),
            "sigma_psi_deg": q("sigma_psi_deg"),
            "n_static_points": q("n_static_points"),
            "fallback_count": int(sum(1 for r in sub if r["fallback"])),
            "fallback_rate": float(np.mean([r["fallback"] for r in sub])),
            "fallback_reasons": sorted(set(r["fallback_reason"] for r in sub if r["fallback"])),
            "clean_local_min_yaw_frac": float(np.mean([r["clean_local_min_yaw"] for r in sub])),
            "clean_local_min_yaw_r05_frac": float(np.mean(
                [r["clean_local_min_yaw_r05"] for r in sub])),
            "clean_local_min_rot3_frac": float(np.mean([r["clean_local_min_rot3"] for r in sub])),
            "improved_angle_frac": float(np.mean(
                [r["resid_angle_deg"] < r["baseline_angle_deg"] for r in sub])),
            "improved_angle_frac_nonfallback": (float(np.mean(
                [r["resid_angle_deg"] < r["baseline_angle_deg"] for r in nf])) if nf else None),
            "harmed_angle_frac": float(np.mean(
                [r["resid_angle_deg"] > r["baseline_angle_deg"] + 1e-9 for r in sub])),
            "harmed_gt_0p5deg_frac": float(np.mean(
                [r["resid_angle_deg"] > r["baseline_angle_deg"] + 0.5 for r in sub])),
            "frac_of_headroom_captured": float(np.median(
                [(r["baseline_angle_deg"] - r["resid_angle_deg"]) / r["headroom_deg"]
                 for r in sub if r["headroom_deg"] > 1e-6])),
            "seconds_per_pair": q("seconds"),
        }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["by_scale"], indent=2), flush=True)
    print("[c2v2] wrote " + out_dir, flush=True)


if __name__ == "__main__":
    main()
