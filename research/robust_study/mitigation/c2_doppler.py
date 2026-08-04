"""C2 redesign, variant DOP: radar->ego extrinsic YAW from static-target Doppler.

Motivation. The original C2 score (mean image-gradient magnitude at radar-point pixel
projections) failed its unit test in 0/100 cases because it is texture-seeking: it is
monotone in "points land on strong image response", so the optimiser walks points off the
horizon band onto foliage and buildings, and the clean extrinsic is not a local maximum.
This variant abandons the image entirely.

Mechanism (prior art). For a STATIC world target the radial (Doppler) velocity measured by
the radar is the negative projection of the sensor own velocity onto the line of sight:

    v_r(phi) = -dhat(phi)^T * v_S,     v_S = R^T (v_ego + omega x t)

so the distribution of radial velocity over azimuth is a cosine whose PHASE is the direction
of travel expressed in the radar azimuth frame. A mounting-yaw error shifts that phase by
exactly the yaw error. This is the Kellner et al. (ITSC 2013 / ICRA 2014) instantaneous
ego-motion formulation read backwards: with the ego twist known from odometry, the residual
of the same cosine fit identifies the radar mounting yaw. It uses no image, so it cannot be
texture-seeking, and it is the standard basis of Doppler-based radar mounting-angle
estimation (Doer and Trommer 2020; Wise et al. ICRA 2021; arXiv:2511.01431).

Observability. nuScenes radar points carry z == 0 in the sensor frame, so every line of
sight dhat lies in the sensor xy-plane, and v_S is near-horizontal in that same frame. The
first-order sensitivity of v_r to a rotation perturbation r is d(v_r) = r . (u x dhat) with
u = R^T v_s; the cross product of two in-plane vectors is along the sensor z axis, hence
ONLY the z component (yaw) of r is observable from Doppler. Roll and pitch are not, at any
signal-to-noise ratio. This variant therefore estimates yaw only and leaves roll, pitch and
translation untouched, which is a deliberate, structural choice, not a shortcut.

Why yaw is the component worth having: for a point p = (x, y, 0) in the sensor frame, a
rotation error r displaces it in the ego frame by R (r x p). With p in the sensor xy-plane,
the z component of r moves p WITHIN that plane (a horizontal / BEV displacement of |r_z||p|)
while the x and y components move it perpendicular to the plane (a purely vertical
displacement). Doppler recovers exactly the component that displaces radar returns in BEV.

Failure-safe. The estimate is accompanied by its Cramer-Rao standard deviation
sigma_psi = sigma_res / sqrt(I), where I = sum_i (d r_i / d psi)^2 is the Fisher information
of the yaw parameter under the cosine model. I collapses to zero exactly when the geometry
degenerates -- ego vehicle stationary (v_S = 0 makes every a_i, b_i zero), too few static
returns, or all returns at one azimuth -- so the gate fires on geometric degradation rather
than on the score failing to improve. This is the degenerate-direction test of Zhang, Kaess
and Singh (ICRA 2016) specialised to a one-parameter problem.
"""

import os
import time
from dataclasses import dataclass

import numpy as np
from pyquaternion import Quaternion

# .pcd column indices, from the nuScenes devkit RadarPointCloud header:
# x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state
# x_rms y_rms invalid_state pdh0 vx_rms vy_rms
IX, IY, IDYN, IRCS, IVX, IVY, IAMB, IINV = 0, 1, 3, 5, 6, 7, 11, 14

# dyn_prop values that assert a STATIONARY cluster: 1 stationary, 3 stationary candidate,
# 5 crossing stationary. Moving / oncoming / crossing-moving / stopped are dropped.
STATIC_DYNPROP = (1, 3, 5)
# invalid_state values the devkit documents as VALID clusters.
VALID_INVALID = (0, 4, 8, 9, 10, 11, 12, 15, 16, 17)


@dataclass
class DopplerConfig:
    use_sweeps: bool = False        # every radar cycle in the scene, not just keyframes
    min_range: float = 1.0          # m, devkit remove_close, matches the production loader
    trunc: float = 2.0              # m/s, truncated-quadratic cutoff for the coarse stage
    coarse_span: float = 0.35       # rad, +/- (3.5 sigma at the severe level)
    coarse_step: float = 0.005      # rad
    fine_span: float = 0.02         # rad
    fine_step: float = 2.5e-4       # rad
    huber_floor: float = 0.20       # m/s, lower bound on the robust scale
    huber_k: float = 3.0            # robust scale = huber_k * 1.4826 * MAD
    # --- failure-safe thresholds ---
    min_points: int = 300           # static returns in the whole scene
    min_inliers: int = 200          # points inside the robust scale at the optimum
    min_speed: float = 1.0          # m/s, median |v_S| over the frames used
    max_sigma_psi: float = 0.02     # rad, Cramer-Rao 1-sigma of the yaw estimate
    max_psi: float = 0.35           # rad, divergence bound


def _ego_twist(nusc, sd):
    """Linear velocity (ego frame, m/s) and angular velocity (ego frame, rad/s) at this
    sample_data, by central difference over the sample_data prev/next chain. Radar
    sample_data run at ~13 Hz, so the differencing interval is ~150 ms."""
    prev = nusc.get("sample_data", sd["prev"]) if sd["prev"] else sd
    nxt = nusc.get("sample_data", sd["next"]) if sd["next"] else sd
    if prev["token"] == nxt["token"]:
        return None
    p0 = nusc.get("ego_pose", prev["ego_pose_token"])
    p1 = nusc.get("ego_pose", nxt["ego_pose_token"])
    pc = nusc.get("ego_pose", sd["ego_pose_token"])
    dt = (p1["timestamp"] - p0["timestamp"]) * 1e-6
    if dt <= 1e-3:
        return None
    v_glob = (np.array(p1["translation"]) - np.array(p0["translation"])) / dt
    Rc = Quaternion(pc["rotation"]).rotation_matrix
    v_ego = Rc.T @ v_glob
    dR = Quaternion(p0["rotation"]).rotation_matrix.T @ Quaternion(p1["rotation"]).rotation_matrix
    ang = float(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
    ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
    n = np.linalg.norm(ax)
    w = (ax / n * ang / dt) if n > 1e-12 else np.zeros(3)
    return v_ego, w


class SensorDopplerCache(object):
    """Per (scene, sensor). Holds the extrinsic-INDEPENDENT measurements (azimuth, radial
    speed, frame index) and the per-frame ego twist; the extrinsic only enters through the
    per-frame sensor velocity u = R_cur^T (v_ego + omega x t_cur), recomputed by
    ``set_extrinsic``."""

    def __init__(self, nusc, scene_token, channel, cfg=None):
        from nuscenes.utils.data_classes import RadarPointCloud

        self.cfg = cfg or DopplerConfig()
        self.channel = channel
        self.scene_token = scene_token
        scene = nusc.get("scene", scene_token)
        self.scene_name = scene["name"]

        phis, vrs, fidx, rng = [], [], [], []
        v_ego, omega, R_cl, t_cl = [], [], [], []
        # Frame list: the 40 annotated keyframes, or -- with use_sweeps -- every radar
        # sample_data in the scene (about 13 Hz, ~260 cycles). The extrinsic under test is
        # the same record either way; sweeps only add measurement cycles.
        sds = []
        if self.cfg.use_sweeps:
            cur = nusc.get("sample_data",
                           nusc.get("sample", scene["first_sample_token"])["data"][channel])
            while cur:
                sds.append(cur)
                cur = nusc.get("sample_data", cur["next"]) if cur["next"] else None
        else:
            tok = scene["first_sample_token"]
            while tok:
                smp = nusc.get("sample", tok)
                sds.append(nusc.get("sample_data", smp["data"][channel]))
                tok = smp["next"]
        k = 0
        n_raw = 0
        for sd in sds:
            tw = _ego_twist(nusc, sd)
            if tw is None:
                continue
            RadarPointCloud.disable_filters()
            pc = RadarPointCloud.from_file(os.path.join(nusc.dataroot, sd["filename"]))
            RadarPointCloud.default_filters()
            P = pc.points
            n_raw += P.shape[1]
            r = np.hypot(P[IX], P[IY])
            keep = (np.isin(P[IDYN].astype(int), STATIC_DYNPROP)
                    & np.isin(P[IINV].astype(int), VALID_INVALID)
                    & (P[IAMB].astype(int) == 3)
                    & (r > self.cfg.min_range))
            if keep.sum() == 0:
                continue
            x, y, rr = P[IX, keep], P[IY, keep], r[keep]
            dx, dy = x / rr, y / rr
            # (vx, vy) projected on the line of sight IS the Doppler radial speed; the
            # tangential part of the reported vector carries no measurement.
            vr = P[IVX, keep] * dx + P[IVY, keep] * dy
            phis.append(np.arctan2(y, x))
            vrs.append(vr)
            rng.append(rr)
            fidx.append(np.full(rr.size, k, dtype=np.int64))
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            R_cl.append(Quaternion(cs["rotation"]).rotation_matrix)
            t_cl.append(np.array(cs["translation"], dtype=np.float64))
            v_ego.append(tw[0])
            omega.append(tw[1])
            k += 1

        self.n_frames = k
        self.n_raw_points = n_raw
        if k == 0:
            self.phi = np.zeros(0)
            self.vr = np.zeros(0)
            self.rng = np.zeros(0)
            self.fidx = np.zeros(0, dtype=np.int64)
            self.R_clean = np.zeros((0, 3, 3))
            self.t_clean = np.zeros((0, 3))
            self.v_ego = np.zeros((0, 3))
            self.omega = np.zeros((0, 3))
            self.n_static = 0
            return
        self.phi = np.concatenate(phis)
        self.vr = np.concatenate(vrs)
        self.rng = np.concatenate(rng)
        self.fidx = np.concatenate(fidx)
        self.R_clean = np.stack(R_cl)
        self.t_clean = np.stack(t_cl)
        self.v_ego = np.stack(v_ego)
        self.omega = np.stack(omega)
        self.n_static = int(self.phi.size)
        self.set_extrinsic(self.R_clean, self.t_clean)

    def set_extrinsic(self, R, t):
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        if R.ndim == 2:
            R = np.broadcast_to(R, (self.n_frames, 3, 3))
        if t.ndim == 1:
            t = np.broadcast_to(t, (self.n_frames, 3))
        self.R_cur = np.array(R)
        self.t_cur = np.array(t)
        v_s = self.v_ego + np.cross(self.omega, self.t_cur)          # (F,3) ego frame
        u = np.einsum("fji,fj->fi", self.R_cur, v_s)                 # R^T v_s, sensor frame
        self.u = u
        self.speed = np.linalg.norm(v_s, axis=1)
        if self.n_static == 0:
            return
        c, s = np.cos(self.phi), np.sin(self.phi)
        ux, uy = u[self.fidx, 0], u[self.fidx, 1]
        # residual(psi) = vr + a cos(psi) + b sin(psi)
        self.a = ux * c + uy * s
        self.b = uy * c - ux * s

    def perturbed_extrinsic(self, rvec, dt):
        """The (d2) fault applied to the clean extrinsic: R @ expm(r), t + dt."""
        import cv2

        dR, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        return np.einsum("fij,jk->fik", self.R_clean, np.asarray(dR)), self.t_clean + np.asarray(dt)

    def residuals(self, psi):
        return self.vr + self.a * np.cos(psi) + self.b * np.sin(psi)

    def cost_trunc(self, psi):
        r2 = self.residuals(psi) ** 2
        return float(np.mean(np.minimum(r2, self.cfg.trunc ** 2)))

    def cost_huber(self, psi, scale):
        r = np.abs(self.residuals(psi))
        q = np.where(r <= scale, 0.5 * r * r, scale * (r - 0.5 * scale))
        return float(np.mean(q))


def _robust_scale(res, cfg):
    mad = float(np.median(np.abs(res - np.median(res))))
    return max(cfg.huber_floor, cfg.huber_k * 1.4826 * mad)


def estimate_yaw(cache, cfg=None, log=None):
    """Estimate the yaw correction psi for one (scene, sensor).

    Returns a dict. ``psi`` is the correction to apply to the extrinsic currently set on the
    cache: R_new = R_cur @ Rz(psi). ``fallback`` True means psi is forced to 0.
    """
    cfg = cfg or cache.cfg
    say = log if log is not None else (lambda *a: None)
    t0 = time.time()
    out = {
        "channel": cache.channel,
        "scene_name": cache.scene_name,
        "scene_token": cache.scene_token,
        "n_frames": cache.n_frames,
        "n_raw_points": cache.n_raw_points,
        "n_static_points": cache.n_static,
        "fallback": False,
        "fallback_reason": "",
        "psi": 0.0,
    }

    def bail(reason, extra=None):
        out["fallback"] = True
        out["fallback_reason"] = reason
        out["psi"] = 0.0
        if extra:
            out.update(extra)
        out["seconds"] = time.time() - t0
        say("    FALLBACK to identity: " + reason)
        return out

    if cache.n_static < cfg.min_points:
        return bail("degenerate: {} static returns < {}".format(cache.n_static, cfg.min_points))
    speed_med = float(np.median(cache.speed))
    out["speed_median"] = speed_med
    if speed_med < cfg.min_speed:
        return bail("degenerate: median sensor speed {:.3f} m/s < {}".format(
            speed_med, cfg.min_speed))

    # --- stage 1: truncated-quadratic grid over the full search span --------------------
    grid = np.arange(-cfg.coarse_span, cfg.coarse_span + 1e-12, cfg.coarse_step)
    ca, sa = np.cos(grid)[:, None], np.sin(grid)[:, None]
    R2 = (cache.vr[None, :] + cache.a[None, :] * ca + cache.b[None, :] * sa) ** 2
    c_coarse = np.minimum(R2, cfg.trunc ** 2).mean(axis=1)
    psi0 = float(grid[int(np.argmin(c_coarse))])
    out["cost_coarse_at_opt"] = float(c_coarse.min())
    out["cost_coarse_at_zero"] = float(c_coarse[int(np.argmin(np.abs(grid)))])

    # --- stage 2: Huber refinement on a fine grid, scale from the coarse optimum ---------
    scale = _robust_scale(cache.residuals(psi0), cfg)
    out["robust_scale"] = scale
    fg = np.arange(psi0 - cfg.fine_span, psi0 + cfg.fine_span + 1e-12, cfg.fine_step)
    rr = cache.vr[None, :] + cache.a[None, :] * np.cos(fg)[:, None] + cache.b[None, :] * np.sin(fg)[:, None]
    ar = np.abs(rr)
    q = np.where(ar <= scale, 0.5 * ar * ar, scale * (ar - 0.5 * scale)).mean(axis=1)
    psi = float(fg[int(np.argmin(q))])
    out["psi_raw"] = psi
    out["cost_at_opt"] = float(q.min())

    # --- observability / uncertainty: Fisher information of psi under the cosine model --
    res = cache.residuals(psi)
    inl = np.abs(res) <= scale
    n_inl = int(inl.sum())
    out["n_inliers"] = n_inl
    out["inlier_frac"] = float(n_inl) / float(cache.n_static)
    if n_inl < cfg.min_inliers:
        return bail("degenerate: {} inliers < {}".format(n_inl, cfg.min_inliers), out)
    d = -cache.a[inl] * np.sin(psi) + cache.b[inl] * np.cos(psi)     # d residual / d psi
    fisher = float(np.dot(d, d))
    sigma_res = float(np.sqrt(np.mean(res[inl] ** 2)))
    out["fisher_psi"] = fisher
    out["sigma_res_mps"] = sigma_res
    sigma_psi = float(sigma_res / np.sqrt(fisher)) if fisher > 0 else np.inf
    out["sigma_psi_rad"] = sigma_psi
    out["sigma_psi_deg"] = float(np.degrees(sigma_psi)) if np.isfinite(sigma_psi) else np.inf

    # contrast of the optimum against a null of random yaw offsets in the search box
    rng = np.random.default_rng(12345)
    null = rng.uniform(-cfg.coarse_span, cfg.coarse_span, 64)
    cn = np.array([cache.cost_huber(float(p), scale) for p in null])
    out["null_cost_median"] = float(np.median(cn))
    out["null_margin"] = float((np.median(cn) - out["cost_at_opt"]) /
                               max(1e-12, np.subtract(*np.percentile(cn, [75, 25]))))

    if not np.isfinite(sigma_psi) or sigma_psi > cfg.max_sigma_psi:
        return bail("unobservable: sigma_psi={:.4f} rad > {}".format(sigma_psi, cfg.max_sigma_psi), out)
    if abs(psi) > cfg.max_psi:
        return bail("divergence: |psi|={:.3f} rad > {}".format(abs(psi), cfg.max_psi), out)

    out["psi"] = psi
    out["seconds"] = time.time() - t0
    say("    psi={:+.4f} rad (sigma {:.4f}) scale={:.2f} inl={}/{} cost {:.4f} null {:.4f}".format(
        psi, sigma_psi, scale, n_inl, cache.n_static, out["cost_at_opt"], out["null_cost_median"]))
    return out


def cost_rvec(cache, rvec, scale=None):
    """Doppler cost for a FULL 3-vector correction, used only by the diagnostics to probe
    whether the clean extrinsic is a local minimum along each rotation axis."""
    import cv2

    dR, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    dh = np.stack([np.cos(cache.phi), np.sin(cache.phi), np.zeros(cache.phi.size)], axis=1)
    dh = dh @ np.asarray(dR).T
    pred = -(dh * cache.u[cache.fidx]).sum(axis=1)
    r = cache.vr - pred
    if scale is None:
        return float(np.mean(np.minimum(r * r, cache.cfg.trunc ** 2)))
    ar = np.abs(r)
    return float(np.mean(np.where(ar <= scale, 0.5 * ar * ar, scale * (ar - 0.5 * scale))))


def _cost_psi(self, psi, scale=None):
    """Doppler cost for a yaw-only correction, used by the local-optimality probe."""
    return cost_rvec(self, [0.0, 0.0, float(psi)], scale)


SensorDopplerCache.cost_rvec_or_psi = _cost_psi


# ---------------------------------------------------------------------------------------
# Variant DOP-B: the DOP estimator plus two corrections whose constants are measured on a
# DISJOINT set of clean TRAIN scenes (no injected fault, no evaluation scene), so they are an
# offline property of the estimator on this sensor suite rather than tuning on the answer.
#
#   bias[channel]  -- the median yaw the estimator reports on the CLEAN extrinsic. It is
#                     non-zero because the nuScenes nominal radar yaw and the Doppler-
#                     consistent yaw differ by a few tenths of a degree per sensor.
#   shrink_lambda  -- soft threshold. The estimator error after bias removal has a scale of
#                     sigma_floor; correcting by less than that can only add noise, so the
#                     applied correction is sign(psi) * max(0, |psi| - lambda). This is the
#                     usual soft-thresholding/MAP-under-Laplace-prior rule and it is what
#                     stops the mitigation from harming faults whose yaw content is below
#                     the estimator noise floor.
# ---------------------------------------------------------------------------------------


def calibrate_bias(nusc, scenes, channels, cfg=None, log=None, est=None):
    """Median reported yaw on the CLEAN extrinsic, per channel, over ``scenes``."""
    cfg = cfg or DopplerConfig()
    say = log if log is not None else (lambda *a: None)
    est = est or estimate_yaw
    open_cfg = DopplerConfig(**dict(vars(cfg), max_sigma_psi=1e9))
    bias, scatter, raw = {}, {}, {}
    for ch in channels:
        vals = []
        for sc in scenes:
            cache = SensorDopplerCache(nusc, sc["token"], ch, cfg=open_cfg)
            if cache.n_static < cfg.min_points:
                continue
            r = est(cache, cfg=open_cfg)
            if not r["fallback"]:
                vals.append(r["psi"])
        v = np.array(vals)
        raw[ch] = v.tolist()
        if v.size >= 5:
            bias[ch] = float(np.median(v))
            scatter[ch] = float(1.4826 * np.median(np.abs(v - np.median(v))))
        else:
            bias[ch] = 0.0
            scatter[ch] = float("nan")
        say("  [bias] {:<18} n={:2d} bias={:+.4f} rad ({:+.3f} deg) scatter={:.4f} rad".format(
            ch, v.size, bias[ch], np.degrees(bias[ch]), scatter[ch]))
    return {"bias": bias, "scatter": scatter, "raw": raw,
            "scenes": [s["name"] for s in scenes]}


def apply_corrections(res, bias=0.0, shrink_lambda=0.0, sigma_floor=0.0):
    """Turn the raw estimate in ``res`` into the yaw actually applied. Mutates and returns res."""
    if res["fallback"]:
        res["psi_debiased"] = 0.0
        res["psi_applied"] = 0.0
        res["sigma_eff_rad"] = float("inf")
        res["psi"] = 0.0
        return res
    psi_raw = float(res["psi"])
    psi_db = psi_raw - float(bias)
    res["psi_debiased"] = psi_db
    crb = float(res.get("sigma_psi_rad", 0.0))
    res["sigma_eff_rad"] = float(np.hypot(crb, sigma_floor))
    res["psi_applied"] = float(np.sign(psi_db) * max(0.0, abs(psi_db) - float(shrink_lambda)))
    res["psi_bias_used"] = float(bias)
    res["shrink_lambda"] = float(shrink_lambda)
    res["psi"] = res["psi_applied"]
    return res


# ---------------------------------------------------------------------------------------
# Variant DOP-K: Kellner-style PER-CYCLE velocity fit, then robust aggregation over frames.
#
# Instead of one global cosine fit over the whole scene, each radar measurement cycle gets
# its own two-parameter least-squares fit of the sensor velocity vector w from its static
# returns alone -- v_r,i = -dhat_i . w -- which is exactly the instantaneous ego-motion
# estimator of Kellner et al. (ITSC 2013, ICRA 2014). Because
#     v_r = -(Rz(psi) dhat)^T u   =>   w = Rz(-psi) u,
# the yaw error is the angle between the radar-derived velocity direction w and the
# odometry-derived one u, measured independently in every cycle:
#     psi_k = atan2(u_y, u_x) - atan2(w_y, w_x).
#
# Three things this buys over the global fit:
#   1. per-cycle IRLS rejects returns that are moving despite a stationary dyn_prop, which a
#      single scene-wide robust scale handles poorly;
#   2. the speed consistency check | |w| - |u_xy| | rejects whole cycles whose odometry and
#      radar disagree, before they can bias the answer;
#   3. the spread of psi_k over cycles is an EMPIRICAL uncertainty. The Cramer-Rao bound of
#      the global fit is about 0.06 deg while the measured error is about 0.8 deg, so the CRB
#      is optimistic by more than an order of magnitude and cannot be used as a gate. The
#      across-cycle spread is the honest replacement and it degenerates in exactly the cases
#      that matter (stationary ego, few returns, one-sided azimuth coverage).
# ---------------------------------------------------------------------------------------


def _fit_cycle(phi, vr, n_irls=4, floor=0.25):
    """Least-squares sensor velocity w (2-vector) from one cycle, with IRLS reweighting.
    Returns (w, sigma, n_inliers) or None."""
    if phi.size < 6:
        return None
    A = np.stack([-np.cos(phi), -np.sin(phi)], axis=1)
    wt = np.ones(phi.size)
    w = np.zeros(2)
    for _ in range(n_irls):
        Aw = A * wt[:, None]
        try:
            w = np.linalg.lstsq(Aw, vr * wt, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        r = vr - A @ w
        s = max(floor, 1.4826 * float(np.median(np.abs(r - np.median(r)))))
        wt = 1.0 / np.sqrt(1.0 + (r / (2.0 * s)) ** 2)          # Cauchy-ish IRLS weight
    r = vr - A @ w
    s = max(floor, 1.4826 * float(np.median(np.abs(r - np.median(r)))))
    inl = np.abs(r) <= 2.0 * s
    return w, s, int(inl.sum())


def estimate_yaw_kellner(cache, cfg=None, log=None, min_cycle_points=8,
                         speed_tol=0.35, min_cycles=8):
    """Per-cycle Kellner fit plus robust aggregation. Same return contract as estimate_yaw."""
    cfg = cfg or cache.cfg
    say = log if log is not None else (lambda *a: None)
    t0 = time.time()
    out = {"channel": cache.channel, "scene_name": cache.scene_name,
           "scene_token": cache.scene_token, "n_frames": cache.n_frames,
           "n_raw_points": cache.n_raw_points, "n_static_points": cache.n_static,
           "fallback": False, "fallback_reason": "", "psi": 0.0, "method": "kellner"}

    def bail(reason):
        out["fallback"] = True
        out["fallback_reason"] = reason
        out["psi"] = 0.0
        out["seconds"] = time.time() - t0
        say("    FALLBACK to identity: " + reason)
        return out

    if cache.n_static < cfg.min_points:
        return bail("degenerate: {} static returns < {}".format(cache.n_static, cfg.min_points))
    speed_med = float(np.median(cache.speed))
    out["speed_median"] = speed_med
    if speed_med < cfg.min_speed:
        return bail("degenerate: median sensor speed {:.3f} m/s < {}".format(
            speed_med, cfg.min_speed))

    psis, wts, nrej_speed, nrej_pts = [], [], 0, 0
    for k in range(cache.n_frames):
        sel = cache.fidx == k
        if sel.sum() < min_cycle_points:
            nrej_pts += 1
            continue
        u = cache.u[k]
        su = float(np.hypot(u[0], u[1]))
        if su < cfg.min_speed:
            nrej_speed += 1
            continue
        fit = _fit_cycle(cache.phi[sel], cache.vr[sel])
        if fit is None:
            nrej_pts += 1
            continue
        w, s, n_inl = fit
        sw = float(np.hypot(w[0], w[1]))
        if sw < 1e-6 or abs(sw - su) / su > speed_tol:
            nrej_speed += 1
            continue
        d = np.arctan2(u[1], u[0]) - np.arctan2(w[1], w[0])
        d = (d + np.pi) % (2 * np.pi) - np.pi
        psis.append(d)
        wts.append(float(n_inl))
    out["n_cycles_used"] = len(psis)
    out["n_cycles_rejected_speed"] = nrej_speed
    out["n_cycles_rejected_points"] = nrej_pts
    if len(psis) < min_cycles:
        return bail("degenerate: {} usable cycles < {}".format(len(psis), min_cycles))

    p = np.array(psis)
    psi = float(np.median(p))
    mad = float(1.4826 * np.median(np.abs(p - psi)))
    sigma_emp = float(mad / np.sqrt(p.size))
    out["psi_raw"] = psi
    out["psi_cycle_mad_rad"] = mad
    out["sigma_psi_rad"] = sigma_emp
    out["sigma_psi_deg"] = float(np.degrees(sigma_emp))
    out["psi_cycle_iqr_deg"] = float(np.degrees(np.subtract(*np.percentile(p, [75, 25]))))
    # keep a robust scale for the diagnostic cost probes
    out["robust_scale"] = max(cfg.huber_floor, 1.4826 * float(
        np.median(np.abs(cache.residuals(psi) - np.median(cache.residuals(psi))))) * cfg.huber_k)

    if not np.isfinite(sigma_emp) or sigma_emp > cfg.max_sigma_psi:
        return bail("unobservable: empirical sigma_psi={:.4f} rad > {}".format(
            sigma_emp, cfg.max_sigma_psi))
    if abs(psi) > cfg.max_psi:
        return bail("divergence: |psi|={:.3f} rad > {}".format(abs(psi), cfg.max_psi))

    out["psi"] = psi
    out["seconds"] = time.time() - t0
    say("    kellner psi={:+.4f} rad (emp sigma {:.4f}, cycle MAD {:.4f}) cycles {}/{} "
        "rej(spd {}, pts {})".format(psi, sigma_emp, mad, len(psis), cache.n_frames,
                                     nrej_speed, nrej_pts))
    return out
