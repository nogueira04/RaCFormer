"""C2 mitigation candidate: test-time radar->ego extrinsic re-calibration.

Model-free. Nothing in this module loads a checkpoint, imports the detector, or runs a
forward pass. It estimates a correction delta in SE(3) for a radar sensor extrinsic by
maximising an image-gradient alignment score of the sensor frame-t radar points projected
into the six cameras of a scene.

Conventions (matched to the (d2) fault spec):
  * the extrinsic under test is (R, t) with R the radar->ego rotation and t the radar->ego
    translation, exactly the ``calibrated_sensor`` record the devkit uses to map a radar
    sweep into the ego frame;
  * a candidate correction is a rotation vector r (axis-angle, radians) and a translation
    vector dt (metres), applied as   R_new = R @ expm(r),   t_new = t + dt.
    This is the same right-multiplied sensor-frame rotation / additive ego-frame translation
    parameterisation the fault injection uses, so a fault (r_f, dt_f) is exactly undone by
    the correction (-r_f expressed in the rotated frame, -dt_f).

Score
  S(r, dt) = ( sum over samples, cameras, points of the image-gradient magnitude sampled at
               the pixel the point projects to ) / (number of frame-t points in the scene)
  Points that project behind a camera or outside its image contribute zero. The denominator
  is FIXED (it does not depend on the candidate), so the score cannot be inflated by pushing
  points out of view.

Optimiser
  Three stages, coarse to fine, then coordinate descent; identity fallback on any sanity-bound
  violation. See ``optimise``.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from pyquaternion import Quaternion

RADAR_CHANNELS = (
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
)
CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

# (downsample scale, gaussian sigma in pixels AT THAT SCALE). Index 0 is coarsest.
# Effective full-resolution smoothing: 16 px, 6 px, 2 px.
DEFAULT_LEVELS = ((0.25, 4.0), (0.5, 3.0), (1.0, 2.0))


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """expm of the skew matrix of rvec (axis-angle, radians) -> 3x3 rotation."""
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return np.asarray(R, dtype=np.float64)


def rotation_angle_deg(R: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix, in degrees."""
    c = (np.trace(np.asarray(R, dtype=np.float64)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


@dataclass
class ScoreConfig:
    levels: Tuple[Tuple[float, float], ...] = DEFAULT_LEVELS
    min_depth: float = 0.1        # metres; a point nearer than this does not count
    grad_clip: float = 255.0      # gradient magnitudes are clipped here and stored uint8
    min_distance: float = 1.0     # devkit remove_close, matches the production loader
    use_default_filters: bool = True


@dataclass
class SearchConfig:
    """Search schedule. Sized for the SEVERE (d2) level and reused unchanged at medium:
    a deployed mitigation does not know the severity it faces."""
    coarse_rot_span: float = 0.30     # rad, +/- per rotation-vector component (3 sigma at severe)
    coarse_rot_steps: int = 13        # -> 0.05 rad grid, 13**3 = 2197 evaluations
    mid_rot_span: float = 0.05
    mid_rot_steps: int = 5
    mid_trans_span: float = 0.03      # m, 3 sigma at severe
    mid_trans_steps: int = 3
    cd_rot_step0: float = 0.02
    cd_trans_step0: float = 0.02
    cd_min_rot_step: float = 5e-4
    cd_min_trans_step: float = 5e-4
    cd_max_passes: int = 60
    max_rot_norm: float = 0.5         # divergence bound, rad
    max_trans_norm: float = 0.5       # divergence bound, m
    min_valid_pairs: int = 200        # degenerate-landscape bound at identity
    timeout_s: float = 600.0


class SceneCameraCache(object):
    """Per-scene, camera-side only: gradient pyramids and the ego->pixel geometry.

    Shared by all five radar sensors and by every severity level, because none of it
    depends on the radar extrinsic.
    """

    def __init__(self, nusc, scene_token, cfg=None, cameras=CAMERA_CHANNELS):
        self.cfg = cfg or ScoreConfig()
        self.scene_token = scene_token
        self.cameras = tuple(cameras)
        scene = nusc.get("scene", scene_token)
        self.scene_name = scene["name"]

        tokens = []
        tok = scene["first_sample_token"]
        while tok:
            tokens.append(tok)
            tok = nusc.get("sample", tok)["next"]
        self.sample_tokens = tokens
        ns, nc = len(tokens), len(self.cameras)

        # M and v map a point already in the ego frame OF THE RADAR SAMPLE_DATA into
        # homogeneous pixel coordinates of camera c:  x = M @ p_ego_radar + v.
        self.M = np.zeros((ns, nc, 3, 3), dtype=np.float64)
        self.v = np.zeros((ns, nc, 3), dtype=np.float64)
        self.img_wh = np.zeros((nc, 2), dtype=np.int64)

        grads = [[] for _ in self.cfg.levels]
        self.level_shapes = []
        t_build = time.time()
        for si, stok in enumerate(tokens):
            smp = nusc.get("sample", stok)
            # ego pose at the RADAR keyframe timestamp (all five radars share the keyframe,
            # their poses differ by <1 ms; the per-sensor pose is used in build_sensor_points)
            for ci, cam in enumerate(self.cameras):
                sd = nusc.get("sample_data", smp["data"][cam])
                cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                pose = nusc.get("ego_pose", sd["ego_pose_token"])
                K = np.array(cs["camera_intrinsic"], dtype=np.float64)
                R_cs = Quaternion(cs["rotation"]).rotation_matrix
                t_cs = np.array(cs["translation"], dtype=np.float64)
                R_ec = Quaternion(pose["rotation"]).rotation_matrix
                t_ec = np.array(pose["translation"], dtype=np.float64)
                # p_cam = R_cs.T @ (R_ec.T @ (p_glob - t_ec) - t_cs)
                self.M[si, ci] = K @ R_cs.T @ R_ec.T
                self.v[si, ci] = K @ R_cs.T @ (-R_ec.T @ t_ec - t_cs)
                self.img_wh[ci] = (sd["width"], sd["height"])

                img = cv2.imread(os.path.join(nusc.dataroot, sd["filename"]), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise IOError("unreadable image: " + str(sd["filename"]))
                gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
                g = cv2.magnitude(gx, gy)
                for li, (scale, sigma) in enumerate(self.cfg.levels):
                    gl = g if scale == 1.0 else cv2.resize(g, None, fx=scale, fy=scale,
                                                           interpolation=cv2.INTER_AREA)
                    gl = cv2.GaussianBlur(gl, (0, 0), sigma)
                    grads[li].append(np.clip(gl, 0.0, self.cfg.grad_clip).astype(np.uint8))
        self.build_seconds = time.time() - t_build

        self.G = []
        for li in range(len(self.cfg.levels)):
            arr = np.stack(grads[li]).reshape(ns, nc, grads[li][0].shape[0], grads[li][0].shape[1])
            self.G.append(np.ascontiguousarray(arr))
            self.level_shapes.append((arr.shape[2], arr.shape[3]))
        del grads
        self.n_samples = ns
        # broadcastable index helpers
        self._si = np.arange(ns).reshape(ns, 1, 1)
        self._ci = np.arange(nc).reshape(1, nc, 1)

    def memory_mb(self):
        return sum(a.nbytes for a in self.G) / 1e6


class SensorPointCache(object):
    """Per (scene, sensor): the frame-t radar points, plus the extrinsic currently in force."""

    def __init__(self, nusc, scene_cache, channel, cfg=None):
        from nuscenes.utils.data_classes import RadarPointCloud

        self.cfg = cfg or ScoreConfig()
        self.channel = channel
        self.scene = scene_cache
        ns = scene_cache.n_samples

        pts, R_list, t_list = [], [], []
        for stok in scene_cache.sample_tokens:
            smp = nusc.get("sample", stok)
            sd = nusc.get("sample_data", smp["data"][channel])
            if self.cfg.use_default_filters:
                RadarPointCloud.default_filters()
            else:
                RadarPointCloud.disable_filters()
            pc = RadarPointCloud.from_file(os.path.join(nusc.dataroot, sd["filename"]))
            RadarPointCloud.default_filters()
            pc.remove_close(self.cfg.min_distance)
            pts.append(np.array(pc.points[:3, :].T, dtype=np.float64))
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            pose = nusc.get("ego_pose", sd["ego_pose_token"])
            R_list.append(Quaternion(cs["rotation"]).rotation_matrix)
            t_list.append(np.array(cs["translation"], dtype=np.float64))
            # fold the radar-time ego pose into M/v: the camera-side M maps GLOBAL->pixel,
            # so pre-compose the radar ego->global here.
            self_pose_R = Quaternion(pose["rotation"]).rotation_matrix
            self_pose_t = np.array(pose["translation"], dtype=np.float64)
            R_list[-1] = (self_pose_R, R_list[-1])
            t_list[-1] = (self_pose_t, t_list[-1])

        self.counts = np.array([p.shape[0] for p in pts], dtype=np.int64)
        self.n_total = int(self.counts.sum())
        nmax = int(max(1, self.counts.max()))
        self.P = np.zeros((ns, nmax, 3), dtype=np.float64)
        self.mask = np.zeros((ns, nmax), dtype=bool)
        for i, p in enumerate(pts):
            self.P[i, : p.shape[0]] = p
            self.mask[i, : p.shape[0]] = True
        self.nmax = nmax

        # E maps radar-ego -> global for each sample:  p_glob = Rp @ p_ego + tp
        self.Rp = np.stack([r[0] for r in R_list])
        self.tp = np.stack([t[0] for t in t_list])
        self.R_clean = np.stack([r[1] for r in R_list])
        self.t_clean = np.stack([t[1] for t in t_list])
        # camera-side M/v pre-composed with the radar ego->global transform:
        #   x = M @ (Rp @ (R p + t) + tp) + v  =  (M Rp R) p + M (Rp t + tp) + v
        self.MRp = np.einsum("scij,sjk->scik", scene_cache.M, self.Rp)          # (S,C,3,3)
        self.vg = scene_cache.v + np.einsum("scij,sj->sci", scene_cache.M, self.tp)  # (S,C,3)
        self.set_extrinsic(self.R_clean, self.t_clean)

    def set_extrinsic(self, R, t):
        """R: (S,3,3) or (3,3); t: (S,3) or (3,). The extrinsic the optimiser starts from."""
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        if R.ndim == 2:
            R = np.broadcast_to(R, (self.P.shape[0], 3, 3))
        if t.ndim == 1:
            t = np.broadcast_to(t, (self.P.shape[0], 3))
        self.R_cur = np.array(R)
        self.t_cur = np.array(t)
        self.C = np.einsum("scij,sjk->scik", self.MRp, self.R_cur)               # (S,C,3,3)
        self.d = self.vg + np.einsum("scij,sj->sci", self.MRp, self.t_cur)       # (S,C,3)

    def perturbed_extrinsic(self, rvec, dt):
        """The (d2) fault applied to the clean extrinsic: R@expm(r), t+dt."""
        dR = rodrigues(rvec)
        return np.einsum("sij,jk->sik", self.R_clean, dR), self.t_clean + np.asarray(dt)

    def score(self, rvec, dt, level):
        """S(r, dt) and the number of contributing (point, camera) pairs."""
        dR = rodrigues(rvec)
        Q = self.P @ dR.T                                                        # (S,N,3)
        X = np.einsum("scij,snj->scni", self.C, Q)
        off = self.d + np.einsum("scij,j->sci", self.MRp, np.asarray(dt, dtype=np.float64))
        X = X + off[:, :, None, :]
        z = X[..., 2]
        ok = self.mask[:, None, :] & (z >= self.cfg.min_depth)
        zz = np.where(ok, z, 1.0)
        u = X[..., 0] / zz
        vv = X[..., 1] / zz
        H, W = self.scene.level_shapes[level]
        sc = float(self.scene.cfg.levels[level][0])
        ui = np.floor(u * sc).astype(np.int32)
        vi = np.floor(vv * sc).astype(np.int32)
        ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        np.clip(ui, 0, W - 1, out=ui)
        np.clip(vi, 0, H - 1, out=vi)
        g = self.scene.G[level][self.scene._si, self.scene._ci, vi, ui]
        return float(g[ok].sum()) / max(1, self.n_total), int(ok.sum())


def _grid(span, steps):
    if steps <= 1:
        return np.array([0.0])
    return np.linspace(-span, span, steps)


def optimise(sensor_cache, search=None, log=None):
    """Estimate delta for one (scene, sensor). Returns a dict; never raises on failure.

    The returned ``rvec``/``dt`` are the correction to apply to the extrinsic currently set
    on ``sensor_cache`` (``set_extrinsic``), i.e. R_new = R_cur @ expm(rvec), t_new = t_cur + dt.
    """
    s = search or SearchConfig()
    say = log if log is not None else (lambda *a: None)
    t0 = time.time()
    nlev = len(sensor_cache.scene.cfg.levels)
    fine = nlev - 1
    out = {
        "channel": sensor_cache.channel,
        "scene_name": sensor_cache.scene.scene_name,
        "scene_token": sensor_cache.scene.scene_token,
        "n_points_total": sensor_cache.n_total,
        "n_samples": int(sensor_cache.P.shape[0]),
        "points_per_frame_median": float(np.median(sensor_cache.counts)),
        "fallback": False,
        "fallback_reason": "",
        "n_evals": 0,
    }
    zero = np.zeros(3)

    s_id_fine, n_id = sensor_cache.score(zero, zero, fine)
    out["score_identity"] = s_id_fine
    out["n_valid_pairs_identity"] = n_id

    def bail(reason):
        out["fallback"] = True
        out["fallback_reason"] = reason
        out["rvec"] = [0.0, 0.0, 0.0]
        out["dt"] = [0.0, 0.0, 0.0]
        out["score_best"] = s_id_fine
        out["seconds"] = time.time() - t0
        say("    FALLBACK to identity: " + reason)
        return out

    if n_id < s.min_valid_pairs:
        return bail("degenerate landscape: {} valid pairs at identity < {}".format(
            n_id, s.min_valid_pairs))

    n_evals = [0]

    def ev(r, d, lv):
        n_evals[0] += 1
        val, _ = sensor_cache.score(r, d, lv)
        return val

    # --- stage A: coarse rotation-only lattice on the coarsest gradient level ------------
    ax = _grid(s.coarse_rot_span, s.coarse_rot_steps)
    best_r, best_v = zero.copy(), -np.inf
    for a in ax:
        if time.time() - t0 > s.timeout_s:
            return bail("timeout in stage A")
        for b in ax:
            for c in ax:
                r = np.array([a, b, c])
                val = ev(r, zero, 0)
                if val > best_v:
                    best_v, best_r = val, r
    say("    stage A: S={:.4f} r={} ({} evals, {:.1f}s)".format(
        best_v, np.round(best_r, 4).tolist(), n_evals[0], time.time() - t0))

    # --- stage B: local 6-DoF lattice on the middle level --------------------------------
    rg = _grid(s.mid_rot_span, s.mid_rot_steps)
    tg = _grid(s.mid_trans_span, s.mid_trans_steps)
    mid = 1 if nlev > 2 else fine
    centre_r = best_r.copy()
    best_v = ev(centre_r, zero, mid)
    best_d = zero.copy()
    for da in rg:
        if time.time() - t0 > s.timeout_s:
            return bail("timeout in stage B")
        for db in rg:
            for dc in rg:
                r = centre_r + np.array([da, db, dc])
                for ta in tg:
                    for tb in tg:
                        for tc in tg:
                            d = np.array([ta, tb, tc])
                            val = ev(r, d, mid)
                            if val > best_v:
                                best_v = val
                                best_r = r.copy()
                                best_d = d.copy()
    say("    stage B: S={:.4f} r={} dt={} ({} evals, {:.1f}s)".format(
        best_v, np.round(best_r, 4).tolist(), np.round(best_d, 4).tolist(),
        n_evals[0], time.time() - t0))

    # --- stage C: coordinate descent on the finest level ---------------------------------
    r = best_r.copy()
    d = best_d.copy()
    cur = ev(r, d, fine)
    rstep, tstep = s.cd_rot_step0, s.cd_trans_step0
    for _ in range(s.cd_max_passes):
        if time.time() - t0 > s.timeout_s:
            return bail("timeout in stage C")
        improved = False
        for k in range(6):
            step = rstep if k < 3 else tstep
            for sign in (1.0, -1.0):
                rr, dd = r.copy(), d.copy()
                if k < 3:
                    rr[k] += sign * step
                else:
                    dd[k - 3] += sign * step
                val = ev(rr, dd, fine)
                if val > cur:
                    cur, r, d, improved = val, rr, dd, True
        if not improved:
            rstep *= 0.5
            tstep *= 0.5
            if rstep < s.cd_min_rot_step and tstep < s.cd_min_trans_step:
                break

    out["n_evals"] = n_evals[0]
    out["seconds"] = time.time() - t0
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(d)) and np.isfinite(cur)):
        return bail("divergence: non-finite optimiser state")
    if np.linalg.norm(r) > s.max_rot_norm:
        return bail("divergence: |r|={:.3f} rad > {}".format(float(np.linalg.norm(r)), s.max_rot_norm))
    if np.linalg.norm(d) > s.max_trans_norm:
        return bail("divergence: |dt|={:.3f} m > {}".format(float(np.linalg.norm(d)), s.max_trans_norm))
    if cur <= s_id_fine:
        return bail("no improvement over S(identity): {:.6f} <= {:.6f}".format(cur, s_id_fine))

    out["rvec"] = [float(x) for x in r]
    out["dt"] = [float(x) for x in d]
    out["score_best"] = float(cur)
    say("    stage C: S={:.4f} (identity {:.4f}) r={} dt={} ({} evals, {:.1f}s)".format(
        cur, s_id_fine, np.round(r, 4).tolist(), np.round(d, 5).tolist(),
        n_evals[0], out["seconds"]))
    return out
