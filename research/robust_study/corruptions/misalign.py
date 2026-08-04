"""(d1) radar-camera async and (d2) radar extrinsic miscalibration.

Both families are implemented as subclasses of the frozen loader pipeline classes and
registered under new names, so a cell is activated by swapping two entries in the test
pipeline. Nothing under ``loaders/`` is modified.

Anchors in the frozen checkout (tag robust-study-freeze-20260802 = 1ca50b4):
  * ``loaders/pipelines/loading.py:749``  ``Loadnuradarpoints``          (frame-t radar, element 0)
  * ``loaders/pipelines/loading.py:829``  ``LoadradarpointsFromMultiSweeps`` (sweep stack)
  * ``loaders/pipelines/loading.py:871``  test-mode ``choices`` computation
  * ``loaders/nuscenes_dataset.py:358``   ``get_nu_radar`` (module-level; bound onto both
    loaders as ``self.get_nu_radar`` in their ``__init__``)
  * ``loaders/nuscenes_dataset.py:410``   radar ``calibrated_sensor`` -> velocity rotation
  * ``loaders/nuscenes_dataset.py:483``   radar ``calibrated_sensor`` -> ``car_from_current``
  * ``configs/racformer_r50_nuimg_704x256_f8.py:227,:228`` the two test-pipeline entries

Both families deliberately add no filtering of their own: the existing projection and
out-of-view handling in the loader processes perturbed points exactly as production would
(fault-families (d2) spec item 7).
"""

import contextlib
import hashlib

import numpy as np
from mmdet.datasets.builder import PIPELINES
from pyquaternion import Quaternion

import loaders.nuscenes_dataset as _nuscenes_dataset
from loaders.nuscenes_dataset import RadarPointCloud_v2, renusc
from loaders.pipelines.loading import Loadnuradarpoints, LoadradarpointsFromMultiSweeps

from research.robust_study.corruptions.attest import SINK

RADAR_CHANNELS = (
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
)

# Registered severity ladder, fault-families.md "(d2) implementation spec" and plan.md 16.4.
D2_LEVELS = {
    "medium": dict(sigma_rot=0.06, sigma_trans=0.006),  # Dong et al. A.1 level 3
    "severe": dict(sigma_rot=0.10, sigma_trans=0.010),  # Dong et al. A.1 level 5
}


# --------------------------------------------------------------------------------------
# (d2) extrinsic miscalibration
# --------------------------------------------------------------------------------------
class ExtrinsicPerturber(object):
    """Draws and caches one (axis, z_rot, z_t) triple per (seed, scene_token, channel).

    Severity scales the SAME standard-normal draws, so the two levels are common-random-
    number coupled by construction (spec item 5). Never touches global RNG state.
    """

    def __init__(self, nusc):
        self.nusc = nusc
        self._cs_channel = None    # calibrated_sensor_token -> radar channel
        self._any_channel = None   # calibrated_sensor_token -> ANY channel (attestation only)
        self._sample_scene = None  # sample_token -> scene_token
        self._draws = {}

    def _build_maps(self):
        if self._cs_channel is not None:
            return
        sensor_channel = {s["token"]: s["channel"] for s in self.nusc.sensor}
        cs_channel = {}
        any_channel = {}
        for cs in self.nusc.calibrated_sensor:
            channel = sensor_channel[cs["sensor_token"]]
            any_channel[cs["token"]] = channel
            if channel in RADAR_CHANNELS:
                cs_channel[cs["token"]] = channel
        self._cs_channel = cs_channel
        self._any_channel = any_channel
        self._sample_scene = {s["token"]: s["scene_token"] for s in self.nusc.sample}

    def channel_of(self, cs_token):
        self._build_maps()
        return self._cs_channel.get(cs_token)

    def any_channel_of(self, cs_token):
        """Channel of ANY calibrated_sensor, radar or not.

        Used only by the attestation, which has to be able to say "LIDAR_TOP was looked up N
        times and passed through N times". ``channel_of`` deliberately returns None for
        non-radar sensors and so cannot distinguish LIDAR_TOP from an unknown token.
        """
        self._build_maps()
        return self._any_channel.get(cs_token)

    def scene_of(self, sample_token):
        self._build_maps()
        return self._sample_scene[sample_token]

    def draws(self, seed, scene_token, channel):
        key = (seed, scene_token, channel)
        cached = self._draws.get(key)
        if cached is not None:
            return cached
        payload = "{}:{}:{}".format(seed, scene_token, channel).encode("utf-8")
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))
        axis = rng.standard_normal(3)
        norm = np.linalg.norm(axis)
        # Measure-zero branch; kept so the draw order stays fixed either way.
        axis = np.array([0.0, 0.0, 1.0]) if norm == 0.0 else axis / norm
        z_rot = float(rng.standard_normal())
        z_t = rng.standard_normal(3)
        self._draws[key] = (axis, z_rot, z_t)
        return self._draws[key]

    def perturb(self, record, channel, seed, scene_token, sigma_rot, sigma_trans):
        """Return a perturbed COPY of a radar calibrated_sensor record.

        R' = R . exp([theta . a]x)   applied in the SENSOR frame (spec item 2)
        t' = t + sigma_trans * z_t   added in the EGO frame       (spec item 4)
        """
        axis, z_rot, z_t = self.draws(seed, scene_token, channel)
        theta = sigma_rot * z_rot
        rotation = Quaternion(record["rotation"]) * Quaternion(axis=axis, angle=theta)
        translation = np.asarray(record["translation"], dtype=np.float64) + sigma_trans * z_t
        out = dict(record)
        out["rotation"] = list(rotation.elements)
        out["translation"] = translation.tolist()
        return out


PERTURBER = ExtrinsicPerturber(renusc)

# Ambient injection context. Set by a (d2) pipeline class around the frozen loader call and
# cleared in a finally; None everywhere else, so the hook is a pass-through outside a cell.
_CONTEXT = None
_ORIGINAL_GET = None


def _install_hook(nusc):
    """Wrap ``nusc.get`` so radar calibrated_sensor lookups return a perturbed copy.

    The devkit reads the radar extrinsic in exactly two places on this path, the velocity
    rotation at nuscenes_dataset.py:410 and ``car_from_current`` at :483, and both go
    through ``nusc.get('calibrated_sensor', ...)``. Wrapping the accessor perturbs both with
    one hook and leaves the in-memory tables untouched: a copy is returned, never a mutation.
    LIDAR_TOP is the reference frame for those same call sites and is never in
    RADAR_CHANNELS, so the reference transform stays clean.
    """
    global _ORIGINAL_GET
    if _ORIGINAL_GET is not None:
        return
    _ORIGINAL_GET = nusc.get

    def get(table_name, token):
        record = _ORIGINAL_GET(table_name, token)
        ctx = _CONTEXT
        if table_name != "calibrated_sensor":
            return record

        # Runtime attestation: every calibrated_sensor lookup is classified by channel and
        # outcome, so "only the radar extrinsics moved" and "LIDAR_TOP was 100% pass-through"
        # are counted facts rather than claims about the code above.
        attest = SINK.enabled
        if ctx is None and not attest:
            return record  # unchanged fast path when attestation is off
        channel = PERTURBER.channel_of(token)
        if ctx is None or channel is None:
            if attest:
                name = PERTURBER.any_channel_of(token) or "UNKNOWN"
                outcome = "passthrough_no_context" if ctx is None else "passthrough_in_context"
                SINK.add(tags={"cs": "%s:%s" % (name, outcome)})
            return record

        if attest:
            SINK.add(tags={"cs": "%s:perturbed" % channel},
                     sets={"perturbed_scene_channel": "%s|%s" % (ctx["scene_token"], channel),
                           "scenes": ctx["scene_token"]})
        return PERTURBER.perturb(record, channel, ctx["seed"], ctx["scene_token"],
                                 ctx["sigma_rot"], ctx["sigma_trans"])

    nusc.get = get


@contextlib.contextmanager
def perturbed_extrinsics(sample_token, seed, sigma_rot, sigma_trans):
    """Activate the (d2) fault for one sample. Nestable via save/restore."""
    global _CONTEXT
    _install_hook(renusc)
    previous = _CONTEXT
    _CONTEXT = dict(
        seed=seed,
        scene_token=PERTURBER.scene_of(sample_token),
        sigma_rot=sigma_rot,
        sigma_trans=sigma_trans,
    )
    try:
        yield _CONTEXT
    finally:
        _CONTEXT = previous


class _D2Mixin(object):
    """Config knobs shared by the two (d2) loader wrappers."""

    def _attest_application(self, sample_token):
        """One (d2) application = one loader __call__ that opens a perturbation context.

        Recorded BEFORE the context opens, so a cell whose context manager silently stopped
        perturbing still reports its applications and fails on the lookup counters instead of
        vanishing from the evidence entirely.
        """
        if not SINK.enabled:
            return
        SINK.set_identity(family="d2_extrinsic", level=self.d2_level, seed=self.d2_seed,
                          sigma_rot=self.d2_sigma_rot, sigma_trans=self.d2_sigma_trans)
        SINK.application(sample_token, tags={"loader": type(self).__name__})

    def _init_d2(self, level=None, seed=0, sigma_rot=None, sigma_trans=None):
        if level is not None:
            if level not in D2_LEVELS:
                raise ValueError("unknown (d2) level {!r}; expected one of {}".format(
                    level, sorted(D2_LEVELS)))
            sigma_rot = D2_LEVELS[level]["sigma_rot"] if sigma_rot is None else sigma_rot
            sigma_trans = D2_LEVELS[level]["sigma_trans"] if sigma_trans is None else sigma_trans
        if sigma_rot is None or sigma_trans is None:
            raise ValueError("(d2) needs either level= or both sigma_rot= and sigma_trans=")
        if seed not in (0, 1, 2):
            raise ValueError("(d2) registered seeds are 0, 1, 2; got {!r}".format(seed))
        self.d2_level = level
        self.d2_seed = int(seed)
        self.d2_sigma_rot = float(sigma_rot)
        self.d2_sigma_trans = float(sigma_trans)


@PIPELINES.register_module()
class D2MiscalibLoadnuradarpoints(Loadnuradarpoints, _D2Mixin):
    """Frame-t radar (element 0) loaded through a perturbed radar->ego extrinsic."""

    def __init__(self, *args, **kwargs):
        d2 = {k: kwargs.pop(k) for k in ("level", "seed", "sigma_rot", "sigma_trans")
              if k in kwargs}
        super(D2MiscalibLoadnuradarpoints, self).__init__(*args, **kwargs)
        self._init_d2(**d2)

    def __call__(self, results):
        try:
            with perturbed_extrinsics(results["sample_idx"], self.d2_seed,
                                      self.d2_sigma_rot, self.d2_sigma_trans):
                return super(D2MiscalibLoadnuradarpoints, self).__call__(results)
        finally:
            # Recorded AFTER the context closes so this sample's calibrated_sensor lookups ride
            # on its own evidence line, and in a `finally` so an exception still records the
            # application rather than silently shrinking the coverage count.
            self._attest_application(results["sample_idx"])


@PIPELINES.register_module()
class D2MiscalibLoadradarpointsFromMultiSweeps(LoadradarpointsFromMultiSweeps, _D2Mixin):
    """Sweep stack loaded through the SAME perturbed extrinsic as element 0.

    The perturbation is keyed by (seed, scene_token, channel) only, so every sweep element
    of every sample in a scene sees one identical draw: the quasi-static drift model of
    spec item 1.
    """

    def __init__(self, *args, **kwargs):
        d2 = {k: kwargs.pop(k) for k in ("level", "seed", "sigma_rot", "sigma_trans")
              if k in kwargs}
        super(D2MiscalibLoadradarpointsFromMultiSweeps, self).__init__(*args, **kwargs)
        self._init_d2(**d2)

    def __call__(self, results):
        try:
            with perturbed_extrinsics(results["sample_idx"], self.d2_seed,
                                      self.d2_sigma_rot, self.d2_sigma_trans):
                return super(D2MiscalibLoadradarpointsFromMultiSweeps, self).__call__(results)
        finally:
            self._attest_application(results["sample_idx"])


# --------------------------------------------------------------------------------------
# (d1) radar-camera async -- AMENDED 2026-08-03 (injection point re-registered)
# --------------------------------------------------------------------------------------
# Re-registered injection point (fault-families.md (d1), "AMENDED 2026-08-03"): walk
# `current_sd_rec['prev']` on the TRUE sweep chain feeding the aggregation loop of
# RadarPointCloud_v2.from_file_multisweep (nuscenes_dataset.py:472-503), so every element of
# `radar_points` -- the frame-t element included -- is the sweep k steps older. The nuScenes
# radar sweep period is about 77 ms, so k in {1, 2, 3} realises about {77, 154, 231} ms.
#
# The superseded mechanism (shifting `choices`, i.e. trimming results['sweeps']['prev']) has
# been REMOVED rather than kept as an option: it was measured to be a no-op, because prev
# entries resolve through SAMPLE records on the ~0.5 s keyframe grid
# (nuscenes_dataset.py:388-395 -> :470) and never touch the sweep grid.
#
# The walk moves the aggregation START ONLY -- `sample_rec['data'][chan]`, read once at
# nuscenes_dataset.py:470 -- and leaves the loop's own `prev` stride at :503 untouched, so the
# stack keeps its length and spacing and simply slides k sweeps into the past.
# `ref_sample_rec` is deliberately NOT touched, so `ref_from_car`, `car_from_global` and
# `ref_time` stay anchored on keyframe t (:456-463): the devkit then compensates stale sweeps
# against the pose and clock of t, which IS the async artifact. `sample_idx` is likewise
# passed through unshifted.
#
# Clamp rule (registered intent): if the prev chain runs out before k steps -- the sensor
# stream start at the beginning of a scene -- the walk stops at the oldest sweep available
# rather than failing or wrapping. Every clamp is counted in D1_CLAMP_STATS so the probe can
# disclose how often it fires.

D1_SWEEP_PERIOD_S = 0.077  # nominal nuScenes radar sweep period; reporting only

D1_CLAMP_STATS = dict(walks=0, clamped_walks=0, steps_requested=0, steps_taken=0)

# Per-walk detail for the probe payload of the CURRENT application; drained by
# _D1Mixin._attest_application, so one loader __call__ carries exactly its own walks.
_D1_WALK_LOG = []


def reset_clamp_stats():
    for key in D1_CLAMP_STATS:
        D1_CLAMP_STATS[key] = 0
    del _D1_WALK_LOG[:]


_D1_OFFSET = None


def _record_walk(channel, requested, taken, lag_s):
    """Record one sweep-chain walk, both for the probe counters and for the evidence sink.

    `lag_s` is measured from the sample_data records the walk already fetched: the timestamp
    of the CLEAN aggregation start minus the timestamp of the shifted one. Positive means
    older, which is the whole point of the family, so the sign is recorded rather than
    assumed -- the superseded mechanism failed precisely by producing NEWER data.

    Evidence is held with SINK.add and rides out on the next application line
    (attest.py:128-137), so a stack of 40 walks costs 1 line, not 40.
    """
    D1_CLAMP_STATS["walks"] += 1
    D1_CLAMP_STATS["steps_requested"] += requested
    D1_CLAMP_STATS["steps_taken"] += taken
    clamped = taken < requested
    if clamped:
        D1_CLAMP_STATS["clamped_walks"] += 1
    if not SINK.enabled:
        return
    if lag_s > 1e-9:
        direction = "older"
    elif lag_s < -1e-9:
        direction = "newer"
    else:
        direction = "zero"
    sums = {"d1_walks": 1, "d1_steps_requested": requested, "d1_steps_taken": taken,
            "d1_lag_sum_s": lag_s}
    if clamped:
        sums["d1_clamped_walks"] = 1
        sums["d1_clamped_steps_taken"] = taken
        sums["d1_clamped_abs_lag_sum_s"] = abs(lag_s)
    else:
        sums["d1_full_walks"] = 1
        sums["d1_full_lag_sum_s"] = lag_s
        sums["d1_full_lag_per_step_sum_s"] = lag_s / taken if taken else 0.0
    SINK.add(sums=sums,
             tags={"d1": "clamped" if clamped else "shifted", "d1_dir": direction},
             sets={"d1_channels": channel})
    _D1_WALK_LOG.append({"channel": channel, "requested": requested, "taken": taken,
                         "lag_s": round(lag_s, 6)})


class _SweepShiftedRadarPointCloud(RadarPointCloud_v2):
    """Starts the sweep aggregation k steps earlier on the real sample_data prev chain.

    Only the start token changes; everything else is the frozen devkit implementation,
    reached through super(). Nothing is mutated in place -- `sample_rec` and its `data` dict
    are shallow-copied before the substitution -- so the in-memory nuScenes tables and the
    caller's records are left untouched.
    """

    @classmethod
    def from_file_multisweep(cls, nusc, sample_rec, ref_sample_rec, chan, ref_chan,
                             nsweeps=5, min_distance=1.0):
        offset = _D1_OFFSET
        if offset:
            token = sample_rec["data"][chan]
            start_record = nusc.get("sample_data", token)
            record = start_record
            taken = 0
            for _ in range(offset):
                if not record["prev"]:
                    break  # clamp to the oldest sweep this sensor has
                token = record["prev"]
                record = nusc.get("sample_data", token)
                taken += 1
            _record_walk(chan, offset, taken,
                         (start_record["timestamp"] - record["timestamp"]) * 1e-6)
            data = dict(sample_rec["data"])
            data[chan] = token
            sample_rec = dict(sample_rec)
            sample_rec["data"] = data
        return super(_SweepShiftedRadarPointCloud, cls).from_file_multisweep(
            nusc, sample_rec, ref_sample_rec, chan, ref_chan, nsweeps, min_distance)


@contextlib.contextmanager
def sweep_offset(offset):
    """Activate the (d1) fault for one sample.

    `get_nu_radar` resolves `RadarPointCloud_v2` as a module global at call time
    (nuscenes_dataset.py:399 and :404), so swapping that name is what routes both radar
    loader paths through the shifted aggregation. Both the name and the offset are restored
    in a finally, leaving the module exactly as found outside a cell.
    """
    global _D1_OFFSET
    previous_offset = _D1_OFFSET
    previous_class = _nuscenes_dataset.RadarPointCloud_v2
    _D1_OFFSET = offset
    _nuscenes_dataset.RadarPointCloud_v2 = _SweepShiftedRadarPointCloud
    try:
        yield
    finally:
        _D1_OFFSET = previous_offset
        _nuscenes_dataset.RadarPointCloud_v2 = previous_class


class _D1Mixin(object):

    def _attest_application(self, sample_token):
        """One (d1) application = one loader __call__ that ran with a non-zero offset.

        Recorded AFTER the frozen __call__ returns, unlike (d2)'s before-the-context record:
        here the evidence that decides the verdict is the realised per-walk lag, which only
        exists once the walks have happened. A cell whose hook silently stopped shifting
        still emits its application -- the __call__ completes either way -- and then fails on
        `d1_walks == 0` rather than vanishing from the evidence.
        """
        if not SINK.enabled:
            del _D1_WALK_LOG[:]
            return
        walks = list(_D1_WALK_LOG)
        del _D1_WALK_LOG[:]
        lags = [w["lag_s"] for w in walks]
        SINK.set_identity(family="d1_async", offset=self.d1_offset,
                          sweep_period_s=D1_SWEEP_PERIOD_S)
        SINK.application(
            sample_token,
            tags={"loader": type(self).__name__},
            probe={"loader": type(self).__name__, "offset": self.d1_offset,
                   "walks": len(walks),
                   "expected_lag_s": round(self.d1_offset * D1_SWEEP_PERIOD_S, 6),
                   "lag_s_min": min(lags) if lags else None,
                   "lag_s_max": max(lags) if lags else None,
                   "per_walk": walks[:12]})

    def _init_d1(self, offset=0):
        offset = int(offset)
        if offset < 0:
            raise ValueError("(d1) offset must be >= 0; got {!r}".format(offset))
        self.d1_offset = offset


@PIPELINES.register_module()
class D1AsyncLoadnuradarpoints(Loadnuradarpoints, _D1Mixin):
    """Frame-t radar element, aggregated from k sweeps earlier.

    Shifting this path as well as the stack is what closes the partial-injection trap: it
    supplies element 0 of `radar_points`, which the sweep loader never revisits.
    """

    def __init__(self, *args, **kwargs):
        offset = kwargs.pop("offset", 0)
        super(D1AsyncLoadnuradarpoints, self).__init__(*args, **kwargs)
        self._init_d1(offset)

    def __call__(self, results):
        if self.d1_offset == 0:
            return super(D1AsyncLoadnuradarpoints, self).__call__(results)
        try:
            with sweep_offset(self.d1_offset):
                results = super(D1AsyncLoadnuradarpoints, self).__call__(results)
        except Exception:
            del _D1_WALK_LOG[:]  # a raised __call__ must not leak walks onto the next sample
            raise
        self._attest_application(results["sample_idx"])
        return results


@PIPELINES.register_module()
class D1AsyncLoadradarpointsFromMultiSweeps(LoadradarpointsFromMultiSweeps, _D1Mixin):
    """Every sweep-stack element aggregated from k sweeps earlier; cameras untouched.

    `results['sweeps']` is neither read nor written by this class -- the stack still selects
    the same prev entries the clean pipeline selects, and only the sweep chain underneath
    each one slides. The camera loaders run earlier in the pipeline and are unaffected.
    """

    def __init__(self, *args, **kwargs):
        offset = kwargs.pop("offset", 0)
        super(D1AsyncLoadradarpointsFromMultiSweeps, self).__init__(*args, **kwargs)
        self._init_d1(offset)

    def __call__(self, results):
        if self.d1_offset == 0:
            return super(D1AsyncLoadradarpointsFromMultiSweeps, self).__call__(results)
        try:
            with sweep_offset(self.d1_offset):
                results = super(D1AsyncLoadradarpointsFromMultiSweeps, self).__call__(results)
        except Exception:
            del _D1_WALK_LOG[:]  # a raised __call__ must not leak walks onto the next sample
            raise
        self._attest_application(results["sample_idx"])
        return results
