"""Radar corruption pipeline wrappers for the robustness study.

Families implemented here:
  (b) radar point dropout      -- uniform random removal of radar points, p per sweep
  (c) radar Doppler/RCS noise  -- additive N(0, sigma) on raw rcs / vx_comp / vy_comp

NEW FILE. Nothing tracked is modified: every class below subclasses a frozen loader
pipeline class and only wraps ``self.get_nu_radar``, the single function through which
BOTH radar loader paths obtain their points.

Injection point
---------------
``loaders.nuscenes_dataset.get_nu_radar`` returns a (18, N) float32 tensor whose rows are
the nuScenes RadarPointCloud fields in devkit order
(``nuscenes/utils/data_classes.py:315``):

    0 x  1 y  2 z  3 dyn_prop  4 id  5 rcs  6 vx  7 vy  8 vx_comp  9 vy_comp
    10 is_quality_valid  11 ambig_state  12 x_rms  13 y_rms  14 invalid_state
    15 pdh0  16 vx_rms  17 vy_rms

The loader pipelines then keep columns [0,1,2,5,8,9,18] = x, y, z, rcs, vx_comp, vy_comp,
dt (loading.py:812 and :862/:896). We corrupt rows 5/8/9 of the (18, N) array, i.e. the
raw dBsm / m-per-s values, before any downstream use. RaCFormer applies no dataset-level
normalisation to these fields at all -- they enter ``PillarFeatureNet(in_channels=7)``
directly (configs/racformer_r50_nuimg_704x256_f8.py:129-136) -- so "raw, pre-normalisation"
is unambiguous in this codebase.

vx_comp/vy_comp have already been rotated into the LIDAR_TOP reference frame by
``get_nu_radar`` (nuscenes_dataset.py:412-418) at the point we see them. That rotation is
inside the frozen function and cannot be wrapped without editing tracked code.

Partial-injection trap
----------------------
The frame-t radar set is loaded by ``Loadnuradarpoints`` and the sweep stack by
``LoadradarpointsFromMultiSweeps``; they are separate pipeline entries
(base config :227 and :228 for the test pipeline, :212/:213 for train). A corruption
applied to only one of them is only partially applied. ``build_corrupted_val_pipeline``
below rewrites both and asserts it found exactly one of each.

Determinism / common random numbers / sweep identity
----------------------------------------------------
Draws are keyed by WHAT WAS MEASURED, not by which pipeline call asked for it.
``sweep_identity`` mirrors ``get_nu_radar``'s own channel loop
(nuscenes_dataset.py:383-400) and resolves, per radar channel, the sample_data token that
``from_file_multisweep`` will actually start reading from -- ``renusc.get('sample',
sd_record['sample_token'])['data'][chan]`` (nuscenes_dataset.py:394 and :470). Two calls
whose resolved token tuple, reference sample, sweep count and filter flag all agree
produce a bit-identical point array, and therefore get bit-identical corruption.

This matters because the frozen loader duplicates sweeps: when a sample has no previous
sweeps it stacks the keyframe set N times (loading.py:855-868), and the padding branch
clamps several stack slots onto the same sweep. Keying per call would hand each copy an
independent mask, so a point dropped in one copy would survive in its siblings and the
fault would be diluted into an averaging effect. Keying on measurement identity means a
dropped point is dropped in every copy of the sweep it belongs to.

The key is hashed with SHA-256 and the first 8 bytes seed a PCG64 generator; no global RNG
state is read or written, and nothing depends on dataset order or worker sharding. Within
a family the draws do not depend on the severity, so:
  (b) the uniforms u are identical across p, and {u >= 0.25} superset {u >= 0.50} superset {u >= 0.75}
  (c) the standard normals z are identical across sigma, so the applied deltas are
      exactly proportional (sigma_a * z vs sigma_b * z).
"""

import hashlib

import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES

from loaders.nuscenes_dataset import renusc
from loaders.pipelines.loading import (
    Loadnuradarpoints,
    LoadradarpointsFromMultiSweeps,
)

from research.robust_study.corruptions.attest import SINK

# Row indices into the (18, N) raw array returned by get_nu_radar.
ROW_RCS = 5
ROW_VX_COMP = 8
ROW_VY_COMP = 9

# Mirrors the default list in get_nu_radar (nuscenes_dataset.py:369-372).
DEFAULT_RADAR_TYPES = (
    "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT",
)

# Registered severity ladders (fault-families.md). Nothing above sigma=5 is a published
# eval point; the spec forbids inventing one.
DROPOUT_P_LEVELS = (0.25, 0.50, 0.75)
NOISE_SIGMA_LEVELS = (1.0, 3.0, 5.0)
MAX_SIGMA = 5.0
SEEDS = (0, 1, 2)


def sweep_identity(sam_idx, num_sweeps, filter_flag, radar_sample_rec, radar_types):
    """Content key for one get_nu_radar call.

    Mirrors the channel loop at nuscenes_dataset.py:383-400 and resolves the sample_data
    token each channel will start reading from, exactly as from_file_multisweep does at
    :470. Calls that would return the same points get the same key.
    """
    ref_sample_rec = renusc.get("sample", sam_idx)
    datas = ref_sample_rec["data"]
    rad_types = list(radar_types) if radar_types is not None else list(DEFAULT_RADAR_TYPES)

    starts = []
    for token in datas.keys():
        if token not in rad_types:
            continue
        if radar_sample_rec is not None:
            sd_record = radar_sample_rec[token]
            if sd_record is None:
                continue
        else:
            sd_record = renusc.get("sample_data", datas[token])
        chan = sd_record["channel"]
        sample_rec = renusc.get("sample", sd_record["sample_token"])
        starts.append("{}={}".format(chan, sample_rec["data"][chan]))

    return "{}|n{}|f{}|{}".format(sam_idx, int(num_sweeps), int(bool(filter_flag)), ",".join(starts))


def derive_rng(family, seed, identity):
    """Counter-based RNG: fully determined by (family, seed, measurement identity)."""
    key = "{}:{}:{}".format(family, int(seed), identity)
    digest = hashlib.sha256(key.encode("utf-8")).digest()[:8]
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest, "big")))


class _RadarCorruptionMixin(object):
    """Wraps ``self.get_nu_radar`` on a frozen loader class.

    Stateless: no per-call counters, so the draws do not depend on call order, on which
    pipeline class issued the call, or on DataLoader worker sharding.
    Subclasses set FAMILY and implement ``corrupt``.
    """

    FAMILY = None

    def _install(self, corrupt_seed):
        if self.FAMILY is None:
            raise ValueError("FAMILY must be set on the subclass")
        self.corrupt_seed = int(corrupt_seed)
        self._inner_get_nu_radar = self.get_nu_radar
        self.get_nu_radar = self._corrupting_get_nu_radar

    def _corrupting_get_nu_radar(self, sam_idx, mutil_sweep=True, num_sweeps=6, filter=True,
                                 radar_sample_rec=None, radar_types=None, **kwargs):
        # Signature mirrors get_nu_radar (nuscenes_dataset.py:358) so the identity can be
        # computed from the same arguments the frozen function will act on.
        points, radar_tokens, times = self._inner_get_nu_radar(
            sam_idx, mutil_sweep, num_sweeps, filter=filter,
            radar_sample_rec=radar_sample_rec, radar_types=radar_types, **kwargs)
        identity = sweep_identity(sam_idx, num_sweeps, filter, radar_sample_rec, radar_types)
        rng = derive_rng(self.FAMILY, self.corrupt_seed, identity)
        before = points
        points, times = self.corrupt(points, times, rng)
        # Runtime attestation. Neither `corrupt` implementation writes into its argument
        # (dropout slices, noise clones first), so `before` is still the pre-corruption tensor
        # here -- and if that ever stopped being true the evidence would show a zero delta and
        # the cell would FAIL rather than pass quietly.
        if SINK.enabled:
            self._attest(SINK, sam_idx, before, points, times)
        return points, radar_tokens, times

    def corrupt(self, points, times, rng):
        raise NotImplementedError

    def _attest(self, sink, sam_idx, before, after, times):
        """Emit one application record. Subclasses add the family-specific evidence."""
        raise NotImplementedError


class _DropoutMixin(_RadarCorruptionMixin):
    """(b) uniform random point removal."""

    FAMILY = "radar_dropout"

    def _install_dropout(self, drop_p, corrupt_seed):
        drop_p = float(drop_p)
        if not (0.0 <= drop_p < 1.0):
            raise ValueError("drop_p must be in [0, 1), got {}".format(drop_p))
        self.drop_p = drop_p
        self._install(corrupt_seed)

    def corrupt(self, points, times, rng):
        n = int(points.shape[1])
        if n == 0:
            return points, times
        u = rng.random(n)
        keep = torch.from_numpy(np.asarray(u >= self.drop_p))
        return points[:, keep], times[:, keep]

    def _attest(self, sink, sam_idx, before, after, times):
        n_before = int(before.shape[1])
        n_after = int(after.shape[1])
        tags = {}
        if n_before == 0:
            tags["empty"] = 1
        if n_after > n_before:
            tags["grew"] = 1
        if int(times.shape[1]) != n_after:
            tags["times_mismatch"] = 1
        sink.set_identity(family=self.FAMILY, drop_p=self.drop_p,
                          corrupt_seed=self.corrupt_seed)
        sink.application(
            sam_idx,
            sums={"points_before": n_before, "points_after": n_after},
            tags=tags,
            probe={"loader": type(self).__name__, "points_before": n_before,
                   "points_after": n_after,
                   "realised_drop": (1.0 - n_after / n_before) if n_before else None})


class _DopplerRcsNoiseMixin(_RadarCorruptionMixin):
    """(c) additive Gaussian on raw rcs / vx_comp / vy_comp; positions untouched."""

    FAMILY = "radar_doppler_rcs_noise"

    def _install_noise(self, sigma, corrupt_seed):
        sigma = float(sigma)
        if sigma < 0.0:
            raise ValueError("sigma must be >= 0, got {}".format(sigma))
        if sigma > MAX_SIGMA:
            raise ValueError(
                "sigma={} exceeds the registered maximum {}; nothing above it is a "
                "published eval level (fault-families.md, family (c))".format(sigma, MAX_SIGMA)
            )
        self.sigma = sigma
        self._install(corrupt_seed)

    def corrupt(self, points, times, rng):
        n = int(points.shape[1])
        if n == 0:
            return points, times
        z = rng.standard_normal((3, n))
        delta = torch.from_numpy(self.sigma * z).to(points.dtype)
        points = points.clone()
        points[ROW_RCS] = points[ROW_RCS] + delta[0]
        points[ROW_VX_COMP] = points[ROW_VX_COMP] + delta[1]
        points[ROW_VY_COMP] = points[ROW_VY_COMP] + delta[2]
        return points, times

    #: rows the noise is allowed to move, in the order the attestation reports them
    ATTEST_ROWS = (("rcs", ROW_RCS), ("vx_comp", ROW_VX_COMP), ("vy_comp", ROW_VY_COMP))

    def _attest(self, sink, sam_idx, before, after, times):
        n = int(before.shape[1])
        tags = {"points_dtype": str(before.dtype)}
        if n == 0:
            tags["empty"] = 1
            sink.application(sam_idx, tags=tags)
            return

        # Measure what LANDED in the tensor, not what was drawn: a delta truncated by the
        # `.to(points.dtype)` cast above reports a shrunken SD here even though the draw was
        # correct. Accumulate in float64 so the pooled sums stay exact over a full cell.
        sums = {}
        detail = {}
        for name, row in self.ATTEST_ROWS:
            delta = (after[row].to(torch.float64) - before[row].to(torch.float64))
            sums["%s_n" % name] = n
            sums["%s_sum" % name] = float(delta.sum())
            sums["%s_sqsum" % name] = float((delta * delta).sum())
            detail[name] = {"n": n, "sd": float(delta.std(unbiased=False)),
                            "mean": float(delta.mean())}

        moved_rows = [r for _, r in self.ATTEST_ROWS]
        untouched = [r for r in range(int(before.shape[0])) if r not in moved_rows]
        tags["positions"] = ("IDENTICAL" if torch.equal(before[0:3], after[0:3]) else "CHANGED")
        tags["untouched_rows"] = (
            "IDENTICAL" if torch.equal(before[untouched], after[untouched]) else "CHANGED")

        sink.set_identity(family=self.FAMILY, sigma=self.sigma, corrupt_seed=self.corrupt_seed)
        sink.application(sam_idx, sums=sums, tags=tags,
                         probe=dict(detail, loader=type(self).__name__, points=n,
                                    dtype=str(before.dtype), positions=tags["positions"],
                                    untouched_rows=tags["untouched_rows"]))


@PIPELINES.register_module()
class RadarDropoutLoadnuradarpoints(_DropoutMixin, Loadnuradarpoints):

    def __init__(self, drop_p, corrupt_seed, **kwargs):
        super(RadarDropoutLoadnuradarpoints, self).__init__(**kwargs)
        self._install_dropout(drop_p, corrupt_seed)


@PIPELINES.register_module()
class RadarDropoutLoadradarpointsFromMultiSweeps(_DropoutMixin, LoadradarpointsFromMultiSweeps):

    def __init__(self, drop_p, corrupt_seed, **kwargs):
        super(RadarDropoutLoadradarpointsFromMultiSweeps, self).__init__(**kwargs)
        self._install_dropout(drop_p, corrupt_seed)


@PIPELINES.register_module()
class RadarNoiseLoadnuradarpoints(_DopplerRcsNoiseMixin, Loadnuradarpoints):

    def __init__(self, sigma, corrupt_seed, **kwargs):
        super(RadarNoiseLoadnuradarpoints, self).__init__(**kwargs)
        self._install_noise(sigma, corrupt_seed)


@PIPELINES.register_module()
class RadarNoiseLoadradarpointsFromMultiSweeps(_DopplerRcsNoiseMixin, LoadradarpointsFromMultiSweeps):

    def __init__(self, sigma, corrupt_seed, **kwargs):
        super(RadarNoiseLoadradarpointsFromMultiSweeps, self).__init__(**kwargs)
        self._install_noise(sigma, corrupt_seed)


_SWAP = {
    "radar_dropout": {
        "Loadnuradarpoints": "RadarDropoutLoadnuradarpoints",
        "LoadradarpointsFromMultiSweeps": "RadarDropoutLoadradarpointsFromMultiSweeps",
    },
    "radar_doppler_rcs_noise": {
        "Loadnuradarpoints": "RadarNoiseLoadnuradarpoints",
        "LoadradarpointsFromMultiSweeps": "RadarNoiseLoadradarpointsFromMultiSweeps",
    },
}


def corrupt_pipeline(pipeline, corruption):
    """Return a copy of ``pipeline`` with BOTH radar loader entries swapped.

    ``corruption`` is a dict: {"family": ..., "corrupt_seed": ..., plus "drop_p" or "sigma"}.
    Raises if either radar entry is missing or duplicated -- a partially corrupted pipeline
    must never run silently.
    """
    corruption = dict(corruption)
    family = corruption.pop("family")
    if family not in _SWAP:
        raise ValueError("unknown corruption family {!r}".format(family))
    swap = _SWAP[family]

    out = []
    hits = {k: 0 for k in swap}
    for entry in pipeline:
        entry = dict(entry)
        old = entry.get("type")
        if old in swap:
            hits[old] += 1
            entry["type"] = swap[old]
            entry.update(corruption)
        out.append(entry)

    for old, count in hits.items():
        if count != 1:
            raise AssertionError(
                "expected exactly 1 {!r} entry in the pipeline, found {} -- refusing to "
                "build a partially corrupted pipeline".format(old, count)
            )
    return out


def build_corrupted_val_pipeline(base_config, corruption):
    """Read ``base_config``, return its ``data.val.pipeline`` with both radar entries swapped."""
    from mmcv import Config

    cfg = Config.fromfile(base_config)
    pipeline = [dict(entry) for entry in cfg.data.val.pipeline]
    return corrupt_pipeline(pipeline, corruption)
