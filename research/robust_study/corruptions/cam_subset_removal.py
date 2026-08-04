"""(a) camera-subset removal — the GATE-B G4 black-frame mechanism restricted to a camera subset.

NEW FILE. Nothing tracked is modified. The substitution semantics are not re-implemented here:
this module imports the frozen GATE-B runner
(``research/robust_study/tools/gate_b_removal.py``) and calls its own ``_zero_image_arg`` --
the single function that DEFINES what "the camera is off" means for the ``--removal table9``
cell (gate_b_removal.py:246-252, invoked from its G4 branch at gate_b_removal.py:295). The
substituted pixels a subset cell writes are therefore, by construction, the exact tensor G4
would have written for those view-frames; the only thing this module adds is *which* view-frames
receive them.

Stage, tensor and units are G4's, unchanged: the wrapper sits on ``RaCFormer.extract_feat``
(models/racformer.py:179), i.e. the RAW image tensor in sensor units, BEFORE the frozen
normalization at models/racformer.py:202-212. A black frame, not a zero activation.

--------------------------------------------------------------------------------------------
Camera indexing: why this module resolves channels from filenames instead of slicing
--------------------------------------------------------------------------------------------
``extract_feat`` receives ``img`` with a flat view-frame axis NT = num_frames * num_cams, laid
out frame-major (models/racformer.py:313 slices ``lidar2img[i*num_cams:(i+1)*num_cams]`` for
frame i). The camera order WITHIN a frame is NOT constant across frames:

  * frame 0 is ordered by the pkl's ``info['cams']`` dict insertion order
    (loaders/nuscenes_dataset.py:207-213 -> :234), observed
    [CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT];
  * frames 1..T-1 are ordered by the hardcoded ``cam_types`` list in
    ``LoadMultiViewImageFromMultiSweeps.load_offline`` (loaders/pipelines/loading.py:632-636
    -> :672), which is
    [CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT].

The two orders differ. A fixed-index slice such as ``img[:, cam_idx::num_cams]`` would black out
CAM_FRONT in frame 0 and CAM_FRONT_LEFT in frames 1..7 -- a different fault from the one
registered, silently. (The exception is a scene-start sample, where the loader has no previous
sweeps and copies frame 0 T times, loading.py:640-646; there the orders coincide, which is why
the discrepancy is invisible on sample 0 of the val split.)

This module therefore derives the channel of every view-frame from ``img_metas['filename']``,
which the loader appends in lockstep with the image itself in both branches
(loading.py:645 and :672), and refuses to run if a filename does not parse to a known channel.
The per-call attestation then asserts that each camera was seen exactly num_frames times, so a
layout change cannot degrade the cell into partial coverage without failing loudly.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
GATE_B_RUNNER = os.path.join(REPO_ROOT, "research", "robust_study", "tools", "gate_b_removal.py")

# The GATE-B runner is a script, not a package module. It is loaded by absolute path under a
# private name and cached in sys.modules, so repeated imports (config parse + runner) reuse one
# module object and one set of function identities.
_GATE_B_MODULE_NAME = "robust_study_gate_b_removal"


def load_gate_b_removal():
    """Import the frozen GATE-B runner. Never copied, never modified."""
    cached = sys.modules.get(_GATE_B_MODULE_NAME)
    if cached is not None:
        return cached
    if not os.path.isfile(GATE_B_RUNNER):
        raise RuntimeError(
            "GATE-B runner not found at %s; the (a)-subset mechanism is defined as a "
            "restriction of it and cannot run without it." % GATE_B_RUNNER
        )
    spec = importlib.util.spec_from_file_location(_GATE_B_MODULE_NAME, GATE_B_RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_GATE_B_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


gate_b = load_gate_b_removal()

# Validation set only. NEVER used as an ordering: see the module docstring.
CAMERA_CHANNELS = frozenset((
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
))

# The registered (a)-ladder cells. The all-6 endpoint is NOT here: it is the existing
# GATE-B G4 + g4_repeat pair and is never re-run (plan.md 16.4).
CELLS = {
    "A1": ("CAM_FRONT",),
    "A2": ("CAM_BACK",),
}


def channel_from_path(path):
    """nuScenes filenames are '<log>__<CHANNEL>__<timestamp>.jpg'. Strict by design."""
    base = os.path.basename(str(path))
    parts = base.split("__")
    if len(parts) < 3 or parts[1] not in CAMERA_CHANNELS:
        raise ValueError(
            "cannot resolve a camera channel from image filename %r; the (a)-subset "
            "mechanism refuses to guess which view-frame belongs to which camera" % (path,)
        )
    return parts[1]


def resolve_channels(img_metas, batch_index, n_view_frames):
    """Channel of every view-frame of one batch element, in tensor order."""
    try:
        filenames = img_metas[batch_index]["filename"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "img_metas[%d] carries no 'filename'; the (a)-subset mechanism needs it to map "
            "view-frames to cameras (Collect3D meta_keys, "
            "configs/racformer_r50_nuimg_704x256_f8.py:219-220)" % batch_index
        ) from exc
    if len(filenames) != n_view_frames:
        raise RuntimeError(
            "img_metas[%d]['filename'] has %d entries but the image tensor has %d view-frames"
            % (batch_index, len(filenames), n_view_frames)
        )
    return [channel_from_path(p) for p in filenames]


def build_mask(channels_per_batch, cameras, torch, device):
    """Per batch element, a bool tensor over the view-frame axis: True = black-frame this view."""
    masks = []
    target = set(cameras)
    for channels in channels_per_batch:
        flags = [channel in target for channel in channels]
        masks.append(torch.tensor(flags, dtype=torch.bool, device=device))
    return masks


def apply_subset(img, masks, torch):
    """Write G4's substituted values into the masked view-frames only.

    The substituted values come from the GATE-B runner's own ``_zero_image_arg``, so they are
    literally the ones the ``--removal table9`` cell writes, for the view-frames selected.
    Dtype, device and the tensor-vs-list layout all come from it unchanged.
    """
    blacked = gate_b._zero_image_arg(img)
    if torch.is_tensor(img):
        out = img.clone()
        for b in range(int(out.shape[0])):
            out[b][masks[b]] = blacked[b][masks[b]]
        return out
    out = []
    for b, per_sample in enumerate(img):
        clone = per_sample.clone()
        clone[masks[b]] = blacked[b][masks[b]]
        out.append(clone)
    return out


class SubsetAttestation(gate_b.Attestation):
    """GATE-B's attestation, with the coverage arithmetic of a camera SUBSET.

    The base class is reused for the branch-hit counter, the shape cross-checks, the probe-set
    bookkeeping and the digest function, so the numbers a subset cell reports are produced by
    the same code that produced GATE-B's. Only ``report`` is overridden: G4's rule "every
    view-frame altered, whole tensor zero" is false for a subset cell by definition, and is
    replaced by the per-camera rules below.
    """

    def __init__(self, cell, cameras, cfg_num_cams, cfg_num_frames, expect_samples):
        # The stage string the base class attaches to every probe is keyed by removal mode;
        # "table9" is the correct one here because this cell's stage IS G4's stage
        # (extract_feat entry, raw image tensor). The family is tracked separately.
        super(SubsetAttestation, self).__init__(
            "table9", cfg_num_cams, cfg_num_frames, expect_samples)
        self.family = "a_camera_subset"
        self.cell = cell
        self.cameras = tuple(cameras)
        self.channel_seen = {}
        self.channel_altered = {}
        self.order_signatures = {}

    def note_channels(self, channels_per_batch, masks):
        for channels, mask in zip(channels_per_batch, masks):
            n_cams = self.cfg_num_cams
            for f in range(len(channels) // n_cams):
                frame = tuple(channels[f * n_cams:(f + 1) * n_cams])
                self.order_signatures[frame] = self.order_signatures.get(frame, 0) + 1
            flags = [bool(x) for x in mask.tolist()]
            for channel, altered in zip(channels, flags):
                self.channel_seen[channel] = self.channel_seen.get(channel, 0) + 1
                if altered:
                    self.channel_altered[channel] = self.channel_altered.get(channel, 0) + 1

    def note_altered(self, n_view_frames):
        self.altered_view_frames += int(n_view_frames)

    def probe_per_camera(self, side, img, channels_per_batch, torch):
        """Digest each camera's view-frames separately, so 'only this camera changed' is checkable."""
        if self.current_probe is None:
            return
        for b, channels in enumerate(channels_per_batch):
            per_sample = img[b]
            buckets = {}
            for view_index, channel in enumerate(channels):
                buckets.setdefault(channel, []).append(per_sample[view_index])
            for channel, slices in buckets.items():
                self.probe(side, "b%d:%s" % (b, channel), torch.stack(slices, dim=0))

    def report(self, driver_n_total=None):
        expect_view_frames = self.expect_samples * self.cfg_num_cams * self.cfg_num_frames
        per_camera_view_frames = self.expect_samples * self.cfg_num_frames
        expect_altered = per_camera_view_frames * len(self.cameras)
        failures = []

        if self.calls != self.expect_samples:
            failures.append(
                "branch-hit count %d != expected n_samples %d" % (self.calls, self.expect_samples))
        if driver_n_total is not None and driver_n_total != self.expect_samples:
            failures.append(
                "driver n_total %s != expected n_samples %d" % (driver_n_total, self.expect_samples))
        if self.covered_view_frames != expect_view_frames:
            failures.append(
                "covered view-frames %d != expected %d (partial coverage)"
                % (self.covered_view_frames, expect_view_frames))
        failures.extend(self.shape_errors[:5])

        if self.altered_view_frames != expect_altered:
            failures.append(
                "altered view-frames %d != expected %d (= %d camera(s) x %d frames x %d samples)"
                % (self.altered_view_frames, expect_altered, len(self.cameras),
                   self.cfg_num_frames, self.expect_samples))

        # Every camera must have been SEEN the full number of times -- this is what catches a
        # loader layout change that silently drops or duplicates a channel.
        for channel in sorted(CAMERA_CHANNELS):
            seen = self.channel_seen.get(channel, 0)
            if seen != per_camera_view_frames:
                failures.append(
                    "camera %s appeared in %d view-frames, expected %d"
                    % (channel, seen, per_camera_view_frames))
        for channel in sorted(CAMERA_CHANNELS):
            altered = self.channel_altered.get(channel, 0)
            want = per_camera_view_frames if channel in self.cameras else 0
            if altered != want:
                failures.append(
                    "camera %s had %d view-frames blacked, expected %d"
                    % (channel, altered, want))

        if len(self.probes) != gate_b.PROBE_SAMPLES:
            failures.append(
                "probe set has %d samples, expected %d"
                % (len(self.probes), gate_b.PROBE_SAMPLES))
        for probe in self.probes:
            names = sorted(set(probe["pre"]) | set(probe["post"]))
            if not names:
                failures.append("probe call %d recorded no digests" % probe["call_index"])
            for name in names:
                pre = probe["pre"].get(name, {}).get("sha256")
                post = probe["post"].get(name, {}).get("sha256")
                if pre is None or post is None:
                    failures.append(
                        "probe call %d tensor %s is missing a pre or post digest"
                        % (probe["call_index"], name))
                    continue
                if name == "raw_img":
                    if pre == post:
                        failures.append(
                            "probe call %d: the whole image tensor is unchanged; the cell "
                            "blacked nothing" % probe["call_index"])
                    continue
                channel = name.split(":", 1)[1]
                if channel in self.cameras:
                    if pre == post:
                        failures.append(
                            "probe call %d camera %s is unchanged; it is a target camera"
                            % (probe["call_index"], channel))
                    if probe["post"][name]["abs_max"] != 0.0:
                        failures.append(
                            "probe call %d camera %s is not black after the intervention "
                            "(abs_max=%r)" % (probe["call_index"], channel,
                                              probe["post"][name]["abs_max"]))
                elif pre != post:
                    failures.append(
                        "probe call %d camera %s changed; only %s may change"
                        % (probe["call_index"], channel, ", ".join(self.cameras)))

        return {
            "schema": "a_subset_intervention_attestation/1",
            "family": self.family,
            "cell": self.cell,
            "cameras": list(self.cameras),
            "mechanism": {
                "stage": gate_b.PROBE_STAGE["table9"],
                "source": "research/robust_study/tools/gate_b_removal.py:_zero_image_arg "
                          "(:246-252), the same function the --removal table9 branch calls "
                          "at :295",
                "camera_resolution": "img_metas['filename'] per view-frame; no fixed index",
            },
            "expected": {
                "n_samples": self.expect_samples,
                "num_cams": self.cfg_num_cams,
                "num_cams_source": "models/racformer.py:45 ctor default; the config does not set it",
                "num_frames": self.cfg_num_frames,
                "num_frames_source": "config num_frames",
                "view_frames": expect_view_frames,
                "altered_view_frames": expect_altered,
                "per_camera_view_frames": per_camera_view_frames,
            },
            "observed": {
                "branch_hits": self.calls,
                "driver_n_total": driver_n_total,
                "covered_view_frames": self.covered_view_frames,
                "altered_view_frames": self.altered_view_frames,
                "channel_seen": dict(sorted(self.channel_seen.items())),
                "channel_altered": dict(sorted(self.channel_altered.items())),
                "frame_camera_orders": [
                    {"order": list(order), "frames": count}
                    for order, count in sorted(self.order_signatures.items())
                ],
            },
            "probe_set": self.probes,
            "failures": failures,
            "verdict": "PASS" if not failures else "FAIL",
        }


def install_subset_intervention(cameras, att):
    """Wrap ``RaCFormer.extract_feat`` so only ``cameras`` receive G4's black frame.

    Returns the un-patched method so a caller (the probe) can restore it.
    """
    import torch

    from models.racformer import RaCFormer

    cameras = tuple(cameras)
    unknown = sorted(set(cameras) - CAMERA_CHANNELS)
    if unknown:
        raise ValueError("unknown camera channel(s): %s" % ", ".join(unknown))
    if not cameras:
        raise ValueError("the (a)-subset cell needs at least one camera")

    original_extract_feat = RaCFormer.extract_feat

    def patched_extract_feat(self, img, radar_points, radar_depth, radar_rcs, img_metas):
        batch, n_view_frames = gate_b._batch_and_view_frames(img)
        att.on_extract_feat(batch, n_view_frames, int(self.num_cams))
        try:
            att.probe_identity(img_metas[0].get("filename"))
        except (AttributeError, IndexError, TypeError):
            pass

        channels = [resolve_channels(img_metas, b, n_view_frames) for b in range(batch)]
        device = gate_b._first_tensor(img).device
        masks = build_mask(channels, cameras, torch, device)
        att.note_channels(channels, masks)

        att.probe("pre", "raw_img", gate_b._first_tensor(img))
        att.probe_per_camera("pre", img, channels, torch)

        img = apply_subset(img, masks, torch)
        att.note_altered(sum(int(mask.sum()) for mask in masks))

        att.probe("post", "raw_img", gate_b._first_tensor(img))
        att.probe_per_camera("post", img, channels, torch)

        return original_extract_feat(self, img, radar_points, radar_depth, radar_rcs, img_metas)

    RaCFormer.extract_feat = patched_extract_feat
    return original_extract_feat
