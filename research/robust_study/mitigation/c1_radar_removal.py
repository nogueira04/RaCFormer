#!/usr/bin/env python
"""C1 mitigation candidate: radar-branch removal (modality-drop fallback).

The radar-side analog of the GATE-B camera machinery in
`research/robust_study/tools/gate_b_removal.py`. Same contract: no frozen file is modified, the
interventions are installed as wrappers on `RaCFormer` methods before the driver builds the model,
and everything else -- config, weights, dataset, dataloader, seeds, inference loop, submission
writing, metric computation -- is the frozen driver
`research/night_gen_phase1/eval_by_condition.py` invoked in-process through its own `main()`.

Radar reaches RaCFormer through TWO independent paths, so "disable the radar branch" is two
interventions, not one:

  1. BEV pillar branch. `radar_points` -> `extract_pts_feat` (models/racformer.py:129-148) ->
     `radar_bev_feats`, consumed by `models/racformer_transformer.py:248 sampling_radar_bev`.
  2. Perspective-view branch. `radar_depth` / `radar_rcs` -> `extract_feat`
     (models/racformer.py:179, reshaped :226-227, passed to the view transformer at :320/:328/:334)
     -> `models/necks/view_transformer_racformer.py:682-697`, where they condition the LSS depth
     net. Zeroing only the BEV branch leaves this one live and is NOT camera-only.

  --radar-removal none   Strict pass-through. Wrappers and counters installed, no tensor touched.
                         Its submission must reproduce the unmitigated cell bit-for-bit.
  --radar-removal bev    BEV branch only: `extract_pts_feat` is short-circuited (see below).
  --radar-removal pv     PV branch only: `radar_depth` and `radar_rcs` are zeroed at the entry of
                         `extract_feat`.
  --radar-removal both   The registered C1 arm: BEV + PV, i.e. the model runs camera-only.

Why the BEV branch is bypassed rather than fed zeroed points
------------------------------------------------------------
The frozen checkpoint's tree (`models/racformer.py:139-148`) predates the fix on branch
`fix/racformer-empty-radar-voxelization` (commit 836dc47, NOT an ancestor of the freeze), and that
code has TWO distinct breakages on the empty/degenerate radar path:

  * `:142  batch_size = coors[-1, 0] + 1`  -- IndexError when `coors` is empty (zero voxels);
  * `:144  radar_features.squeeze()`       -- `PillarFeatureNet.forward` already returns (N, 64),
    so at N == 1 this collapses to (64,) and `PointPillarsScatter.forward_batch`, which expects
    (N, C), breaks.

Eval runs at batch size 1 and `extract_pts_feat:136` already forces z = 0, so a shape-preserving
zero-fill puts every point in ONE voxel -> N == 1 -> the second breakage fires. In the frozen tree
an empty tensor AND a naive zero-fill both crash. C1 therefore never enters the voxelizer at all:
`_bypass_radar_bev` reproduces exactly the state commit 836dc47 installs for an empty batch --
`radar_bev_conv` applied to an all-zero scatter canvas -- without importing the fix and without
touching a tracked file. `radar_voxelize` is wrapped with a counter so the attestation can prove
the voxelizer was never reached.

Note the conv is APPLIED to the zero canvas rather than having its output zeroed. `radar_bev_conv`
is a `ConvModule` stack with bias and BatchNorm (models/racformer.py:81-99), so conv(0) != 0.
"No radar returns" means an empty scatter canvas fed through the frozen conv; a zeroed output
feature map would be a different, unmotivated intervention. The attestation therefore does NOT
assert abs_max == 0 on the BEV output -- only on the PV maps, where zero is the correct target.

Runtime intervention attestation (written to <out-dir>/intervention_attestation.json; a FAIL exits
non-zero so the calling job records validity=INVALID):

  (i)   branch-hit count == n_samples, cross-checked against the driver's own `n_total`;
  (ii)  covered radar view-frames == n_samples x num_cams x num_frames, and for `pv`/`both` the
        altered count must equal it -- partial coverage is INVALID, not a warning;
  (iii) `extract_pts_feat` bypass count == its call count for `bev`/`both`, and the
        `radar_voxelize` counter must be exactly 0 for those modes and non-zero otherwise;
  (iv)  paired pre/post digests over a fixed 3-sample probe set: bit-identical for `none`,
        and for `pv`/`both` the PV maps must be zero afterwards.

CUDA is required. This script never runs inference on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time

REMOVALS = ("none", "bev", "pv", "both")

# Which branch each mode touches. Purely descriptive; the runner derives the actual digests and
# counters from the tensors it intervenes on.
PROBE_STAGE = {
    "none": "extract_feat entry, radar_depth/radar_rcs (no-op probe)",
    "bev": "extract_pts_feat return, radar BEV feature map (voxelizer bypassed)",
    "pv": "extract_feat entry, radar_depth/radar_rcs",
    "both": "extract_feat entry radar_depth/radar_rcs + extract_pts_feat return",
}

PROBE_SAMPLES = 3

ZEROES_PV = ("pv", "both")
BYPASSES_BEV = ("bev", "both")


def _digest(tensor):
    import numpy as np
    import torch

    array = tensor.detach().to(dtype=torch.float32).cpu().numpy()
    array = np.ascontiguousarray(array)
    return {
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "shape": [int(d) for d in tensor.shape],
        "dtype": str(tensor.dtype),
        "abs_max": float(abs(array).max()) if array.size else 0.0,
    }


class Attestation:
    """Collects the coverage counters and probe digests, then judges them."""

    def __init__(self, removal, cfg_num_cams, cfg_num_frames, expect_samples):
        self.removal = removal
        self.cfg_num_cams = cfg_num_cams
        self.cfg_num_frames = cfg_num_frames
        self.expect_samples = expect_samples

        self.calls = 0
        self.covered_view_frames = 0
        self.altered_pv_view_frames = 0
        self.pts_calls = 0
        self.pts_bypassed = 0
        self.voxelize_calls = 0
        self.probes = []
        self.shape_errors = []

        self.current_b = None
        self.current_nt = None
        self.current_probe = None

    # -- called from the wrappers -------------------------------------------------------------
    def on_extract_feat(self, batch, n_view_frames, runtime_num_cams):
        self.calls += 1
        self.current_b = batch
        self.current_nt = n_view_frames
        self.covered_view_frames += batch * n_view_frames

        if runtime_num_cams != self.cfg_num_cams:
            self.shape_errors.append(
                "call %d: runtime num_cams=%s != expected %s"
                % (self.calls, runtime_num_cams, self.cfg_num_cams)
            )
        elif n_view_frames % runtime_num_cams != 0:
            self.shape_errors.append(
                "call %d: NT=%s is not a multiple of num_cams=%s"
                % (self.calls, n_view_frames, runtime_num_cams)
            )
        elif n_view_frames // runtime_num_cams != self.cfg_num_frames:
            self.shape_errors.append(
                "call %d: frames-per-sample=%s != config num_frames=%s"
                % (self.calls, n_view_frames // runtime_num_cams, self.cfg_num_frames)
            )

        if self.calls <= PROBE_SAMPLES:
            self.current_probe = {
                "call_index": self.calls - 1,
                "stage": PROBE_STAGE[self.removal],
                "pre": {},
                "post": {},
            }
            self.probes.append(self.current_probe)
        else:
            self.current_probe = None

    def note_altered_pv(self):
        self.altered_pv_view_frames += (self.current_b or 0) * (self.current_nt or 0)

    def on_extract_pts_feat(self):
        self.pts_calls += 1

    def note_pts_bypassed(self):
        self.pts_bypassed += 1

    def note_voxelize(self):
        self.voxelize_calls += 1

    def probe(self, side, name, tensor):
        if self.current_probe is not None:
            self.current_probe[side][name] = _digest(tensor)

    def probe_identity(self, filenames):
        if self.current_probe is not None and filenames:
            self.current_probe["first_camera_file"] = str(filenames[0])

    # -- verdict -------------------------------------------------------------------------------
    def report(self, driver_n_total):
        expect_view_frames = self.expect_samples * self.cfg_num_cams * self.cfg_num_frames
        failures = []

        if self.calls != self.expect_samples:
            failures.append(
                "branch-hit count %d != expected n_samples %d" % (self.calls, self.expect_samples)
            )
        if driver_n_total is not None and driver_n_total != self.expect_samples:
            failures.append(
                "driver n_total %s != expected n_samples %d" % (driver_n_total, self.expect_samples)
            )
        if self.covered_view_frames != expect_view_frames:
            failures.append(
                "covered view-frames %d != expected %d (partial coverage)"
                % (self.covered_view_frames, expect_view_frames)
            )
        failures.extend(self.shape_errors[:5])

        # -- PV branch coverage ---------------------------------------------------------------
        if self.removal in ZEROES_PV:
            if self.altered_pv_view_frames != expect_view_frames:
                failures.append(
                    "altered PV view-frames %d != expected %d (partial coverage)"
                    % (self.altered_pv_view_frames, expect_view_frames)
                )
        elif self.altered_pv_view_frames != 0:
            failures.append(
                "--radar-removal %s altered %d PV view-frames; it must alter none"
                % (self.removal, self.altered_pv_view_frames)
            )

        # -- BEV branch coverage --------------------------------------------------------------
        if self.pts_calls == 0:
            failures.append("extract_pts_feat was never called; the radar branch was not exercised")
        if self.removal in BYPASSES_BEV:
            if self.pts_bypassed != self.pts_calls:
                failures.append(
                    "extract_pts_feat bypassed %d of %d calls (partial coverage)"
                    % (self.pts_bypassed, self.pts_calls)
                )
            if self.voxelize_calls != 0:
                failures.append(
                    "radar_voxelize ran %d times under --radar-removal %s; the bypass exists "
                    "precisely so the frozen voxelizer is never reached"
                    % (self.voxelize_calls, self.removal)
                )
        else:
            if self.pts_bypassed != 0:
                failures.append(
                    "--radar-removal %s bypassed %d extract_pts_feat calls; it must bypass none"
                    % (self.removal, self.pts_bypassed)
                )
            if self.voxelize_calls != self.pts_calls:
                failures.append(
                    "radar_voxelize ran %d times for %d extract_pts_feat calls; the unbypassed "
                    "path must voxelize exactly once per call"
                    % (self.voxelize_calls, self.pts_calls)
                )

        # -- probe digests --------------------------------------------------------------------
        if len(self.probes) != PROBE_SAMPLES:
            failures.append(
                "probe set has %d samples, expected %d" % (len(self.probes), PROBE_SAMPLES)
            )
        for probe in self.probes:
            names = sorted(set(probe["pre"]) | set(probe["post"]))
            if not names:
                failures.append("probe call %d recorded no digests" % probe["call_index"])
            for name in names:
                pre = probe["pre"].get(name, {}).get("sha256")
                post = probe["post"].get(name, {}).get("sha256")
                # The BEV probe is recorded post-only by construction: computing a "pre" would
                # mean running the voxelizer the bypass exists to avoid.
                if name == "radar_bev_feats" and pre is None and post is not None:
                    continue
                if pre is None or post is None:
                    failures.append(
                        "probe call %d tensor %s is missing a pre or post digest"
                        % (probe["call_index"], name)
                    )
                    continue
                same = pre == post
                pre_abs_max = probe["pre"][name]["abs_max"]
                if self.removal == "none" and not same:
                    failures.append(
                        "probe call %d tensor %s changed under --radar-removal none"
                        % (probe["call_index"], name)
                    )
                if self.removal in ZEROES_PV and same and pre_abs_max != 0.0:
                    failures.append(
                        "probe call %d tensor %s is unchanged under --radar-removal %s"
                        % (probe["call_index"], name, self.removal)
                    )
                if self.removal in ZEROES_PV and probe["post"][name]["abs_max"] != 0.0:
                    failures.append(
                        "probe call %d tensor %s is not zero after the intervention (abs_max=%r)"
                        % (probe["call_index"], name, probe["post"][name]["abs_max"])
                    )

        return {
            "schema": "c1_radar_intervention_attestation/1",
            "radar_removal": self.removal,
            "expected": {
                "n_samples": self.expect_samples,
                "num_cams": self.cfg_num_cams,
                "num_cams_source": "models/racformer.py:45 ctor default; the config does not set it",
                "num_frames": self.cfg_num_frames,
                "num_frames_source": "config num_frames",
                "view_frames": expect_view_frames,
            },
            "observed": {
                "branch_hits": self.calls,
                "driver_n_total": driver_n_total,
                "covered_view_frames": self.covered_view_frames,
                "altered_pv_view_frames": self.altered_pv_view_frames,
                "extract_pts_feat_calls": self.pts_calls,
                "extract_pts_feat_bypassed": self.pts_bypassed,
                "radar_voxelize_calls": self.voxelize_calls,
            },
            "probe_set": self.probes,
            "failures": failures,
            "verdict": "PASS" if not failures else "FAIL",
        }


def _batch_and_view_frames(img):
    """(B, NT) for either layout, without allocating. See models/racformer.py:180-185."""
    import torch

    if torch.is_tensor(img):
        return int(img.shape[0]), int(img.shape[1])
    return len(img), int(img[0].shape[0])


def bypass_radar_bev(model, radar_points):
    """Return the radar BEV features for "no radar returns", without voxelizing.

    Reproduces the state commit 836dc47 installs for an empty voxel batch: an all-zero
    PointPillarsScatter canvas fed through the frozen `radar_bev_conv`. Shapes come from the
    built middle encoder (`in_channels`, `ny`, `nx`) rather than being hardcoded, so a config
    change cannot silently desynchronise the bypass from the real path.
    """
    import torch

    batch_size = len(radar_points)
    encoder = model.radar_middle_encoder
    param = next(model.radar_bev_conv.parameters())
    canvas = torch.zeros(
        batch_size,
        int(encoder.in_channels),
        int(encoder.ny),
        int(encoder.nx),
        device=param.device,
        dtype=torch.float32,
    )
    return model.radar_bev_conv(canvas)


def install_intervention(removal, att):
    """Wrap the three RaCFormer methods the radar path funnels through.

    All three wrappers are installed for every mode, including `none`: the counters and the probe
    set have to be produced by the same code in every cell, otherwise the attestation would not be
    evidence about the cells it is attesting.

    Returns a callable that restores the original unbound methods, so a caller can measure several
    modes in one process (used by the smoke test; the cell runner ignores it).
    """
    import torch

    from models.racformer import RaCFormer

    original_extract_feat = RaCFormer.extract_feat
    original_extract_pts_feat = RaCFormer.extract_pts_feat
    original_radar_voxelize = RaCFormer.radar_voxelize

    def patched_extract_feat(self, img, radar_points, radar_depth, radar_rcs, img_metas):
        batch, n_view_frames = _batch_and_view_frames(img)
        att.on_extract_feat(batch, n_view_frames, int(self.num_cams))
        try:
            att.probe_identity(img_metas[0].get("filename"))
        except (AttributeError, IndexError, TypeError):
            pass

        att.probe("pre", "radar_depth", radar_depth)
        att.probe("pre", "radar_rcs", radar_rcs)
        if removal in ZEROES_PV:
            radar_depth = torch.zeros_like(radar_depth)
            radar_rcs = torch.zeros_like(radar_rcs)
            att.note_altered_pv()
        att.probe("post", "radar_depth", radar_depth)
        att.probe("post", "radar_rcs", radar_rcs)

        out = original_extract_feat(self, img, radar_points, radar_depth, radar_rcs, img_metas)
        att.probe("post", "radar_bev_feats", out[2])
        return out

    def patched_extract_pts_feat(self, radar_points=None):
        att.on_extract_pts_feat()
        if removal in BYPASSES_BEV:
            att.note_pts_bypassed()
            return bypass_radar_bev(self, radar_points)
        return original_extract_pts_feat(self, radar_points=radar_points)

    def patched_radar_voxelize(self, points):
        att.note_voxelize()
        return original_radar_voxelize(self, points)

    RaCFormer.extract_feat = patched_extract_feat
    RaCFormer.extract_pts_feat = patched_extract_pts_feat
    RaCFormer.radar_voxelize = patched_radar_voxelize

    def uninstall():
        RaCFormer.extract_feat = original_extract_feat
        RaCFormer.extract_pts_feat = original_extract_pts_feat
        RaCFormer.radar_voxelize = original_radar_voxelize

    return uninstall


def load_driver(driver_path):
    spec = importlib.util.spec_from_file_location("c1_eval_driver", driver_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_config_expectations(config_path):
    from mmcv import Config

    cfg = Config.fromfile(config_path)
    num_frames = int(cfg.num_frames)
    transformer_frames = int(cfg.model.pts_bbox_head.transformer.num_frames)
    if transformer_frames != num_frames:
        raise SystemExit(
            "config is internally inconsistent: num_frames=%d but "
            "model.pts_bbox_head.transformer.num_frames=%d" % (num_frames, transformer_frames)
        )
    num_cams = int(cfg.model.get("num_cams", 6))
    return num_cams, num_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--radar-removal", required=True, choices=REMOVALS)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expect-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; C1 never evaluates on CPU.")

    num_cams, num_frames = read_config_expectations(args.config)
    att = Attestation(args.radar_removal, num_cams, num_frames, args.expect_samples)

    importlib.import_module("models")
    importlib.import_module("loaders")
    install_intervention(args.radar_removal, att)

    driver_path = os.path.join(repo, "research/night_gen_phase1/eval_by_condition.py")
    driver = load_driver(driver_path)

    os.makedirs(args.out_dir, exist_ok=True)
    argv = [
        driver_path,
        "--config",
        args.config,
        "--weights",
        args.weights,
        "--out-dir",
        args.out_dir,
        "--full-val",
        "--batch-size",
        str(args.batch_size),
    ]
    print("[c1] radar_removal=%s" % args.radar_removal, flush=True)
    print("[c1] driver argv: %s" % " ".join(argv[1:]), flush=True)

    saved_argv = sys.argv
    started = time.time()
    try:
        sys.argv = argv
        driver.main()
    finally:
        sys.argv = saved_argv

    driver_n_total = None
    driver_json = os.path.join(args.out_dir, "eval_by_condition.json")
    if os.path.isfile(driver_json):
        with open(driver_json) as handle:
            driver_n_total = json.load(handle).get("n_total")

    report = att.report(driver_n_total)
    report["elapsed_s"] = round(time.time() - started, 3)
    report["config"] = os.path.abspath(args.config)
    report["weights"] = os.path.abspath(args.weights)
    report["driver"] = driver_path

    report_path = os.path.join(args.out_dir, "intervention_attestation.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print("[c1] attestation: %s" % report_path, flush=True)
    print("[c1] verdict=%s" % report["verdict"], flush=True)
    for failure in report["failures"]:
        print("[c1] FAILURE: %s" % failure, file=sys.stderr, flush=True)

    if report["verdict"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
