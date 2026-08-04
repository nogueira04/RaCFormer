#!/usr/bin/env python
"""Unit-level smoke for the C1 radar-removal arm. CLEAN input only, no detection metrics.

This is candidate unit-testing, not an outcome run: it touches ONE clean val sample, computes no
detection metrics, writes no submission, and never sees corrupted input. It answers four questions
about `research/robust_study/mitigation/c1_radar_removal.py`:

  1. Do all four modes return the same extract_feat output SHAPES as the unpatched model?
  2. Is `--radar-removal none` BIT-IDENTICAL to the unpatched model (sha256 over every returned
     tensor)? That is the no-op check the registration requires.
  3. Do `bev`/`both` really never reach the frozen voxelizer (radar_voxelize counter == 0), while
     `none`/`pv` reach it exactly once per extract_pts_feat call?
  4. Does the bypass survive the degenerate voxel counts that break the frozen tree -- N == 0
     (empty radar tensor) and N == 1 (all points in one voxel, what a naive zero-fill produces)?
     The unpatched path is expected to RAISE on both; that expectation is the reason C1 bypasses
     the voxelizer instead of feeding it zeroed points. A raise there is recorded as evidence,
     not as a smoke failure.

The degenerate-N probe calls `extract_pts_feat` directly with synthetic point tensors. Those are
shape probes on the radar branch, not corrupted evaluation input: no detection is run on them and
no metric is derived from them.

CUDA is required; this script never runs on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback

MODES = ("unpatched", "none", "bev", "pv", "both")


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


def _digest_outputs(out):
    """Digest the four things extract_feat returns (models/racformer.py:348)."""
    img_feats, bev_feats, radar_bev_feats, depth = out
    digests = {"bev_feats": _digest(bev_feats), "radar_bev_feats": _digest(radar_bev_feats)}
    if depth is not None:
        digests["depth"] = _digest(depth)
    for level, feat in enumerate(img_feats):
        digests["img_feats_l%d" % level] = _digest(feat)
    return digests


def install_recorder(sink):
    """Wrap whatever RaCFormer.extract_feat currently is and record its outputs."""
    from models.racformer import RaCFormer

    current = RaCFormer.extract_feat

    def recording_extract_feat(self, img, radar_points, radar_depth, radar_rcs, img_metas):
        out = current(self, img, radar_points, radar_depth, radar_rcs, img_metas)
        sink.append(_digest_outputs(out))
        return out

    RaCFormer.extract_feat = recording_extract_feat

    def restore():
        RaCFormer.extract_feat = current

    return restore


def degenerate_probe(model, n_points, bypassed, device):
    """Call extract_pts_feat with a synthetic radar tensor of exactly n_points points.

    n_points == 0 exercises the empty-coors breakage at models/racformer.py:142; n_points == 1
    exercises the single-voxel `.squeeze()` breakage at :144, which is also what a shape-preserving
    zero-fill degenerates to (extract_pts_feat:136 forces z = 0, so co-located points share a
    voxel). Returns a record; never raises.
    """
    import torch

    points = torch.zeros(n_points, 7, device=device, dtype=torch.float32)
    record = {"n_points": n_points, "bypassed": bypassed}
    try:
        with torch.no_grad():
            out = model.extract_pts_feat(radar_points=[points])
        record["ok"] = True
        record["out"] = _digest(out)
    except Exception as exc:  # noqa: BLE001 -- the raise IS the evidence being collected
        record["ok"] = False
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)[:400]
        frames = traceback.format_exc().strip().splitlines()
        record["traceback"] = [line.strip()[:200] for line in frames]
        # The frozen frame that actually raised, e.g. `models/racformer.py:142`.
        racformer_frames = [
            line.strip() for line in frames if "models/racformer.py" in line and ", line " in line
        ]
        record["frozen_frame"] = racformer_frames[-1][:200] if racformer_frames else None
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)

    import importlib

    import torch
    import torch.backends.cudnn as cudnn
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_model

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; the C1 smoke never runs on CPU.")
    torch.cuda.set_device(0)

    sys.path.insert(0, os.path.join(repo, "research/robust_study/mitigation"))
    import c1_radar_removal as c1

    importlib.import_module("models")
    importlib.import_module("loaders")

    cfg = Config.fromfile(args.config)
    num_cams, num_frames = c1.read_config_expectations(args.config)

    set_random_seed(0, deterministic=True)
    cudnn.benchmark = False

    print("[smoke] building val dataset (one sample will be used)", flush=True)
    dataset = build_dataset(cfg.data.val)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )
    def fresh_sample():
        """A pristine copy of val sample 0.

        The forward pass MUTATES img_metas in place -- models/racformer_transformer.py:109
        replaces `lidar2img` with a CUDA tensor and :102 adds `time_diff`, and
        models/racformer.py:216-217 rewrites `img_shape`/`ori_shape`. Reusing one batch across
        modes therefore feeds the second mode a tensor where the frozen
        `get_mlp_input` (models/necks/view_transformer_racformer.py:591) expects a numpy array.
        The loader is deterministic here (shuffle=False, workers_per_gpu=0, seed=0), so
        re-iterating yields the same sample.
        """
        return next(iter(loader))

    print("[smoke] building model %s" % cfg.model.type, flush=True)
    model = build_model(cfg.model)
    model = MMDataParallel(model.cuda(), [0])
    load_checkpoint(model, args.weights, map_location="cuda", strict=True)
    model.eval()

    results = {}
    attestations = {}
    started = time.time()

    sample_token = None
    for mode in MODES:
        sink = []
        uninstall_c1 = None
        att = None
        if mode != "unpatched":
            att = c1.Attestation(mode, num_cams, num_frames, expect_samples=1)
            uninstall_c1 = c1.install_intervention(mode, att)
        restore_recorder = install_recorder(sink)
        sample = fresh_sample()
        if sample_token is None:
            try:
                sample_token = str(sample["img_metas"][0].data[0][0].get("sample_idx"))
            except Exception:  # noqa: BLE001 -- provenance nicety only
                pass
        # The offline test path does not read these, but clearing them removes any cross-mode
        # coupling through the temporal caches (models/racformer.py:58-62).
        for cache in ("memory", "memory_bev", "memory_radar_bev", "memory_dep"):
            getattr(model.module, cache).clear()
        set_random_seed(0, deterministic=True)
        try:
            with torch.no_grad():
                model(return_loss=False, rescale=True, **sample)
        finally:
            restore_recorder()
            if uninstall_c1 is not None:
                uninstall_c1()

        if len(sink) != 1:
            raise SystemExit(
                "mode %s: extract_feat ran %d times for one sample; expected 1" % (mode, len(sink))
            )
        results[mode] = sink[0]
        if att is not None:
            attestations[mode] = {
                "extract_feat_calls": att.calls,
                "covered_view_frames": att.covered_view_frames,
                "altered_pv_view_frames": att.altered_pv_view_frames,
                "extract_pts_feat_calls": att.pts_calls,
                "extract_pts_feat_bypassed": att.pts_bypassed,
                "radar_voxelize_calls": att.voxelize_calls,
                "probe_set": att.probes,
            }
        print(
            "[smoke] %-9s radar_bev_feats sha=%s voxelize_calls=%s"
            % (
                mode,
                results[mode]["radar_bev_feats"]["sha256"][:16],
                "n/a" if att is None else att.voxelize_calls,
            ),
            flush=True,
        )

    # -- degenerate voxel counts ---------------------------------------------------------------
    raw_model = model.module
    device = next(raw_model.radar_bev_conv.parameters()).device
    degenerate = []
    for n_points in (0, 1):
        degenerate.append(degenerate_probe(raw_model, n_points, bypassed=False, device=device))
    att_bev = c1.Attestation("bev", num_cams, num_frames, expect_samples=1)
    uninstall_bev = c1.install_intervention("bev", att_bev)
    try:
        for n_points in (0, 1):
            degenerate.append(degenerate_probe(raw_model, n_points, bypassed=True, device=device))
    finally:
        uninstall_bev()
    for record in degenerate:
        print(
            "[smoke] degenerate N=%d bypassed=%s ok=%s %s"
            % (
                record["n_points"],
                record["bypassed"],
                record["ok"],
                record.get("error_type", record.get("out", {}).get("shape")),
            ),
            flush=True,
        )

    # -- assertions ----------------------------------------------------------------------------
    failures = []
    names = sorted(results["unpatched"])

    for mode in MODES:
        if sorted(results[mode]) != names:
            failures.append("mode %s returned a different set of tensors than unpatched" % mode)
            continue
        for name in names:
            if results[mode][name]["shape"] != results["unpatched"][name]["shape"]:
                failures.append(
                    "mode %s tensor %s shape %s != unpatched %s"
                    % (mode, name, results[mode][name]["shape"], results["unpatched"][name]["shape"])
                )

    for name in names:
        if results["none"][name]["sha256"] != results["unpatched"][name]["sha256"]:
            failures.append("--radar-removal none is not bit-identical to unpatched on %s" % name)

    for mode in ("bev", "both"):
        if attestations[mode]["radar_voxelize_calls"] != 0:
            failures.append(
                "mode %s entered the voxelizer %d times; the bypass must reach it zero times"
                % (mode, attestations[mode]["radar_voxelize_calls"])
            )
        if attestations[mode]["extract_pts_feat_bypassed"] != attestations[mode]["extract_pts_feat_calls"]:
            failures.append("mode %s bypassed only some extract_pts_feat calls" % mode)
        if results[mode]["radar_bev_feats"]["sha256"] == results["unpatched"]["radar_bev_feats"]["sha256"]:
            failures.append("mode %s left radar_bev_feats unchanged; the bypass did nothing" % mode)

    for mode in ("none", "pv"):
        pts_calls = attestations[mode]["extract_pts_feat_calls"]
        if attestations[mode]["radar_voxelize_calls"] != pts_calls:
            failures.append(
                "mode %s voxelized %d times for %d extract_pts_feat calls"
                % (mode, attestations[mode]["radar_voxelize_calls"], pts_calls)
            )
        if attestations[mode]["extract_pts_feat_bypassed"] != 0:
            failures.append("mode %s bypassed extract_pts_feat; it must not" % mode)

    for mode in ("pv", "both"):
        for probe in attestations[mode]["probe_set"]:
            for name in ("radar_depth", "radar_rcs"):
                post = probe["post"].get(name)
                if post is None:
                    failures.append("mode %s recorded no post digest for %s" % (mode, name))
                elif post["abs_max"] != 0.0:
                    failures.append(
                        "mode %s left %s non-zero (abs_max=%r)" % (mode, name, post["abs_max"])
                    )
        if results[mode]["bev_feats"]["sha256"] == results["unpatched"]["bev_feats"]["sha256"]:
            failures.append(
                "mode %s left the camera-BEV stack unchanged; zeroing the PV radar maps should "
                "perturb the LSS depth net" % mode
            )

    for record in degenerate:
        if record["bypassed"] and not record["ok"]:
            failures.append(
                "bypass failed at N=%d: %s" % (record["n_points"], record.get("error_type"))
            )

    report = {
        "schema": "c1_radar_removal_smoke/1",
        "config": os.path.abspath(args.config),
        "weights": os.path.abspath(args.weights),
        "sample_token": sample_token,
        "digests": results,
        "attestation_counters": attestations,
        "degenerate_voxel_probe": degenerate,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
        "elapsed_s": round(time.time() - started, 3),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "c1_smoke.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print("[smoke] report: %s" % report_path, flush=True)
    print("[smoke] verdict=%s" % report["verdict"], flush=True)
    for failure in failures:
        print("[smoke] FAILURE: %s" % failure, file=sys.stderr, flush=True)

    if failures:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
