#!/usr/bin/env python
"""GATE-B camera-removal runner — one code path, four removal mechanisms.

The four GATE-B cells differ in exactly one thing: which `--removal` branch runs. Everything else
-- config, weights, dataset, dataloader, seeds, inference loop, submission writing, metric
computation -- is the frozen driver `research/night_gen_phase1/eval_by_condition.py`, invoked
in-process through its own `main()`. That is deliberate: GATE-B exists to decide whether three
implementations of "the camera is off" disagree, and that question is only meaningful if nothing
else can differ between the cells.

No frozen file is modified. The interventions are installed as wrappers on two `RaCFormer` methods
before the driver builds the model, so the model source, the configs and the checkpoint stay
exactly as they are on disk.

  --removal none     G1. True no-op. The wrappers are installed and counted, no tensor is touched.
                     G1 exists to prove the instrumentation itself costs nothing: its submission
                     must reproduce the clean E1 run.
  --removal phase0   G2. Feature-level mask, replicating
                     research/paper_goal_20260520/tools/phase0_sensor_baseline.py:87-89 --
                     `extract_feat` runs normally and its image-branch OUTPUTS are zeroed
                     (all FPN levels + the camera-BEV stack).
  --removal input    G3. Post-normalization input zeroing: the normalized image tensor is zeroed
                     at the entry of `extract_img_feat` (models/racformer.py:107), i.e. zero in the
                     network's own units.
  --removal table9   G4. Pre-normalization input zeroing: the raw image tensor is zeroed at the
                     entry of `extract_feat` (models/racformer.py:179), i.e. zero in sensor units
                     -- a black frame, which the frozen normalization at models/racformer.py:211-212
                     then maps to -mean/std.

G3 and G4 differ by exactly the normalization affine and by nothing else. See
research/robust_study/gate_b/MANIFEST.md for why that is the axis the paper leaves open, and for
the alternative reading that was rejected.

Runtime intervention attestation (written to <out-dir>/intervention_attestation.json, and a FAIL
exits non-zero so the calling job records validity=INVALID):

  (i)   branch-hit count == n_samples, cross-checked against the driver's own `n_total`;
  (ii)  covered view-frame count == n_samples x num_cams x num_frames computed from the config --
        partial coverage is INVALID, not a warning;
  (iii) paired pre/post tensor digests at the cell's own intervention stage for a fixed 3-sample
        probe set: bit-identical for G1, different for G2-G4.

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

REMOVALS = ("none", "phase0", "input", "table9")

# Which tensors the probe digests cover, per removal mode. Purely descriptive; the runner derives
# the actual digests from the tensors it intervenes on.
PROBE_STAGE = {
    "none": "extract_feat entry, raw image tensor (no-op probe)",
    "phase0": "extract_feat outputs: img_feats[level 0] and camera-BEV stack",
    "input": "extract_img_feat entry, normalized image tensor",
    "table9": "extract_feat entry, raw image tensor",
}

PROBE_SAMPLES = 3


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
        self.altered_view_frames = 0
        self.altered_bev_frames = 0
        self.probes = []
        self.shape_errors = []

        # Set by the extract_feat wrapper on every call so the extract_img_feat wrapper (which sees
        # a flattened B*NT batch) can recover B and NT. Guaranteed fresh: extract_img_feat is only
        # reached from inside extract_feat.
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

    def note_altered_views(self):
        self.altered_view_frames += (self.current_b or 0) * (self.current_nt or 0)

    def note_altered_bev(self, n_frames):
        self.altered_bev_frames += (self.current_b or 0) * int(n_frames)

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

        if self.removal == "none":
            if self.altered_view_frames != 0:
                failures.append(
                    "--removal none altered %d view-frames; it must alter none"
                    % self.altered_view_frames
                )
        elif self.altered_view_frames != expect_view_frames:
            failures.append(
                "altered view-frames %d != expected %d (partial coverage)"
                % (self.altered_view_frames, expect_view_frames)
            )

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
                if pre is None or post is None:
                    failures.append(
                        "probe call %d tensor %s is missing a pre or post digest"
                        % (probe["call_index"], name)
                    )
                    continue
                same = pre == post
                if self.removal == "none" and not same:
                    failures.append(
                        "probe call %d tensor %s changed under --removal none"
                        % (probe["call_index"], name)
                    )
                if self.removal != "none" and same:
                    failures.append(
                        "probe call %d tensor %s is unchanged under --removal %s"
                        % (probe["call_index"], name, self.removal)
                    )
                if self.removal != "none" and probe["post"][name]["abs_max"] != 0.0:
                    failures.append(
                        "probe call %d tensor %s is not zero after the intervention (abs_max=%r)"
                        % (probe["call_index"], name, probe["post"][name]["abs_max"])
                    )

        return {
            "schema": "gate_b_intervention_attestation/1",
            "removal": self.removal,
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
                "altered_view_frames": self.altered_view_frames,
                "altered_bev_frames": self.altered_bev_frames,
            },
            "probe_set": self.probes,
            "failures": failures,
            "verdict": "PASS" if not failures else "FAIL",
        }


def _zero_image_arg(img):
    """Zero the image argument, preserving whether it is a tensor or a list of tensors."""
    import torch

    if torch.is_tensor(img):
        return torch.zeros_like(img)
    return [torch.zeros_like(x) for x in img]


def _batch_and_view_frames(img):
    """(B, NT) for either layout, without allocating. See models/racformer.py:180-185."""
    import torch

    if torch.is_tensor(img):
        return int(img.shape[0]), int(img.shape[1])
    return len(img), int(img[0].shape[0])


def _first_tensor(img):
    import torch

    return img if torch.is_tensor(img) else img[0]


def install_intervention(removal, att):
    """Wrap the two RaCFormer methods that the eval path funnels through.

    Both wrappers are installed for every mode, including `none`: the counters and the probe set
    have to be produced by the same code in every cell, otherwise the attestation would not be
    evidence about the cells it is attesting.
    """
    import torch

    from models.racformer import RaCFormer

    original_extract_feat = RaCFormer.extract_feat
    original_extract_img_feat = RaCFormer.extract_img_feat

    def patched_extract_feat(self, img, radar_points, radar_depth, radar_rcs, img_metas):
        batch, n_view_frames = _batch_and_view_frames(img)
        att.on_extract_feat(batch, n_view_frames, int(self.num_cams))
        try:
            att.probe_identity(img_metas[0].get("filename"))
        except (AttributeError, IndexError, TypeError):
            pass

        if removal in ("none", "table9"):
            att.probe("pre", "raw_img", _first_tensor(img))
        if removal == "table9":
            img = _zero_image_arg(img)
            att.note_altered_views()
        if removal in ("none", "table9"):
            att.probe("post", "raw_img", _first_tensor(img))

        out = original_extract_feat(self, img, radar_points, radar_depth, radar_rcs, img_metas)

        if removal == "phase0":
            img_feats, bev_feats, radar_bev_feats, depth = out
            att.probe("pre", "img_feats_l0", img_feats[0])
            att.probe("pre", "bev_feats", bev_feats)
            # Same masking as phase0_sensor_baseline.py:87-89: image FPN levels and camera BEV.
            img_feats = [torch.zeros_like(feat) for feat in img_feats]
            bev_feats = torch.zeros_like(bev_feats)
            att.probe("post", "img_feats_l0", img_feats[0])
            att.probe("post", "bev_feats", bev_feats)
            att.note_altered_views()
            att.note_altered_bev(bev_feats.shape[1])
            out = (img_feats, bev_feats, radar_bev_feats, depth)

        return out

    def patched_extract_img_feat(self, img):
        if removal == "input":
            att.probe("pre", "normalized_img", img)
            img = torch.zeros_like(img)
            att.note_altered_views()
            att.probe("post", "normalized_img", img)
        return original_extract_img_feat(self, img)

    RaCFormer.extract_feat = patched_extract_feat
    RaCFormer.extract_img_feat = patched_extract_img_feat


def load_driver(driver_path):
    spec = importlib.util.spec_from_file_location("gate_b_eval_driver", driver_path)
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
    # The config does not set model.num_cams; the value in force is the ctor default. Read it from
    # the config when present so an override would be honoured, and cross-check the runtime
    # attribute on every forward pass.
    num_cams = int(cfg.model.get("num_cams", 6))
    return num_cams, num_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--removal", required=True, choices=REMOVALS)
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
        raise SystemExit("CUDA is not available; GATE-B never evaluates on CPU.")

    num_cams, num_frames = read_config_expectations(args.config)
    att = Attestation(args.removal, num_cams, num_frames, args.expect_samples)

    # Import the custom modules so the registry decorators run, then patch the class before the
    # driver's build_model() instantiates it. The driver re-imports `models` from the module cache.
    importlib.import_module("models")
    importlib.import_module("loaders")
    install_intervention(args.removal, att)

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
    print("[gate_b] removal=%s" % args.removal, flush=True)
    print("[gate_b] driver argv: %s" % " ".join(argv[1:]), flush=True)

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
    print("[gate_b] attestation: %s" % report_path, flush=True)
    print("[gate_b] verdict=%s" % report["verdict"], flush=True)
    for failure in report["failures"]:
        print("[gate_b] FAILURE: %s" % failure, file=sys.stderr, flush=True)

    if report["verdict"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
