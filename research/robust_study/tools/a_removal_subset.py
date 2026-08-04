#!/usr/bin/env python
"""(a)-family camera-SUBSET removal runner — the GATE-B G4 cell restricted to one camera.

NEW FILE. It is the GATE-B runner with one thing changed: which view-frames get the black
frame. Everything that decides what a "removed camera" IS -- the stage, the tensor, the units,
the substituted values -- is imported from research/robust_study/tools/gate_b_removal.py and
executed there, not restated here:

  * the substitution itself is `gate_b_removal._zero_image_arg` (:246-252), the function the
    `--removal table9` branch calls at :295;
  * the driver load is `gate_b_removal.load_driver` (:329-333);
  * the config expectations are `gate_b_removal.read_config_expectations` (:336-351);
  * the attestation counters, shape cross-checks and digest function are
    `gate_b_removal.Attestation` (:85-243), subclassed for subset arithmetic in
    research/robust_study/corruptions/cam_subset_removal.py.

The ladder this cell sits on (plan.md 16.4): frontal-camera drop (A1, CAM_FRONT) <
worst-sector single-camera drop (A2, CAM_BACK) < all-6 drop. The all-6 endpoint is the
EXISTING GATE-B G4 + g4_repeat pair and is NOT re-run by this script.

Camera scope comes from the cell config's `cam_removal` block, not from a flag, so a cell is
one file and its md5 pins the camera set. Running the frozen driver on such a config directly
would silently produce a CLEAN run -- this script is the only thing that installs the fault --
so the config is refused if it carries no `cam_removal` block.

Runtime intervention attestation (written to <out-dir>/intervention_attestation.json; FAIL
exits non-zero so the calling job records validity=INVALID):

  (i)   branch-hit count == n_samples, cross-checked against the driver's own `n_total`;
  (ii)  covered view-frames == n_samples x num_cams x num_frames (unchanged from GATE-B), AND
        altered view-frames == n_samples x |cameras| x num_frames;
  (iii) every one of the 6 channels seen exactly n_samples x num_frames times -- a loader
        layout change cannot silently degrade the cell into partial coverage;
  (iv)  per-camera paired pre/post digests on a fixed 3-sample probe set: the target camera(s)
        changed and are exactly zero, the other cameras bit-identical.

CUDA is required. This script never runs inference on CPU.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True,
                        help="a cell config carrying a `cam_removal` block")
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
        raise SystemExit("CUDA is not available; the (a)-family never evaluates on CPU.")

    from research.robust_study.corruptions import cam_subset_removal

    gate_b = cam_subset_removal.gate_b

    from mmcv import Config

    cfg = Config.fromfile(args.config)
    cam_removal = cfg.get("cam_removal", None)
    if not cam_removal or not cam_removal.get("cameras"):
        raise SystemExit(
            "config %s carries no `cam_removal.cameras`; refusing to run, because without it "
            "this would be a clean eval wearing an (a)-cell name." % args.config)
    cameras = tuple(cam_removal["cameras"])
    cell = cam_removal.get("cell", "?")

    num_cams, num_frames = gate_b.read_config_expectations(args.config)
    if len(cameras) >= num_cams:
        raise SystemExit(
            "cell %s names %d of %d cameras; the all-6 endpoint is the existing GATE-B G4 run "
            "and must not be re-run here (plan.md 16.4)." % (cell, len(cameras), num_cams))

    att = cam_subset_removal.SubsetAttestation(
        cell, cameras, num_cams, num_frames, args.expect_samples)

    # Import the custom modules so the registry decorators run, then patch the class before the
    # driver's build_model() instantiates it. The driver re-imports `models` from the cache.
    importlib.import_module("models")
    importlib.import_module("loaders")
    cam_subset_removal.install_subset_intervention(cameras, att)

    driver_path = os.path.join(repo, "research/night_gen_phase1/eval_by_condition.py")
    driver = gate_b.load_driver(driver_path)

    os.makedirs(args.out_dir, exist_ok=True)
    argv = [
        driver_path,
        "--config", args.config,
        "--weights", args.weights,
        "--out-dir", args.out_dir,
        "--full-val",
        "--batch-size", str(args.batch_size),
    ]
    print("[a_removal] cell=%s cameras=%s" % (cell, ",".join(cameras)), flush=True)
    print("[a_removal] driver argv: %s" % " ".join(argv[1:]), flush=True)

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
    report["gate_b_runner"] = cam_subset_removal.GATE_B_RUNNER

    report_path = os.path.join(args.out_dir, "intervention_attestation.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print("[a_removal] attestation: %s" % report_path, flush=True)
    print("[a_removal] verdict=%s" % report["verdict"], flush=True)
    for failure in report["failures"]:
        print("[a_removal] FAILURE: %s" % failure, file=sys.stderr, flush=True)

    if report["verdict"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
