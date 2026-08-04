#!/usr/bin/env python
"""Runner for the radar corruption cells (b), (c) and (d2), with runtime attestation.

NEW FILE. Same contract as research/robust_study/tools/a_removal_subset.py: the frozen driver
research/night_gen_phase1/eval_by_condition.py does all the work, this script only installs the
evidence collection around it and judges the result.

Why a runner is needed at all. The radar cells activate through a config pipeline swap, so the
frozen driver could be pointed straight at a cell config and would run the corruption. What it
would NOT produce is any evidence that the corruption ran: a cell config whose classes were
constructed but never invoked yields a submission indistinguishable from a clean one, and a
"robust model" is exactly what that looks like. This runner sets ROBUST_ATTEST_DIR, so every
corruption application writes evidence from inside the DataLoader worker that applied it, then
aggregates the workers' snapshots and FAILS the cell if the coverage or the realised magnitude
does not match what the config declares.

  exit 0  attestation PASS
  exit 3  attestation FAIL -- the calling job records validity=INVALID

The family, its parameters and the expected per-sample application count are all read from the
cell config, so the config file's md5 pins the whole cell.

CUDA is required. This script never runs inference on CPU.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

# How many corruption applications one sample must produce, per family. Both radar loader
# entries are swapped in every cell, so a sample passes through both:
#   (b)/(c): Loadnuradarpoints calls get_nu_radar once (loading.py:805) and
#            LoadradarpointsFromMultiSweeps calls it sweeps_num times (loading.py:857 in the
#            no-prev-sweeps branch, :891 otherwise), with sweeps_num = num_frames - 1.
#   (d2):    one application per loader __call__, i.e. exactly 2.
# The value is asserted, not assumed: a cell whose count differs FAILS rather than rescaling.
APPS_PER_SAMPLE = {
    "radar_dropout": lambda cfg: 1 + (int(cfg.num_frames) - 1),
    "radar_doppler_rcs_noise": lambda cfg: 1 + (int(cfg.num_frames) - 1),
    "d2_extrinsic": lambda cfg: 2,
    #   (d1):    one application per loader __call__, i.e. exactly 2 (the walk evidence for a
    #            sample's 5 channels x 8 elements rides on those 2 lines).
    "d1_async": lambda cfg: 2,
}

# Cell class name -> (family, parameter keys carried on the pipeline entry).
CELL_CLASSES = {
    "RadarDropoutLoadnuradarpoints": ("radar_dropout", ("drop_p", "corrupt_seed")),
    "RadarDropoutLoadradarpointsFromMultiSweeps": ("radar_dropout", ("drop_p", "corrupt_seed")),
    "RadarNoiseLoadnuradarpoints": ("radar_doppler_rcs_noise", ("sigma", "corrupt_seed")),
    "RadarNoiseLoadradarpointsFromMultiSweeps": ("radar_doppler_rcs_noise",
                                                 ("sigma", "corrupt_seed")),
    "D2MiscalibLoadnuradarpoints": ("d2_extrinsic", ("level", "seed")),
    "D2MiscalibLoadradarpointsFromMultiSweeps": ("d2_extrinsic", ("level", "seed")),
    "D1AsyncLoadnuradarpoints": ("d1_async", ("offset",)),
    "D1AsyncLoadradarpointsFromMultiSweeps": ("d1_async", ("offset",)),
}


def identify_cell(cfg, config_path):
    """Read the family and its parameters off the swapped pipeline entries."""
    entries = [e for e in cfg.data.val.pipeline if e.get("type") in CELL_CLASSES]
    if len(entries) != 2:
        raise SystemExit(
            "config %s has %d corruption pipeline entries, expected exactly 2; refusing to run, "
            "because a partially swapped pipeline is a partial fault and a fully unswapped one "
            "is a clean eval wearing a cell name." % (config_path, len(entries)))
    families = set(CELL_CLASSES[e["type"]][0] for e in entries)
    if len(families) != 1:
        raise SystemExit("config %s mixes corruption families %s" % (config_path, sorted(families)))
    family = families.pop()

    params = {}
    for entry in entries:
        for key in CELL_CLASSES[entry["type"]][1]:
            if key in entry:
                if key in params and params[key] != entry[key]:
                    raise SystemExit(
                        "config %s gives %s=%r on one radar entry and %r on the other; both "
                        "entries must carry the same parameters or the fault is inconsistent "
                        "between the frame-t radar and the sweep stack"
                        % (config_path, key, params[key], entry[key]))
                params[key] = entry[key]
    return family, params


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True, help="a (b)/(c)/(d2) cell config")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expect-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--probe-samples", type=int, default=3,
                        help="how many val sample tokens get full per-call detail recorded")
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; the robustness cells never evaluate on CPU.")

    os.makedirs(args.out_dir, exist_ok=True)
    attest_dir = os.path.join(args.out_dir, "attestation_evidence")
    if os.path.isdir(attest_dir) and os.listdir(attest_dir):
        raise SystemExit("evidence dir %s already has contents; refusing to mix runs" % attest_dir)
    os.makedirs(attest_dir, exist_ok=True)

    # MUST be set before the corruption module is imported anywhere: the sink samples the
    # environment once, at construction, and every DataLoader worker inherits it by fork.
    os.environ["ROBUST_ATTEST_DIR"] = attest_dir

    from mmcv import Config

    cfg = Config.fromfile(args.config)
    family, params = identify_cell(cfg, args.config)
    if family not in APPS_PER_SAMPLE:
        raise SystemExit("no attestation is registered for family %r" % family)
    apps_per_sample = APPS_PER_SAMPLE[family](cfg)

    from research.robust_study.corruptions import attest

    # Pin the probe set to the FIRST n val sample tokens in dataset order, so the detail rows
    # are comparable across cells and across seeds rather than being whichever samples a worker
    # happened to reach first.
    importlib.import_module("models")
    importlib.import_module("loaders")
    from mmdet3d.datasets import build_dataset

    # data_infos is the dataset's own index order and `sample_idx` IS info['token']
    # (loaders/nuscenes_dataset.py:181-192), which is the key the corruption records under.
    probe_dataset = build_dataset(cfg.data.val)
    probe_tokens = [probe_dataset.data_infos[i]["token"]
                    for i in range(min(args.probe_samples, len(probe_dataset)))]
    # Not via the environment: the sink sampled it at import time, which already happened when
    # the cell config was parsed. Workers are forked after this point and inherit the value.
    attest.SINK.set_probe_tokens(probe_tokens)
    del probe_dataset

    attestation = attest.ATTESTATIONS[family](
        cell=os.path.splitext(os.path.basename(args.config))[0],
        params=params,
        expect_samples=args.expect_samples,
        expect_apps_per_sample=apps_per_sample,
    )

    driver_path = os.path.join(repo, "research/night_gen_phase1/eval_by_condition.py")
    import importlib.util

    spec = importlib.util.spec_from_file_location("radar_cell_eval_driver", driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    argv = [
        driver_path,
        "--config", args.config,
        "--weights", args.weights,
        "--out-dir", args.out_dir,
        "--full-val",
        "--batch-size", str(args.batch_size),
    ]
    print("[radar_cell] family=%s params=%s apps/sample=%d"
          % (family, json.dumps(params, sort_keys=True), apps_per_sample), flush=True)
    print("[radar_cell] evidence dir: %s" % attest_dir, flush=True)
    print("[radar_cell] driver argv: %s" % " ".join(argv[1:]), flush=True)

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

    attestation.load(attest_dir)
    report = attestation.report()
    if driver_n_total is not None and driver_n_total != args.expect_samples:
        report["failures"].append(
            "driver n_total %s != expected n_samples %d" % (driver_n_total, args.expect_samples))
        report["verdict"] = "FAIL"
    report["observed"]["driver_n_total"] = driver_n_total
    report["elapsed_s"] = round(time.time() - started, 3)
    report["config"] = os.path.abspath(args.config)
    report["weights"] = os.path.abspath(args.weights)
    report["driver"] = driver_path
    report["probe_tokens"] = probe_tokens

    report_path = os.path.join(args.out_dir, "intervention_attestation.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print("[radar_cell] attestation: %s" % report_path, flush=True)
    print("[radar_cell] verdict=%s" % report["verdict"], flush=True)
    for failure in report["failures"]:
        print("[radar_cell] FAILURE: %s" % failure, file=sys.stderr, flush=True)

    if report["verdict"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
