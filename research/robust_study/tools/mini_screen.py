#!/usr/bin/env python
"""Mini-screen orchestrator for the Batch-2 cells (fault-families.md, cross-family rule 3).

NEW FILE. Runs every requested cell on the FIRST N val samples (default 300) through the
cell's own runner in --screen-samples mode, plus one clean baseline through the frozen driver,
and writes one screen_summary.json. A screen is a screen: nothing here is a result cell, no
run directory produced here may ever be promoted, and the out-root lives under batch2_prep.

What a screen rejects is RECORDED, not silently dropped (rule 3): a cell is REJECTED when its
runner exits non-zero, its intervention attestation is not PASS, its driver output is missing,
its n_total differs from the screen size, or a headline metric is non-finite. Metric deltas
against the clean baseline are recorded for inspection but are NOT rejection criteria -- no
numeric budget for "how much a corruption may move mAP on 300 samples" is registered anywhere,
so none is applied.

Also verifies, at mechanism level, the (b)-family CRN nesting claim the MANIFEST flags as
"a claim the mini-screen should verify, not assume": same (family, seed, identity) key =>
same u vector from derive_rng, and keep = (u >= p) nests across p. This exercises the real
derive_rng and the real keep rule (radar_noise.py corrupt), not the full loader path; it is
recorded as such.

Modes:
  (default)        orchestrate: clean baseline + all cells, sequentially, on this node.
  --clean-driver   internal: run ONE capped clean eval through the frozen driver. The driver's
                   capped-val guard is re-aimed (cap must equal the screen size), never removed.

CUDA is required. This script never runs inference on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

REPO_DEFAULT = "/srv/nfs/shared/gnmp/RaCFormer"
BASE_CONFIG_REL = "configs/racformer_eval_fullval_research.py"
WEIGHTS_REL = "checkpoints/racformer_r50_f8.pth"
CELL_CONFIG_DIR_REL = "research/robust_study/configs"
TOOLS_REL = "research/robust_study/tools"

ROSTER = (
    ["a_removal_front", "a_removal_back"]
    + ["radar_dropout_p%d_s%d" % (p, s) for p in (25, 50, 75) for s in (0, 1, 2)]
    + ["radar_noise_sig%d_s%d" % (g, s) for g in (1, 3, 5) for s in (0, 1, 2)]
    + ["d2_extrinsic_%s_seed%d" % (lvl, s) for lvl in ("medium", "severe") for s in (0, 1, 2)]
    + ["d1_async_offset%d" % k for k in (1, 2, 3)]
)
assert len(ROSTER) == 29, "cell roster drifted: %d != 29" % len(ROSTER)

PER_CELL_TIMEOUT_S = 1800  # a mini cell has a ~95 s base rate; 30 min means "wedged"


def md5_file(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runner_for(cell):
    if cell.startswith("a_removal_"):
        return os.path.join(TOOLS_REL, "a_removal_subset.py")
    if cell.startswith(("radar_dropout_", "radar_noise_", "d2_extrinsic_", "d1_async_")):
        return os.path.join(TOOLS_REL, "radar_cell_runner.py")
    raise SystemExit("no runner is registered for cell %r" % cell)


def write_mini_config(cfg_dir, name, base_config_abs, n):
    """One derived config per screened cell: _base_ = the committed config, plus the cap.

    The cap is carried by a config FILE (not an in-memory mutation) so the runner's
    config-md5 provenance pins the screen, and the base's md5 is recorded alongside.
    """
    path = os.path.join(cfg_dir, "%s__mini%d.py" % (name, n))
    with open(path, "w") as handle:
        handle.write(
            '# GENERATED mini-screen config -- screens only, never a result cell.\n'
            '_base_ = ["%s"]\n'
            'data = dict(val=dict(max_samples=%d))\n' % (base_config_abs, n))
    return path


def headline_metrics(driver_json_path):
    """Extract n_total plus every */mAP and */NDS key from eval_by_condition.json."""
    if not os.path.isfile(driver_json_path):
        return None
    with open(driver_json_path) as handle:
        data = json.load(handle)
    out = {"n_total": data.get("n_total")}
    for key, value in (data.get("overall") or {}).items():
        if key.endswith("/mAP") or key.endswith("/NDS"):
            out[key.rsplit("/", 1)[1]] = value
    return out


def submission_stats(out_dir):
    path = os.path.join(out_dir, "submission_overall", "pts_bbox", "results_nusc.json")
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        results = json.load(handle).get("results", {})
    return {"n_tokens": len(results),
            "total_boxes": sum(len(boxes) for boxes in results.values())}


def crn_nesting_probe(repo):
    """Mechanism-level check of the (b) CRN nesting claim (MANIFEST (b), Seeds/CRN row).

    Same (family, seed, identity) => derive_rng returns the same u vector, and
    keep = (u >= p) makes the p=0.25 kept set a superset of p=0.50's, superset of p=0.75's
    (the keep rule at radar_noise.py corrupt). Uses the real derive_rng on synthetic
    identities; the full loader path is exercised by the screens themselves.
    """
    spec = importlib.util.spec_from_file_location(
        "mini_screen_radar_noise",
        os.path.join(repo, "research/robust_study/corruptions/radar_noise.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import numpy as np

    report = {"checked_identities": 0, "failures": []}
    for seed in (0, 1, 2):
        for identity in ("tokA|n6|f1|FRONT=x", "tokB|n6|f1|FRONT=y,LEFT=z"):
            n = 4096
            u_first = mod.derive_rng("radar_dropout", seed, identity).random(n)
            u_again = mod.derive_rng("radar_dropout", seed, identity).random(n)
            if not np.array_equal(u_first, u_again):
                report["failures"].append(
                    "derive_rng is not deterministic for seed=%d identity=%r" % (seed, identity))
                continue
            keep = {p: set(np.nonzero(u_first >= p)[0].tolist()) for p in (0.25, 0.50, 0.75)}
            if not (keep[0.50] <= keep[0.25] and keep[0.75] <= keep[0.50]):
                report["failures"].append(
                    "kept sets do not nest for seed=%d identity=%r" % (seed, identity))
            report["checked_identities"] += 1
    report["verdict"] = "PASS" if not report["failures"] else "FAIL"
    return report


def run_clean_driver(args):
    """Internal mode: one capped clean eval through the frozen driver."""
    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; the mini-screen never evaluates on CPU.")

    importlib.import_module("models")
    importlib.import_module("loaders")

    driver_path = os.path.join(repo, "research/night_gen_phase1/eval_by_condition.py")
    spec = importlib.util.spec_from_file_location("mini_screen_clean_driver", driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    def _screen_cap_guard(cfg_, full_val_flag, _n=args.samples):
        cap = getattr(cfg_.data.val, "max_samples", None)
        if cap != _n:
            raise SystemExit(
                "[mini_screen] screen guard: cfg.data.val.max_samples=%r != declared screen "
                "size %d" % (cap, _n))
    driver._abort_if_capped_val = _screen_cap_guard

    argv = [driver_path,
            "--config", args.config,
            "--weights", args.weights,
            "--out-dir", args.out_dir,
            "--batch-size", "1"]
    print("[mini_screen] clean driver argv: %s" % " ".join(argv[1:]), flush=True)
    saved = sys.argv
    try:
        sys.argv = argv
        driver.main()
    finally:
        sys.argv = saved


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=REPO_DEFAULT)
    parser.add_argument("--out-root", help="screen tree, under batch2_prep (orchestrate mode)")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--cells", default="all",
                        help='comma-separated cell names, or "all" (29 cells)')
    parser.add_argument("--skip-clean", action="store_true",
                        help="skip the clean baseline (when another shard on the SAME node "
                             "already ran it)")
    parser.add_argument("--clean-driver", action="store_true",
                        help="internal: run one capped clean eval (needs --config/--out-dir)")
    parser.add_argument("--config", help="clean-driver mode only")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--out-dir", help="clean-driver mode only")
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    if args.weights is None:
        args.weights = os.path.join(repo, WEIGHTS_REL)

    if args.clean_driver:
        if not (args.config and args.out_dir):
            raise SystemExit("--clean-driver needs --config and --out-dir")
        run_clean_driver(args)
        return

    if not args.out_root:
        raise SystemExit("orchestrate mode needs --out-root")
    # radar_noise (nesting probe) imports `loaders`, and the loader resolves data/nuscenes/
    # relative to the cwd (loaders/nuscenes_dataset.py:21) -- same setup the runners do.
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)
    if "batch2_prep" not in os.path.abspath(args.out_root):
        raise SystemExit("screens live under batch2_prep; refusing out-root %r" % args.out_root)

    cells = ROSTER if args.cells == "all" else [c for c in args.cells.split(",") if c]
    unknown = [c for c in cells if c not in ROSTER]
    if unknown:
        raise SystemExit("cells not in the registered roster: %s" % ", ".join(unknown))

    os.makedirs(args.out_root, exist_ok=True)
    cfg_dir = os.path.join(args.out_root, "mini_configs")
    os.makedirs(cfg_dir, exist_ok=True)

    def git(*words):
        return subprocess.run(["git", "-C", repo] + list(words), capture_output=True,
                              text=True).stdout.strip()

    summary = {
        "screen": "batch2 mini-screen (fault-families.md cross-family rule 3)",
        "not_a_result": "screens are screens; no directory here may be promoted to a cell",
        "samples": args.samples,
        "node": os.uname().nodename,
        "repo_head": git("rev-parse", "HEAD"),
        "repo_dirty_paths": git("status", "--porcelain").splitlines(),
        "weights": args.weights,
        "weights_md5": md5_file(args.weights),
        "cells_requested": list(cells),
        "crn_nesting_probe": crn_nesting_probe(repo),
        "clean": None,
        "cells": {},
        "rejected": [],
    }

    def record_reject(name, reason):
        summary["rejected"].append({"cell": name, "reason": reason})
        print("[mini_screen] REJECT %s: %s" % (name, reason), flush=True)

    def run_one(name, cmd, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "screen.log")
        started = time.time()
        try:
            with open(log_path, "w") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      cwd=repo, timeout=PER_CELL_TIMEOUT_S)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout>%ds" % PER_CELL_TIMEOUT_S
        return rc, round(time.time() - started, 1), log_path

    # Clean baseline first: the reference the cell metrics are inspected against.
    if not args.skip_clean:
        clean_cfg = write_mini_config(cfg_dir, "clean_baseline",
                                      os.path.join(repo, BASE_CONFIG_REL), args.samples)
        clean_dir = os.path.join(args.out_root, "clean_baseline")
        rc, elapsed, _ = run_one("clean_baseline", [
            sys.executable, "-u", os.path.abspath(__file__), "--clean-driver",
            "--repo", repo, "--samples", str(args.samples),
            "--config", clean_cfg, "--weights", args.weights, "--out-dir", clean_dir],
            clean_dir)
        metrics = headline_metrics(os.path.join(clean_dir, "eval_by_condition.json"))
        summary["clean"] = {"rc": rc, "elapsed_s": elapsed, "metrics": metrics,
                            "submission": submission_stats(clean_dir),
                            "config_md5": md5_file(clean_cfg)}
        if rc != 0:
            record_reject("clean_baseline", "driver exited %s" % rc)
        elif not metrics or metrics.get("n_total") != args.samples:
            record_reject("clean_baseline",
                          "driver n_total %r != %d" % ((metrics or {}).get("n_total"),
                                                       args.samples))
        else:
            bad = [k for k, v in metrics.items()
                   if isinstance(v, float) and not math.isfinite(v)]
            if bad:
                record_reject("clean_baseline", "non-finite metrics: %s" % ", ".join(bad))

    clean_metrics = (summary["clean"] or {}).get("metrics") or {}

    for name in cells:
        base_cfg = os.path.join(repo, CELL_CONFIG_DIR_REL, name + ".py")
        if not os.path.isfile(base_cfg):
            record_reject(name, "cell config missing: %s" % base_cfg)
            continue
        mini_cfg = write_mini_config(cfg_dir, name, base_cfg, args.samples)
        out_dir = os.path.join(args.out_root, name)
        runner = os.path.join(repo, runner_for(name))
        rc, elapsed, _ = run_one(name, [
            sys.executable, "-u", runner,
            "--repo", repo, "--config", mini_cfg, "--weights", args.weights,
            "--out-dir", out_dir,
            "--expect-samples", str(args.samples),
            "--screen-samples", str(args.samples)],
            out_dir)

        attest_path = os.path.join(out_dir, "intervention_attestation.json")
        attest = None
        if os.path.isfile(attest_path):
            with open(attest_path) as handle:
                report = json.load(handle)
            attest = {"verdict": report.get("verdict"), "failures": report.get("failures")}
        metrics = headline_metrics(os.path.join(out_dir, "eval_by_condition.json"))

        row = {"rc": rc, "elapsed_s": elapsed, "attestation": attest, "metrics": metrics,
               "submission": submission_stats(out_dir),
               "config_md5": md5_file(mini_cfg), "base_config_md5": md5_file(base_cfg)}
        for key in ("mAP", "NDS"):
            if metrics and key in metrics and key in clean_metrics:
                row["delta_vs_clean_" + key] = metrics[key] - clean_metrics[key]
        summary["cells"][name] = row

        if rc != 0:
            record_reject(name, "runner exited %s" % rc)
        elif attest is None:
            record_reject(name, "no intervention_attestation.json written")
        elif attest["verdict"] != "PASS":
            record_reject(name, "attestation %s: %s" % (attest["verdict"], attest["failures"]))
        elif not metrics or metrics.get("n_total") != args.samples:
            record_reject(name, "driver n_total %r != %d"
                          % ((metrics or {}).get("n_total"), args.samples))
        else:
            bad = [k for k, v in metrics.items()
                   if isinstance(v, float) and not math.isfinite(v)]
            if bad:
                record_reject(name, "non-finite metrics: %s" % ", ".join(bad))

    if summary["crn_nesting_probe"]["verdict"] != "PASS":
        record_reject("crn_nesting_probe",
                      str(summary["crn_nesting_probe"]["failures"]))

    summary_path = os.path.join(args.out_root, "screen_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("[mini_screen] === summary ===", flush=True)
    clean_row = summary.get("clean") or {}
    print("[mini_screen] clean: rc=%s metrics=%s" % (clean_row.get("rc"),
                                                     clean_row.get("metrics")), flush=True)
    for name, row in summary["cells"].items():
        print("[mini_screen] %-28s rc=%-3s attest=%-4s mAP=%s NDS=%s dNDS=%s" % (
            name, row["rc"],
            (row["attestation"] or {}).get("verdict"),
            (row["metrics"] or {}).get("mAP"), (row["metrics"] or {}).get("NDS"),
            row.get("delta_vs_clean_NDS")), flush=True)
    print("[mini_screen] rejected: %d -- %s" % (
        len(summary["rejected"]), [r["cell"] for r in summary["rejected"]]), flush=True)
    print("[mini_screen] summary: %s" % summary_path, flush=True)

    if summary["rejected"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
