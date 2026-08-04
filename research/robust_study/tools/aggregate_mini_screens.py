#!/usr/bin/env python3
"""Merge every mini-screen shard + the reconstructed 1807 rows into one coverage report.

Later shards win over the reconstructed 1807 rows for a given cell (a re-run supersedes the
partial). Prints coverage against the 29-cell roster and any cell still missing or rejected.
"""
import glob
import json
import os

BP = "/srv/nfs/shared/gnmp/robust_study_runs/batch2_prep"
ROSTER = (
    ["a_removal_front", "a_removal_back"]
    + ["radar_dropout_p%d_s%d" % (p, s) for p in (25, 50, 75) for s in (0, 1, 2)]
    + ["radar_noise_sig%d_s%d" % (g, s) for g in (1, 3, 5) for s in (0, 1, 2)]
    + ["d2_extrinsic_%s_seed%d" % (l, s) for l in ("medium", "severe") for s in (0, 1, 2)]
    + ["d1_async_offset%d" % k for k in (1, 2, 3)]
)

merged = {}   # cell -> (row, source)
sources = []

# reconstructed 1807 first (lowest priority)
recon = os.path.join(BP, "mini_screens_20260804T020315Z_livenode03",
                     "screen_summary_reconstructed.json")
if os.path.isfile(recon):
    d = json.load(open(recon))
    sources.append(("1807-reconstructed", recon))
    for cell, row in d["cells"].items():
        merged[cell] = (row, "1807-reconstructed")

# every shard's real summary, in mtime order (later wins)
summaries = sorted(glob.glob(os.path.join(BP, "mini_screens_*", "screen_summary.json")),
                   key=os.path.getmtime)
for path in summaries:
    d = json.load(open(path))
    tag = os.path.basename(os.path.dirname(path))
    sources.append((tag, path))
    for cell, row in d.get("cells", {}).items():
        merged[cell] = (row, tag)

def verdict(row):
    att = row.get("attestation")
    att_v = att.get("verdict") if isinstance(att, dict) else att
    m = row.get("metrics") or {}
    if att_v != "PASS":
        return "REJECT(att=%s)" % att_v
    if m.get("n_total") != 300:
        return "REJECT(n=%s)" % m.get("n_total")
    return "PASS"

print("=== mini-screen coverage (29-cell roster) ===")
missing, rejected, passed = [], [], []
for cell in ROSTER:
    if cell not in merged:
        missing.append(cell)
        print("%-24s MISSING" % cell)
        continue
    row, src = merged[cell]
    v = verdict(row)
    m = row.get("metrics") or {}
    dnds = row.get("delta_vs_clean_NDS")
    print("%-24s %-14s NDS=%s dNDS=%s  [%s]" % (
        cell, v,
        ("%.4f" % m["NDS"]) if "NDS" in m else "?",
        ("%+.4f" % dnds) if isinstance(dnds, float) else "?",
        src.replace("mini_screens_", "")))
    (rejected if v.startswith("REJECT") else passed).append(cell)

print("\n=== summary ===")
print("roster: %d  passed: %d  rejected: %d  missing: %d"
      % (len(ROSTER), len(passed), len(rejected), len(missing)))
if rejected:
    print("REJECTED:", rejected)
if missing:
    print("MISSING:", missing)
out = os.path.join(BP, "mini_screen_coverage_merged.json")
json.dump({"roster": ROSTER, "passed": passed, "rejected": rejected, "missing": missing,
           "sources": sources,
           "cells": {c: {"verdict": verdict(r), "source": s, "metrics": r.get("metrics"),
                         "delta_vs_clean_NDS": r.get("delta_vs_clean_NDS")}
                     for c, (r, s) in merged.items()}},
          open(out, "w"), indent=2, sort_keys=True)
print("wrote", out)
