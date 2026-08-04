#!/usr/bin/env python3
"""Build a phase1_t10_seed20260425_ratio<N>_manifest.json for an arbitrary N.

Selection rule (identical to build_ratio12p5_manifest.py):
  1. Load phase1_t10_seed20260425_accepted.json (486 sample-complete tokens).
  2. Concatenate phase1_t10_seed20260425_input_chunk_{1..5}.json's `samples`
     in chunk order, dedup-by-sample_token (yields 500 ordered tokens).
  3. Filter input order to tokens present in `accepted` (yields 486 ordered tokens).
  4. Take the first --n-target-samples tokens (must be <= 486).
  5. Filter accepted entries to those tokens; verify count = N * 6 and 6-cam coverage.
  6. Verify every entry's generated_path exists on disk.

Output dict shape mirrors phase1_t10_seed20260425_ratio12p5_manifest.json so
LoadMultiViewImageFromManifest can consume it without changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

REPO = "/srv/nfs/shared/gnmp/RaCFormer"
ACCEPTED = os.path.join(
    REPO, "research/night_gen_phase1/manifests/phase1_t10_seed20260425_accepted.json"
)
INPUT_CHUNKS_DIR = "/srv/nfs/shared/gnmp/t10_gen/manifests"
INPUT_CHUNK_TPL = "phase1_t10_seed20260425_input_chunk_{i}.json"


def load_input_order() -> tuple[list[str] | None, str]:
    order: list[str] = []
    seen: set[str] = set()
    for i in range(1, 6):
        path = os.path.join(INPUT_CHUNKS_DIR, INPUT_CHUNK_TPL.format(i=i))
        if not os.path.isfile(path):
            return None, f"missing chunk {i}: {path}"
        with open(path) as fh:
            chunk = json.load(fh)
        samples = chunk.get("samples")
        if not isinstance(samples, list) or not samples:
            return None, f"chunk {i} has no samples key"
        for s in samples:
            t = s.get("sample_token")
            if not t or t in seen:
                continue
            seen.add(t)
            order.append(t)
    if len(order) != 500:
        return None, f"expected 500 unique input tokens, got {len(order)}"
    return order, "input_chunks_1_to_5_in_order"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-target-samples", type=int, required=True)
    parser.add_argument("--ratio-label", required=True, help='e.g. "18.75%"')
    parser.add_argument(
        "--ratio-basis",
        required=True,
        help='e.g. "375 samples / 2000 day-only training subset"',
    )
    parser.add_argument("--out", required=True, help="absolute manifest output path")
    args = parser.parse_args()

    n_target = args.n_target_samples
    if not (1 <= n_target <= 486):
        print(f"ERROR: --n-target-samples must be in [1, 486]; got {n_target}")
        return 1

    with open(ACCEPTED) as fh:
        acc = json.load(fh)
    entries = acc["entries"]
    accepted_tokens = {e["sample_token"] for e in entries}
    if len(accepted_tokens) != 486:
        print(
            f"ERROR: accepted manifest has {len(accepted_tokens)} unique tokens, expected 486"
        )
        return 2
    print(f"accepted: {len(entries)} entries, {len(accepted_tokens)} unique tokens")

    input_order, label = load_input_order()
    if input_order is None:
        selection_rule = "lexicographic_sample_token_fallback"
        ordered = sorted(accepted_tokens)
        print(f"WARN: input order unavailable ({label}); fallback to lexicographic")
    else:
        selection_rule = f"input_chunk_order_first_{n_target}_sample_complete"
        ordered = [t for t in input_order if t in accepted_tokens]
        print(f"input order recovered ({label}); {len(ordered)} of 486 in order")
        if len(ordered) != 486:
            print(
                f"ERROR: input-order intersection size {len(ordered)} != 486 (mismatch)"
            )
            return 3

    selected = ordered[:n_target]
    if len(selected) != n_target:
        print(f"ERROR: selected {len(selected)}, expected {n_target}")
        return 4
    selected_set = set(selected)

    filtered = [e for e in entries if e["sample_token"] in selected_set]
    expected_entries = n_target * 6
    print(f"filtered entries: {len(filtered)} (expected {expected_entries})")
    if len(filtered) != expected_entries:
        print("ERROR: entry count mismatch")
        return 5

    per_sample_cams = Counter()
    for e in filtered:
        per_sample_cams[e["sample_token"]] += 1
    bad = {t: c for t, c in per_sample_cams.items() if c != 6}
    if bad:
        print(f"ERROR: {len(bad)} samples not 6-cam complete: {list(bad.items())[:5]}")
        return 6

    missing: list[str] = []
    for e in filtered:
        gp = os.path.join(REPO, e["generated_path"])
        if not os.path.isfile(gp):
            missing.append(gp)
    if missing:
        print(f"ERROR: {len(missing)} missing generated_path files; first 5:")
        for m in missing[:5]:
            print(" ", m)
        return 7
    print(f"all {len(filtered)} generated_path files exist on disk")

    out = {
        "generator": acc["generator"],
        "model": acc["model"],
        "image_size_tier": acc["image_size_tier"],
        "prompt_sha256": acc["prompt_sha256"],
        "generated_at": acc["generated_at"],
        "source_manifest": acc["source_manifest"],
        "merged_from_chunks": acc["merged_from_chunks"],
        "seed": acc["seed"],
        "__derived_from": (
            "research/night_gen_phase1/manifests/phase1_t10_seed20260425_accepted.json"
        ),
        "__selection_rule": selection_rule,
        "__selection_source": label,
        "__derived_at": datetime.now(timezone.utc).isoformat(),
        "ratio_target": args.ratio_label,
        "ratio_basis": args.ratio_basis,
        "n_accepted_samples_pool": 486,
        "n_accepted_entries_pool": len(entries),
        "n_selected_samples": n_target,
        "n_selected_entries": len(filtered),
        "__cluster_rewritten_to": acc["__cluster_rewritten_to"],
        "__path_rewritten_to_repo_relative": acc.get(
            "__path_rewritten_to_repo_relative", True
        ),
        "n_accepted_entries": len(filtered),
        "n_accepted_samples": n_target,
        "entries": filtered,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, args.out)
    print(f"\nwrote {args.out}")
    print(f"  selection_rule = {selection_rule}")
    print(f"  n_selected_samples = {n_target}")
    print(f"  n_selected_entries = {len(filtered)}")
    print(f"  first 5 selected tokens (in selection order):")
    for t in selected[:5]:
        print(f"    {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
