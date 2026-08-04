#!/usr/bin/env python3
"""Build phase1_t10_seed20260425_ratio12p5_manifest.json.

Selection rule: take the first 250 sample-complete tokens in the original
seed-20260425 input/merged manifest order (chunk_1 .. chunk_5 as written
locally and replicated to the cluster). Falls back to lexicographic
sample_token order only if the chunked input order cannot be recovered.

Output: dict-shaped manifest with the same `entries` list-of-dicts schema as
phase1_t10_seed20260425_accepted.json so the existing
LoadMultiViewImageFromManifest pipeline can consume it without changes.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

REPO = "/srv/nfs/shared/gnmp/RaCFormer"
ACCEPTED = os.path.join(
    REPO, "research/night_gen_phase1/manifests/phase1_t10_seed20260425_accepted.json"
)
OUT = os.path.join(
    REPO,
    "research/night_gen_phase1/manifests/phase1_t10_seed20260425_ratio12p5_manifest.json",
)
INPUT_CHUNKS_DIR = "/srv/nfs/shared/gnmp/t10_gen/manifests"
INPUT_CHUNK_TPL = "phase1_t10_seed20260425_input_chunk_{i}.json"
N_TARGET_SAMPLES = 250


def load_input_order() -> tuple[list[str] | None, str]:
    """Return (ordered_token_list, source_label) or (None, reason)."""
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
    with open(ACCEPTED) as fh:
        acc = json.load(fh)
    entries = acc["entries"]
    accepted_tokens = {e["sample_token"] for e in entries}
    if len(accepted_tokens) != 486:
        print(
            f"ERROR: accepted manifest has {len(accepted_tokens)} unique tokens, expected 486"
        )
        return 2
    print(
        f"accepted manifest: {len(entries)} entries, {len(accepted_tokens)} unique tokens"
    )

    input_order, label = load_input_order()
    if input_order is None:
        # fallback: lexicographic sample_token
        selection_rule = "lexicographic_sample_token_fallback"
        ordered = sorted(accepted_tokens)
        print(f"WARN: input order unavailable ({label}); falling back to lexicographic")
    else:
        selection_rule = "input_chunk_order_first_250_sample_complete"
        ordered = [t for t in input_order if t in accepted_tokens]
        print(
            f"input order recovered ({label}); {len(ordered)} of 486 sample-complete tokens in input order"
        )
        if len(ordered) != 486:
            print(
                f"ERROR: input-order intersection size {len(ordered)} != 486 — input/accepted mismatch"
            )
            return 3

    selected = ordered[:N_TARGET_SAMPLES]
    if len(selected) != N_TARGET_SAMPLES:
        print(f"ERROR: selected {len(selected)} samples, expected {N_TARGET_SAMPLES}")
        return 4
    selected_set = set(selected)

    # filter entries, preserving accepted-manifest entry order within each sample
    filtered = [e for e in entries if e["sample_token"] in selected_set]
    print(
        f"filtered entries: {len(filtered)}  (expected {N_TARGET_SAMPLES * 6} = {N_TARGET_SAMPLES} samples * 6 cams)"
    )
    if len(filtered) != N_TARGET_SAMPLES * 6:
        print("ERROR: entry count mismatch")
        return 5

    # cam-coverage check
    from collections import Counter

    per_sample_cams = Counter()
    for e in filtered:
        per_sample_cams[e["sample_token"]] += 1
    bad = {t: c for t, c in per_sample_cams.items() if c != 6}
    if bad:
        print(
            f"ERROR: {len(bad)} selected samples don't have all 6 cams: {list(bad.items())[:5]}"
        )
        return 6

    # generated_path existence check (every entry must point at an extant png)
    missing: list[str] = []
    for e in filtered:
        gp = os.path.join(REPO, e["generated_path"])
        if not os.path.isfile(gp):
            missing.append(gp)
    if missing:
        print(f"ERROR: {len(missing)} generated_path entries missing on disk; first 5:")
        for m in missing[:5]:
            print(" ", m)
        return 7
    print(f"all {len(filtered)} generated_path files exist on disk")

    out = {
        # carry parent fields for traceability
        "generator": acc["generator"],
        "model": acc["model"],
        "image_size_tier": acc["image_size_tier"],
        "prompt_sha256": acc["prompt_sha256"],
        "generated_at": acc["generated_at"],
        "source_manifest": acc["source_manifest"],
        "merged_from_chunks": acc["merged_from_chunks"],
        "seed": acc["seed"],
        # ratio12p5-specific provenance
        "__derived_from": "research/night_gen_phase1/manifests/phase1_t10_seed20260425_accepted.json",
        "__selection_rule": selection_rule,
        "__selection_source": label,
        "__derived_at": datetime.now(timezone.utc).isoformat(),
        "ratio_target": "12.5%",
        "ratio_basis": "250 samples / 2000 day-only training subset",
        "n_accepted_samples_pool": 486,
        "n_accepted_entries_pool": 2916,
        "n_selected_samples": N_TARGET_SAMPLES,
        "n_selected_entries": len(filtered),
        "__cluster_rewritten_to": acc["__cluster_rewritten_to"],
        "__path_rewritten_to_repo_relative": acc.get(
            "__path_rewritten_to_repo_relative", True
        ),
        "n_accepted_entries": len(filtered),
        "n_accepted_samples": N_TARGET_SAMPLES,
        "entries": filtered,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT}")
    print(f"  selection_rule = {selection_rule}")
    print(f"  n_selected_samples = {N_TARGET_SAMPLES}")
    print(f"  n_selected_entries = {len(filtered)}")
    print(f"  first 5 selected tokens (in selection order):")
    for t in selected[:5]:
        print(f"    {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
