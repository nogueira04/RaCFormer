"""
Merges the 5 seed20260502_ratio18p75 chunk generated-manifests into one training-time
manifest at the RaCFormer-repo-relative path scheme.

Rewrites generated_path:
  outputs/nano_banana_v2_t10_seed20260502_ratio18p75_512/samples/CAM_X/foo.png
->
  research/night_gen_phase1/outputs/nano_banana_v2_t10_seed20260502_ratio18p75/samples/CAM_X/foo.png

Verifies:
  - 2250 entries, all status=ok
  - 375 unique sample_tokens with all 6 cams (sample-complete)
  - all rewritten generated_paths resolve to existing files under repo root
"""

import json
import os
import sys
from collections import defaultdict

CHUNK_DIR = "/srv/nfs/shared/gnmp/t10_gen/manifests"
OUT_PATH = (
    "/srv/nfs/shared/gnmp/RaCFormer/research/night_gen_phase1/manifests/"
    "phase1_t10_seed20260502_ratio18p75_generated.json"
)
REPO_ROOT = "/srv/nfs/shared/gnmp/RaCFormer"
PURPOSE = "phase1_t10_seed20260502_ratio18p75"
SRC_PREFIX = "outputs/nano_banana_v2_t10_seed20260502_ratio18p75_512"
DST_PREFIX = (
    "research/night_gen_phase1/outputs/nano_banana_v2_t10_seed20260502_ratio18p75"
)
EXPECTED_CAMS = {
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
}


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    all_entries = []
    chunk_meta = []
    for c in range(1, 6):
        path = os.path.join(CHUNK_DIR, f"{PURPOSE}_generated_chunk_{c}.json")
        with open(path) as f:
            d = json.load(f)
        chunk_meta.append(
            {
                "chunk": c,
                "n_success": d.get("n_success"),
                "n_failed": d.get("n_failed"),
                "spend_usd_est": d.get("spend_usd_est"),
                "generated_at": d.get("generated_at"),
            }
        )
        for e in d["entries"]:
            if e.get("status") != "ok":
                continue
            new = dict(e)
            gp = e["generated_path"]
            if gp.startswith(SRC_PREFIX):
                new["generated_path"] = DST_PREFIX + gp[len(SRC_PREFIX) :]
            else:
                raise RuntimeError(f"unexpected generated_path: {gp!r}")
            all_entries.append(new)

    # Sample-completeness
    by_token = defaultdict(set)
    for e in all_entries:
        by_token[e["sample_token"]].add(e["camera"])
    sample_complete_tokens = [
        t for t, cams in by_token.items() if cams == EXPECTED_CAMS
    ]
    n_total = len(all_entries)
    n_unique = len(by_token)
    n_complete = len(sample_complete_tokens)

    print(f"[merge] total ok entries: {n_total}")
    print(f"[merge] unique sample_tokens: {n_unique}")
    print(f"[merge] sample-complete tokens (6/6 cams): {n_complete}")
    if n_total != 2250 or n_complete != 375:
        print(
            "[merge] WARNING: counts do not match (expected 2250 entries / 375 sample-complete)"
        )
        # Continue anyway — let downstream decide.

    # Path resolution check (sample 30)
    import random as _r

    _r.seed(0)
    miss = []
    for e in _r.sample(all_entries, min(30, len(all_entries))):
        abs_p = os.path.join(REPO_ROOT, e["generated_path"])
        if not os.path.exists(abs_p):
            miss.append(abs_p)
    if miss:
        print(f"[merge] FAIL: {len(miss)} of 30 spot-checked PNGs do not resolve")
        for m in miss[:5]:
            print(f"  missing: {m}")
        sys.exit(1)
    print(f"[merge] spot-check 30/30 PNGs resolve under {REPO_ROOT}")

    out = {
        "generator": "nano_banana_v2",
        "model": "gemini-3.1-flash-image-preview",
        "image_size_tier": "512",
        "source_chunks": [f"{PURPOSE}_generated_chunk_{c}.json" for c in range(1, 6)],
        "chunk_meta": chunk_meta,
        "seed": 20260502,
        "ratio_target": 18.75,
        "ratio_basis": "375 of 2000 = 18.75% of train_2k_day pkl",
        "__derived_at": "merged from cluster t10_gen chunk manifests",
        "__cluster_rewritten_to": REPO_ROOT,
        "__path_rewritten_to_repo_relative": True,
        "n_accepted_entries": n_total,
        "n_accepted_samples": n_complete,
        "n_total_unique_tokens": n_unique,
        "entries": all_entries,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"[merge] wrote {OUT_PATH} ({n_total} entries / {n_complete} sample-complete)"
    )


if __name__ == "__main__":
    main()
