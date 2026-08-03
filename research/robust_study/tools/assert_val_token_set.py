"""Post-run assertion: a submission JSON covers exactly the official nuScenes val split.

Checks, in order (any failure exits non-zero and the calling job fails):
  1. the JSON parses and has a `results` field;
  2. its token set is SET-EQUAL to `nuscenes.utils.splits.val` resolved to sample tokens;
  3. the token count is the expected 6019;
  4. every token maps to a list (>= 0 boxes; an empty list is legal, a missing key is not).

Emits `token_set.json` with the token-set sha256 and the staged counts, which the provenance
writer folds into `provenance.json`.

Scene-name -> sample-token resolution reads the NuScenes metadata JSONs directly rather than
constructing a NuScenes object: the membership test is the same one
nuscenes/eval/common/loaders.py::load_gt applies (scene_record['name'] in splits[eval_split]),
and skipping the full DB load keeps this assertion cheap enough to run at the end of every job.
"""

import argparse
import hashlib
import json
import os
import sys

EXPECTED_N_VAL_SAMPLES = 6019


def token_set_sha256(tokens):
    return hashlib.sha256("\n".join(sorted(tokens)).encode("utf-8")).hexdigest()


def official_val_tokens(dataroot, version):
    from nuscenes.utils.splits import val as val_scene_names  # noqa: WPS433

    meta = os.path.join(dataroot, version)
    with open(os.path.join(meta, "scene.json")) as f:
        scenes = json.load(f)
    with open(os.path.join(meta, "sample.json")) as f:
        samples = json.load(f)

    wanted = set(val_scene_names)
    val_scene_tokens = {s["token"] for s in scenes if s["name"] in wanted}
    missing_scenes = wanted - {s["name"] for s in scenes}
    if missing_scenes:
        raise SystemExit("[assert_val_token_set] FATAL: %d val scenes absent from %s: %s"
                         % (len(missing_scenes), meta, sorted(missing_scenes)[:5]))
    return {s["token"] for s in samples if s["scene_token"] in val_scene_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out", required=True, help="path to write token_set.json")
    ap.add_argument("--expected-n", type=int, default=EXPECTED_N_VAL_SAMPLES)
    args = ap.parse_args()

    with open(args.submission) as f:
        data = json.load(f)
    if "results" not in data:
        raise SystemExit("[assert_val_token_set] FATAL: no `results` field in %s" % args.submission)
    results = data["results"]

    bad_values = [t for t, v in results.items() if not isinstance(v, list)]
    if bad_values:
        raise SystemExit("[assert_val_token_set] FATAL: %d tokens map to a non-list value, e.g. %s"
                         % (len(bad_values), bad_values[:3]))

    submission_tokens = set(results.keys())
    official = official_val_tokens(args.dataroot, args.version)

    only_sub = submission_tokens - official
    only_off = official - submission_tokens
    n_boxes = sum(len(v) for v in results.values())
    empty_tokens = sum(1 for v in results.values() if not v)

    out = {
        "submission": os.path.abspath(args.submission),
        "n_submission_tokens": len(submission_tokens),
        "n_official_val_tokens": len(official),
        "expected_n": args.expected_n,
        "set_equal": not only_sub and not only_off,
        "n_only_in_submission": len(only_sub),
        "n_only_in_official": len(only_off),
        "sample_only_in_submission": sorted(only_sub)[:10],
        "sample_only_in_official": sorted(only_off)[:10],
        "token_set_sha256": token_set_sha256(submission_tokens),
        "official_token_set_sha256": token_set_sha256(official),
        "n_boxes_total": n_boxes,
        "n_tokens_with_zero_boxes": empty_tokens,
        "meta": data.get("meta"),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    problems = []
    if not out["set_equal"]:
        problems.append("token set is NOT set-equal to the official val split "
                        "(%d only-in-submission, %d only-in-official)" % (len(only_sub), len(only_off)))
    if len(official) != args.expected_n:
        problems.append("official val resolved to %d tokens, expected %d" % (len(official), args.expected_n))
    if len(submission_tokens) != args.expected_n:
        problems.append("submission covers %d tokens, expected %d" % (len(submission_tokens), args.expected_n))

    print("[assert_val_token_set] tokens=%d official=%d boxes=%d empty_tokens=%d sha256=%s"
          % (len(submission_tokens), len(official), n_boxes, empty_tokens, out["token_set_sha256"]))
    if problems:
        for problem in problems:
            print("[assert_val_token_set] FAIL: %s" % problem, file=sys.stderr)
        sys.exit(1)
    print("[assert_val_token_set] OK")


if __name__ == "__main__":
    main()
