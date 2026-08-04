"""
T1 — build the Phase 1 derived pkls from the existing source pkls.

Outputs (all written under --out, defaults to repo root):
  nuscenes_infos_train_2k_day.pkl
  nuscenes_infos_train_2k_mixed.pkl
  nuscenes_infos_train_2k_mixed_oversampled.pkl    # T6-A: physical duplication
  nuscenes_infos_val_day_matched.pkl
  research/night_gen_phase1/manifests/phase1_subset_report.json

Source pkls (must exist; not mutated):
  nuscenes_infos_train_sweep.pkl
  nuscenes_infos_val_sweep.pkl

Scene condition function is re-implemented inline (matches loaders/nuscenes_dataset.py:
_get_scene_condition) to avoid the module-level `NuScenes(...)` import in that file.

Usage (cluster, repo root /srv/nfs/shared/gnmp/RaCFormer/, racformerfix env):
    conda run -n racformerfix python research/night_gen_phase1/build_phase1_pkls.py \
        --train-pkl nuscenes_infos_train_sweep.pkl \
        --val-pkl   nuscenes_infos_val_sweep.pkl \
        --dataroot  /srv/nfs/shared/shared/nuscenes/ \
        --nusc-version v1.0-trainval \
        --out       . \
        --report    research/night_gen_phase1/manifests/phase1_subset_report.json \
        --seed      20260425 \
        --oversample-target-n 2000 \
        --oversample-target-night 0.32

The oversampled pkl is built by sampling with replacement from the mixed pkl: night
samples receive ~640/2000 (32%) of the entries, the remaining ~1360 entries are split
between day and rain in proportion to their representation in the mixed pkl. This keeps
the epoch length and effective LR schedule comparable to S0/S1/S3 while raising night
exposure ~3.2x.
"""

import argparse
import json
import os
import pickle
import random
from collections import Counter, defaultdict


def get_scene_condition(nusc, sample_token):
    """Re-implementation of CustomNuScenesDataset_radar._get_scene_condition."""
    try:
        sample = nusc.get("sample", sample_token)
        scene = nusc.get("scene", sample["scene_token"])
        desc = scene.get("description", "").lower()
        if "night" in desc:
            return "night"
        if "rain" in desc or "rainy" in desc:
            return "rain"
        return "day"
    except Exception:
        return "unknown"


def get_scene_location(nusc, scene_token):
    try:
        scene = nusc.get("scene", scene_token)
        log = nusc.get("log", scene["log_token"])
        return log.get("location", "unknown")
    except Exception:
        return "unknown"


def load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict) and "infos" in d:
        return d, d["infos"]
    return {"infos": d}, d


def save_pkl(orig_wrapper, new_infos, path):
    out = {k: v for k, v in orig_wrapper.items() if k != "infos"}
    out["infos"] = new_infos
    with open(path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


def tag_conditions(infos, nusc):
    """Annotate each info dict with scene_condition (in-place; preserves all other fields)."""
    counts = Counter()
    for info in infos:
        cond = get_scene_condition(nusc, info["token"])
        info["scene_condition"] = cond
        counts[cond] += 1
    return counts


def round_robin_pick(infos_by_cond, target_n, seed, condition_quotas=None):
    """
    Pick `target_n` infos with a deterministic seed, optionally honoring per-condition
    quotas. If quotas is None, picks proportional to availability.
    """
    rng = random.Random(seed)
    picked = []
    if condition_quotas is None:
        # All conditions; sample uniformly without replacement up to target_n
        all_infos = []
        for v in infos_by_cond.values():
            all_infos.extend(v)
        rng.shuffle(all_infos)
        return all_infos[:target_n]

    for cond, quota in condition_quotas.items():
        pool = list(infos_by_cond.get(cond, []))
        rng.shuffle(pool)
        picked.extend(pool[:quota])
    rng.shuffle(picked)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pkl", required=True)
    ap.add_argument("--val-pkl", required=True)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--nusc-version", default="v1.0-trainval")
    ap.add_argument("--out", default=".")
    ap.add_argument("--report", required=True)
    ap.add_argument("--seed", type=int, default=20260425)
    ap.add_argument(
        "--oversample-target-n",
        type=int,
        default=2000,
        help="Total number of entries in the oversampled pkl (sampled with replacement).",
    )
    ap.add_argument(
        "--oversample-target-night",
        type=float,
        default=0.32,
        help="Target night fraction in the oversampled pkl. Must land in [0.30, 0.35].",
    )
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes  # noqa: WPS433

    print(f"[t1] loading NuScenes({args.nusc_version}) from {args.dataroot}")
    nusc = NuScenes(version=args.nusc_version, dataroot=args.dataroot, verbose=False)

    print(f"[t1] loading train pkl: {args.train_pkl}")
    train_wrapper, train_infos = load_pkl(args.train_pkl)
    print(f"[t1] {len(train_infos)} train infos")

    print(f"[t1] loading val pkl: {args.val_pkl}")
    val_wrapper, val_infos = load_pkl(args.val_pkl)
    print(f"[t1] {len(val_infos)} val infos")

    # Tag every info with scene_condition (in-memory; NOT written back to source pkls)
    print("[t1] tagging train conditions ...")
    train_counts = tag_conditions(train_infos, nusc)
    print(f"[t1] train conditions: {dict(train_counts)}")
    print("[t1] tagging val conditions ...")
    val_counts = tag_conditions(val_infos, nusc)
    print(f"[t1] val conditions: {dict(val_counts)}")

    # Tag locations on val for day-matched filtering
    val_scene_to_loc = {}
    for info in val_infos:
        if info["scene_token"] not in val_scene_to_loc:
            val_scene_to_loc[info["scene_token"]] = get_scene_location(
                nusc, info["scene_token"]
            )

    # --- T9 fix: drop samples missing the 5-sweep radar/cam history.
    # An empty sweeps list (len==0) crashes racformer.py at iter 1 because the
    # model assumes uniform batch dim across all 5 radar sweeps and >=1 radar
    # point per sample. ~40 day / ~57 mixed samples have len(sweeps)==0.
    n_pre = len(train_infos)
    train_infos = [i for i in train_infos if len(i.get("sweeps", [])) == 5]
    n_dropped = n_pre - len(train_infos)
    print(f"[t1] dropped {n_dropped} train samples with len(sweeps)<5; {len(train_infos)} remain")

    train_by_cond = defaultdict(list)
    for info in train_infos:
        train_by_cond[info["scene_condition"]].append(info)

    rng = random.Random(args.seed)

    # --- 1. nuscenes_infos_train_2k_day.pkl --------------------------------
    n_target_day = 2000
    day_pool = list(train_by_cond.get("day", []))
    rng.shuffle(day_pool)
    if len(day_pool) < n_target_day:
        raise RuntimeError(
            f"only {len(day_pool)} day samples in train pkl; cannot build 2K day pkl"
        )
    day_pkl_infos = day_pool[:n_target_day]
    day_path = os.path.join(args.out, "nuscenes_infos_train_2k_day.pkl")
    save_pkl(train_wrapper, day_pkl_infos, day_path)
    print(f"[t1] wrote {day_path}: {len(day_pkl_infos)} samples (all day)")

    # --- 2. nuscenes_infos_train_2k_mixed.pkl ------------------------------
    # Preserve natural condition ratios within a 2K cap.
    n_target_mixed = 2000
    total_train = sum(train_counts.values())
    quotas = {}
    for cond, n in train_counts.items():
        if cond == "unknown":
            continue
        quotas[cond] = max(1, round(n_target_mixed * n / total_train))
    quota_sum = sum(quotas.values())
    if quota_sum != n_target_mixed:
        # Adjust the largest bucket to keep total at exactly 2000
        biggest = max(quotas, key=lambda k: quotas[k])
        quotas[biggest] += n_target_mixed - quota_sum
    print(f"[t1] mixed pkl quotas: {quotas}")
    mixed_infos = round_robin_pick(
        {c: list(v) for c, v in train_by_cond.items()},
        n_target_mixed,
        args.seed + 1,
        quotas,
    )
    rng.shuffle(mixed_infos)
    mixed_counts = Counter(i["scene_condition"] for i in mixed_infos)
    mixed_path = os.path.join(args.out, "nuscenes_infos_train_2k_mixed.pkl")
    save_pkl(train_wrapper, mixed_infos, mixed_path)
    print(
        f"[t1] wrote {mixed_path}: {len(mixed_infos)} samples, conditions={dict(mixed_counts)}"
    )

    # --- 3. nuscenes_infos_train_2k_mixed_oversampled.pkl  (T6-A revised) --
    # Exactly target_n entries, sampled with replacement, with target night fraction
    # in [0.30, 0.35]. Day:rain ratio in the non-night portion preserves the mixed
    # pkl distribution, so non-night samples are statistically equivalent to S0/S1/S3.
    target_n = args.oversample_target_n
    target_night = args.oversample_target_night
    if not (0.30 <= target_night <= 0.35):
        raise ValueError(
            f"--oversample-target-night must be in [0.30, 0.35], got {target_night}"
        )
    n_night = int(round(target_n * target_night))
    n_other = target_n - n_night

    night_pool = [i for i in mixed_infos if i["scene_condition"] == "night"]
    day_pool = [i for i in mixed_infos if i["scene_condition"] == "day"]
    rain_pool = [i for i in mixed_infos if i["scene_condition"] == "rain"]
    if not night_pool:
        raise RuntimeError("mixed pkl has 0 night entries — cannot oversample")
    if not day_pool:
        raise RuntimeError("mixed pkl has 0 day entries — cannot oversample")

    if rain_pool:
        ratio_day = len(day_pool) / (len(day_pool) + len(rain_pool))
    else:
        ratio_day = 1.0
    n_day = int(round(n_other * ratio_day))
    n_rain = n_other - n_day

    rng_os = random.Random(args.seed + 7)
    oversampled_infos = (
        [rng_os.choice(night_pool) for _ in range(n_night)]
        + [rng_os.choice(day_pool) for _ in range(n_day)]
        + [rng_os.choice(rain_pool) for _ in range(n_rain)]
    )
    rng_os.shuffle(oversampled_infos)
    if len(oversampled_infos) != target_n:
        raise RuntimeError(
            f"oversampled length mismatch: got {len(oversampled_infos)}, expected {target_n}"
        )

    oversampled_counts = Counter(i["scene_condition"] for i in oversampled_infos)
    realized_night_fraction = oversampled_counts["night"] / target_n
    if not (0.30 <= realized_night_fraction <= 0.35):
        raise RuntimeError(
            f"realized night fraction {realized_night_fraction:.4f} outside [0.30, 0.35]"
        )

    unique_night_count = len({id(i) for i in night_pool})
    oversampled_path = os.path.join(
        args.out, "nuscenes_infos_train_2k_mixed_oversampled.pkl"
    )
    save_pkl(train_wrapper, oversampled_infos, oversampled_path)
    print(
        f"[t1] wrote {oversampled_path}: {len(oversampled_infos)} samples, "
        f"conditions={dict(oversampled_counts)}, "
        f"realized_night_fraction={realized_night_fraction:.4f}, "
        f"unique_night_pool={unique_night_count}, strategy=with_replacement"
    )

    # --- 4. nuscenes_infos_val_day_matched.pkl -----------------------------
    # Day-condition val tokens drawn only from scenes/locations that also contain
    # at least one night sample in val.
    night_locations = {
        val_scene_to_loc[info["scene_token"]]
        for info in val_infos
        if info["scene_condition"] == "night"
    }
    night_scene_tokens = {
        info["scene_token"] for info in val_infos if info["scene_condition"] == "night"
    }
    val_day_matched = [
        info
        for info in val_infos
        if info["scene_condition"] == "day"
        and (
            info["scene_token"] in night_scene_tokens  # same scene (rare)
            or val_scene_to_loc.get(info["scene_token"]) in night_locations
        )
    ]
    val_dm_path = os.path.join(args.out, "nuscenes_infos_val_day_matched.pkl")
    save_pkl(val_wrapper, val_day_matched, val_dm_path)
    print(
        f"[t1] wrote {val_dm_path}: {len(val_day_matched)} day-matched val samples "
        f"(from {len(night_locations)} night locations, {len(night_scene_tokens)} night scenes)"
    )

    # --- subset report ----------------------------------------------------
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    report = {
        "seed": args.seed,
        "train_source": args.train_pkl,
        "val_source": args.val_pkl,
        "train_total_infos": len(train_infos),
        "val_total_infos": len(val_infos),
        "train_condition_counts": dict(train_counts),
        "val_condition_counts": dict(val_counts),
        "train_2k_day": {
            "path": day_path,
            "n": len(day_pkl_infos),
            "conditions": {"day": len(day_pkl_infos)},
        },
        "train_2k_mixed": {
            "path": mixed_path,
            "n": len(mixed_infos),
            "conditions": dict(mixed_counts),
            "quotas": quotas,
        },
        "train_2k_mixed_oversampled": {
            "path": oversampled_path,
            "n": len(oversampled_infos),
            "conditions": dict(oversampled_counts),
            "oversample_strategy": "with_replacement",
            "target_n": target_n,
            "target_night": target_night,
            "realized_night_fraction": realized_night_fraction,
            "unique_night_count": unique_night_count,
            "n_night_replicas": n_night,
            "n_day_replicas": n_day,
            "n_rain_replicas": n_rain,
        },
        "val_day_matched": {
            "path": val_dm_path,
            "n": len(val_day_matched),
            "n_night_locations": len(night_locations),
            "n_night_scenes": len(night_scene_tokens),
        },
        "unknown_count_train": train_counts.get("unknown", 0),
        "unknown_count_val": val_counts.get("unknown", 0),
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[t1] wrote {args.report}")
    print("[t1] done")


if __name__ == "__main__":
    main()
