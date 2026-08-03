"""Independent nuScenes devkit cross-check of a detection submission JSON.

Runs the OFFICIAL `DetectionEval` (config `detection_cvpr_2019`, split `val`) from a dedicated,
separately-installed devkit venv and compares its numbers against the values the evaluation
wrapper reported for the same artifact. The point is independence, not parity: nothing in this
file imports the wrapper, mmdet3d, or torch.

Invocation contract (enforced by assertions below, see `run_devkit_crosscheck.sh`):
    PYTHONNOUSERSITE=1 <venv>/bin/python -I devkit_crosscheck.py ...
`python -I` gives isolated mode (no user site, no inherited PYTHONPATH, no cwd on sys.path); the
script additionally asserts that `nuscenes.__file__` resolves inside the venv, so an accidental
run under any other interpreter fails loudly instead of silently cross-checking against itself.

Outputs `crosscheck.json` (full precision, no rounding). Exit status is the verdict: 0 = every
comparison row passed, non-zero = at least one row failed (or an unverifiable row was present and
`--allow-unverifiable-rows` was not given).
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict

# --- Interpreter / devkit isolation assertions -------------------------------------------------
# These run before anything else so a mis-launched process cannot produce a crosscheck.json.


def _assert_isolated_interpreter(venv_root):
    problems = []
    if not sys.flags.isolated:
        problems.append("interpreter not started with -I (sys.flags.isolated == 0)")
    if not sys.flags.no_user_site:
        problems.append("user site-packages not disabled (sys.flags.no_user_site == 0)")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        problems.append("PYTHONNOUSERSITE is not '1' in the environment")
    inherited_pythonpath = os.environ.get("PYTHONPATH")
    if inherited_pythonpath:
        problems.append("PYTHONPATH is set in the environment: %r" % inherited_pythonpath)
    if problems:
        raise SystemExit("[devkit_crosscheck] FATAL isolation failure:\n  - " + "\n  - ".join(problems))

    import nuscenes  # noqa: WPS433  (imported here so the checks above run first)

    venv_real = os.path.realpath(os.path.expanduser(venv_root))
    devkit_real = os.path.realpath(nuscenes.__file__)
    if not devkit_real.startswith(venv_real + os.sep):
        raise SystemExit(
            "[devkit_crosscheck] FATAL: nuscenes resolves to %s which is OUTSIDE the cross-check "
            "venv %s. Refusing to run: the whole point of this script is an independently "
            "installed devkit." % (devkit_real, venv_real)
        )
    return devkit_real


def _devkit_version():
    try:
        import importlib.metadata as md  # noqa: WPS433

        return md.version("nuscenes-devkit")
    except Exception as exc:  # noqa: BLE001
        return "UNKNOWN (%s)" % exc


# --- Small helpers -----------------------------------------------------------------------------

PP = 100.0  # devkit reports AP/NDS as fractions; the study reports percentage points.


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def token_set_sha256(tokens):
    """Order-independent digest of a token set: sha256 over newline-joined sorted tokens."""
    payload = "\n".join(sorted(tokens)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _jsonable(value):
    """JSON cannot hold +/-inf; NaN is kept (json emits NaN) because the NaN policy forbids
    substituting it with a sentinel. +inf only ever appears as a max-F1 threshold and is encoded
    as null, documented in the frozen-thresholds file."""
    if isinstance(value, float) and value == float("inf"):
        return None
    return value


# --- Official-val token resolution -------------------------------------------------------------


def official_split_tokens(nusc, split):
    """Resolve `nuscenes.utils.splits.<split>` scene names to sample tokens.

    Mirrors nuscenes/eval/common/loaders.py::load_gt (scene-name membership test), so the token
    set compared here is the same one the devkit itself evaluates on.
    """
    from nuscenes.utils.splits import create_splits_scenes  # noqa: WPS433

    scenes = set(create_splits_scenes()[split])
    tokens = []
    for sample in nusc.sample:
        scene = nusc.get("scene", sample["scene_token"])
        if scene["name"] in scenes:
            tokens.append(sample["token"])
    return set(tokens)


# --- Staged box counts -------------------------------------------------------------------------


def raw_json_counts(submission_path):
    """Box counts straight out of the submission file, before the devkit touches anything."""
    with open(submission_path) as f:
        data = json.load(f)
    results = data["results"]
    per_class = {}
    total = 0
    max_per_sample = 0
    for token, boxes in results.items():
        total += len(boxes)
        if len(boxes) > max_per_sample:
            max_per_sample = len(boxes)
        for box in boxes:
            name = box["detection_name"]
            per_class[name] = per_class.get(name, 0) + 1
    return {
        "tokens": set(results.keys()),
        "total": total,
        "per_class": per_class,
        "max_boxes_per_sample": max_per_sample,
        "meta": data.get("meta"),
    }


def recompute_devkit_postfilter(nusc, submission_path, eval_cfg):
    """Independently re-apply the devkit's own prediction filter rules to the raw JSON.

    Mirrors, in order, nuscenes/eval/common/loaders.py::filter_eval_boxes (v1.1.11):
      1. distance:  keep if box.ego_dist < class_range[detection_name]   (strict <, 2D xy)
      2. points:    keep if box.num_pts != 0  (predictions carry the -1 default -> always kept)
      3. bike rack: drop bicycle/motorcycle whose center falls inside a static bicycle-rack box
    `max_boxes_per_sample` is an assertion in load_prediction, not a cap, so it is checked here as
    a limit rather than applied as a truncation.

    Recomputed from the JSON + the NuScenes DB, never from the devkit's own filtered structures,
    so agreement with the devkit's internal count is real evidence rather than a tautology.
    """
    import numpy as np  # noqa: WPS433
    from pyquaternion import Quaternion  # noqa: WPS433
    from nuscenes.utils.data_classes import Box  # noqa: WPS433
    from nuscenes.utils.geometry_utils import points_in_box  # noqa: WPS433

    with open(submission_path) as f:
        results = json.load(f)["results"]

    stages = {"total": 0, "after_distance": 0, "after_points": 0, "after_bikerack": 0}
    per_class = {}
    over_cap_samples = []

    for token, boxes in results.items():
        if len(boxes) > eval_cfg.max_boxes_per_sample:
            over_cap_samples.append(token)
        stages["total"] += len(boxes)

        sample_rec = nusc.get("sample", token)
        sd_record = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])
        ego_xyz = pose_record["translation"]

        kept = []
        for box in boxes:
            name = box["detection_name"]
            dx = box["translation"][0] - ego_xyz[0]
            dy = box["translation"][1] - ego_xyz[1]
            ego_dist = float(np.sqrt(dx * dx + dy * dy))
            if ego_dist < eval_cfg.class_range[name]:
                kept.append(box)
        stages["after_distance"] += len(kept)

        # num_pts defaults to -1 for deserialized predictions, so this stage is a no-op for them;
        # it is spelled out because the devkit applies it unconditionally.
        kept = [b for b in kept if b.get("num_pts", -1) != 0]
        stages["after_points"] += len(kept)

        needs_bikerack = any(b["detection_name"] in ("bicycle", "motorcycle") for b in kept)
        if needs_bikerack:
            bikerack_boxes = []
            for ann_token in sample_rec["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                if ann["category_name"] == "static_object.bicycle_rack":
                    bikerack_boxes.append(Box(ann["translation"], ann["size"], Quaternion(ann["rotation"])))
            survivors = []
            for box in kept:
                if box["detection_name"] in ("bicycle", "motorcycle") and bikerack_boxes:
                    center = np.expand_dims(np.array(box["translation"]), axis=1)
                    in_rack = any(np.sum(points_in_box(br, center)) > 0 for br in bikerack_boxes)
                    if in_rack:
                        continue
                survivors.append(box)
            kept = survivors
        stages["after_bikerack"] += len(kept)

        for box in kept:
            per_class[box["detection_name"]] = per_class.get(box["detection_name"], 0) + 1

    return {
        "stages": stages,
        "total": stages["after_bikerack"],
        "per_class": per_class,
        "samples_over_max_boxes_per_sample": over_cap_samples,
    }


# --- TP/FP/FN at dist_th_tp --------------------------------------------------------------------


def match_class(gt_boxes, pred_boxes, class_name, dist_fcn, dist_th):
    """Replicate the devkit's greedy matching for one class at one distance threshold.

    Line-for-line mirror of nuscenes/eval/detection/algo.py::accumulate (v1.1.11) up to the point
    where it discards the cumulative tp/fp arrays in favour of an interpolated PR curve. The
    devkit never exposes raw counts, so they are recomputed here; `ap_replicated` (below) proves
    the replication is faithful by re-deriving the class AP from these arrays and comparing it to
    the devkit's own AP for the same class and threshold.

    Returns None when the devkit would have returned `no_predictions()`.
    """
    import numpy as np  # noqa: WPS433

    npos = len([1 for gt in gt_boxes.all if gt.detection_name == class_name])
    if npos == 0:
        return None

    pred_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_list]
    # Devkit sort order, verbatim: ascending (score, index), then reversed. Equal scores therefore
    # come out in DESCENDING index order, and equal-score runs are contiguous.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    tp, fp, conf = [], [], []
    taken = set()
    for ind in sortind:
        pred_box = pred_list[ind]
        min_dist = np.inf
        match_gt_idx = None
        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx
        if min_dist < dist_th:
            taken.add((pred_box.sample_token, match_gt_idx))
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
        conf.append(pred_box.detection_score)

    if not tp:
        return {"npos": npos, "n_pred": 0, "tp_cum": None, "fp_cum": None, "conf": None}

    return {
        "npos": npos,
        "n_pred": len(tp),
        "tp_cum": np.cumsum(tp).astype(float),
        "fp_cum": np.cumsum(fp).astype(float),
        "conf": np.array(conf, dtype=float),
    }


def ap_replicated(match, min_recall, min_precision):
    """Re-derive the class AP from the replicated cumulative arrays, using the devkit's own math
    (algo.py interpolation block + calc_ap). Compared against the devkit's AP as a proof that
    `match_class` reproduces the official matching."""
    import numpy as np  # noqa: WPS433

    if match is None or match["tp_cum"] is None or match["tp_cum"][-1] == 0:
        return 0.0
    tp, fp, conf, npos = match["tp_cum"], match["fp_cum"], match["conf"], match["npos"]
    prec = tp / (fp + tp)
    rec = tp / float(npos)
    rec_interp = np.linspace(0, 1, 101)
    prec = np.interp(rec_interp, rec, prec, right=0)
    prec = np.copy(prec)[round(100 * min_recall) + 1:]
    prec -= min_precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def counts_at_threshold(match, threshold):
    """TP/FP/FN for one class at a score cut.

    Greedy matching is sequential in descending score, so keeping only predictions with
    score >= threshold is exactly a prefix of the matched list: the cumulative arrays can be read
    off directly with no re-matching.
    """
    import numpy as np  # noqa: WPS433

    if match is None:
        return None
    npos = match["npos"]
    if match["tp_cum"] is None or threshold == float("inf"):
        return {"TP": 0, "FP": 0, "FN": npos, "GT": npos, "recall": 0.0, "precision": float("nan")}
    keep = np.nonzero(match["conf"] >= threshold)[0]
    if len(keep) == 0:
        return {"TP": 0, "FP": 0, "FN": npos, "GT": npos, "recall": 0.0, "precision": float("nan")}
    idx = int(keep[-1])
    tp = int(match["tp_cum"][idx])
    fp = int(match["fp_cum"][idx])
    fn = npos - tp
    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else float("nan")
    return {"TP": tp, "FP": fp, "FN": fn, "GT": npos, "recall": float(tp) / npos, "precision": precision}


def max_f1_threshold(match):
    """Per-class score threshold maximising F1 over the dist_th_tp match set.

    Candidate set = the class's unique prediction scores. Ties in F1 are broken by taking the
    LOWEST score among the maxima (keeps more predictions). An empty candidate set yields +inf,
    at which every count is 0.
    """
    import numpy as np  # noqa: WPS433

    if match is None or match["tp_cum"] is None or match["n_pred"] == 0:
        return float("inf"), float("nan")

    conf = match["conf"]
    tp_cum, fp_cum, npos = match["tp_cum"], match["fp_cum"], match["npos"]
    # conf is sorted descending; the last index of each equal-score run is the prefix end.
    unique_scores = np.unique(conf)  # ascending, unique
    best_f1 = -1.0
    best_threshold = float("inf")
    for score in unique_scores:  # ascending => the first maximum found is the lowest score
        keep = np.nonzero(conf >= score)[0]
        idx = int(keep[-1])
        tp = float(tp_cum[idx])
        fp = float(fp_cum[idx])
        fn = float(npos) - tp
        denom = 2.0 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(score)
    return best_threshold, best_f1


# --- Mechanical comparison ---------------------------------------------------------------------


class Row(object):
    """One row of the pre-registered tolerance table."""

    def __init__(self, name, status, detail):
        self.name = name
        self.status = status  # PASS | FAIL | UNVERIFIABLE
        self.detail = detail

    def as_dict(self):
        out = OrderedDict([("row", self.name), ("status", self.status)])
        out.update(self.detail)
        return out


def cmp_abs_pp(name, ours, theirs, tol_pp):
    if not (_finite(ours) and _finite(theirs)):
        return Row(name, "FAIL", {"reason": "non-finite value (Phase E forbids NaN)",
                                  "devkit": ours, "wrapper": theirs})
    diff = abs(ours - theirs)
    return Row(name, "PASS" if diff <= tol_pp else "FAIL",
               {"devkit_pp": ours, "wrapper_pp": theirs, "abs_diff_pp": diff, "tol_abs_pp": tol_pp})


def cmp_rel(name, ours, theirs, tol_rel, abs_guard):
    if not (_finite(ours) and _finite(theirs)):
        return Row(name, "FAIL", {"reason": "non-finite value (Phase E forbids NaN)",
                                  "devkit": ours, "wrapper": theirs})
    abs_diff = abs(ours - theirs)
    rel = abs_diff / abs(theirs) if theirs != 0 else float("inf")
    ok = rel <= tol_rel or abs_diff <= abs_guard
    return Row(name, "PASS" if ok else "FAIL",
               {"devkit": ours, "wrapper": theirs, "abs_diff": abs_diff, "rel_diff": rel,
                "tol_rel": tol_rel, "abs_guard": abs_guard})


TP_ERROR_KEYS = ["trans_err", "scale_err", "orient_err", "vel_err", "attr_err"]


def build_comparison(report, p2_metrics, expect_submission_sha256, official_tokens, expect_token_sha):
    rows = []

    if p2_metrics is not None:
        rows.append(cmp_abs_pp("mAP", report["metrics"]["mAP_pp"], p2_metrics["mean_ap"] * PP, 0.1))
        rows.append(cmp_abs_pp("NDS", report["metrics"]["NDS_pp"], p2_metrics["nd_score"] * PP, 0.1))
        for name in sorted(report["metrics"]["per_class_AP_pp"]):
            rows.append(cmp_abs_pp(
                "AP:%s" % name,
                report["metrics"]["per_class_AP_pp"][name],
                p2_metrics["mean_dist_aps"][name] * PP,
                0.1,
            ))
        for key in TP_ERROR_KEYS:
            rows.append(cmp_rel("tp_error:%s" % key, report["metrics"]["tp_errors"][key],
                                p2_metrics["tp_errors"][key], 0.001, 1e-4))
    else:
        rows.append(Row("wrapper_metrics", "UNVERIFIABLE",
                        {"reason": "--p2-metrics not supplied; no wrapper-reported values to compare"}))

    ours_sha = report["inputs"]["submission_sha256"]
    if expect_submission_sha256:
        rows.append(Row("submission_sha256", "PASS" if ours_sha == expect_submission_sha256 else "FAIL",
                        {"read_by_crosscheck": ours_sha, "written_by_wrapper": expect_submission_sha256}))
    else:
        rows.append(Row("submission_sha256", "UNVERIFIABLE",
                        {"reason": "--expect-submission-sha256 not supplied",
                         "read_by_crosscheck": ours_sha}))

    submission_tokens = report["_submission_tokens"]
    set_equal = submission_tokens == official_tokens
    detail = {"n_submission": len(submission_tokens), "n_official_split": len(official_tokens),
              "set_equal": set_equal}
    if not set_equal:
        detail["only_in_submission"] = sorted(submission_tokens - official_tokens)[:20]
        detail["only_in_official"] = sorted(official_tokens - submission_tokens)[:20]
    rows.append(Row("token_set", "PASS" if set_equal else "FAIL", detail))

    if expect_token_sha:
        ours_tok_sha = report["tokens"]["token_set_sha256"]
        rows.append(Row("token_set_sha256", "PASS" if ours_tok_sha == expect_token_sha else "FAIL",
                        {"crosscheck": ours_tok_sha, "wrapper_provenance": expect_token_sha}))

    # Wrapper-side pre-format prediction count: structurally unavailable from the frozen driver,
    # so this row can never close. It is reported UNVERIFIABLE rather than skipped, and the
    # default exit status is non-zero so it cannot pass unnoticed.
    rows.append(Row("wrapper_pred_count_vs_raw_json", "UNVERIFIABLE", {
        "reason": ("the evaluation wrapper records no pre-format prediction count: "
                   "research/night_gen_phase1/eval_by_condition.py writes only sample counts and "
                   "metrics to eval_by_condition.json, and mmdet3d runs NuScenesEval with "
                   "verbose=False (mmdet3d/datasets/nuscenes_dataset.py:405) so filter_eval_boxes' "
                   "staged box counts are never printed either"),
        "format_stage_transform": ("mmdet3d/datasets/nuscenes_dataset.py:622 lidar_nusc_box_to_global "
                                   "drops boxes with ego-frame radius > class_range (line 651)"),
        "why_delta_not_recomputable": ("the dropped boxes are absent from the submission JSON by "
                                       "construction, and the pre-format count is not a constant: the "
                                       "NMSFreeCoder applies score_threshold=0.05 and post_center_range "
                                       "before max_num=300 (configs/racformer_r50_nuimg_704x256_f8.py:168-174)"),
        "one_sided_check_performed": "format_stage_keeprule",
        "raw_json_total": report["box_counts"]["raw_json_total"],
    }))

    keeprule = report["box_counts"]["format_stage_keeprule"]
    rows.append(Row("format_stage_keeprule", "PASS" if keeprule["violations"] == 0 else "FAIL", {
        "note": ("one-sided: every surviving box must satisfy the format-stage keep rule "
                 "(ego-FRAME radius <= class_range). Cannot prove nothing extra was dropped."),
        "convention": keeprule["convention"],
        "violations": keeprule["violations"],
        "worst_excess_m": keeprule["worst_excess_m"],
        "violations_under_global_delta_convention": keeprule["violations_global_delta"],
    }))

    devkit_n = report["box_counts"]["devkit_postfilter_total"]
    recomputed_n = report["box_counts"]["recomputed_postfilter_total"]
    rows.append(Row("devkit_postfilter_count", "PASS" if devkit_n == recomputed_n else "FAIL",
                    {"devkit_internal": devkit_n, "recomputed": recomputed_n,
                     "per_class_mismatches": report["box_counts"]["postfilter_per_class_mismatches"]}))

    repl = report["counts_at_dist_th_tp"]["match_replication_check"]
    rows.append(Row("match_replication", "PASS" if repl["ok"] else "FAIL",
                    {"max_abs_ap_diff": repl["max_abs_diff"], "tol": repl["tol"],
                     "note": "TP/FP/FN counts are only trustworthy if the replicated matching "
                             "reproduces the devkit's own AP at dist_th_tp"}))

    return rows


# --- Main --------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--submission", required=True, help="path to results_nusc.json")
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--split", required=True, choices=["val"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nuscenes-version", default="v1.0-trainval")
    ap.add_argument("--venv", default="~/venvs/nusc-devkit-check")
    ap.add_argument("--p2-metrics", default=None,
                    help="wrapper-written metrics_summary.json to compare against")
    ap.add_argument("--expect-submission-sha256", default=None)
    ap.add_argument("--expect-token-set-sha256", default=None)
    ap.add_argument("--freeze-thresholds", default=None,
                    help="write the per-class max-F1 thresholds computed from THIS run to this path "
                         "(clean reference run only; the file is then committed and reused)")
    ap.add_argument("--thresholds", default=None,
                    help="read frozen per-class max-F1 thresholds from this path instead of computing")
    ap.add_argument("--allow-unverifiable-rows", action="store_true",
                    help="exit 0 when the only non-passing rows are UNVERIFIABLE (default: fail closed)")
    args = ap.parse_args()

    devkit_path = _assert_isolated_interpreter(args.venv)

    if args.freeze_thresholds and args.thresholds:
        raise SystemExit("[devkit_crosscheck] FATAL: --freeze-thresholds and --thresholds are exclusive.")

    os.makedirs(args.out_dir, exist_ok=True)

    import numpy as np  # noqa: WPS433
    from nuscenes import NuScenes  # noqa: WPS433
    from nuscenes.eval.detection.config import config_factory  # noqa: WPS433
    from nuscenes.eval.detection.evaluate import DetectionEval  # noqa: WPS433

    started = time.time()
    eval_cfg = config_factory("detection_cvpr_2019")
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.dataroot, verbose=False)

    raw = raw_json_counts(args.submission)
    official_tokens = official_split_tokens(nusc, args.split)

    print("[devkit_crosscheck] running official DetectionEval ...", flush=True)
    nusc_eval = DetectionEval(nusc, config=eval_cfg, result_path=os.path.abspath(args.submission),
                              eval_set=args.split, output_dir=os.path.join(args.out_dir, "devkit_eval"),
                              verbose=True)
    metrics, _md_list = nusc_eval.evaluate()
    serialized = metrics.serialize()

    devkit_postfilter_per_class = {}
    for token in nusc_eval.pred_boxes.sample_tokens:
        for box in nusc_eval.pred_boxes[token]:
            devkit_postfilter_per_class[box.detection_name] = (
                devkit_postfilter_per_class.get(box.detection_name, 0) + 1
            )
    devkit_postfilter_total = sum(devkit_postfilter_per_class.values())

    print("[devkit_crosscheck] recomputing the devkit's own post-filter count ...", flush=True)
    recomputed = recompute_devkit_postfilter(nusc, args.submission, eval_cfg)

    # One-sided check on the wrapper's format-stage keep rule.
    #
    # Two different "distance to ego" conventions are in play and they are NOT interchangeable:
    #   * the devkit filters on DetectionBox.ego_dist, the xy-norm of the GLOBAL-frame difference
    #     (data_classes.py:54-56) -- that is what recompute_devkit_postfilter uses;
    #   * mmdet3d's format stage filters on the xy-norm of the box centre expressed in the EGO
    #     frame (nuscenes_dataset.py:650), i.e. after undoing the ego2global rotation.
    # Because ego2global carries roll and pitch, the two norms differ by centimetres. Checking the
    # format-stage rule with the devkit's convention produces spurious violations, so the ego
    # rotation is undone here. The global-delta count is kept alongside as a diagnostic.
    from pyquaternion import Quaternion  # noqa: WPS433

    keeprule_violations = 0
    keeprule_violations_global = 0
    worst_excess = 0.0
    with open(args.submission) as f:
        results_raw = json.load(f)["results"]
    for token, boxes in results_raw.items():
        if not boxes:
            continue
        sample_rec = nusc.get("sample", token)
        sd_record = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", sd_record["ego_pose_token"])
        ego_xyz = np.array(pose["translation"])
        ego_rot_inv = Quaternion(pose["rotation"]).inverse.rotation_matrix
        for box in boxes:
            delta = np.array(box["translation"]) - ego_xyz
            limit = eval_cfg.class_range[box["detection_name"]]
            radius_ego = float(np.linalg.norm(ego_rot_inv.dot(delta)[:2], 2))
            if radius_ego > limit:
                keeprule_violations += 1
                worst_excess = max(worst_excess, radius_ego - limit)
            if float(np.linalg.norm(delta[:2], 2)) > limit:
                keeprule_violations_global += 1

    print("[devkit_crosscheck] replicating matching at dist_th_tp=%s ..." % eval_cfg.dist_th_tp, flush=True)
    dist_th_tp = eval_cfg.dist_th_tp
    matches = {}
    ap_diffs = {}
    for class_name in eval_cfg.class_names:
        match = match_class(nusc_eval.gt_boxes, nusc_eval.pred_boxes, class_name,
                            eval_cfg.dist_fcn_callable, dist_th_tp)
        matches[class_name] = match
        ours_ap = ap_replicated(match, eval_cfg.min_recall, eval_cfg.min_precision)
        theirs_ap = metrics.get_label_ap(class_name, dist_th_tp)
        ap_diffs[class_name] = abs(ours_ap - theirs_ap)

    repl_tol = 1e-9
    max_ap_diff = max(ap_diffs.values()) if ap_diffs else 0.0

    if args.thresholds:
        with open(args.thresholds) as f:
            frozen = json.load(f)
        thresholds = {k: (float("inf") if v is None else float(v)) for k, v in frozen["thresholds"].items()}
        thresholds_source = "frozen:%s" % os.path.abspath(args.thresholds)
        thresholds_f1 = frozen.get("f1_at_threshold", {})
    else:
        thresholds = {}
        thresholds_f1 = {}
        for class_name in eval_cfg.class_names:
            threshold, f1 = max_f1_threshold(matches[class_name])
            thresholds[class_name] = threshold
            thresholds_f1[class_name] = f1
        thresholds_source = "computed_from_this_run"

    op_i = {c: counts_at_threshold(matches[c], -float("inf")) for c in eval_cfg.class_names}
    op_ii = {c: counts_at_threshold(matches[c], thresholds[c]) for c in eval_cfg.class_names}

    submission_tokens = raw["tokens"]
    # Per-class AP is the mean over the four matching distances, exactly as DetectionMetrics
    # .mean_dist_aps computes it (data_classes.py:221-223).
    per_class_ap_pp = {c: float(np.mean(list(serialized["label_aps"][c].values()))) * PP
                       for c in eval_cfg.class_names}

    per_class_mismatches = {}
    all_names = set(devkit_postfilter_per_class) | set(recomputed["per_class"])
    for name in sorted(all_names):
        a = devkit_postfilter_per_class.get(name, 0)
        b = recomputed["per_class"].get(name, 0)
        if a != b:
            per_class_mismatches[name] = {"devkit_internal": a, "recomputed": b}

    report = OrderedDict()
    report["schema"] = "devkit_crosscheck/1"
    report["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["elapsed_s"] = None  # filled at the end
    report["interpreter"] = OrderedDict([
        ("path", sys.executable),
        ("version", sys.version.split()[0]),
        ("isolated", bool(sys.flags.isolated)),
        ("no_user_site", bool(sys.flags.no_user_site)),
        ("PYTHONNOUSERSITE", os.environ.get("PYTHONNOUSERSITE")),
        ("PYTHONPATH", os.environ.get("PYTHONPATH")),
    ])
    report["devkit"] = OrderedDict([
        ("module_path", devkit_path),
        ("version", _devkit_version()),
        ("venv_root", os.path.realpath(os.path.expanduser(args.venv))),
    ])
    report["inputs"] = OrderedDict([
        ("submission", os.path.abspath(args.submission)),
        ("submission_sha256", sha256_file(args.submission)),
        ("submission_meta", raw["meta"]),
        ("dataroot", os.path.abspath(args.dataroot)),
        ("split", args.split),
        ("nuscenes_version", args.nuscenes_version),
        ("eval_config", "detection_cvpr_2019"),
    ])
    report["tokens"] = OrderedDict([
        ("n_submission_tokens", len(submission_tokens)),
        ("n_official_split_tokens", len(official_tokens)),
        ("set_equal", submission_tokens == official_tokens),
        ("token_set_sha256", token_set_sha256(submission_tokens)),
        ("official_token_set_sha256", token_set_sha256(official_tokens)),
    ])
    report["box_counts"] = OrderedDict([
        ("raw_json_total", raw["total"]),
        ("raw_json_per_class", raw["per_class"]),
        ("raw_json_max_boxes_per_sample", raw["max_boxes_per_sample"]),
        ("max_boxes_per_sample_limit", eval_cfg.max_boxes_per_sample),
        ("samples_over_limit", recomputed["samples_over_max_boxes_per_sample"]),
        ("devkit_postfilter_total", devkit_postfilter_total),
        ("devkit_postfilter_per_class", devkit_postfilter_per_class),
        ("recomputed_postfilter_total", recomputed["total"]),
        ("recomputed_postfilter_per_class", recomputed["per_class"]),
        ("recomputed_stages", recomputed["stages"]),
        ("postfilter_per_class_mismatches", per_class_mismatches),
        ("format_stage_keeprule", {
            "convention": "xy-norm of the box centre in the EGO frame (mmdet3d nuscenes_dataset.py:650)",
            "violations": keeprule_violations,
            "worst_excess_m": worst_excess,
            "violations_global_delta": keeprule_violations_global,
            "note_global_delta": ("same boxes measured with the devkit's ego_dist convention "
                                  "(global-frame xy delta); a nonzero count here with zero above is "
                                  "expected and is only the ego roll/pitch difference"),
        }),
    ])
    report["metrics"] = OrderedDict([
        ("mAP_pp", serialized["mean_ap"] * PP),
        ("NDS_pp", serialized["nd_score"] * PP),
        ("per_class_AP_pp", per_class_ap_pp),
        ("tp_errors", serialized["tp_errors"]),
        ("per_class_tp_errors", serialized["label_tp_errors"]),
        ("label_aps_by_dist_th", serialized["label_aps"]),
        ("raw_fractions", {"mean_ap": serialized["mean_ap"], "nd_score": serialized["nd_score"]}),
        ("eval_time_s", serialized.get("eval_time")),
    ])
    report["counts_at_dist_th_tp"] = OrderedDict([
        ("dist_th", dist_th_tp),
        ("rationale", "dist_th_tp is the threshold the devkit itself uses for the TP-error metrics"),
        ("operating_point_i_no_score_cut", op_i),
        ("operating_point_ii_max_f1", op_ii),
        ("max_f1_thresholds", {k: _jsonable(v) for k, v in thresholds.items()}),
        ("max_f1_values", thresholds_f1),
        ("thresholds_source", thresholds_source),
        ("near_zero_recall_at_op_i", {c: (op_i[c] is not None and op_i[c]["recall"] < 0.01)
                                      for c in eval_cfg.class_names}),
        ("match_replication_check", {"per_class_abs_ap_diff": ap_diffs, "max_abs_diff": max_ap_diff,
                                     "tol": repl_tol, "ok": max_ap_diff <= repl_tol}),
    ])
    report["_submission_tokens"] = submission_tokens  # internal, stripped before writing

    p2_metrics = None
    if args.p2_metrics:
        with open(args.p2_metrics) as f:
            p2_metrics = json.load(f)

    rows = build_comparison(report, p2_metrics, args.expect_submission_sha256, official_tokens,
                            args.expect_token_set_sha256)
    del report["_submission_tokens"]

    n_fail = sum(1 for r in rows if r.status == "FAIL")
    n_unver = sum(1 for r in rows if r.status == "UNVERIFIABLE")
    if n_fail:
        verdict = "FAIL"
    elif n_unver:
        verdict = "PASS_WITH_UNVERIFIABLE"
    else:
        verdict = "PASS"

    report["comparison"] = [r.as_dict() for r in rows]
    report["verdict"] = verdict
    report["n_failed_rows"] = n_fail
    report["n_unverifiable_rows"] = n_unver
    report["elapsed_s"] = time.time() - started

    out_path = os.path.join(args.out_dir, "crosscheck.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=False)
    print("[devkit_crosscheck] wrote %s" % out_path)

    if args.freeze_thresholds:
        frozen = OrderedDict([
            ("schema", "maxf1_thresholds/1"),
            ("note", "null threshold means +inf (class had no predictions); all counts are 0 there"),
            ("dist_th", dist_th_tp),
            ("source_submission_sha256", report["inputs"]["submission_sha256"]),
            ("source_crosscheck", out_path),
            ("generated_utc", report["generated_utc"]),
            ("thresholds", {k: _jsonable(v) for k, v in thresholds.items()}),
            ("f1_at_threshold", thresholds_f1),
        ])
        with open(args.freeze_thresholds, "w") as f:
            json.dump(frozen, f, indent=2)
        print("[devkit_crosscheck] froze max-F1 thresholds to %s" % args.freeze_thresholds)

    print("\n=== comparison rows ===")
    for row in rows:
        print("  [%-14s] %s" % (row.status, row.name))
        if row.status != "PASS":
            for key, value in row.detail.items():
                print("      %s: %s" % (key, value))
    print("VERDICT: %s (%d failed, %d unverifiable)" % (verdict, n_fail, n_unver))

    if n_fail:
        sys.exit(1)
    if n_unver and not args.allow_unverifiable_rows:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
