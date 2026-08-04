"""
Phase 1 post-processing — recompute condition-aware mAP/NDS from existing predictions.

Reuses the 6019-sample full-val results_nusc.json produced by eval_by_condition.py's overall
pass. Filters predictions and GT to per-condition sample-token sets (day/night/rain), then runs
the standard nuScenes detection metric pipeline directly. Sidesteps the NuScenesEval assertion
that crashes the in-place per-split path inside eval_by_condition.py.

Self-check: also recomputes overall metrics from the same predictions; aborts (exit 2) if the
recomputed overall mean_ap or nd_score deviates from the existing metrics_summary.json by more
than --tolerance (default 0.001). If the self-check fails, no per-split metrics are trusted and
nothing further is written.

Usage (cluster, racformerfix env, cwd = /srv/nfs/shared/gnmp/RaCFormer):
    conda run -n racformerfix --no-capture-output python -u \
      research/night_gen_phase1/eval_per_condition_post.py \
        --pred research/night_gen_phase1/reports/eval_by_condition/submission_overall/pts_bbox/results_nusc.json \
        --val-pkl /srv/nfs/shared/gnmp/RaCFormer/nuscenes_infos_val_sweep.pkl \
        --reference-summary research/night_gen_phase1/reports/eval_by_condition/submission_overall/pts_bbox/metrics_summary.json \
        --out-dir research/night_gen_phase1/reports/eval_by_condition_post \
        --dataroot data/nuscenes/ \
        --version v1.0-trainval
"""

import argparse
import json
import logging
import os
import pickle
import sys

# Repo root must be on sys.path so any incidental loaders import resolves.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _build_token_to_condition(infos, nusc):
    """Map sample_token -> {'day','night','rain','unknown'}.

    Prefer pre-tagged `scene_condition` on each info; fall back to scene-description parsing
    (the same logic build_phase1_pkls.py uses).
    """
    out = {}
    for info in infos:
        tok = info["token"]
        cond = info.get("scene_condition")
        if cond is None:
            try:
                scene = nusc.get("scene", info["scene_token"])
                desc = scene.get("description", "").lower()
                # Match build_phase1_pkls.get_scene_condition: night > rain > day.
                if "night" in desc:
                    cond = "night"
                elif "rain" in desc or "rainy" in desc:
                    cond = "rain"
                else:
                    cond = "day"
            except Exception:  # noqa: BLE001
                cond = "unknown"
        out[tok] = cond
    return out


def _filter_by_tokens(boxes, allowed_tokens):
    from nuscenes.eval.common.data_classes import EvalBoxes

    out = EvalBoxes()
    for tok in boxes.sample_tokens:
        if tok in allowed_tokens:
            out.add_boxes(tok, list(boxes[tok]))
    return out


def _build_subset_evaluator(nusc, cfg, pred_boxes, gt_boxes, output_dir, verbose):
    """Construct a NuScenesEval-equivalent that takes pre-loaded EvalBoxes.

    Replicates DetectionEval.__init__ post-load steps (add_center_dist + filter_eval_boxes) but
    skips load_prediction/load_gt and the strict sample-token-set equality assertion.
    """
    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
    from nuscenes.eval.detection.evaluate import NuScenesEval

    class _PreloadedEval(NuScenesEval):
        def __init__(self):
            self.nusc = nusc
            self.cfg = cfg
            self.eval_set = "val"
            self.output_dir = output_dir
            self.verbose = verbose
            self.result_path = ""
            self.meta = {}
            os.makedirs(output_dir, exist_ok=True)
            self.plot_dir = os.path.join(output_dir, "plots")
            os.makedirs(self.plot_dir, exist_ok=True)

            self.pred_boxes = pred_boxes
            self.gt_boxes = gt_boxes
            self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
            self.gt_boxes = add_center_dist(nusc, self.gt_boxes)
            self.pred_boxes = filter_eval_boxes(
                nusc, self.pred_boxes, cfg.class_range, verbose=verbose
            )
            self.gt_boxes = filter_eval_boxes(
                nusc, self.gt_boxes, cfg.class_range, verbose=verbose
            )
            self.sample_tokens = self.gt_boxes.sample_tokens

    return _PreloadedEval()


def _evaluate_split(nusc, cfg, pred_full, gt_full, allowed, base_dir, label, verbose):
    """Filter (or pass-through for overall) and run the NuScenesEval metric pipeline."""
    if allowed is None:
        pred_split, gt_split = pred_full, gt_full
    else:
        pred_split = _filter_by_tokens(pred_full, allowed)
        gt_split = _filter_by_tokens(gt_full, allowed)

    if len(pred_split.sample_tokens) == 0 or len(gt_split.sample_tokens) == 0:
        return {
            "label": label,
            "n_pred_samples": len(pred_split.sample_tokens),
            "n_gt_samples": len(gt_split.sample_tokens),
            "skipped": True,
            "reason": "empty split",
        }

    split_dir = os.path.join(base_dir, label)
    os.makedirs(split_dir, exist_ok=True)
    evaluator = _build_subset_evaluator(
        nusc, cfg, pred_split, gt_split, split_dir, verbose=verbose
    )
    summary = evaluator.main(plot_examples=0, render_curves=False)
    summary["n_pred_samples"] = len(pred_split.sample_tokens)
    summary["n_gt_samples"] = len(gt_split.sample_tokens)
    summary["label"] = label
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred",
        required=True,
        help="Path to results_nusc.json (full-val predictions).",
    )
    ap.add_argument(
        "--val-pkl",
        required=True,
        help="Path to nuscenes_infos_val_sweep.pkl for token-to-condition mapping.",
    )
    ap.add_argument(
        "--reference-summary",
        required=True,
        help="Existing overall metrics_summary.json for self-check.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataroot", default="data/nuscenes/")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--tolerance", type=float, default=0.001)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("eval_per_condition_post")

    from nuscenes.eval.common.loaders import load_gt, load_prediction
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.nuscenes import NuScenes

    log.info("loading NuScenes (%s, dataroot=%s)", args.version, args.dataroot)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    cfg = config_factory("detection_cvpr_2019")

    log.info("loading predictions %s", args.pred)
    pred_full, _meta = load_prediction(
        args.pred, cfg.max_boxes_per_sample, DetectionBox, verbose=args.verbose
    )
    log.info("pred sample_tokens: %d", len(pred_full.sample_tokens))

    log.info("loading GT for eval_set=val")
    gt_full = load_gt(nusc, "val", DetectionBox, verbose=args.verbose)
    log.info("gt sample_tokens: %d", len(gt_full.sample_tokens))

    log.info("loading val pkl %s", args.val_pkl)
    with open(args.val_pkl, "rb") as f:
        val_data = pickle.load(f)
    if isinstance(val_data, dict) and "infos" in val_data:
        infos = val_data["infos"]
    else:
        infos = val_data
    log.info("val pkl infos: %d", len(infos))

    cond_map = _build_token_to_condition(infos, nusc)
    day_tokens = {t for t, c in cond_map.items() if c == "day"}
    night_tokens = {t for t, c in cond_map.items() if c == "night"}
    rain_tokens = {t for t, c in cond_map.items() if c == "rain"}
    unknown_tokens = {t for t, c in cond_map.items() if c == "unknown"}
    log.info(
        "split counts day=%d night=%d rain=%d unknown=%d total_in_pkl=%d",
        len(day_tokens),
        len(night_tokens),
        len(rain_tokens),
        len(unknown_tokens),
        len(infos),
    )

    os.makedirs(args.out_dir, exist_ok=True)

    log.info("computing overall postprocess (self-check)")
    overall_summary = _evaluate_split(
        nusc, cfg, pred_full, gt_full, None, args.out_dir, "overall", args.verbose
    )

    with open(args.reference_summary) as f:
        ref = json.load(f)
    ref_map = float(ref["mean_ap"])
    ref_nds = float(ref["nd_score"])
    new_map = float(overall_summary["mean_ap"])
    new_nds = float(overall_summary["nd_score"])
    delta_map = abs(new_map - ref_map)
    delta_nds = abs(new_nds - ref_nds)
    log.info(
        "overall self-check: ref(mAP=%.4f, NDS=%.4f) post(mAP=%.4f, NDS=%.4f) "
        "Δ_mAP=%.5f Δ_NDS=%.5f tol=%.4f",
        ref_map,
        ref_nds,
        new_map,
        new_nds,
        delta_map,
        delta_nds,
        args.tolerance,
    )
    self_check = {
        "passed": delta_map <= args.tolerance and delta_nds <= args.tolerance,
        "ref": {"mean_ap": ref_map, "nd_score": ref_nds},
        "post": {"mean_ap": new_map, "nd_score": new_nds},
        "delta_mean_ap": delta_map,
        "delta_nd_score": delta_nds,
        "tolerance": args.tolerance,
    }
    if not self_check["passed"]:
        log.error(
            "overall postprocess does NOT match reference within tol=%.4f. ABORTING.",
            args.tolerance,
        )
        out = {
            "self_check": self_check,
            "split_counts": {
                "overall_pred": len(pred_full.sample_tokens),
                "overall_gt": len(gt_full.sample_tokens),
                "day": len(day_tokens),
                "night": len(night_tokens),
                "rain": len(rain_tokens),
                "day_matched": 0,
            },
            "splits": {"overall": overall_summary},
        }
        with open(os.path.join(args.out_dir, "eval_per_condition_post.json"), "w") as f:
            json.dump(out, f, indent=2, default=float)
        sys.exit(2)

    splits = {"overall": overall_summary}
    for label, allowed in (
        ("day", day_tokens),
        ("night", night_tokens),
        ("rain", rain_tokens),
    ):
        log.info("computing split %s (n=%d)", label, len(allowed))
        splits[label] = _evaluate_split(
            nusc, cfg, pred_full, gt_full, allowed, args.out_dir, label, args.verbose
        )
    splits["day_matched"] = {
        "label": "day_matched",
        "n_pred_samples": 0,
        "n_gt_samples": 0,
        "skipped": True,
        "reason": "empty split (Phase 1 accepts as N/A)",
    }

    out = {
        "self_check": self_check,
        "split_counts": {
            "overall_pred": len(pred_full.sample_tokens),
            "overall_gt": len(gt_full.sample_tokens),
            "day": len(day_tokens),
            "night": len(night_tokens),
            "rain": len(rain_tokens),
            "day_matched": 0,
        },
        "splits": splits,
    }
    with open(os.path.join(args.out_dir, "eval_per_condition_post.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    log.info("done; wrote %s/eval_per_condition_post.json", args.out_dir)


if __name__ == "__main__":
    main()
