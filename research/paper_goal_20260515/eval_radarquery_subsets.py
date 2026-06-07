"""Post-hoc radar/long-range subset evaluation for RaCFormer prediction JSONs.

This script is intended to run after `eval_by_condition.py` has written a standard
nuScenes `results_nusc.json`, for example:

    conda run -n racformerfix --no-capture-output python -u \
      research/paper_goal_20260515/eval_radarquery_subsets.py \
      --pred-json research/night_gen_phase1/results/S6_radarquery/eval/submission_overall/pts_bbox/results_nusc.json \
      --out-dir research/night_gen_phase1/results/S6_radarquery/subset_eval

It runs on existing prediction files only. It does not run model inference or mutate
datasets/checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


TokenSet = Sequence[str]
BoxPredicate = Callable[[object], bool]


DYNAMIC_CLASSES = {
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
}


def _load_infos(path: Path) -> List[dict]:
    with path.open("rb") as fh:
        data = pickle.load(fh)
    return data["infos"] if isinstance(data, dict) and "infos" in data else data


def _center_dist_xy(box: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(box[:2], dtype=float)))


def _build_sample_subsets(infos: List[dict]) -> Dict[str, List[str]]:
    radar_counts = []
    dyn_radar_counts = []
    far30_counts = []
    far40_counts = []
    moving_counts = []
    for info in infos:
        names = np.asarray(info["gt_names"])
        gt_boxes = np.asarray(info["gt_boxes"], dtype=float)
        num_radar = np.asarray(info["num_radar_pts"], dtype=float)
        gt_velocity = np.nan_to_num(np.asarray(info["gt_velocity"], dtype=float), nan=0.0)
        speed = np.linalg.norm(gt_velocity, axis=1)
        dynamic = np.isin(names, list(DYNAMIC_CLASSES))
        dists = np.asarray([_center_dist_xy(box) for box in gt_boxes], dtype=float)
        radar_counts.append(int((num_radar > 0).sum()))
        dyn_radar_counts.append(int(((num_radar > 0) & dynamic).sum()))
        far30_counts.append(int((dists >= 30.0).sum()))
        far40_counts.append(int((dists >= 40.0).sum()))
        moving_counts.append(int((speed > 0.5).sum()))

    def top_quartile_mask(values: Sequence[int]) -> List[bool]:
        positive = [v for v in values if v > 0]
        if not positive:
            return [False for _ in values]
        threshold = float(np.quantile(np.asarray(positive, dtype=float), 0.75))
        return [v >= threshold and v > 0 for v in values]

    tokens = [info["token"] for info in infos]
    subsets = {
        "all_samples": tokens,
        "radar_supported_any_sample": [
            tok for tok, count in zip(tokens, radar_counts) if count > 0
        ],
        "dynamic_radar_supported_any_sample": [
            tok for tok, count in zip(tokens, dyn_radar_counts) if count > 0
        ],
        "radar_rich_top_quartile_sample": [
            tok for tok, keep in zip(tokens, top_quartile_mask(radar_counts)) if keep
        ],
        "long_range_any_ge30_sample": [
            tok for tok, count in zip(tokens, far30_counts) if count > 0
        ],
        "long_range_any_ge40_sample": [
            tok for tok, count in zip(tokens, far40_counts) if count > 0
        ],
        "moving_any_gt_speed_gt_0p5_sample": [
            tok for tok, count in zip(tokens, moving_counts) if count > 0
        ],
    }
    return subsets


def _add_boxes_preserving_empty(
    eval_boxes,
    tokens: TokenSet,
    source_boxes,
    predicate: Optional[BoxPredicate] = None,
):
    for token in tokens:
        boxes = list(source_boxes[token]) if token in source_boxes.sample_tokens else []
        if predicate is not None:
            boxes = [box for box in boxes if predicate(box)]
        eval_boxes.add_boxes(token, boxes)
    return eval_boxes


def _subset_eval_boxes(source_boxes, tokens: TokenSet, predicate: Optional[BoxPredicate] = None):
    from nuscenes.eval.common.data_classes import EvalBoxes

    return _add_boxes_preserving_empty(EvalBoxes(), tokens, source_boxes, predicate)


def _distance_predicate(min_dist: Optional[float], max_dist: Optional[float]) -> BoxPredicate:
    def keep(box) -> bool:
        dist = getattr(box, "ego_dist", None)
        if dist is None:
            return True
        if min_dist is not None and dist < min_dist:
            return False
        if max_dist is not None and dist >= max_dist:
            return False
        return True

    return keep


def _make_preloaded_eval(nusc, eval_cfg, pred_boxes, gt_boxes, output_dir: Path):
    from nuscenes.eval.common.loaders import filter_eval_boxes
    from nuscenes.eval.detection.evaluate import NuScenesEval

    class PreloadedEval(NuScenesEval):
        def __init__(self):
            self.nusc = nusc
            self.cfg = eval_cfg
            self.eval_set = "val"
            self.output_dir = str(output_dir)
            self.verbose = False
            self.result_path = ""
            self.meta = {}
            os.makedirs(self.output_dir, exist_ok=True)
            self.plot_dir = os.path.join(self.output_dir, "plots")
            os.makedirs(self.plot_dir, exist_ok=True)
            self.pred_boxes = filter_eval_boxes(nusc, pred_boxes, eval_cfg.class_range, verbose=False)
            self.gt_boxes = filter_eval_boxes(nusc, gt_boxes, eval_cfg.class_range, verbose=False)
            self.sample_tokens = self.gt_boxes.sample_tokens

    return PreloadedEval()


def _run_subset(nusc, eval_cfg, name: str, pred_boxes, gt_boxes, out_dir: Path) -> dict:
    evaluator = _make_preloaded_eval(nusc, eval_cfg, pred_boxes, gt_boxes, out_dir / name)
    summary = evaluator.main(plot_examples=0, render_curves=False)
    summary["label"] = name
    summary["n_eval_tokens"] = len(evaluator.sample_tokens)
    summary["n_gt_boxes"] = sum(len(evaluator.gt_boxes[token]) for token in evaluator.sample_tokens)
    summary["n_pred_boxes"] = sum(len(evaluator.pred_boxes[token]) for token in evaluator.sample_tokens)
    return summary


def _metric_row(summary: dict) -> dict:
    return {
        "mean_ap": float(summary["mean_ap"]),
        "nd_score": float(summary["nd_score"]),
        "n_eval_tokens": int(summary["n_eval_tokens"]),
        "n_gt_boxes": int(summary["n_gt_boxes"]),
        "n_pred_boxes": int(summary["n_pred_boxes"]),
    }


def _write_markdown(path: Path, rows: Dict[str, dict]) -> None:
    lines = [
        "# Radar-Query Subset Evaluation",
        "",
        "| Subset | Samples | GT boxes | Pred boxes | mAP | NDS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        lines.append(
            "| {name} | {samples} | {gt} | {pred} | {map:.4f} | {nds:.4f} |".format(
                name=name,
                samples=row["n_eval_tokens"],
                gt=row["n_gt_boxes"],
                pred=row["n_pred_boxes"],
                map=row["mean_ap"],
                nds=row["nd_score"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-pkl", default="nuscenes_infos_val_sweep.pkl")
    parser.add_argument("--dataroot", default="data/nuscenes/")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--eval-set", default="val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.loaders import add_center_dist, load_gt, load_prediction
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.nuscenes import NuScenes

    infos = _load_infos(Path(args.val_pkl))
    all_tokens = [info["token"] for info in infos]
    sample_subsets = _build_sample_subsets(infos)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    eval_cfg = config_factory("detection_cvpr_2019")
    pred_full, meta = load_prediction(args.pred_json, eval_cfg.max_boxes_per_sample, DetectionBox, verbose=False)
    gt_full = load_gt(nusc, args.eval_set, DetectionBox, verbose=False)
    pred_full = add_center_dist(nusc, pred_full)
    gt_full = add_center_dist(nusc, gt_full)

    summaries = {}

    for name, tokens in sample_subsets.items():
        pred_subset = _subset_eval_boxes(pred_full, tokens)
        gt_subset = _subset_eval_boxes(gt_full, tokens)
        summaries[name] = _run_subset(nusc, eval_cfg, name, pred_subset, gt_subset, out_dir)

    object_splits: Dict[str, Tuple[Optional[float], Optional[float]]] = {
        "object_near_lt30m": (None, 30.0),
        "object_far_ge30m": (30.0, None),
        "object_far_ge40m": (40.0, None),
    }
    for name, (min_dist, max_dist) in object_splits.items():
        predicate = _distance_predicate(min_dist, max_dist)
        pred_subset = _subset_eval_boxes(pred_full, all_tokens, predicate)
        gt_subset = _subset_eval_boxes(gt_full, all_tokens, predicate)
        summaries[name] = _run_subset(nusc, eval_cfg, name, pred_subset, gt_subset, out_dir)

    rows = {name: _metric_row(summary) for name, summary in summaries.items()}
    payload = {
        "pred_json": args.pred_json,
        "val_pkl": args.val_pkl,
        "dataroot": args.dataroot,
        "meta": meta,
        "metrics": rows,
        "note": (
            "Sample subsets evaluate all objects in selected samples. Object-distance "
            "subsets filter both GT and predictions by nuScenes ego_dist. Radar-supported "
            "subsets are sample-level because prediction boxes do not carry GT radar support."
        ),
    }
    (out_dir / "subset_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_markdown(out_dir / "subset_metrics.md", rows)
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
