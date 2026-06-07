"""
T5 — Phase 1 condition-aware evaluation wrapper.

Wraps the standard `dataset.evaluate(results)` from val.py to also report mAP/NDS and
per-class AP for each condition split: {day, night, rain, day_matched}, by re-running the
nuScenes detection eval restricted to the relevant sample tokens.

Hard rule: this script will ABORT if `cfg.data.val.max_samples` is set to a finite value
AND `--full-val` is not passed. The 300-sample in-training val is for fast feedback only;
reported metrics MUST be computed on the full 6 019-sample val.

Usage (cluster, racformerfix env):
    conda run -n racformerfix python research/night_gen_phase1/eval_by_condition.py \
        --config configs/racformer_eval_fullval_research.py \
        --weights checkpoints/racformer_r50_f8.pth \
        --out-dir research/night_gen_phase1/reports/eval_by_condition

Or override at runtime:
    conda run -n racformerfix python research/night_gen_phase1/eval_by_condition.py \
        --config configs/racformer_train2k_day_research.py --full-val \
        --weights outputs/<run>/epoch_12.pth \
        --out-dir research/night_gen_phase1/reports/eval_by_condition
"""

import argparse
import importlib
import json
import logging
import os
import sys

# Repo root must be on sys.path so `import models` / `import loaders` resolve.
# This script sits at <repo>/research/night_gen_phase1/, so go two levels up
# (matches the pattern used by smoke_build_datasets.py).
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _coerce_full_val(cfg):
    """Force full-val data block when --full-val is passed."""
    cfg.data.val.ann_file = (
        "/srv/nfs/shared/gnmp/RaCFormer/nuscenes_infos_val_sweep.pkl"
    )
    cfg.data.val.max_samples = None


def _abort_if_capped_val(cfg, full_val_flag):
    cap = getattr(cfg.data.val, "max_samples", None)
    if cap is not None and not full_val_flag:
        print(
            f"[eval_by_condition] FATAL: cfg.data.val.max_samples = {cap} but --full-val "
            "was not passed. Reported Phase 1 metrics MUST be computed on the full val. "
            "Pass --full-val or use configs/racformer_eval_fullval_research.py.",
            file=sys.stderr,
        )
        sys.exit(2)


def _build_token_to_condition_map(dataset):
    """Map sample_token -> {'day','night','rain'} from the dataset's data_infos."""
    out = {}
    for info in dataset.data_infos:
        # CustomNuScenesDataset_radar.get_data_info already plumbs scene_condition; the
        # info dict itself may carry it once tagged by build_phase1_pkls.py. Fall back
        # to the dataset method if a token isn't pre-tagged.
        cond = info.get("scene_condition")
        if cond is None and hasattr(dataset, "_get_scene_condition"):
            try:
                cond = dataset._get_scene_condition(info["token"])
            except Exception:  # noqa: BLE001
                cond = "unknown"
        out[info["token"]] = cond or "unknown"
    return out


def _build_day_matched_set(token_to_cond, dataset, nusc=None):
    """Day-condition tokens whose scene shares a location with at least one night token."""
    if nusc is None:
        from nuscenes.nuscenes import NuScenes  # noqa: WPS433

        nusc = NuScenes(
            version="v1.0-trainval", dataroot="data/nuscenes/", verbose=False
        )
    night_locations = set()
    night_scenes = set()
    for info in dataset.data_infos:
        if token_to_cond.get(info["token"]) == "night":
            night_scenes.add(info["scene_token"])
            try:
                scene = nusc.get("scene", info["scene_token"])
                log = nusc.get("log", scene["log_token"])
                night_locations.add(log.get("location"))
            except Exception:  # noqa: BLE001
                pass
    matched = set()
    for info in dataset.data_infos:
        tok = info["token"]
        if token_to_cond.get(tok) != "day":
            continue
        if info["scene_token"] in night_scenes:
            matched.add(tok)
            continue
        try:
            scene = nusc.get("scene", info["scene_token"])
            log = nusc.get("log", scene["log_token"])
            if log.get("location") in night_locations:
                matched.add(tok)
        except Exception:  # noqa: BLE001
            pass
    return matched


def _filter_eval_boxes_by_tokens(boxes, allowed_tokens):
    from nuscenes.eval.common.data_classes import EvalBoxes  # noqa: WPS433

    out = EvalBoxes()
    for tok in boxes.sample_tokens:
        if tok in allowed_tokens:
            out.add_boxes(tok, list(boxes[tok]))
    return out


def _make_subset_evaluator(
    nusc, eval_cfg, pred_boxes, gt_boxes, output_dir, verbose=False
):
    """NuScenesEval that takes pre-loaded EvalBoxes and skips the strict pred==gt assertion.

    Replicates DetectionEval.__init__'s post-load steps (add_center_dist + filter_eval_boxes)
    so per-condition splits with pred ⊊ full-val gt evaluate correctly. Verified bit-exact on
    the unfiltered overall set against dataset.evaluate (Δ_mAP=0, Δ_NDS=0) by
    eval_per_condition_post.py.
    """
    from nuscenes.eval.common.loaders import (  # noqa: WPS433
        add_center_dist,
        filter_eval_boxes,
    )
    from nuscenes.eval.detection.evaluate import NuScenesEval  # noqa: WPS433

    class _PreloadedEval(NuScenesEval):
        def __init__(self):
            self.nusc = nusc
            self.cfg = eval_cfg
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
                nusc, self.pred_boxes, eval_cfg.class_range, verbose=verbose
            )
            self.gt_boxes = filter_eval_boxes(
                nusc, self.gt_boxes, eval_cfg.class_range, verbose=verbose
            )
            self.sample_tokens = self.gt_boxes.sample_tokens

    return _PreloadedEval()


def _eval_subset(
    dataset, results, keep_indices, label, out_dir, *, nusc, eval_cfg, gt_full
):
    """Format predictions for a per-condition subset and run nuScenes detection eval.

    Bypasses dataset.evaluate's call into NuScenesEval.__init__, which asserts that
    pred sample tokens == gt sample tokens loaded for `eval_set='val'`. For per-condition
    subsets pred ⊊ gt, so the assertion fires. Instead, format predictions for the subset,
    load them via load_prediction, filter gt_full to the same tokens, and run an evaluator
    with the strict assertion skipped.
    """
    if not keep_indices:
        return None
    sub_results = [results[i] for i in keep_indices]
    sub_data_infos = [dataset.data_infos[i] for i in keep_indices]

    saved = dataset.data_infos
    dataset.data_infos = sub_data_infos
    try:
        prefix = os.path.join(out_dir, f"submission_{label}")
        os.makedirs(out_dir, exist_ok=True)
        result_files, _tmp = dataset.format_results(sub_results, jsonfile_prefix=prefix)
    finally:
        dataset.data_infos = saved

    from nuscenes.eval.common.loaders import load_prediction  # noqa: WPS433
    from nuscenes.eval.detection.data_classes import DetectionBox  # noqa: WPS433

    if isinstance(result_files, dict):
        result_path = result_files.get("pts_bbox", next(iter(result_files.values())))
    else:
        result_path = result_files

    pred_boxes, _meta = load_prediction(
        result_path, eval_cfg.max_boxes_per_sample, DetectionBox, verbose=False
    )
    allowed_tokens = {info["token"] for info in sub_data_infos}
    gt_split = _filter_eval_boxes_by_tokens(gt_full, allowed_tokens)

    split_eval_dir = os.path.join(out_dir, f"eval_{label}")
    evaluator = _make_subset_evaluator(
        nusc, eval_cfg, pred_boxes, gt_split, split_eval_dir, verbose=False
    )
    summary = evaluator.main(plot_examples=0, render_curves=False)
    summary["n_pred_samples"] = len(pred_boxes.sample_tokens)
    summary["n_gt_samples"] = len(gt_split.sample_tokens)
    summary["label"] = label
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--full-val", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    import torch  # noqa: WPS433
    import torch.backends.cudnn as cudnn  # noqa: WPS433
    from mmcv import Config  # noqa: WPS433
    from mmcv.parallel import MMDataParallel  # noqa: WPS433
    from mmcv.runner import load_checkpoint  # noqa: WPS433
    from mmdet.apis import set_random_seed, single_gpu_test  # noqa: WPS433
    from mmdet3d.datasets import build_dataset, build_dataloader  # noqa: WPS433
    from mmdet3d.models import build_model  # noqa: WPS433

    cfg = Config.fromfile(args.config)
    if args.full_val:
        _coerce_full_val(cfg)
    _abort_if_capped_val(cfg, args.full_val)

    # Custom modules must be imported so registry decorators run.
    importlib.import_module("models")
    importlib.import_module("loaders")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info(
        "[eval_by_condition] config=%s weights=%s val_pkl=%s val_cap=%s",
        args.config,
        args.weights,
        cfg.data.val.ann_file,
        getattr(cfg.data.val, "max_samples", None),
    )

    assert torch.cuda.is_available(), "CUDA required; never fall back to CPU."
    torch.cuda.set_device(0)
    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True

    val_dataset = build_dataset(cfg.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    logging.info("[eval_by_condition] building model %s", cfg.model.type)
    model = build_model(cfg.model)
    model = MMDataParallel(model.cuda(), [0])
    load_checkpoint(model, args.weights, map_location="cuda", strict=True)

    logging.info(
        "[eval_by_condition] running inference on %d samples", len(val_dataset)
    )
    results = single_gpu_test(model, val_loader)

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Overall ------------------------------------------------------
    overall = val_dataset.evaluate(
        results, jsonfile_prefix=os.path.join(args.out_dir, "submission_overall")
    )

    # ---- Pre-load NuScenes / eval cfg / full-val GT once for per-split eval ----
    from nuscenes.eval.common.loaders import load_gt  # noqa: WPS433
    from nuscenes.eval.detection.config import config_factory  # noqa: WPS433
    from nuscenes.eval.detection.data_classes import DetectionBox  # noqa: WPS433
    from nuscenes.nuscenes import NuScenes  # noqa: WPS433

    logging.info("[eval_by_condition] loading NuScenes for per-split eval")
    nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuscenes/", verbose=False)
    eval_cfg = config_factory("detection_cvpr_2019")
    logging.info("[eval_by_condition] loading GT for eval_set=val")
    gt_full = load_gt(nusc, "val", DetectionBox, verbose=False)

    # ---- Condition splits --------------------------------------------
    token_to_cond = _build_token_to_condition_map(val_dataset)
    by_split = {"day": [], "night": [], "rain": []}
    for i, info in enumerate(val_dataset.data_infos):
        cond = token_to_cond.get(info["token"], "unknown")
        if cond in by_split:
            by_split[cond].append(i)

    day_matched_tokens = _build_day_matched_set(token_to_cond, val_dataset, nusc=nusc)
    by_split["day_matched"] = [
        i
        for i, info in enumerate(val_dataset.data_infos)
        if info["token"] in day_matched_tokens
    ]

    split_metrics = {}
    for label, idx in by_split.items():
        logging.info("[eval_by_condition] split %s n=%d", label, len(idx))
        m = _eval_subset(
            val_dataset,
            results,
            idx,
            label,
            args.out_dir,
            nusc=nusc,
            eval_cfg=eval_cfg,
            gt_full=gt_full,
        )
        split_metrics[label] = m

    out = {
        "config": args.config,
        "weights": args.weights,
        "n_total": len(val_dataset),
        "split_counts": {k: len(v) for k, v in by_split.items()},
        "overall": overall,
        "splits": split_metrics,
    }
    out_path = os.path.join(args.out_dir, "eval_by_condition.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logging.info("[eval_by_condition] wrote %s", out_path)


if __name__ == "__main__":
    main()
