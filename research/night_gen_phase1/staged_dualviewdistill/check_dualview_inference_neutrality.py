#!/usr/bin/env python3
"""Check Branch G inference neutrality on one validation sample."""

import argparse
import copy
import importlib
import json
import sys
from pathlib import Path

import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import Config
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from loaders.builder import build_dataloader


def detach_logits(output):
    keys = ("all_cls_scores", "all_bbox_preds")
    return {key: output[key].detach().float().cpu() for key in keys if key in output}


def compare_logits(lhs, rhs):
    result = {"shape_match": True, "max_abs_diff": 0.0, "shapes": {}}
    for key in sorted(set(lhs) | set(rhs)):
        if key not in lhs or key not in rhs:
            result["shape_match"] = False
            result["shapes"][key] = {
                "baseline": None if key not in lhs else list(lhs[key].shape),
                "candidate": None if key not in rhs else list(rhs[key].shape),
            }
            continue
        result["shapes"][key] = {"baseline": list(lhs[key].shape), "candidate": list(rhs[key].shape)}
        if lhs[key].shape != rhs[key].shape:
            result["shape_match"] = False
            continue
        result["max_abs_diff"] = max(result["max_abs_diff"], float((lhs[key] - rhs[key]).abs().max().item()))
    result["within_tolerance"] = result["shape_match"] and result["max_abs_diff"] <= 1e-6
    return result


def prepare_cfg(config_path, mode):
    cfg = Config.fromfile(config_path)
    cfg = copy.deepcopy(cfg)
    cfg.batch_size = 1
    cfg.data.workers_per_gpu = 0
    cfg.data.val.max_samples = 1
    if mode == "baseline":
        cfg.model.pop("dualview_distill", None)
    elif mode == "zero":
        cfg.model.dualview_distill.loss_weight = 0.0
    elif mode == "positive":
        pass
    else:
        raise ValueError(mode)
    return cfg


def first_val_batch(cfg):
    dataset = build_dataset(cfg.data.val)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )
    return next(iter(loader)), len(dataset)


def build_wrapped_model(cfg, checkpoint):
    set_random_seed(0, deterministic=True)
    model = build_model(cfg.model)
    model.init_weights()
    model.cuda()
    load_checkpoint(model, checkpoint, map_location="cpu", strict=False)
    return MMDataParallel(model, [0])


def capture_forward_test_logits(model, batch):
    captured = {}
    batch = copy.deepcopy(batch)

    def hook(_module, _inputs, output):
        captured["logits"] = detach_logits(output)

    handle = model.module.pts_bbox_head.register_forward_hook(hook)
    try:
        model.eval()
        set_random_seed(0, deterministic=True)
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **batch)
    finally:
        handle.remove()
    if "logits" not in captured:
        raise RuntimeError("pts_bbox_head hook did not capture forward_test logits")
    return captured["logits"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    set_random_seed(0, deterministic=True)
    importlib.import_module("models")
    importlib.import_module("loaders")

    baseline_cfg = prepare_cfg(args.config, "baseline")
    zero_cfg = prepare_cfg(args.config, "zero")
    positive_cfg = prepare_cfg(args.config, "positive")
    batch, val_len = first_val_batch(zero_cfg)

    baseline_model = build_wrapped_model(baseline_cfg, args.checkpoint)
    baseline_logits = capture_forward_test_logits(baseline_model, batch)
    del baseline_model
    torch.cuda.empty_cache()

    zero_model = build_wrapped_model(zero_cfg, args.checkpoint)
    zero_logits = capture_forward_test_logits(zero_model, batch)
    zero_compare = compare_logits(baseline_logits, zero_logits)
    del zero_model
    torch.cuda.empty_cache()

    positive_model = build_wrapped_model(positive_cfg, args.checkpoint)
    positive_logits = capture_forward_test_logits(positive_model, batch)
    positive_compare = compare_logits(baseline_logits, positive_logits)

    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "val_samples_in_check": val_len,
        "baseline_vs_zero_loss_weight": zero_compare,
        "baseline_vs_positive_loss_weight": positive_compare,
        "pass": zero_compare["within_tolerance"] and positive_compare["within_tolerance"],
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(args.out_json)
    if not payload["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
