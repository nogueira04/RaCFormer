#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import importlib
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import Config
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from loaders.builder import build_dataloader


REFERENCE_COVERAGE = 0.86346435546875
REFERENCE_PER_FRAME_MIN = 0.771484375
REFERENCE_PER_FRAME_MAX = 0.78790283203125
LOGIT_TOL = 1e-6
MAX_MEMORY_GB = 22.0


def finite_float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


def parse_losses(losses: dict) -> tuple[torch.Tensor, dict[str, float]]:
    parsed = {}
    for name, value in losses.items():
        if torch.is_tensor(value):
            parsed[name] = value.mean()
        elif isinstance(value, list):
            parsed[name] = sum(v.mean() for v in value)
        else:
            raise TypeError(f"{name} has unsupported loss type {type(value)}")
    total = sum(value for name, value in parsed.items() if "loss" in name)
    return total, {name: finite_float(value) for name, value in parsed.items()}


def detach_logits(output):
    keys = ("all_cls_scores", "all_bbox_preds")
    return {key: output[key].detach().float().cpu() for key in keys if key in output}


def compare_logits(lhs: dict, rhs: dict) -> dict:
    result = {"shape_match": True, "max_abs_diff": 0.0, "shapes": {}}
    for key in sorted(set(lhs) | set(rhs)):
        if key not in lhs or key not in rhs:
            result["shape_match"] = False
            result["shapes"][key] = {"lhs": None if key not in lhs else list(lhs[key].shape),
                                     "rhs": None if key not in rhs else list(rhs[key].shape)}
            continue
        result["shapes"][key] = {"lhs": list(lhs[key].shape), "rhs": list(rhs[key].shape)}
        if lhs[key].shape != rhs[key].shape:
            result["shape_match"] = False
            continue
        diff = (lhs[key] - rhs[key]).abs().max().item()
        result["max_abs_diff"] = max(result["max_abs_diff"], float(diff))
    result["within_tolerance"] = result["shape_match"] and result["max_abs_diff"] <= LOGIT_TOL
    return result


def load_first_batch(cfg: Config, split: str):
    dataset = build_dataset(getattr(cfg.data, split))
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


def prepare_cfg(config_path: str, mode: str) -> Config:
    cfg = Config.fromfile(config_path)
    cfg = copy.deepcopy(cfg)
    cfg.batch_size = 1
    cfg.data.workers_per_gpu = 0
    if hasattr(cfg.data, "val"):
        cfg.data.val.max_samples = 1
    if mode == "baseline":
        cfg.model.pop("dualview_distill", None)
    elif mode == "zero":
        cfg.model.dualview_distill.loss_weight = 0.0
    elif mode == "positive":
        pass
    else:
        raise ValueError(f"unknown mode {mode}")
    return cfg


def build_wrapped_model(cfg: Config, checkpoint: str) -> MMDataParallel:
    set_random_seed(0, deterministic=True)
    torch.manual_seed(0)
    model = build_model(cfg.model)
    model.init_weights()
    model.cuda()
    load_checkpoint(model, checkpoint, map_location="cpu", strict=False)
    return MMDataParallel(model, [0])


def capture_forward_test_logits(model: MMDataParallel, batch: dict) -> dict:
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


def grad_norms(module: torch.nn.Module) -> dict[str, float | None]:
    norms = {}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            norms[name] = None
        else:
            norms[name] = float(param.grad.detach().float().norm().cpu())
    return norms


def static_coverage(repo_root: Path, train_ann_file: str) -> dict:
    diag_path = repo_root / "research/night_gen_phase1/teachers/diagnostics/branch_g_dinov2_dualview_diag.py"
    spec = importlib.util.spec_from_file_location("branch_g_dinov2_dualview_diag", str(diag_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {diag_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ann_path = Path(train_ann_file)
    if not ann_path.is_absolute():
        ann_path = repo_root / ann_path
    with ann_path.open("rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) else data
    return module.coverage_for_infos(infos, 50)


def run_gradient_smoke(model: MMDataParallel, train_batch: dict) -> dict:
    model.train()
    train_batch = copy.deepcopy(train_batch)
    torch.cuda.reset_peak_memory_stats()
    set_random_seed(0, deterministic=True)
    losses = model(return_loss=True, **train_batch)
    total_loss, log_vars = parse_losses(losses)
    total_loss.backward()
    stats = dict(model.module.dualview_distill.last_stats)
    grads = grad_norms(model.module.dualview_distill)
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    finite_losses = all(torch.isfinite(torch.tensor(value)).item() for value in log_vars.values())
    finite_total = bool(torch.isfinite(total_loss).all().item())
    return {
        "losses": log_vars,
        "total_loss": finite_float(total_loss),
        "aux_stats": stats,
        "grad_norms": grads,
        "all_new_trainable_gradients_nonzero": all(value is not None and value > 0 for value in grads.values()),
        "finite_loss_tensors": finite_losses,
        "finite_total_loss": finite_total,
        "peak_memory_gb": float(peak_gb),
    }


def write_report(report_path: Path, payload: dict) -> None:
    checks = payload["checks"]
    lines = [
        "# Branch G Stage 3A DualViewDistill Smoke",
        "",
        f"UTC: {payload['utc']}",
        f"Host: `{payload['host']}`",
        f"Config: `{payload['config']}`",
        f"Checkpoint: `{payload['checkpoint']}`",
        "",
        "## Scope",
        "",
        (
            "Executed only the Branch G Stage 3A smoke: one train2k batch forward/backward "
            "with a training-only DualViewDistill auxiliary loss on `all_bev_feats[:, 0]`, "
            "plus identity and forward_test neutrality checks. No 12-epoch training was started."
        ),
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence |",
        "|---|---|---|",
    ]
    for key, item in checks.items():
        lines.append(f"| {key} | {item['status']} | {item['evidence']} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON evidence: `{payload['json_path']}`",
            f"- Slurm job id: `{payload.get('slurm_job_id', 'unknown')}`",
            "",
            "## Decision",
            "",
            payload["decision"],
            "",
        ]
    )
    report_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report-dir", default="research/paper_goal_20260516")
    parser.add_argument("--json-dir", default="research/night_gen_phase1/teachers/diagnostics")
    args = parser.parse_args()

    started = time.time()
    repo_root = Path.cwd()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Branch G Stage 3A smoke; refusing CPU fallback.")
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    set_random_seed(0, deterministic=True)
    importlib.import_module("models")
    importlib.import_module("loaders")

    utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_dir = repo_root / args.report_dir
    json_dir = repo_root / args.json_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"G_SMOKE_{utc}.md"
    json_path = json_dir / f"G_SMOKE_{utc}.json"

    positive_cfg = prepare_cfg(args.config, "positive")
    baseline_cfg = prepare_cfg(args.config, "baseline")
    zero_cfg = prepare_cfg(args.config, "zero")
    train_batch, train_len = load_first_batch(positive_cfg, "train")
    val_batch, val_len = load_first_batch(positive_cfg, "val")
    coverage50 = static_coverage(repo_root, positive_cfg.data.train.ann_file)

    baseline_model = build_wrapped_model(baseline_cfg, args.checkpoint)
    baseline_logits = capture_forward_test_logits(baseline_model, val_batch)
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    zero_model = build_wrapped_model(zero_cfg, args.checkpoint)
    zero_logits = capture_forward_test_logits(zero_model, val_batch)
    zero_compare = compare_logits(baseline_logits, zero_logits)
    del zero_model
    gc.collect()
    torch.cuda.empty_cache()

    positive_model = build_wrapped_model(positive_cfg, args.checkpoint)
    positive_logits = capture_forward_test_logits(positive_model, val_batch)
    positive_compare = compare_logits(baseline_logits, positive_logits)
    gradient = run_gradient_smoke(positive_model, train_batch)

    aux_stats = gradient["aux_stats"]
    dino_shape_ok = all(shape == [1, 1024, 18, 50] for shape in aux_stats.get("per_camera_dino_shapes", []))
    dino_load = aux_stats.get("teacher_load_info", {})
    dino_load_ok = dino_load.get("missing_count") == 0 and dino_load.get("unexpected_count") == 0
    coverage_ratio = aux_stats.get("coverage_ratio")
    coverage_text = "None" if coverage_ratio is None else f"{coverage_ratio:.6f}"
    coverage_batch_ok = (
        coverage_ratio is not None
        and REFERENCE_PER_FRAME_MIN - 0.03 <= coverage_ratio <= REFERENCE_PER_FRAME_MAX + 0.03
    )
    coverage50_ok = abs(float(coverage50["coverage_ratio"]) - REFERENCE_COVERAGE) <= 0.03
    aux_loss = gradient["losses"].get("loss_dualview_distill")
    aux_finite = aux_loss is not None and torch.isfinite(torch.tensor(aux_loss)).item()
    total_finite = torch.isfinite(torch.tensor(gradient["total_loss"])).item()
    peak_ok = gradient["peak_memory_gb"] <= MAX_MEMORY_GB

    checks = {
        "a_dino_pv_shape": {
            "status": "PASS" if dino_shape_ok and dino_load_ok else "FAIL",
            "evidence": (
                f"per-camera shapes={aux_stats.get('per_camera_dino_shapes')}; "
                f"teacher_load={dino_load}"
            ),
        },
        "b_lss_pooling_coverage": {
            "status": "PASS" if coverage_batch_ok and coverage50_ok else "FAIL",
            "evidence": (
                f"batch coverage={coverage_text}; 50-frame train coverage="
                f"{coverage50['coverage_ratio']:.6f} ({coverage50['covered_cells']}/"
                f"{coverage50['total_cells']}), reference={REFERENCE_COVERAGE:.6f}"
            ),
        },
        "c_aux_gradients": {
            "status": "PASS" if gradient["all_new_trainable_gradients_nonzero"] else "FAIL",
            "evidence": f"new trainable grad norms={gradient['grad_norms']}",
        },
        "d_finite_losses": {
            "status": "PASS" if aux_finite and total_finite and gradient["finite_loss_tensors"] else "FAIL",
            "evidence": f"aux={aux_loss}, total={gradient['total_loss']}, losses={gradient['losses']}",
        },
        "e_aux_weight_zero_identity": {
            "status": "PASS" if zero_compare["within_tolerance"] else "FAIL",
            "evidence": f"shape_match={zero_compare['shape_match']}, max_abs_diff={zero_compare['max_abs_diff']:.3e}",
        },
        "f_peak_memory": {
            "status": "PASS" if peak_ok else "FAIL",
            "evidence": f"peak={gradient['peak_memory_gb']:.3f} GB, limit={MAX_MEMORY_GB:.1f} GB",
        },
        "g_forward_test_neutrality": {
            "status": "PASS" if positive_compare["within_tolerance"] else "FAIL",
            "evidence": (
                f"shape_match={positive_compare['shape_match']}, "
                f"max_abs_diff={positive_compare['max_abs_diff']:.3e}"
            ),
        },
    }
    all_pass = all(item["status"] == "PASS" for item in checks.values())
    payload = {
        "utc": utc,
        "host": os.uname().nodename,
        "elapsed_seconds": time.time() - started,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "train_dataset_len": train_len,
        "val_dataset_len": val_len,
        "coverage50": coverage50,
        "zero_compare": zero_compare,
        "positive_compare": positive_compare,
        "gradient": gradient,
        "checks": checks,
        "decision": (
            "PASS: halt for user review. Do not start 12-epoch training."
            if all_pass
            else "FAIL: halt and request user diagnosis. Auto-retry is forbidden."
        ),
        "report_path": str(report_path.relative_to(repo_root)),
        "json_path": str(json_path.relative_to(repo_root)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(report_path, payload)
    print(json.dumps({"report": str(report_path), "json": str(json_path), "checks": checks}, indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
