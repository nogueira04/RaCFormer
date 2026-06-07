# pyright: reportMissingImports=false
"""Offline keyframe propagation experiments for RaCFormer Orin.

This versions the ad hoc May 2026 shell-history snippets that evaluated
keyframe schedules from saved prediction pickles. It does not run model
inference; it loads a full-frame prediction file, keeps selected keyframe
predictions, propagates boxes to skipped frames with ego pose and velocity, and
re-runs nuScenes mini evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
from pathlib import Path
from typing import Iterable

import importlib
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmdet.registry import MODELS as MMDET_MODELS
from mmdet.registry import TASK_UTILS as MMDET_TASK_UTILS
from mmdet3d.registry import DATASETS
from mmdet3d.registry import MODELS as MMDET3D_MODELS
from mmdet3d.registry import TASK_UTILS as MMDET3D_TASK_UTILS
from mmdet3d.registry import TRANSFORMS as MMDET3D_TRANSFORMS
from mmdet3d.structures import LiDARInstance3DBoxes
from pyquaternion import Quaternion


os.environ.setdefault("NUSCENES_VERSION", "v1.0-mini")


def register_project() -> None:
    importlib.import_module("models")
    importlib.import_module("loaders")
    import mmdet3d.datasets.transforms  # noqa: F401
    for name, module in MMDET3D_TRANSFORMS.module_dict.items():
        if name not in MMENGINE_TRANSFORMS.module_dict:
            MMENGINE_TRANSFORMS.register_module(name=name, module=module)
    for name, module in MMDET_MODELS.module_dict.items():
        if name not in MMDET3D_MODELS.module_dict:
            MMDET3D_MODELS.register_module(name=name, module=module)
    for name, module in MMDET_TASK_UTILS.module_dict.items():
        if name not in MMDET3D_TASK_UTILS.module_dict:
            MMDET3D_TASK_UTILS.register_module(name=name, module=module)
    for name, module in MMDET3D_TASK_UTILS.module_dict.items():
        if name not in MMDET_TASK_UTILS.module_dict:
            MMDET_TASK_UTILS.register_module(name=name, module=module)


def lidar2global(info: dict) -> np.ndarray:
    lidar2ego = np.eye(4, dtype=np.float64)
    lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(info["lidar2ego_translation"], dtype=np.float64)
    ego2global = np.eye(4, dtype=np.float64)
    ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    ego2global[:3, 3] = np.asarray(info["ego2global_translation"], dtype=np.float64)
    return ego2global @ lidar2ego


class OfflinePropagator:
    def __init__(self, dataset, base_results):
        self.dataset = dataset
        self.base_results = base_results

    def actual(self, idx: int) -> dict:
        pred = self.base_results[idx]["pts_bbox"]
        boxes = pred["boxes_3d"]
        return {
            "pts_bbox": {
                "boxes_3d": LiDARInstance3DBoxes(boxes.tensor.detach().float().cpu().clone(), box_dim=boxes.box_dim),
                "scores_3d": pred["scores_3d"].detach().float().cpu().clone(),
                "labels_3d": pred["labels_3d"].detach().long().cpu().clone(),
            }
        }

    def propagated(self, src_idx: int, dst_idx: int, score_scale: float = 1.0) -> dict:
        if src_idx == dst_idx:
            return self.actual(dst_idx)

        src_info = self.dataset.data_infos[src_idx]
        dst_info = self.dataset.data_infos[dst_idx]
        pred = self.base_results[src_idx]["pts_bbox"]
        src_boxes = pred["boxes_3d"]
        box = src_boxes.tensor.detach().float().cpu().numpy().copy()
        scores = (pred["scores_3d"].detach().float().cpu().clone() * score_scale).clamp(max=1.0)
        labels = pred["labels_3d"].detach().long().cpu().clone()

        src_l2g = lidar2global(src_info)
        dst_g2l = np.linalg.inv(lidar2global(dst_info))
        dt = (dst_info["timestamp"] - src_info["timestamp"]) / 1e6

        centers = np.concatenate([box[:, :3], np.ones((box.shape[0], 1), dtype=np.float64)], axis=1)
        centers_global = (src_l2g @ centers.T).T[:, :3]
        vel_lidar = np.zeros((box.shape[0], 3), dtype=np.float64)
        if box.shape[1] >= 9:
            vel_lidar[:, :2] = box[:, 7:9]
        vel_global = (src_l2g[:3, :3] @ vel_lidar.T).T
        centers_global = centers_global + vel_global * dt
        centers_dst = (
            dst_g2l
            @ np.concatenate([centers_global, np.ones((box.shape[0], 1), dtype=np.float64)], axis=1).T
        ).T[:, :3]
        vel_dst = (dst_g2l[:3, :3] @ vel_global.T).T

        src_q = Quaternion(src_info["ego2global_rotation"]) * Quaternion(src_info["lidar2ego_rotation"])
        dst_q_inv = (Quaternion(dst_info["ego2global_rotation"]) * Quaternion(dst_info["lidar2ego_rotation"])).inverse
        box[:, :3] = centers_dst
        box[:, 6] = np.asarray(
            [(dst_q_inv * src_q * Quaternion(axis=[0, 0, 1], angle=float(yaw))).yaw_pitch_roll[0] for yaw in box[:, 6]],
            dtype=box.dtype,
        )
        if box.shape[1] >= 9:
            box[:, 7:9] = vel_dst[:, :2]

        return {
            "pts_bbox": {
                "boxes_3d": LiDARInstance3DBoxes(torch.from_numpy(box).float(), box_dim=src_boxes.box_dim),
                "scores_3d": scores,
                "labels_3d": labels,
            }
        }

    def make_results(self, mask: list[bool], score_scale: float = 1.0) -> tuple[list[dict], int, int]:
        results = []
        last_key = None
        keyframes = 0
        propagated = 0
        for idx, is_key in enumerate(mask):
            if is_key or last_key is None:
                results.append(self.actual(idx))
                last_key = idx
                keyframes += 1
            else:
                results.append(self.propagated(last_key, idx, score_scale=score_scale))
                propagated += 1
        return results, keyframes, propagated


def stride_mask(dataset, stride: int, *, extra_keys: Iterable[int] = (), drop_keys: Iterable[int] = ()) -> list[bool]:
    extra = set(extra_keys)
    drop = set(drop_keys)
    mask = []
    scene = None
    pos = 0
    for idx, info in enumerate(dataset.data_infos):
        if info["scene_token"] != scene:
            scene = info["scene_token"]
            pos = 0
        is_key = (pos % stride == 0) or idx in extra
        if idx in drop:
            is_key = False
        if idx == 0:
            is_key = True
        mask.append(is_key)
        pos += 1
    return mask


def evaluate(dataset, results: list[dict], out_dir: Path, report: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        metrics = dataset.evaluate(results, jsonfile_prefix=str(out_dir / "submission"))
    full_report = {**report, **metrics}
    (out_dir / "metrics.json").write_text(json.dumps(full_report, indent=2, sort_keys=True))
    print(f"{out_dir.name}: {json.dumps(full_report, sort_keys=True)}", flush=True)
    return full_report


def load_context(config: Path, predictions: Path):
    register_project()
    cfg = Config.fromfile(str(config))
    dataset = DATASETS.build(cfg.data.val)
    with predictions.open("rb") as handle:
        base_results = pickle.load(handle)
    return dataset, OfflinePropagator(dataset, base_results)


def run_decay_and_scale(args) -> None:
    dataset, propagator = load_context(args.config, args.predictions)
    for stride, decay in [(2, 1.0), (2, 0.95), (2, 0.90), (3, 1.0)]:
        mask = stride_mask(dataset, stride)
        results, keyframes, copied = propagator.make_results(mask, score_scale=decay)
        name = f"f4_keystride{stride}_prop_decay{str(decay).replace('.', 'p')}_mini"
        evaluate(
            dataset,
            results,
            args.out_root / name,
            {
                "stride": stride,
                "score_decay": decay,
                "keyframes": keyframes,
                "copied": copied,
                "amortized_ms": args.base_latency_ms * keyframes / len(dataset),
            },
        )

    for scale in [1.02, 1.05, 1.10, 1.20, 1.50]:
        mask = stride_mask(dataset, 2)
        results, keyframes, copied = propagator.make_results(mask, score_scale=scale)
        name = f"f4_keystride2_prop_scale{str(scale).replace('.', 'p')}_mini"
        evaluate(
            dataset,
            results,
            args.out_root / name,
            {
                "stride": 2,
                "copied_score_scale": scale,
                "keyframes": keyframes,
                "copied": copied,
                "amortized_ms": args.base_latency_ms * keyframes / len(dataset),
            },
        )


def run_topk500_rm70(args) -> None:
    dataset, propagator = load_context(args.config, args.predictions)
    mask = stride_mask(dataset, 2, extra_keys=[3, 33], drop_keys=[70])
    results, keyframes, copied = propagator.make_results(mask)
    evaluate(
        dataset,
        results,
        args.out_root / "f4_topk500_add3_33_rm70_offline_mini",
        {
            "keyframes": keyframes,
            "copied": copied,
            "amortized_ms": args.base_latency_ms * keyframes / len(dataset),
            "extra_keys": [3, 33],
            "drop_keys": [70],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/orin/racformer_f4_3layer_q900_thrnone_mini.py"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("eval_results"))
    parser.add_argument("--base-latency-ms", type=float, default=385.45)
    parser.add_argument("--mode", choices=["decay-and-scale", "topk500-rm70"], required=True)
    args = parser.parse_args()

    if args.mode == "decay-and-scale":
        run_decay_and_scale(args)
    elif args.mode == "topk500-rm70":
        run_topk500_rm70(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
