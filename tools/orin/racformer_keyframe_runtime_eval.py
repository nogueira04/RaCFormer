# pyright: reportMissingImports=false
"""Run RaCFormer validation with keyframe inference and box propagation.

This is the versioned replacement for the temporary
``/tmp/racformer_keyframe_runtime_eval.py`` wrapper used on the Orin May 11
experiments. It patches ``val.single_gpu_test`` so keyframes run normal model
inference while skipped frames reuse the last keyframe detections propagated by
ego pose and predicted velocity.

Default schedule reproduces the passing mini-val run:

* stride-2 keyframes by scene-local sample index
* extra global keyframes: 3, 33
* dropped global keyframes: 26, 70, 72

The decoder top-k patch is loaded separately from ``racformer_topk_decoder_eval``.
Set ``RACFORMER_TOPK_PATCH`` if the patch file is not beside this script.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from mmdet3d.structures import LiDARInstance3DBoxes
from pyquaternion import Quaternion
from tqdm import tqdm


def _parse_int_set(value: str | None, default: set[int]) -> set[int]:
    if value is None or value.strip() == "":
        return set(default)
    return {int(item.strip()) for item in value.split(",") if item.strip()}


KEYFRAME_STRIDE = int(os.environ.get("RACFORMER_KEYFRAME_STRIDE", "2"))
EXTRA_KEYS = _parse_int_set(os.environ.get("RACFORMER_EXTRA_KEYS"), {3, 33})
DROP_KEYS = _parse_int_set(os.environ.get("RACFORMER_DROP_KEYS"), {26, 70, 72})


def _load_topk_patch() -> None:
    local_patch = Path(__file__).with_name("racformer_topk_decoder_eval.py")
    patch_path = Path(os.environ.get("RACFORMER_TOPK_PATCH", str(local_patch)))
    if not patch_path.exists():
        legacy_tmp = Path("/tmp/racformer_topk_decoder_eval.py")
        if legacy_tmp.exists():
            patch_path = legacy_tmp
    if not patch_path.exists():
        raise FileNotFoundError(
            f"Could not find top-k decoder patch at {patch_path}. "
            "Set RACFORMER_TOPK_PATCH to racformer_topk_decoder_eval.py."
        )

    spec = importlib.util.spec_from_file_location("racformer_topk_decoder_eval", patch_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load top-k decoder patch from {patch_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_patch()


def _lidar2global(info: dict) -> np.ndarray:
    lidar2ego = np.eye(4, dtype=np.float64)
    lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(info["lidar2ego_translation"], dtype=np.float64)

    ego2global = np.eye(4, dtype=np.float64)
    ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    ego2global[:3, 3] = np.asarray(info["ego2global_translation"], dtype=np.float64)
    return ego2global @ lidar2ego


def _to_cpu_result(result: dict) -> dict:
    pred = result["pts_bbox"]
    boxes = pred["boxes_3d"]
    return {
        "pts_bbox": {
            "boxes_3d": LiDARInstance3DBoxes(boxes.tensor.detach().float().cpu().clone(), box_dim=boxes.box_dim),
            "scores_3d": pred["scores_3d"].detach().float().cpu().clone(),
            "labels_3d": pred["labels_3d"].detach().long().cpu().clone(),
        }
    }


def _propagate_result(
    src_result: dict,
    src_info: dict,
    dst_info: dict,
    src_l2g: np.ndarray,
    dst_g2l: np.ndarray,
) -> dict:
    pred = src_result["pts_bbox"]
    src_boxes = pred["boxes_3d"]
    box = src_boxes.tensor.detach().float().cpu().numpy().copy()
    scores = pred["scores_3d"].detach().float().cpu().clone()
    labels = pred["labels_3d"].detach().long().cpu().clone()

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

    rel_rot = dst_g2l[:3, :3] @ src_l2g[:3, :3]
    yaw = box[:, 6].astype(np.float64)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    yaw_num = rel_rot[1, 0] * cos_yaw + rel_rot[1, 1] * sin_yaw
    yaw_den = rel_rot[0, 0] * cos_yaw + rel_rot[0, 1] * sin_yaw
    box[:, 6] = np.arctan2(yaw_num, yaw_den).astype(box.dtype)

    box[:, :3] = centers_dst
    if box.shape[1] >= 9:
        box[:, 7:9] = vel_dst[:, :2]

    return {
        "pts_bbox": {
            "boxes_3d": LiDARInstance3DBoxes(torch.from_numpy(box).float(), box_dim=src_boxes.box_dim),
            "scores_3d": scores,
            "labels_3d": labels,
        }
    }


def _key_mask(dataset) -> list[bool]:
    mask: list[bool] = []
    scene = None
    pos = 0
    for idx, info in enumerate(dataset.data_infos):
        if info["scene_token"] != scene:
            scene = info["scene_token"]
            pos = 0
        is_key = (pos % KEYFRAME_STRIDE == 0)
        if idx in EXTRA_KEYS:
            is_key = True
        if idx in DROP_KEYS:
            is_key = False
        if idx == 0:
            is_key = True
        mask.append(is_key)
        pos += 1
    return mask


def _unwrap_sample(sample):
    if isinstance(sample, dict):
        return sample
    if isinstance(sample, (list, tuple)):
        while isinstance(sample, (list, tuple)) and len(sample) == 1:
            sample = sample[0]
        if isinstance(sample, dict):
            return sample
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            return sample[0] if isinstance(sample[0], dict) else {"inputs": sample[0], "data_samples": sample[1]}
    return None


def _cuda_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def keyframe_single_gpu_test(model, data_loader, show=False, out_dir=None, autocast_dtype=None):
    import val

    model.eval()
    dataset = data_loader.dataset
    mask = _key_mask(dataset)
    keyframes = sum(mask)
    l2g_mats = [_lidar2global(info) for info in dataset.data_infos]
    g2l_mats = [np.linalg.inv(mat) for mat in l2g_mats]
    print(
        "Keyframe runtime schedule: "
        f"{keyframes}/{len(mask)} keyframes, stride={KEYFRAME_STRIDE}, "
        f"extras={sorted(EXTRA_KEYS)}, drops={sorted(DROP_KEYS)}"
    )

    results = []
    inference_times = []
    prog_bar = tqdm(total=len(dataset))
    warmup_done = False
    sample_idx = 0
    last_key_idx = None
    last_key_result = None

    for batch in data_loader:
        for sample in batch:
            data = _unwrap_sample(sample)
            if data is None:
                continue

            is_key = mask[sample_idx] or last_key_result is None
            if is_key:
                for key in ["img_metas", "img", "radar_points", "radar_depth", "radar_rcs", "gt_depth"]:
                    if key in data and not isinstance(data[key], list):
                        data[key] = [data[key]]

                device = next((model.module if hasattr(model, "module") else model).parameters()).device
                data = val.move_to_device(data, device)

                if not warmup_done:
                    warmup_data = copy.deepcopy(data)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext():
                            _ = model(return_loss=False, rescale=True, **warmup_data)
                    _cuda_synchronize()
                    warmup_done = True

                _cuda_synchronize()
                start_time = time.perf_counter()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext():
                        result = model(return_loss=False, rescale=True, **data)
                _cuda_synchronize()
                end_time = time.perf_counter()
                inference_times.append((end_time - start_time) * 1000.0)

                key_result = result[0] if isinstance(result, list) else result
                key_result = _to_cpu_result(key_result)
                results.append(key_result)
                last_key_idx = sample_idx
                last_key_result = key_result
            else:
                assert last_key_idx is not None
                assert last_key_result is not None
                start_time = time.perf_counter()
                propagated = _propagate_result(
                    last_key_result,
                    dataset.data_infos[last_key_idx],
                    dataset.data_infos[sample_idx],
                    l2g_mats[last_key_idx],
                    g2l_mats[sample_idx],
                )
                end_time = time.perf_counter()
                inference_times.append((end_time - start_time) * 1000.0)
                results.append(propagated)

            sample_idx += 1
            prog_bar.update(1)

    prog_bar.close()
    timing_stats = {}
    if inference_times:
        timing_stats = {
            "mean_inference_ms": float(np.mean(inference_times)),
            "std_inference_ms": float(np.std(inference_times)),
            "min_inference_ms": float(np.min(inference_times)),
            "max_inference_ms": float(np.max(inference_times)),
            "median_inference_ms": float(np.median(inference_times)),
            "fps": float(1000.0 / np.mean(inference_times)),
            "num_samples": len(inference_times),
            "keyframes": keyframes,
            "propagated_frames": len(mask) - keyframes,
            "keyframe_stride": KEYFRAME_STRIDE,
            "extra_keys": sorted(EXTRA_KEYS),
            "drop_keys": sorted(DROP_KEYS),
        }
    return results, timing_stats


def main() -> int:
    _load_topk_patch()
    import val

    val.single_gpu_test = keyframe_single_gpu_test
    return val.main()


if __name__ == "__main__":
    sys.exit(main())
