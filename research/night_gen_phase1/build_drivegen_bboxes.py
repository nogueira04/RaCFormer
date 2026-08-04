#!/usr/bin/env python3
"""Export RaCFormer nuScenes samples to DriveGEN's 2D bbox JSON format."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points


CAMERA_NAMES = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a DriveGEN organized_2d_bboxes JSON from a RaCFormer nuScenes info pkl."
    )
    parser.add_argument("--ann-file", default="nuscenes_infos_train_2k_day.pkl")
    parser.add_argument("--dataroot", default="/mnt/nfs/shared/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--manifest", help="Optional NB2 manifest; only tokens in the manifest are exported.")
    parser.add_argument("--limit-tokens", type=int, default=0, help="Maximum number of sample tokens to export.")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset after manifest filtering.")
    parser.add_argument("--min-area", type=float, default=25.0, help="Minimum 2D bbox area in pixels.")
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def load_infos(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    infos = payload.get("infos", payload) if isinstance(payload, dict) else payload
    if not isinstance(infos, list):
        raise TypeError(f"{path} did not contain a list of infos")
    return infos


def manifest_tokens(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    tokens: set[str] = set()

    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("entries") or payload.get("samples") or payload.get("items") or []
        for key in ("token", "sample_token"):
            value = payload.get(key)
            if isinstance(value, str):
                tokens.add(value)
    else:
        entries = []

    if isinstance(entries, dict):
        entries = entries.values()

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key in ("token", "sample_token"):
            value = entry.get(key)
            if isinstance(value, str):
                tokens.add(value)
        for value in entry.values():
            if isinstance(value, dict):
                for key in ("token", "sample_token"):
                    nested = value.get(key)
                    if isinstance(nested, str):
                        tokens.add(nested)

    if not tokens:
        raise ValueError(f"no sample tokens found in manifest {path}")
    return tokens


def relative_image_key(data_path: str) -> str:
    path = Path(data_path)
    parts = path.parts
    if "samples" in parts:
        idx = parts.index("samples")
        return str(Path(*parts[idx:]))
    return data_path


def post_process_coords(corner_coords: np.ndarray, image_size: tuple[int, int]) -> list[float] | None:
    inside = (
        (corner_coords[:, 0] >= 0)
        & (corner_coords[:, 0] < image_size[0])
        & (corner_coords[:, 1] >= 0)
        & (corner_coords[:, 1] < image_size[1])
    )
    if not inside.any():
        return None

    x1 = float(np.clip(corner_coords[:, 0].min(), 0, image_size[0] - 1))
    y1 = float(np.clip(corner_coords[:, 1].min(), 0, image_size[1] - 1))
    x2 = float(np.clip(corner_coords[:, 0].max(), 0, image_size[0] - 1))
    y2 = float(np.clip(corner_coords[:, 1].max(), 0, image_size[1] - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def camera_boxes(
    nusc: NuScenes,
    sample_data_token: str,
    dataroot: Path,
    min_area: float,
) -> tuple[str, list[dict[str, Any]]]:
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(
        sample_data_token, box_vis_level=BoxVisibility.ANY
    )
    image_path = Path(data_path)
    if not image_path.is_absolute():
        image_path = dataroot / image_path
    image_size = Image.open(image_path).size
    key = relative_image_key(str(data_path))

    entries: list[dict[str, Any]] = []
    for box in boxes:
        detection_name = category_to_detection_name(box.name)
        if detection_name is None:
            continue
        corners = view_points(box.corners(), np.array(camera_intrinsic), normalize=True)[:2, :].T
        bbox = post_process_coords(corners, image_size)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        entries.append(
            {
                "category_name": detection_name,
                "bbox_corners": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )
    return key, entries


def main() -> None:
    args = parse_args()
    infos = load_infos(Path(args.ann_file))
    allowed_tokens = manifest_tokens(Path(args.manifest)) if args.manifest else None

    filtered = [info for info in infos if allowed_tokens is None or info.get("token") in allowed_tokens]
    if args.start_index:
        filtered = filtered[args.start_index :]
    if args.limit_tokens:
        filtered = filtered[: args.limit_tokens]
    if not filtered:
        raise ValueError("no infos matched the requested filters")

    dataroot = Path(args.dataroot)
    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=False)
    output: dict[str, list[dict[str, Any]]] = {}

    for info in filtered:
        sample = nusc.get("sample", info["token"])
        for camera_name in CAMERA_NAMES:
            sample_data_token = sample["data"][camera_name]
            key, boxes = camera_boxes(nusc, sample_data_token, dataroot, args.min_area)
            output[key] = boxes

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    total_boxes = sum(len(boxes) for boxes in output.values())
    print(f"wrote {out_path} with {len(filtered)} samples, {len(output)} images, {total_boxes} boxes")


if __name__ == "__main__":
    main()
