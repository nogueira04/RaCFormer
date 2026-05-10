#!/usr/bin/env python3
"""Adapt an 8-frame RaCFormer checkpoint for the latency200 mini config.

This is a checkpoint surgery utility for latency profiling. It slices temporal
weights down to the current-frame shape expected by
configs/racformer_r50_nuimg_704x256_latency200_mini.py. The resulting checkpoint
must be validated for mAP/NDS before any accuracy-sensitive use.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


SAMPLING_WEIGHT = "pts_bbox_head.transformer.decoder.decoder_layer.sampling.scale_weights.weight"
SAMPLING_BIAS = "pts_bbox_head.transformer.decoder.decoder_layer.sampling.scale_weights.bias"
MIXING_PREFIX = "pts_bbox_head.transformer.decoder.decoder_layer.mixing.parameter_generator"
BEV_QUEUE_PREFIXES = [
    "pts_bbox_head.transformer.decoder.decoder_layer.sampling_radar_bev.attention.bev_queue_weight",
    "pts_bbox_head.transformer.decoder.decoder_layer.sampling_lss_bev.attention.bev_queue_weight",
]


def adapt_mixing_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Slice AdaptiveMixing rows from 8-frame in_points=96 to 1-frame in_points=12."""
    if tensor.ndim == 2:
        grouped = tensor.reshape(4, 4096 + 128 * 96, 256)
        channel_mix = grouped[:, :4096, :]
        point_mix = grouped[:, 4096:, :].reshape(4, 128, 96, 256)[:, :, :12, :]
        return torch.cat([channel_mix, point_mix.reshape(4, 128 * 12, 256)], dim=1).reshape(
            4 * (4096 + 128 * 12), 256
        ).contiguous()

    grouped = tensor.reshape(4, 4096 + 128 * 96)
    channel_mix = grouped[:, :4096]
    point_mix = grouped[:, 4096:].reshape(4, 128, 96)[:, :, :12]
    return torch.cat([channel_mix, point_mix.reshape(4, 128 * 12)], dim=1).reshape(
        4 * (4096 + 128 * 12)
    ).contiguous()


def adapt_checkpoint(source: Path, output: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu")
    adapted = copy.deepcopy(checkpoint)
    state_dict = adapted["state_dict"] if isinstance(adapted, dict) and "state_dict" in adapted else adapted

    state_dict[SAMPLING_WEIGHT] = state_dict[SAMPLING_WEIGHT].reshape(4, 8, 12, 4, 256)[:, :1].reshape(
        192, 256
    ).contiguous()
    state_dict[SAMPLING_BIAS] = state_dict[SAMPLING_BIAS].reshape(4, 8, 12, 4)[:, :1].reshape(192).contiguous()

    for prefix in BEV_QUEUE_PREFIXES:
        state_dict[f"{prefix}.weight"] = state_dict[f"{prefix}.weight"][:1].contiguous()
        state_dict[f"{prefix}.bias"] = state_dict[f"{prefix}.bias"][:1].contiguous()

    for suffix in ["weight", "bias"]:
        key = f"{MIXING_PREFIX}.{suffix}"
        state_dict[key] = adapt_mixing_tensor(state_dict[key])

    if isinstance(adapted, dict):
        adapted.setdefault("meta", {})
        adapted["meta"]["latency200_adaptation"] = (
            "Current-frame slice for configs/racformer_r50_nuimg_704x256_latency200_mini.py; "
            "validate mAP before use."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapted, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original 8-frame checkpoint")
    parser.add_argument("output", type=Path, help="Output latency200-adapted checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapt_checkpoint(args.source, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
