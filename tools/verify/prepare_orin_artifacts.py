# pyright: reportMissingImports=false
"""Prepare stable RaCFormer Orin experiment artifacts.

The original May 2026 Orin experiments used temporary files under ``/tmp``.
This script regenerates the important transient configs and checkpoint from
versioned inputs so evaluation commands can use stable paths without a top-level
``repro/`` tree.

Run from the RaCFormer repository root on ``orin5``:

    python tools/verify/prepare_orin_artifacts.py
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path


GENERATED_CONFIGS = {
    "f2_2layer_q900_thrnone": {
        "source": "configs/racformer_r50_nuimg_704x256_f2_2layer_q900_mini.py",
        "target": "configs/orin/racformer_f2_2layer_q900_thrnone_mini.py",
        "replacements": [
            (r"score_threshold=0\.05", "score_threshold=None"),
        ],
    },
    "f2_1layer_q900_max500": {
        "source": "configs/racformer_r50_nuimg_704x256_f2_1layer_q900_mini.py",
        "target": "configs/orin/racformer_f2_1layer_q900_max500_mini.py",
        "replacements": [
            (r"max_num=300", "max_num=500"),
            (r"score_threshold=0\.05", "score_threshold=None"),
        ],
    },
    "f2_2layer_q900_max500": {
        "source": "configs/racformer_r50_nuimg_704x256_f2_2layer_q900_mini.py",
        "target": "configs/orin/racformer_f2_2layer_q900_max500_mini.py",
        "replacements": [
            (r"max_num=300", "max_num=500"),
            (r"score_threshold=0\.05", "score_threshold=None"),
        ],
    },
    "f2_2layer_q900_d0807_thrnone": {
        "source": "configs/racformer_r50_nuimg_704x256_f2_2layer_q900_mini.py",
        "target": "configs/orin/racformer_f2_2layer_q900_d0807_thrnone_mini.py",
        "replacements": [
            (r"d_region_list = \[0\.08, 0\.03\]", "d_region_list = [0.08, 0.07]"),
            (r"score_threshold=0\.05", "score_threshold=None"),
        ],
    },
    "f2_2layer_q900_d0808_thrnone": {
        "source": "configs/racformer_r50_nuimg_704x256_f2_2layer_q900_mini.py",
        "target": "configs/orin/racformer_f2_2layer_q900_d0808_thrnone_mini.py",
        "replacements": [
            (r"d_region_list = \[0\.08, 0\.03\]", "d_region_list = [0.08, 0.08]"),
            (r"score_threshold=0\.05", "score_threshold=None"),
        ],
    },
    "f2_2layer_q900_d1010_thrnone": {
        "source": "configs/racformer_r50_nuimg_704x256_f2_2layer_q900_mini.py",
        "target": "configs/orin/racformer_f2_2layer_q900_d1010_thrnone_mini.py",
        "replacements": [
            (r"d_region_list = \[0\.08, 0\.03\]", "d_region_list = [0.10, 0.10]"),
            (r"score_threshold=0\.05", "score_threshold=None"),
        ],
    },
    "f2_3layer_q900": {
        "source": "configs/racformer_r50_nuimg_704x256_f2_2layer_q900_mini.py",
        "target": "configs/orin/racformer_f2_3layer_q900_mini.py",
        "replacements": [
            (r"num_layers = 2", "num_layers = 3"),
            (r"d_region_list = \[0\.08, 0\.03\]", "d_region_list = [0.08, 0.05, 0.03]"),
        ],
    },
    "f2_4layer_q900": {
        "source": "configs/racformer_r50_nuimg_704x256_f2_2layer_q900_mini.py",
        "target": "configs/orin/racformer_f2_4layer_q900_mini.py",
        "replacements": [
            (r"num_layers = 2", "num_layers = 4"),
            (r"d_region_list = \[0\.08, 0\.03\]", "d_region_list = [0.08, 0.06, 0.04, 0.03]"),
        ],
    },
    "f2_6layer_q900": {
        "source": "configs/racformer_r50_nuimg_704x256_f2_2layer_q900_mini.py",
        "target": "configs/orin/racformer_f2_6layer_q900_mini.py",
        "replacements": [
            (r"num_layers = 2", "num_layers = 6"),
            (
                r"d_region_list = \[0\.08, 0\.03\]",
                "d_region_list = [0.08, 0.07, 0.06, 0.05, 0.04, 0.03]",
            ),
        ],
    },
    "f4_1layer_q900_thrnone": {
        "source": "configs/racformer_r50_nuimg_704x256_f2_1layer_q900_mini.py",
        "target": "configs/orin/racformer_f4_1layer_q900_thrnone_mini.py",
        "replacements": [
            (r"num_frames = 2", "num_frames = 4"),
            (r"score_threshold=0\.05", "score_threshold=None"),
        ],
    },
    "f4_2layer_q900_d0808_thrnone": {
        "source": "configs/racformer_r50_nuimg_704x256_f2_2layer_q900_mini.py",
        "target": "configs/orin/racformer_f4_2layer_q900_d0808_thrnone_mini.py",
        "replacements": [
            (r"num_frames = 2", "num_frames = 4"),
            (r"d_region_list = \[0\.08, 0\.03\]", "d_region_list = [0.08, 0.08]"),
            (r"score_threshold=0\.05", "score_threshold=None"),
        ],
    },
    "f4_3layer_q900_thrnone": {
        "source": "configs/racformer_r50_nuimg_704x256_f2_2layer_q900_mini.py",
        "target": "configs/orin/racformer_f4_3layer_q900_thrnone_mini.py",
        "replacements": [
            (r"num_frames = 2", "num_frames = 4"),
            (r"num_layers = 2", "num_layers = 3"),
            (r"d_region_list = \[0\.08, 0\.03\]", "d_region_list = [0.08, 0.05, 0.03]"),
            (r"score_threshold=0\.05", "score_threshold=None"),
        ],
    },
}


def _replace_once(text: str, pattern: str, replacement: str, target: Path) -> str:
    new_text, count = re.subn(pattern, replacement, text)
    if count == 0:
        raise RuntimeError(f"Pattern not found in {target}: {pattern}")
    return new_text


def generate_configs(repo: Path) -> None:
    for name, spec in GENERATED_CONFIGS.items():
        source = repo / spec["source"]
        target = repo / spec["target"]
        if not source.exists():
            raise FileNotFoundError(f"Missing source config for {name}: {source}")

        text = source.read_text()
        text = "# Auto-generated by tools/verify/prepare_orin_artifacts.py\n" + text
        for pattern, replacement in spec["replacements"]:
            text = _replace_once(text, pattern, replacement, source)

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(f"wrote {target}")


def generate_f4_checkpoint(repo: Path, artifact_root: Path, force: bool = False) -> None:
    import torch

    source = repo / "checkpoints/racformer_r50_f8.pth"
    target = artifact_root / "generated_checkpoints/racformer_r50_f8_f4_1layer_adapted.pth"
    if target.exists() and not force:
        print(f"exists {target}")
        return
    if not source.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {source}")

    checkpoint = torch.load(source, map_location="cpu")
    adapted = copy.deepcopy(checkpoint)
    state = adapted["state_dict"]
    num_frames = 4

    scale_weight = "pts_bbox_head.transformer.decoder.decoder_layer.sampling.scale_weights.weight"
    scale_bias = "pts_bbox_head.transformer.decoder.decoder_layer.sampling.scale_weights.bias"
    state[scale_weight] = (
        state[scale_weight]
        .reshape(4, 8, 12, 4, 256)[:, :num_frames]
        .reshape(192 * num_frames, 256)
        .contiguous()
    )
    state[scale_bias] = (
        state[scale_bias].reshape(4, 8, 12, 4)[:, :num_frames].reshape(192 * num_frames).contiguous()
    )

    for prefix in [
        "pts_bbox_head.transformer.decoder.decoder_layer.sampling_radar_bev.attention.bev_queue_weight",
        "pts_bbox_head.transformer.decoder.decoder_layer.sampling_lss_bev.attention.bev_queue_weight",
    ]:
        state[prefix + ".weight"] = state[prefix + ".weight"][:num_frames].contiguous()
        state[prefix + ".bias"] = state[prefix + ".bias"][:num_frames].contiguous()

    for suffix in ["weight", "bias"]:
        key = f"pts_bbox_head.transformer.decoder.decoder_layer.mixing.parameter_generator.{suffix}"
        tensor = state[key]
        if tensor.ndim == 2:
            grouped = tensor.reshape(4, 4096 + 128 * 96, 256)
            state[key] = (
                torch.cat(
                    [
                        grouped[:, :4096, :],
                        grouped[:, 4096:, :].reshape(4, 128, 96, 256)[:, :, : 12 * num_frames, :].reshape(
                            4, 128 * 12 * num_frames, 256
                        ),
                    ],
                    1,
                )
                .reshape(4 * (4096 + 128 * 12 * num_frames), 256)
                .contiguous()
            )
        else:
            grouped = tensor.reshape(4, 4096 + 128 * 96)
            state[key] = (
                torch.cat(
                    [
                        grouped[:, :4096],
                        grouped[:, 4096:].reshape(4, 128, 96)[:, :, : 12 * num_frames].reshape(
                            4, 128 * 12 * num_frames
                        ),
                    ],
                    1,
                )
                .reshape(4 * (4096 + 128 * 12 * num_frames))
                .contiguous()
            )

    adapted.setdefault("meta", {})["f4_adaptation"] = (
        "Sliced temporal tensors to first 4 frames for reproducible Orin f4 keyframe experiments"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapted, target)
    print(f"wrote {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="RaCFormer repository root")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("../RaCFormer_artifacts/orin"),
        help="Directory for generated heavyweight artifacts outside Git",
    )
    parser.add_argument("--force-checkpoint", action="store_true", help="Regenerate checkpoint even if it exists")
    parser.add_argument("--configs-only", action="store_true", help="Only generate stable configs")
    args = parser.parse_args()

    repo = args.repo.resolve()
    generate_configs(repo)
    if not args.configs_only:
        artifact_root = args.artifact_root
        if not artifact_root.is_absolute():
            artifact_root = (repo / artifact_root).resolve()
        generate_f4_checkpoint(repo, artifact_root=artifact_root, force=args.force_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
