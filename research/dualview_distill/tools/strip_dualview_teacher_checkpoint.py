#!/usr/bin/env python3
"""Create an inference checkpoint without frozen DINOv2 teacher weights."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    before = len(state)
    drop_markers = (".dualview_distill.teacher.", "dualview_distill.teacher.")
    drop_keys = {"dualview_distill.dino_mean", "dualview_distill.dino_std"}
    for key in list(state.keys()):
        if key in drop_keys or any(marker in key for marker in drop_markers):
            del state[key]
    after = len(state)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"wrote {args.output}; removed {before - after} frozen teacher keys; kept {after}")


if __name__ == "__main__":
    main()
