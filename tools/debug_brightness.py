#!/usr/bin/env python
"""
Debug script to check actual brightness values for day vs night images.
This will reveal if brightness detection is working correctly.
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from mmcv import Config
from mmdet3d.datasets import build_dataset, build_dataloader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loaders  # noqa: F401

from nuscenes.nuscenes import NuScenes


def get_scene_condition(nusc, sample_token):
    """Get condition from scene description."""
    try:
        sample = nusc.get('sample', sample_token)
        scene = nusc.get('scene', sample['scene_token'])
        description = scene.get('description', '').lower()

        if 'night' in description:
            return 'night'
        elif 'rain' in description or 'rainy' in description:
            return 'rain'
        else:
            return 'day'
    except Exception:
        return 'day'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--max-samples', type=int, default=100)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val)

    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes/', verbose=False)

    # Collect brightness values by condition
    brightness_by_condition = defaultdict(list)
    raw_stats_by_condition = defaultdict(list)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    condition_counts = {'day': 0, 'night': 0, 'rain': 0}

    print("Collecting brightness statistics...")
    for i, batch in enumerate(tqdm(data_loader)):
        if all(c >= args.max_samples for c in condition_counts.values()):
            break

        sample_token = dataset.data_infos[i]['token']
        condition = get_scene_condition(nusc, sample_token)

        if condition_counts[condition] >= args.max_samples:
            continue
        condition_counts[condition] += 1

        # Get image tensor
        if 'img' in batch:
            img = batch['img'][0].data[0]  # DataContainer -> tensor
        else:
            continue

        # Compute various brightness metrics
        img_float = img.float()

        stats = {
            'mean': img_float.mean().item(),
            'std': img_float.std().item(),
            'min': img_float.min().item(),
            'max': img_float.max().item(),
            'median': img_float.median().item(),
            # Per-channel means (R, G, B)
            'r_mean': img_float[:, 0].mean().item() if img_float.dim() >= 2 else 0,
            'g_mean': img_float[:, 1].mean().item() if img_float.dim() >= 2 else 0,
            'b_mean': img_float[:, 2].mean().item() if img_float.dim() >= 2 else 0,
        }

        raw_stats_by_condition[condition].append(stats)

        # Compute "brightness" using original formula
        mean_val = stats['mean']
        if mean_val < 0:
            brightness = (mean_val + 1.0) / 2.0
        elif mean_val > 1:
            brightness = mean_val / 255.0
        else:
            brightness = mean_val

        brightness_by_condition[condition].append(brightness)

    # Print results
    print("\n" + "="*80)
    print("RAW IMAGE STATISTICS BY CONDITION")
    print("="*80)

    for condition in ['day', 'night', 'rain']:
        if not raw_stats_by_condition[condition]:
            continue

        print(f"\n{condition.upper()} (n={len(raw_stats_by_condition[condition])})")
        print("-"*60)

        for stat in ['mean', 'std', 'min', 'max', 'median']:
            values = [s[stat] for s in raw_stats_by_condition[condition]]
            print(f"  {stat:>8}: mean={np.mean(values):+8.4f}, "
                  f"std={np.std(values):6.4f}, "
                  f"range=[{np.min(values):+.3f}, {np.max(values):+.3f}]")

    print("\n" + "="*80)
    print("COMPUTED BRIGHTNESS VALUES")
    print("="*80)

    for condition in ['day', 'night', 'rain']:
        if not brightness_by_condition[condition]:
            continue
        values = brightness_by_condition[condition]
        print(f"\n{condition.upper()}:")
        print(f"  Brightness: mean={np.mean(values):.4f}, "
              f"std={np.std(values):.4f}, "
              f"range=[{np.min(values):.4f}, {np.max(values):.4f}]")

    print("\n" + "="*80)
    print("SEPARATION ANALYSIS")
    print("="*80)

    if brightness_by_condition['day'] and brightness_by_condition['night']:
        day_mean = np.mean(brightness_by_condition['day'])
        night_mean = np.mean(brightness_by_condition['night'])
        day_std = np.std(brightness_by_condition['day'])
        night_std = np.std(brightness_by_condition['night'])

        # Cohen's d effect size
        pooled_std = np.sqrt((day_std**2 + night_std**2) / 2)
        cohens_d = (day_mean - night_mean) / pooled_std if pooled_std > 0 else 0

        print(f"\nDay vs Night:")
        print(f"  Day mean:   {day_mean:.4f}")
        print(f"  Night mean: {night_mean:.4f}")
        print(f"  Difference: {day_mean - night_mean:+.4f}")
        print(f"  Cohen's d:  {cohens_d:.2f}")

        if abs(cohens_d) < 0.2:
            print("  → NEGLIGIBLE separation - brightness cannot distinguish day/night")
        elif abs(cohens_d) < 0.5:
            print("  → SMALL separation - brightness is a weak signal")
        elif abs(cohens_d) < 0.8:
            print("  → MEDIUM separation - brightness provides some signal")
        else:
            print("  → LARGE separation - brightness is a good discriminator")

        # Threshold analysis
        print(f"\n  Threshold analysis (at 0.4):")
        day_below = sum(1 for v in brightness_by_condition['day'] if v < 0.4)
        night_below = sum(1 for v in brightness_by_condition['night'] if v < 0.4)
        print(f"    Day samples below 0.4:   {day_below}/{len(brightness_by_condition['day'])} "
              f"({100*day_below/len(brightness_by_condition['day']):.1f}%)")
        print(f"    Night samples below 0.4: {night_below}/{len(brightness_by_condition['night'])} "
              f"({100*night_below/len(brightness_by_condition['night']):.1f}%)")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    # Check if the data is ImageNet normalized
    all_means = [s['mean'] for stats in raw_stats_by_condition.values() for s in stats]
    if all_means and np.mean(all_means) < 1 and np.mean(all_means) > -1:
        print("\nImages appear to be normalized (mean close to 0).")
        print("The brightness formula may not work correctly on normalized images.")
        print("\nOptions:")
        print("1. Compute brightness BEFORE normalization (in dataloader)")
        print("2. Use a different condition signal (scene metadata)")
        print("3. Train a small network to predict scene condition")


if __name__ == '__main__':
    main()
