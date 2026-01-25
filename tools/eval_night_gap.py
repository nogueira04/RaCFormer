#!/usr/bin/env python
"""
Night Performance Gap Verification

Evaluates RaCFormer checkpoint separately on day vs night scenes to verify
the performance gap reported in the paper (Table 8: Day 54.1%, Night 31.6%).

Approach:
- Run inference on FULL val set ONCE
- For each condition (day/night/rain), filter both GT and predictions
- Run official nuScenes evaluation on each filtered subset

Usage:
    python tools/eval_night_gap.py \
        configs/racformer_r50_nuimg_704x256_f8.py \
        checkpoints/racformer_r50_f8.pth \
        --out results/night_gap_verification.pkl

Output:
    - Full mAP/NDS metrics per condition (day, night, rain)
    - Per-class AP breakdown
    - Comparison to paper Table 8
    - Markdown report saved to research-notes/racformer/NIGHT_GAP_VERIFICATION.md
"""

import argparse
import os
import sys
import copy
import pickle
import tempfile
import json
from collections import defaultdict
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loaders  # noqa: F401
import models   # noqa: F401

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.loaders import load_gt as original_load_gt
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory


CLASS_NAMES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]


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


def run_inference(model, data_loader):
    """Run inference and return results."""
    model.eval()
    results = []

    for data in tqdm(data_loader, desc="  Inference"):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)

    return results


def evaluate_subset_with_filtering(dataset, full_results, indices, condition_name, nusc):
    """
    Evaluate results for a specific subset by filtering GT and predictions.

    This approach:
    1. Gets the sample tokens for the subset
    2. Filters predictions to only include those samples
    3. Monkeypatches load_gt to filter GT boxes
    4. Runs official nuScenes evaluation
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {condition_name.upper()} ({len(indices)} samples)")
    print("="*70)

    if len(indices) == 0:
        return {'mAP': 0, 'NDS': 0, 'per_class_ap': {}}

    # Get sample tokens for this subset
    valid_tokens = set([dataset.data_infos[i]['token'] for i in indices])

    # Filter results to only include this subset
    subset_results = [full_results[i] for i in indices]

    # Create a temporary dataset-like object with filtered data_infos
    subset_dataset = copy.copy(dataset)
    subset_dataset.data_infos = [dataset.data_infos[i] for i in indices]

    # Create custom load_gt that filters to only our subset
    def custom_load_gt(nusc_obj, eval_split, box_cls, verbose=False):
        gt_boxes = original_load_gt(nusc_obj, eval_split, box_cls, verbose)

        # Filter to only include our subset tokens
        filtered_boxes = EvalBoxes()
        for sample_token in gt_boxes.sample_tokens:
            if sample_token in valid_tokens:
                filtered_boxes.add_boxes(sample_token, gt_boxes[sample_token])

        return filtered_boxes

    # Monkeypatch the load_gt function
    import nuscenes.eval.detection.evaluate as eval_module
    original_func = eval_module.load_gt
    eval_module.load_gt = custom_load_gt

    try:
        # Run evaluation on the subset
        eval_results = subset_dataset.evaluate(subset_results, metric='bbox')
    finally:
        # Restore original function
        eval_module.load_gt = original_func

    # Extract metrics
    mAP = get_metric(eval_results, 'mAP')
    NDS = get_metric(eval_results, 'NDS')
    per_class_ap = extract_per_class_ap(eval_results)

    print(f"\n  {condition_name} Results:")
    print(f"    mAP: {mAP:.4f} ({mAP:.1%})")
    print(f"    NDS: {NDS:.4f} ({NDS:.1%})")

    return {
        'mAP': mAP,
        'NDS': NDS,
        'per_class_ap': per_class_ap,
        'raw_eval': eval_results,
    }


def extract_per_class_ap(eval_results):
    """Extract per-class AP from evaluation results."""
    per_class_ap = {}
    for class_name in CLASS_NAMES:
        # Try different key formats
        for key_fmt in [f'{class_name}_AP', f'pts_bbox_{class_name}_AP',
                        f'pts_bbox_NuScenes/{class_name}_AP']:
            if key_fmt in eval_results:
                per_class_ap[class_name] = eval_results[key_fmt]
                break

        # If not found, try to find it in nested structure
        if class_name not in per_class_ap:
            for key, value in eval_results.items():
                if class_name.lower() in key.lower() and 'ap' in key.lower():
                    per_class_ap[class_name] = value
                    break

    return per_class_ap


def get_metric(results, key):
    """Extract metric with fallback for different key formats."""
    for k in [key, f'pts_bbox_{key}', f'pts_bbox_NuScenes/{key}']:
        if k in results:
            return results[k]
    return 0


def generate_markdown_report(results, output_path):
    """Generate markdown report."""

    # Paper reference values (Table 8)
    paper_day_map = 0.541  # 54.1%
    paper_night_map = 0.316  # 31.6%
    paper_gap = (paper_day_map - paper_night_map) / paper_day_map  # relative gap

    day_metrics = results.get('day', {})
    night_metrics = results.get('night', {})
    rain_metrics = results.get('rain', {})
    overall_metrics = results.get('overall', {})

    day_map = day_metrics.get('mAP', 0)
    day_nds = day_metrics.get('NDS', 0)
    night_map = night_metrics.get('mAP', 0)
    night_nds = night_metrics.get('NDS', 0)
    rain_map = rain_metrics.get('mAP', 0)
    rain_nds = rain_metrics.get('NDS', 0)
    overall_map = overall_metrics.get('mAP', 0)
    overall_nds = overall_metrics.get('NDS', 0)

    # Compute gaps
    if day_map > 0:
        our_gap = (day_map - night_map) / day_map
    else:
        our_gap = 0

    report = f"""# Night Performance Gap Verification

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Checkpoint**: checkpoints/racformer_r50_f8.pth
**Config**: configs/racformer_r50_nuimg_704x256_f8.py

---

## 1. Executive Summary

### Key Finding

| Metric | Our Checkpoint | Paper Table 8 | Match? |
|--------|----------------|---------------|--------|
| Day mAP | {day_map:.1%} | {paper_day_map:.1%} | {'✅ Close' if abs(day_map - paper_day_map) < 0.03 else '❌ Different'} |
| Night mAP | {night_map:.1%} | {paper_night_map:.1%} | {'✅ Close' if abs(night_map - paper_night_map) < 0.03 else '❌ Different'} |
| Relative Gap | {our_gap:.1%} | {paper_gap:.1%} | {'✅ Similar' if abs(our_gap - paper_gap) < 0.05 else '❌ Different'} |

### Interpretation

"""

    if abs(day_map - paper_day_map) < 0.03 and abs(night_map - paper_night_map) < 0.03:
        report += """**VERIFIED**: Our checkpoint matches the paper's reported day/night performance gap.
The ~42% relative gap between day and night is confirmed.
"""
    elif our_gap > 0.3:  # >30% gap
        report += f"""**PARTIAL MATCH**: There is a significant day/night gap ({our_gap:.1%}), though
absolute numbers differ from paper. The relative gap confirms night-time degradation.
"""
    else:
        report += f"""**DISCREPANCY**: Results do not match paper Table 8. Further investigation needed.
Our gap: {our_gap:.1%} vs Paper gap: {paper_gap:.1%}
"""

    report += f"""
---

## 2. Full Metrics by Condition

### Overall Validation Set

| Metric | Value |
|--------|-------|
| mAP | {overall_map:.4f} ({overall_map:.1%}) |
| NDS | {overall_nds:.4f} ({overall_nds:.1%}) |
| Samples | {results.get('sample_counts', {}).get('total', 'N/A')} |

### Day Scenes

| Metric | Value |
|--------|-------|
| mAP | {day_map:.4f} ({day_map:.1%}) |
| NDS | {day_nds:.4f} ({day_nds:.1%}) |
| Samples | {results.get('sample_counts', {}).get('day', 'N/A')} |

### Night Scenes

| Metric | Value |
|--------|-------|
| mAP | {night_map:.4f} ({night_map:.1%}) |
| NDS | {night_nds:.4f} ({night_nds:.1%}) |
| Samples | {results.get('sample_counts', {}).get('night', 'N/A')} |

### Rain Scenes

| Metric | Value |
|--------|-------|
| mAP | {rain_map:.4f} ({rain_map:.1%}) |
| NDS | {rain_nds:.4f} ({rain_nds:.1%}) |
| Samples | {results.get('sample_counts', {}).get('rain', 'N/A')} |

---

## 3. Performance Gaps

| Comparison | mAP Gap | NDS Gap | Relative mAP Gap |
|------------|---------|---------|------------------|
| Day vs Night | {day_map - night_map:+.4f} | {day_nds - night_nds:+.4f} | {our_gap:.1%} |
| Day vs Rain | {day_map - rain_map:+.4f} | {day_nds - rain_nds:+.4f} | {((day_map - rain_map) / day_map * 100) if day_map > 0 else 0:.1f}% |

---

## 4. Per-Class AP Breakdown

"""

    # Add per-class breakdown
    day_per_class = day_metrics.get('per_class_ap', {})
    night_per_class = night_metrics.get('per_class_ap', {})
    rain_per_class = rain_metrics.get('per_class_ap', {})

    if day_per_class or night_per_class:
        report += "| Class | Day AP | Night AP | Rain AP | Day-Night Gap |\n"
        report += "|-------|--------|----------|---------|---------------|\n"

        for cls in CLASS_NAMES:
            day_ap = day_per_class.get(cls, 0)
            night_ap = night_per_class.get(cls, 0)
            rain_ap = rain_per_class.get(cls, 0)
            gap = day_ap - night_ap
            report += f"| {cls} | {day_ap:.4f} | {night_ap:.4f} | {rain_ap:.4f} | {gap:+.4f} |\n"

        report += "\n"

    report += f"""---

## 5. Comparison to Paper Table 8

| Metric | Paper Day | Paper Night | Our Day | Our Night |
|--------|-----------|-------------|---------|-----------|
| mAP | {paper_day_map:.1%} | {paper_night_map:.1%} | {day_map:.1%} | {night_map:.1%} |

### Analysis

"""

    day_diff = day_map - paper_day_map
    night_diff = night_map - paper_night_map

    if abs(day_diff) < 0.02 and abs(night_diff) < 0.02:
        report += """Our checkpoint closely matches the paper's reported performance.
The night-time degradation is real and significant.
"""
    else:
        report += f"""There are differences from paper values:
- Day mAP difference: {day_diff:+.1%}
- Night mAP difference: {night_diff:+.1%}

Possible explanations:
1. Different evaluation protocol or subset
2. Different model checkpoint (training seed, epoch)
3. Paper may report different metric variant
"""

    report += f"""
---

## 6. Conclusion

"""

    if night_map < day_map * 0.7:  # >30% degradation
        report += f"""**CONFIRMED**: Significant night-time performance degradation exists.

- Night mAP is {(1 - night_map/day_map)*100:.1f}% lower than day mAP
- This validates the motivation for night-specific improvements
- However, previous experiments (Path B) showed that inference-time
  adaptive fusion does not help — training-time solutions are needed

**Blocking Status**: Verification complete. Night gap confirmed.
"""
    else:
        report += f"""**INCONCLUSIVE**: Night gap is smaller than expected ({our_gap:.1%}).
Further investigation may be needed.
"""

    report += f"""
---

*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Write report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(description='Verify night performance gap')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='results/night_gap_verification.pkl',
                        help='output pickle file')
    parser.add_argument('--report', default='research-notes/racformer/NIGHT_GAP_VERIFICATION.md',
                        help='output markdown report')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)

    # Load config
    print("Loading config...")
    cfg = Config.fromfile(args.config)

    # Build full dataset
    print("Building dataset...")
    dataset = build_dataset(cfg.data.val)
    print(f"  Total samples: {len(dataset)}")

    # Load nuScenes for condition lookup
    print("\nLoading nuScenes for scene condition lookup...")
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes/', verbose=False)

    # Index samples by condition
    print("Indexing samples by lighting condition...")
    condition_indices = defaultdict(list)
    for i in tqdm(range(len(dataset)), desc="  Indexing"):
        sample_token = dataset.data_infos[i]['token']
        condition = get_scene_condition(nusc, sample_token)
        condition_indices[condition].append(i)

    sample_counts = {
        'total': len(dataset),
        'day': len(condition_indices['day']),
        'night': len(condition_indices['night']),
        'rain': len(condition_indices['rain']),
    }

    print(f"\nDataset split:")
    print(f"  Day:   {sample_counts['day']} samples ({sample_counts['day']/sample_counts['total']*100:.1f}%)")
    print(f"  Night: {sample_counts['night']} samples ({sample_counts['night']/sample_counts['total']*100:.1f}%)")
    print(f"  Rain:  {sample_counts['rain']} samples ({sample_counts['rain']/sample_counts['total']*100:.1f}%)")

    # Build model
    print("\nBuilding model...")
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda(args.gpu_id)
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    model.eval()

    # Build dataloader for FULL dataset
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    # Run inference on FULL dataset ONCE
    print(f"\n{'='*70}")
    print("Running inference on FULL validation set...")
    print("="*70)
    full_results = run_inference(model, data_loader)
    print(f"  Got {len(full_results)} results")

    # Store all results
    all_results = {
        'sample_counts': sample_counts,
        'condition_indices': dict(condition_indices),
    }

    # First, get overall metrics
    print(f"\n{'='*70}")
    print("Evaluating OVERALL (full validation set)")
    print("="*70)
    overall_eval = dataset.evaluate(full_results, metric='bbox')
    overall_mAP = get_metric(overall_eval, 'mAP')
    overall_NDS = get_metric(overall_eval, 'NDS')
    print(f"\n  Overall mAP: {overall_mAP:.4f} ({overall_mAP:.1%})")
    print(f"  Overall NDS: {overall_NDS:.4f} ({overall_NDS:.1%})")

    all_results['overall'] = {
        'mAP': overall_mAP,
        'NDS': overall_NDS,
        'raw_eval': overall_eval,
    }

    # Evaluate each condition with proper filtering
    for condition in ['day', 'night', 'rain']:
        indices = condition_indices[condition]
        if len(indices) == 0:
            print(f"\nSkipping {condition} - no samples")
            continue

        metrics = evaluate_subset_with_filtering(
            dataset, full_results, indices, condition, nusc
        )
        all_results[condition] = metrics

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    day_map = all_results.get('day', {}).get('mAP', 0)
    night_map = all_results.get('night', {}).get('mAP', 0)
    rain_map = all_results.get('rain', {}).get('mAP', 0)

    print(f"\n{'Condition':<12} | {'mAP':>10} | {'NDS':>10} | {'Samples':>8}")
    print("-"*50)
    print(f"{'Day':<12} | {day_map:>10.4f} | {all_results.get('day', {}).get('NDS', 0):>10.4f} | {sample_counts['day']:>8}")
    print(f"{'Night':<12} | {night_map:>10.4f} | {all_results.get('night', {}).get('NDS', 0):>10.4f} | {sample_counts['night']:>8}")
    print(f"{'Rain':<12} | {rain_map:>10.4f} | {all_results.get('rain', {}).get('NDS', 0):>10.4f} | {sample_counts['rain']:>8}")
    print("-"*50)
    print(f"{'Overall':<12} | {overall_mAP:>10.4f} | {overall_NDS:>10.4f} | {sample_counts['total']:>8}")

    # Compare to paper
    print("\n" + "-"*50)
    print("Comparison to Paper Table 8:")
    print(f"  Paper Day mAP:   54.1%")
    print(f"  Our Day mAP:     {day_map:.1%}")
    print(f"  Paper Night mAP: 31.6%")
    print(f"  Our Night mAP:   {night_map:.1%}")

    if day_map > 0:
        our_gap = (day_map - night_map) / day_map
        paper_gap = (0.541 - 0.316) / 0.541
        print(f"\n  Paper relative gap: {paper_gap:.1%}")
        print(f"  Our relative gap:   {our_gap:.1%}")

    # Save pickle
    print(f"\nSaving results to {args.out}...")
    with open(args.out, 'wb') as f:
        pickle.dump(all_results, f)

    # Generate markdown report
    print(f"Generating report to {args.report}...")
    report = generate_markdown_report(all_results, args.report)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"\nResults saved to: {args.out}")
    print(f"Report saved to: {args.report}")


if __name__ == '__main__':
    main()
