#!/usr/bin/env python
"""
Accuracy Validation Script for RaCFormer Optimized Inference

This script validates that optimized inference configurations maintain
acceptable accuracy compared to the baseline FP32 inference.

Accuracy Threshold: NDS > 0.95 × baseline, mAP > 0.95 × baseline

Usage:
    # Validate FP16 accuracy on a subset
    python validate_accuracy.py configs/racformer_r50_nuimg_704x256_f8.py \
        checkpoints/racformer.pth --fp16 --num-samples 100

    # Full validation (all samples)
    python validate_accuracy.py configs/racformer_r50_nuimg_704x256_f8.py \
        checkpoints/racformer.pth --fp16
"""

import os
import sys
import argparse
import importlib
import numpy as np
import torch
import torch.cuda
import torch.backends.cudnn as cudnn
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ['NUSCENES_VERSION'] = os.environ.get('NUSCENES_VERSION', 'v1.0-mini')

from mmengine.config import Config
from mmengine.runner import load_checkpoint, set_random_seed

from mmdet3d.registry import MODELS, DATASETS


def build_dataset(cfg, default_args=None):
    return DATASETS.build(cfg, default_args=default_args)


def pseudo_collate(batch):
    return batch


def build_model(cfg):
    return MODELS.build(cfg)


def move_to_device(data, device, non_blocking=True):
    """Move data tensors to the specified device."""
    for key in ['img', 'radar_depth', 'radar_rcs']:
        if key in data:
            if isinstance(data[key], list):
                data[key] = [d.to(device, non_blocking=non_blocking).float() if isinstance(d, torch.Tensor) else d for d in data[key]]
            elif isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device, non_blocking=non_blocking).float()
    if 'radar_points' in data:
        if isinstance(data['radar_points'], list):
            new_pts = []
            for pts in data['radar_points']:
                if isinstance(pts, list):
                    new_pts.append([p.to(device, non_blocking=non_blocking).float() if isinstance(p, torch.Tensor) else p for p in pts])
                elif isinstance(pts, torch.Tensor):
                    new_pts.append(pts.to(device, non_blocking=non_blocking).float())
                else:
                    new_pts.append(pts)
            data['radar_points'] = new_pts
    return data


def compare_outputs(baseline_outputs, optimized_outputs, threshold=0.01):
    """
    Compare model outputs between baseline and optimized inference.

    Args:
        baseline_outputs: List of baseline inference results
        optimized_outputs: List of optimized inference results
        threshold: Maximum allowed relative difference

    Returns:
        dict with comparison statistics
    """
    if len(baseline_outputs) != len(optimized_outputs):
        return {'match': False, 'error': 'Different number of outputs'}

    total_boxes_baseline = 0
    total_boxes_optimized = 0
    matching_boxes = 0
    score_diffs = []
    position_diffs = []

    for baseline, optimized in zip(baseline_outputs, optimized_outputs):
        if 'pts_bbox' not in baseline or 'pts_bbox' not in optimized:
            continue

        b_bbox = baseline['pts_bbox']
        o_bbox = optimized['pts_bbox']

        # Count boxes
        b_scores = b_bbox.get('scores_3d', torch.tensor([]))
        o_scores = o_bbox.get('scores_3d', torch.tensor([]))

        if hasattr(b_scores, 'cpu'):
            b_scores = b_scores.cpu().numpy()
        if hasattr(o_scores, 'cpu'):
            o_scores = o_scores.cpu().numpy()

        total_boxes_baseline += len(b_scores)
        total_boxes_optimized += len(o_scores)

        # Compare scores for matching detections
        min_len = min(len(b_scores), len(o_scores))
        if min_len > 0:
            # Sort by score for comparison
            b_sorted_idx = np.argsort(b_scores)[::-1][:min_len]
            o_sorted_idx = np.argsort(o_scores)[::-1][:min_len]

            b_top = b_scores[b_sorted_idx]
            o_top = o_scores[o_sorted_idx]

            diff = np.abs(b_top - o_top)
            score_diffs.extend(diff.tolist())

            # Count how many are within threshold
            matching_boxes += np.sum(diff < threshold)

    # Calculate statistics
    if len(score_diffs) > 0:
        mean_diff = np.mean(score_diffs)
        max_diff = np.max(score_diffs)
    else:
        mean_diff = 0.0
        max_diff = 0.0

    match_rate = matching_boxes / max(total_boxes_baseline, 1)

    return {
        'match': match_rate > 0.95,
        'match_rate': match_rate,
        'mean_score_diff': mean_diff,
        'max_score_diff': max_diff,
        'total_boxes_baseline': total_boxes_baseline,
        'total_boxes_optimized': total_boxes_optimized,
        'matching_boxes': matching_boxes,
    }


def run_inference(model, val_loader, device, use_fp16=False, num_samples=None, desc="Inference"):
    """
    Run inference on validation set.

    Returns:
        List of inference results
    """
    if hasattr(model, 'module'):
        m = model.module
    else:
        m = model

    results = []
    data_iter = iter(val_loader)

    if num_samples is None:
        num_samples = len(val_loader)

    pbar = tqdm(range(num_samples), desc=desc)
    for i in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        sample = batch[0]
        if isinstance(sample, (list, tuple)):
            while isinstance(sample, (list, tuple)) and len(sample) == 1:
                sample = sample[0]
        data = sample
        for key in ['img_metas', 'img', 'radar_points', 'radar_depth', 'radar_rcs']:
            if key in data and not isinstance(data[key], list):
                data[key] = [data[key]]
        data = move_to_device(data, device, non_blocking=True)

        with torch.no_grad():
            if use_fp16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    result = m(return_loss=False, rescale=True, **data)
            else:
                result = m(return_loss=False, rescale=True, **data)

        # Convert tensors to CPU for storage
        if isinstance(result, list) and len(result) > 0:
            for r in result:
                if 'pts_bbox' in r:
                    for k, v in r['pts_bbox'].items():
                        if hasattr(v, 'cpu'):
                            r['pts_bbox'][k] = v.cpu()
        results.extend(result if isinstance(result, list) else [result])

    return results


def main():
    parser = argparse.ArgumentParser(description='Validate RaCFormer optimized inference accuracy')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples (None=all)')
    parser.add_argument('--fp16', action='store_true', help='Validate FP16 accuracy')
    parser.add_argument('--threshold', type=float, default=0.05, help='Accuracy threshold (0.05=95%)')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    cfg = Config.fromfile(args.config)

    # Register custom modules
    importlib.import_module('models')
    importlib.import_module('loaders')
    import mmdet3d.datasets.transforms

    from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
    from mmdet3d.registry import TRANSFORMS as MMDET3D_TRANSFORMS
    for name, module in MMDET3D_TRANSFORMS.module_dict.items():
        if name not in MMENGINE_TRANSFORMS.module_dict:
            MMENGINE_TRANSFORMS.register_module(name=name, module=module)

    from mmdet.registry import MODELS as MMDET_MODELS
    from mmdet3d.registry import MODELS as MMDET3D_MODELS
    for name, module in MMDET_MODELS.module_dict.items():
        if name not in MMDET3D_MODELS.module_dict:
            MMDET3D_MODELS.register_module(name=name, module=module)

    from mmdet.registry import TASK_UTILS as MMDET_TASK_UTILS
    from mmdet3d.registry import TASK_UTILS as MMDET3D_TASK_UTILS
    for name, module in MMDET_TASK_UTILS.module_dict.items():
        if name not in MMDET3D_TASK_UTILS.module_dict:
            MMDET3D_TASK_UTILS.register_module(name=name, module=module)
    for name, module in MMDET3D_TASK_UTILS.module_dict.items():
        if name not in MMDET_TASK_UTILS.module_dict:
            MMDET_TASK_UTILS.register_module(name=name, module=module)

    import logging
    logging.getLogger('mmengine').setLevel(logging.WARNING)
    logging.getLogger('mmcv').setLevel(logging.WARNING)

    assert torch.cuda.is_available()
    set_random_seed(0, deterministic=False)
    cudnn.benchmark = True

    # Build model
    print("Building model...")
    model = build_model(cfg.model)

    print(f"Loading checkpoint from {args.checkpoint}")
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = model.to(args.device)
    model.eval()

    # Build dataset and dataloader
    print("Building dataset...")
    val_dataset = build_dataset(cfg.data.val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pseudo_collate,
    )

    num_samples = args.num_samples if args.num_samples else len(val_loader)
    print(f"Validating on {num_samples} samples...")

    # Run baseline (FP32) inference
    print("\n" + "="*60)
    print("Running baseline (FP32) inference...")
    print("="*60)
    baseline_results = run_inference(
        model, val_loader, args.device,
        use_fp16=False, num_samples=num_samples,
        desc="FP32 Baseline"
    )

    # Run optimized inference
    if args.fp16:
        print("\n" + "="*60)
        print("Running FP16 inference...")
        print("="*60)
        optimized_results = run_inference(
            model, val_loader, args.device,
            use_fp16=True, num_samples=num_samples,
            desc="FP16 Optimized"
        )

        # Compare results
        print("\n" + "="*60)
        print("ACCURACY COMPARISON: FP32 vs FP16")
        print("="*60)

        comparison = compare_outputs(baseline_results, optimized_results, threshold=0.01)

        print(f"\nTotal baseline detections: {comparison['total_boxes_baseline']}")
        print(f"Total optimized detections: {comparison['total_boxes_optimized']}")
        print(f"Matching detections: {comparison['matching_boxes']}")
        print(f"Match rate: {comparison['match_rate']*100:.2f}%")
        print(f"Mean score difference: {comparison['mean_score_diff']:.4f}")
        print(f"Max score difference: {comparison['max_score_diff']:.4f}")

        # Check if accuracy is acceptable
        accuracy_ok = comparison['match_rate'] >= (1.0 - args.threshold)

        print("\n" + "-"*60)
        if accuracy_ok:
            print(f"VALIDATION PASSED: FP16 accuracy is within {args.threshold*100:.0f}% of FP32")
        else:
            print(f"VALIDATION FAILED: FP16 accuracy drop exceeds {args.threshold*100:.0f}%")
            print("Consider using FP32 for production or fine-tuning with FP16.")
        print("-"*60)

    else:
        print("\nNo optimization specified. Use --fp16 to validate FP16 accuracy.")

    # Additional validation tips
    print("\n" + "="*60)
    print("VALIDATION NOTES")
    print("="*60)
    print("""
For comprehensive accuracy validation, run the full evaluation:

    python tools/test.py {config} {checkpoint} --eval mAP

This will compute:
- mAP (mean Average Precision) across all classes
- NDS (nuScenes Detection Score)
- Per-class AP metrics

Target thresholds for optimization acceptance:
- NDS >= 0.95 * baseline_NDS
- mAP >= 0.95 * baseline_mAP
    """.format(config=args.config, checkpoint=args.checkpoint))


if __name__ == '__main__':
    main()
