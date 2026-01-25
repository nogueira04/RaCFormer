#!/usr/bin/env python
"""
Oracle Test Evaluation Script for Condition-Specific Analysis

This script evaluates RaCFormer performance broken down by scene conditions
(day, night, rain) to measure the effectiveness of the oracle fusion weights.

Usage:
    python tools/eval_oracle_fusion.py \
        configs/racformer_r50_nuimg_704x256_f8_oracle_fusion.py \
        work_dirs/racformer_oracle/latest.pth \
        --eval mAP

What to measure:
- Overall mAP/NDS
- Per-condition mAP (day, night, rain)  
- Compare to baseline equal weighting

Success criteria: If oracle weights improve night mAP by >1%, learned weights should work.
"""

import argparse
import os
import sys
import json
from collections import defaultdict

import torch
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import loaders to register custom datasets
import loaders  # noqa: F401 - registers CustomNuScenesDataset_radar
import models   # noqa: F401 - registers custom models

from nuscenes.nuscenes import NuScenes


def get_scene_conditions(nusc, dataset):
    """
    Get scene condition for each sample in the dataset.
    
    Returns:
        dict: Maps sample_idx to condition ('day', 'night', 'rain')
        dict: Maps condition to list of indices
    """
    sample_to_condition = {}
    condition_to_indices = defaultdict(list)
    
    for idx in range(len(dataset)):
        info = dataset.data_infos[idx]
        sample_token = info['token']
        
        try:
            sample = nusc.get('sample', sample_token)
            scene = nusc.get('scene', sample['scene_token'])
            description = scene.get('description', '').lower()
            
            if 'night' in description:
                condition = 'night'
            elif 'rain' in description or 'rainy' in description:
                condition = 'rain'
            else:
                condition = 'day'
        except Exception:
            condition = 'day'
        
        sample_to_condition[idx] = condition
        condition_to_indices[condition].append(idx)
    
    return sample_to_condition, condition_to_indices


def evaluate_condition_subset(dataset, results, indices, condition_name, logger=None):
    """
    Evaluate results for a specific subset of samples.
    
    Args:
        dataset: The dataset object
        results: Full results list
        indices: Indices of samples to include
        condition_name: Name of the condition for logging
        logger: Optional logger
        
    Returns:
        dict: Evaluation metrics for this condition
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {condition_name.upper()} condition ({len(indices)} samples)")
    print(f"{'='*60}")
    
    if len(indices) == 0:
        print(f"No samples for condition: {condition_name}")
        return {}
    
    # Filter results to this subset
    subset_results = [results[i] for i in indices]
    
    # Create a subset dataset for evaluation
    # Note: This is a simplified approach; full implementation would need
    # to properly handle the dataset subset for nuScenes evaluation
    
    # For now, we'll report basic statistics
    num_detections = 0
    total_score = 0.0
    
    for res in subset_results:
        if 'pts_bbox' in res:
            res = res['pts_bbox']
        if 'scores_3d' in res and len(res['scores_3d']) > 0:
            num_detections += len(res['scores_3d'])
            total_score += res['scores_3d'].sum().item()
    
    avg_detections = num_detections / len(indices) if indices else 0
    avg_score = total_score / num_detections if num_detections > 0 else 0
    
    print(f"  Average detections per sample: {avg_detections:.2f}")
    print(f"  Average detection score: {avg_score:.4f}")
    
    return {
        'num_samples': len(indices),
        'avg_detections': avg_detections,
        'avg_score': avg_score
    }


def print_fusion_weights_summary():
    """Print the oracle fusion weights being used."""
    print("\n" + "="*60)
    print("ORACLE FUSION WEIGHTS")
    print("="*60)
    print("Condition      | Img Weight | LSS Weight | Radar Weight")
    print("-"*60)
    print("Day/Clear      |    0.4     |    0.3     |     0.3    ")
    print("Night          |    0.2     |    0.2     |     0.6    ")
    print("Rain           |    0.3     |    0.2     |     0.5    ")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate RaCFormer with Oracle Fusion by condition')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file', nargs='?', default=None)
    parser.add_argument('--eval', type=str, nargs='+', default=['mAP'],
                        help='evaluation metrics')
    parser.add_argument('--show', action='store_true',
                        help='show results')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0],
                        help='GPU ids to use')
    parser.add_argument('--condition-split', action='store_true',
                        help='show results split by condition')
    parser.add_argument('--analyze-only', action='store_true',
                        help='only analyze dataset conditions, do not run inference')
    parser.add_argument('--no-oracle', action='store_true',
                        help='disable oracle fusion weights (use baseline equal fusion)')
    args = parser.parse_args()
    
    # Print oracle weights (only if oracle is being used)
    if not args.no_oracle:
        print_fusion_weights_summary()
    else:
        print("\n" + "="*60)
        print("BASELINE MODE (No Oracle Fusion)")
        print("="*60)
        print("Using original concat+linear fusion (no adaptive weights)")
        print("="*60 + "\n")
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Override oracle_fusion if --no-oracle is specified
    if args.no_oracle:
        cfg.model.pts_bbox_head.transformer.oracle_fusion = False
        print("Oracle Fusion: DISABLED (baseline)")
    else:
        # Check if oracle_fusion is enabled
        oracle_enabled = cfg.model.pts_bbox_head.transformer.get('oracle_fusion', False)
        print(f"Oracle Fusion Enabled: {oracle_enabled}")
    
    # Build dataset
    dataset = build_dataset(cfg.data.val)
    
    # Load nuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes/', verbose=False)
    
    # Get conditions
    print("\nAnalyzing scene conditions in validation set...")
    sample_to_condition, condition_to_indices = get_scene_conditions(nusc, dataset)
    
    # Print condition distribution
    print("\n" + "="*60)
    print("CONDITION DISTRIBUTION")
    print("="*60)
    for condition, indices in sorted(condition_to_indices.items()):
        pct = 100.0 * len(indices) / len(dataset)
        print(f"  {condition:10s}: {len(indices):5d} samples ({pct:5.1f}%)")
    print(f"  {'Total':10s}: {len(dataset):5d} samples")
    print("="*60 + "\n")
    
    if args.analyze_only:
        print("Analysis complete. Use --no-analyze-only to run full evaluation.")
        return
    
    if args.checkpoint is None:
        print("No checkpoint provided. Skipping inference.")
        return
    
    # Build model
    model = build_model(cfg.model)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # Use MMDataParallel wrapper (required for proper DataContainer handling)
    from mmcv.parallel import MMDataParallel
    model.cuda(args.gpu_ids[0])
    model = MMDataParallel(model, device_ids=args.gpu_ids)
    model.eval()
    
    # Run inference
    print("\nRunning inference...")
    from mmdet.apis import single_gpu_test
    from mmdet3d.datasets import build_dataloader
    
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )
    
    results = single_gpu_test(model, data_loader)
    
    # Evaluate overall
    print("\n" + "="*60)
    print("OVERALL EVALUATION")
    print("="*60)
    eval_results = dataset.evaluate(results, metric=args.eval)
    print(eval_results)
    
    # Evaluate per condition
    if args.condition_split:
        condition_results = {}
        for condition, indices in sorted(condition_to_indices.items()):
            metrics = evaluate_condition_subset(
                dataset, results, indices, condition)
            condition_results[condition] = metrics
        
        # Print summary
        print("\n" + "="*60)
        print("CONDITION-SPECIFIC SUMMARY")
        print("="*60)
        print(f"{'Condition':<12} | {'Samples':<8} | {'Avg Dets':<10} | {'Avg Score':<10}")
        print("-"*60)
        for condition, metrics in sorted(condition_results.items()):
            print(f"{condition:<12} | {metrics['num_samples']:<8} | "
                  f"{metrics['avg_detections']:<10.2f} | {metrics['avg_score']:<10.4f}")
        print("="*60)


if __name__ == '__main__':
    main()
