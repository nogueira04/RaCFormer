#!/usr/bin/env python
"""
Weight Sweep Script for Oracle Fusion

Tests multiple weight configurations efficiently by:
1. Using subset of samples for faster iteration
2. Testing pre-defined weight configurations
3. Comparing all results at the end

Usage:
    python tools/sweep_oracle_weights.py \
        configs/racformer_r50_nuimg_704x256_f8_oracle_fusion.py \
        checkpoints/racformer_r50_f8.pth \
        --max-samples 500

Estimated time: ~5 min per configuration (with --max-samples 500)
"""

import argparse
import os
import sys
import copy
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import loaders  # noqa: F401
import models   # noqa: F401

from nuscenes.nuscenes import NuScenes


# Weight configurations to test
# Format: name -> {condition: [img, lss, radar]}
WEIGHT_CONFIGS = {
    'equal': {
        'day': [1.0, 1.0, 1.0],
        'night': [1.0, 1.0, 1.0],
        'rain': [1.0, 1.0, 1.0],
    },
    'original_oracle': {
        'day': [0.4, 0.3, 0.3],
        'night': [0.2, 0.2, 0.6],
        'rain': [0.3, 0.2, 0.5],
    },
    'radar_heavy_night': {
        'day': [1.0, 1.0, 1.0],
        'night': [0.2, 0.2, 1.5],  # 1.5x radar at night
        'rain': [0.5, 0.5, 1.2],   # 1.2x radar in rain
    },
    'radar_only_night': {
        'day': [1.0, 1.0, 1.0],
        'night': [0.1, 0.1, 2.0],  # 2x radar, suppress others at night
        'rain': [1.0, 1.0, 1.0],
    },
    'boost_radar_all': {
        'day': [1.0, 1.0, 1.3],
        'night': [1.0, 1.0, 1.5],
        'rain': [1.0, 1.0, 1.4],
    },
    'suppress_image_night': {
        'day': [1.0, 1.0, 1.0],
        'night': [0.5, 1.0, 1.0],  # Half image weight at night
        'rain': [0.8, 1.0, 1.0],
    },
}


def get_scene_conditions(nusc, dataset):
    """Get scene condition for each sample."""
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


def set_runtime_weights(cfg, weights_dict):
    """
    Set oracle weights at runtime by modifying the config.
    
    Instead of hardcoded weights in the model, we pass them via config.
    """
    cfg.model.pts_bbox_head.transformer.oracle_fusion = True
    cfg.model.pts_bbox_head.transformer.oracle_weights = weights_dict


def run_evaluation(cfg, checkpoint_path, gpu_ids, max_samples=None):
    """Run inference and evaluation with current config."""
    from mmcv.parallel import MMDataParallel
    from mmdet.apis import single_gpu_test
    from mmdet3d.datasets import build_dataloader
    
    # Build dataset
    dataset = build_dataset(cfg.data.val)
    
    # Limit samples if requested
    if max_samples and max_samples < len(dataset):
        dataset.data_infos = dataset.data_infos[:max_samples]
    
    # Build model
    model = build_model(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.cuda(gpu_ids[0])
    model = MMDataParallel(model, device_ids=gpu_ids)
    model.eval()
    
    # Build dataloader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )
    
    # Run inference
    results = single_gpu_test(model, data_loader)
    
    # Evaluate
    eval_results = dataset.evaluate(results, metric=['mAP'])
    
    return eval_results, len(dataset)


def main():
    parser = argparse.ArgumentParser(description='Sweep oracle fusion weights')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='max samples to evaluate (for speed)')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0])
    parser.add_argument('--configs', type=str, nargs='+', 
                        default=list(WEIGHT_CONFIGS.keys()),
                        help='which weight configs to test')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ORACLE FUSION WEIGHT SWEEP")
    print("="*70)
    
    # Load base config
    base_cfg = Config.fromfile(args.config)
    
    # Store results
    all_results = {}
    
    # First run baseline (no oracle)
    print("\n[1/{}] Running BASELINE (no oracle)...".format(len(args.configs) + 1))
    cfg = copy.deepcopy(base_cfg)
    cfg.model.pts_bbox_head.transformer.oracle_fusion = False
    
    baseline_results, n_samples = run_evaluation(
        cfg, args.checkpoint, args.gpu_ids, args.max_samples)
    all_results['baseline'] = {
        'mAP': baseline_results.get('pts_bbox_NuScenes/mAP', 0),
        'NDS': baseline_results.get('pts_bbox_NuScenes/NDS', 0),
    }
    print(f"  Baseline: mAP={all_results['baseline']['mAP']:.4f}, NDS={all_results['baseline']['NDS']:.4f}")
    
    # Run each weight configuration
    for i, config_name in enumerate(args.configs):
        if config_name not in WEIGHT_CONFIGS:
            print(f"  Skipping unknown config: {config_name}")
            continue
            
        print(f"\n[{i+2}/{len(args.configs)+1}] Running {config_name}...")
        weights = WEIGHT_CONFIGS[config_name]
        print(f"  Day:   {weights['day']}")
        print(f"  Night: {weights['night']}")
        print(f"  Rain:  {weights['rain']}")
        
        # Need to modify the model's forward to use these weights
        # For now, we'll need to modify the transformer code to accept runtime weights
        # This is a placeholder - actual implementation needs model code changes
        
        cfg = copy.deepcopy(base_cfg)
        set_runtime_weights(cfg, weights)
        
        try:
            results, _ = run_evaluation(
                cfg, args.checkpoint, args.gpu_ids, args.max_samples)
            all_results[config_name] = {
                'mAP': results.get('pts_bbox_NuScenes/mAP', 0),
                'NDS': results.get('pts_bbox_NuScenes/NDS', 0),
            }
            print(f"  {config_name}: mAP={all_results[config_name]['mAP']:.4f}, NDS={all_results[config_name]['NDS']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            all_results[config_name] = {'mAP': 0, 'NDS': 0, 'error': str(e)}
    
    # Print summary
    print("\n" + "="*70)
    print("WEIGHT SWEEP SUMMARY")
    print("="*70)
    print(f"{'Configuration':<25} | {'mAP':>8} | {'NDS':>8} | {'Δ mAP':>8} | {'Δ NDS':>8}")
    print("-"*70)
    
    baseline_map = all_results['baseline']['mAP']
    baseline_nds = all_results['baseline']['NDS']
    
    for config_name, metrics in all_results.items():
        delta_map = metrics['mAP'] - baseline_map if 'error' not in metrics else 0
        delta_nds = metrics['NDS'] - baseline_nds if 'error' not in metrics else 0
        delta_str = f"{delta_map:+.4f}" if config_name != 'baseline' else "---"
        delta_nds_str = f"{delta_nds:+.4f}" if config_name != 'baseline' else "---"
        print(f"{config_name:<25} | {metrics['mAP']:8.4f} | {metrics['NDS']:8.4f} | {delta_str:>8} | {delta_nds_str:>8}")
    
    print("="*70)
    print(f"\nEvaluated on {n_samples} samples")
    
    # Find best config
    best_config = max(all_results.items(), key=lambda x: x[1].get('mAP', 0))
    print(f"Best configuration: {best_config[0]} (mAP: {best_config[1]['mAP']:.4f})")


if __name__ == '__main__':
    main()
