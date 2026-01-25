#!/usr/bin/env python
"""
Feature Quality Analysis by Condition

Analyzes whether image features degrade at night while radar stays stable.
If true, it validates the hypothesis for adaptive fusion.

Usage:
    python tools/analyze_feature_quality.py \
        configs/racformer_r50_nuimg_704x256_f8.py \
        checkpoints/racformer_r50_f8.pth \
        --num-samples 200
"""

import argparse
import os
import sys

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loaders  # noqa: F401
import models   # noqa: F401

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


def analyze_feature_quality(model, dataset, nusc, num_samples=200, gpu_id=0):
    """Compare feature statistics between day and night."""
    
    model.eval()
    stats = defaultdict(lambda: defaultdict(list))
    
    # Storage for captured features
    features = {}
    
    # Hook to capture features before fusion
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                features[name] = output.detach()
            elif isinstance(output, tuple):
                features[name] = output[0].detach()
        return hook
    
    # Get the decoder layer
    decoder = model.module.pts_bbox_head.transformer.decoder.decoder_layer
    
    # Register hooks on the norm layers (they receive the features after processing)
    hooks = []
    hooks.append(decoder.norm2.register_forward_hook(hook_fn('img')))  # After mixing
    hooks.append(decoder.norm_radar_bev.register_forward_hook(hook_fn('radar')))
    hooks.append(decoder.norm_lss_bev.register_forward_hook(hook_fn('lss')))
    
    # Build dataloader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )
    
    sample_count = {'day': 0, 'night': 0, 'rain': 0}
    
    try:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, total=min(num_samples, len(data_loader)))):
                if i >= num_samples:
                    break
                
                # Get condition from metadata
                sample_token = dataset.data_infos[i]['token']
                condition = get_scene_condition(nusc, sample_token)
                sample_count[condition] += 1
                
                # Forward pass
                try:
                    model(return_loss=False, rescale=True, **batch)
                except Exception as e:
                    print(f"Warning: Error in sample {i}: {e}")
                    continue
                
                # Record feature statistics for each modality
                for mod in ['img', 'radar', 'lss']:
                    if mod not in features:
                        continue
                    feat = features[mod]
                    
                    # Compute statistics
                    stats[condition][mod].append({
                        'norm': feat.norm(dim=-1).mean().item(),
                        'std': feat.std(dim=-1).mean().item(),
                        'mean': feat.mean().item(),
                        'max': feat.abs().max().item(),
                        'min': feat.min().item(),
                        'sparsity': (feat.abs() < 0.01).float().mean().item(),  # % near zero
                    })
                
                features.clear()
                
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    # Print results
    print("\n" + "="*80)
    print("FEATURE QUALITY ANALYSIS BY CONDITION")
    print("="*80)
    print(f"Samples analyzed: day={sample_count['day']}, night={sample_count['night']}, rain={sample_count['rain']}")
    
    for mod in ['img', 'radar', 'lss']:
        print(f"\n{'='*80}")
        print(f"{mod.upper()} FEATURES")
        print("="*80)
        print(f"{'Condition':<10} | {'Norm':>10} | {'Std':>10} | {'Mean':>10} | {'Max':>10} | {'Sparsity':>10}")
        print("-"*80)
        
        for condition in ['day', 'night', 'rain']:
            if len(stats[condition][mod]) == 0:
                continue
                
            norm = np.mean([s['norm'] for s in stats[condition][mod]])
            std = np.mean([s['std'] for s in stats[condition][mod]])
            mean = np.mean([s['mean'] for s in stats[condition][mod]])
            max_val = np.mean([s['max'] for s in stats[condition][mod]])
            sparsity = np.mean([s['sparsity'] for s in stats[condition][mod]])
            
            print(f"{condition:<10} | {norm:10.4f} | {std:10.4f} | {mean:10.4f} | {max_val:10.4f} | {sparsity*100:9.1f}%")
    
    # Compare day vs night
    print("\n" + "="*80)
    print("DAY vs NIGHT COMPARISON (Δ%)")
    print("="*80)
    print(f"{'Modality':<10} | {'Δ Norm':>12} | {'Δ Std':>12} | {'Δ Max':>12} | {'Interpretation':<30}")
    print("-"*80)
    
    for mod in ['img', 'radar', 'lss']:
        if len(stats['day'][mod]) == 0 or len(stats['night'][mod]) == 0:
            continue
            
        day_norm = np.mean([s['norm'] for s in stats['day'][mod]])
        night_norm = np.mean([s['norm'] for s in stats['night'][mod]])
        day_std = np.mean([s['std'] for s in stats['day'][mod]])
        night_std = np.mean([s['std'] for s in stats['night'][mod]])
        day_max = np.mean([s['max'] for s in stats['day'][mod]])
        night_max = np.mean([s['max'] for s in stats['night'][mod]])
        
        delta_norm = (night_norm / day_norm - 1) * 100
        delta_std = (night_std / day_std - 1) * 100
        delta_max = (night_max / day_max - 1) * 100
        
        # Interpretation
        if mod == 'img':
            if delta_norm < -10:
                interp = "⚠️ Degrades at night"
            elif delta_norm > 10:
                interp = "Stronger at night (unexpected)"
            else:
                interp = "Stable across conditions"
        elif mod == 'radar':
            if abs(delta_norm) < 15:
                interp = "✓ Stable (as expected)"
            else:
                interp = "Varies with condition"
        else:
            interp = ""
        
        print(f"{mod:<10} | {delta_norm:+11.1f}% | {delta_std:+11.1f}% | {delta_max:+11.1f}% | {interp}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("If IMAGE features degrade at night (lower norm) while RADAR stays stable,")
    print("then adaptive fusion that trusts radar more at night should help.")
    print("="*80)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Analyze feature quality by condition')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='number of samples per condition to analyze')
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args()
    
    # Load config and build model
    cfg = Config.fromfile(args.config)
    
    # Build dataset
    dataset = build_dataset(cfg.data.val)
    
    # Load nuScenes for condition lookup
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes/', verbose=False)
    
    # Pre-index samples by condition
    print("Pre-indexing samples by condition...")
    condition_indices = defaultdict(list)
    for i in tqdm(range(len(dataset))):
        sample_token = dataset.data_infos[i]['token']
        condition = get_scene_condition(nusc, sample_token)
        condition_indices[condition].append(i)
    
    print(f"Found: day={len(condition_indices['day'])}, "
          f"night={len(condition_indices['night'])}, rain={len(condition_indices['rain'])}")
    
    # Sample from each condition
    selected_indices = []
    for condition in ['day', 'night', 'rain']:
        n_select = min(args.num_samples, len(condition_indices[condition]))
        selected_indices.extend(condition_indices[condition][:n_select])
    
    print(f"Analyzing {len(selected_indices)} samples total ({args.num_samples} per condition max)")
    
    # Build model
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda(args.gpu_id)
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    model.eval()
    
    # Run analysis with selected indices
    analyze_feature_quality_indexed(model, dataset, nusc, selected_indices, args.gpu_id)


def analyze_feature_quality_indexed(model, dataset, nusc, indices, gpu_id=0):
    """Compare feature statistics between day and night using pre-selected indices."""
    
    model.eval()
    stats = defaultdict(lambda: defaultdict(list))
    
    # Storage for captured features
    features = {}
    
    # Hook to capture features before fusion
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                features[name] = output.detach()
            elif isinstance(output, tuple):
                features[name] = output[0].detach()
        return hook
    
    # Get the decoder layer
    decoder = model.module.pts_bbox_head.transformer.decoder.decoder_layer
    
    # Register hooks on the norm layers
    hooks = []
    hooks.append(decoder.norm2.register_forward_hook(hook_fn('img')))
    hooks.append(decoder.norm_radar_bev.register_forward_hook(hook_fn('radar')))
    hooks.append(decoder.norm_lss_bev.register_forward_hook(hook_fn('lss')))
    
    # Build dataloader for full dataset
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )
    
    sample_count = {'day': 0, 'night': 0, 'rain': 0}
    indices_set = set(indices)
    
    try:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
                if i not in indices_set:
                    continue
                
                # Get condition from metadata
                sample_token = dataset.data_infos[i]['token']
                condition = get_scene_condition(nusc, sample_token)
                sample_count[condition] += 1
                
                # Forward pass
                try:
                    model(return_loss=False, rescale=True, **batch)
                except Exception as e:
                    print(f"Warning: Error in sample {i}: {e}")
                    continue
                
                # Record feature statistics for each modality
                for mod in ['img', 'radar', 'lss']:
                    if mod not in features:
                        continue
                    feat = features[mod]
                    
                    # Compute statistics
                    stats[condition][mod].append({
                        'norm': feat.norm(dim=-1).mean().item(),
                        'std': feat.std(dim=-1).mean().item(),
                        'mean': feat.mean().item(),
                        'max': feat.abs().max().item(),
                        'min': feat.min().item(),
                        'sparsity': (feat.abs() < 0.01).float().mean().item(),
                    })
                
                features.clear()
                
    finally:
        for hook in hooks:
            hook.remove()
    
    # Print results
    print("\n" + "="*80)
    print("FEATURE QUALITY ANALYSIS BY CONDITION")
    print("="*80)
    print(f"Samples analyzed: day={sample_count['day']}, night={sample_count['night']}, rain={sample_count['rain']}")
    
    for mod in ['img', 'radar', 'lss']:
        print(f"\n{'='*80}")
        print(f"{mod.upper()} FEATURES")
        print("="*80)
        print(f"{'Condition':<10} | {'Norm':>10} | {'Std':>10} | {'Mean':>10} | {'Max':>10} | {'Sparsity':>10}")
        print("-"*80)
        
        for condition in ['day', 'night', 'rain']:
            if len(stats[condition][mod]) == 0:
                continue
                
            norm = np.mean([s['norm'] for s in stats[condition][mod]])
            std = np.mean([s['std'] for s in stats[condition][mod]])
            mean = np.mean([s['mean'] for s in stats[condition][mod]])
            max_val = np.mean([s['max'] for s in stats[condition][mod]])
            sparsity = np.mean([s['sparsity'] for s in stats[condition][mod]])
            
            print(f"{condition:<10} | {norm:10.4f} | {std:10.4f} | {mean:10.4f} | {max_val:10.4f} | {sparsity*100:9.1f}%")
    
    # Compare day vs night
    print("\n" + "="*80)
    print("DAY vs NIGHT COMPARISON (Δ%)")
    print("="*80)
    print(f"{'Modality':<10} | {'Δ Norm':>12} | {'Δ Std':>12} | {'Δ Max':>12} | {'Interpretation':<30}")
    print("-"*80)
    
    for mod in ['img', 'radar', 'lss']:
        if len(stats['day'][mod]) == 0 or len(stats['night'][mod]) == 0:
            continue
            
        day_norm = np.mean([s['norm'] for s in stats['day'][mod]])
        night_norm = np.mean([s['norm'] for s in stats['night'][mod]])
        day_std = np.mean([s['std'] for s in stats['day'][mod]])
        night_std = np.mean([s['std'] for s in stats['night'][mod]])
        day_max = np.mean([s['max'] for s in stats['day'][mod]])
        night_max = np.mean([s['max'] for s in stats['night'][mod]])
        
        delta_norm = (night_norm / day_norm - 1) * 100
        delta_std = (night_std / day_std - 1) * 100
        delta_max = (night_max / day_max - 1) * 100
        
        # Interpretation
        if mod == 'img':
            if delta_norm < -10:
                interp = "⚠️ Degrades at night"
            elif delta_norm > 10:
                interp = "Stronger at night (unexpected)"
            else:
                interp = "Stable across conditions"
        elif mod == 'radar':
            if abs(delta_norm) < 15:
                interp = "✓ Stable (as expected)"
            else:
                interp = "Varies with condition"
        else:
            interp = ""
        
        print(f"{mod:<10} | {delta_norm:+11.1f}% | {delta_std:+11.1f}% | {delta_max:+11.1f}% | {interp}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("If IMAGE features degrade at night (lower norm) while RADAR stays stable,")
    print("then adaptive fusion that trusts radar more at night should help.")
    print("="*80)
    
    return stats


if __name__ == '__main__':
    main()

