#!/usr/bin/env python
"""
Intervention Analysis for Modality Contribution

Tests if changing modality weights helps at night by zeroing/boosting features.
No training required - uses monkey-patching to intervene during inference.

Usage:
    python tools/intervention_analysis.py \
        configs/racformer_r50_nuimg_704x256_f8.py \
        checkpoints/racformer_r50_f8.pth \
        --max-samples 300

What to look for: If boosting radar helps at night but hurts during day,
adaptive fusion is justified.
"""

import argparse
import os
import sys
import copy
from collections import defaultdict
from functools import partial

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

# Global intervention settings (modified by evaluate_with_intervention)
INTERVENTION = {
    'type': 'none',
    'img_scale': 1.0,
    'radar_scale': 1.0,
    'lss_scale': 1.0,
}


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


def patched_forward(self, query_bbox, query_feat, mlvl_feats, lss_bev_feats, radar_bev_feats, attn_mask, img_metas, layer=0):
    """
    Patched forward method that applies interventions before fusion.
    This is a copy of the original forward with intervention hooks.
    """
    query_pos = self.position_encoder(query_bbox[..., :3])
    query_feat = query_feat + query_pos

    query_feat = self.norm1(self.self_attn(query_bbox, query_feat, attn_mask))

    query_radar_feat = self.sampling_radar_bev(query_bbox, query_feat, radar_bev_feats, img_metas, d_region=self.d_region_list[layer])
    query_radar_feat = self.norm_radar_bev(query_radar_feat)
    query_lss_feat = self.sampling_lss_bev(query_bbox, query_feat, lss_bev_feats, img_metas, d_region=self.d_region_list[layer])
    query_lss_feat = self.norm_lss_bev(query_lss_feat)

    sampled_feat = self.sampling(query_bbox, query_feat, mlvl_feats, img_metas, d_region=self.d_region_list[layer])

    query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
    
    # =========== INTERVENTION POINT ===========
    # Apply scaling based on global intervention settings
    global INTERVENTION
    query_feat = query_feat * INTERVENTION['img_scale']
    query_radar_feat = query_radar_feat * INTERVENTION['radar_scale']
    query_lss_feat = query_lss_feat * INTERVENTION['lss_scale']
    # ==========================================
    
    # Standard fusion
    oracle_fusion = getattr(self, 'oracle_fusion', False)
    
    if oracle_fusion:
        condition = img_metas[0].get('scene_condition', 'day')
        oracle_weights = getattr(self, 'oracle_weights', None)
        
        if oracle_weights and condition in oracle_weights:
            weights = oracle_weights[condition]
        else:
            if condition == 'night':
                weights = [0.2, 0.2, 0.6]
            elif condition == 'rain':
                weights = [0.3, 0.2, 0.5]
            else:
                weights = [0.4, 0.3, 0.3]
        
        scaled_img_feat = weights[0] * query_feat
        scaled_lss_feat = weights[1] * query_lss_feat
        scaled_radar_feat = weights[2] * query_radar_feat
        
        query_feat = self.norm_fusion(self.fusion(
            torch.cat((scaled_img_feat, scaled_radar_feat, scaled_lss_feat), dim=-1)))
    else:
        query_feat = self.norm_fusion(self.fusion(torch.cat((query_feat, query_radar_feat, query_lss_feat), dim=-1)))
    
    query_feat = self.norm3(self.ffn(query_feat))

    cls_score = self.cls_branch(query_feat)
    bbox_pred = self.reg_branch(query_feat)
    bbox_pred = self.refine_bbox(query_bbox, bbox_pred)

    time_diff = img_metas[0]['time_diff']
    if time_diff.shape[1] > 1:
        time_diff = time_diff.clone()
        time_diff[time_diff < 1e-5] = 1.0
        bbox_pred[..., 8:] = bbox_pred[..., 8:] / time_diff[:, 1:2, None]

    return query_feat, cls_score, bbox_pred


def evaluate_subset(model, dataset, indices, condition_name):
    """Evaluate on a subset of samples."""
    from mmdet.apis import single_gpu_test
    
    # Create subset
    subset_dataset = copy.copy(dataset)
    subset_dataset.data_infos = [dataset.data_infos[i] for i in indices]
    
    data_loader = build_dataloader(
        subset_dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )
    
    # Run inference
    results = single_gpu_test(model, data_loader)
    
    # Get basic metrics (avg score, avg detections)
    total_dets = 0
    total_score = 0
    
    for res in results:
        if 'pts_bbox' in res:
            res = res['pts_bbox']
        if 'scores_3d' in res and len(res['scores_3d']) > 0:
            total_dets += len(res['scores_3d'])
            total_score += res['scores_3d'].sum().item()
    
    avg_dets = total_dets / len(results) if results else 0
    avg_score = total_score / total_dets if total_dets > 0 else 0
    
    return {
        'num_samples': len(results),
        'avg_detections': avg_dets,
        'avg_score': avg_score,
    }


def main():
    parser = argparse.ArgumentParser(description='Intervention analysis for modality contribution')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--max-samples', type=int, default=300,
                        help='max samples per condition')
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Build dataset
    dataset = build_dataset(cfg.data.val)
    
    # Load nuScenes for condition lookup
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes/', verbose=False)
    
    # Get indices by condition
    condition_indices = defaultdict(list)
    for i in range(len(dataset)):
        sample_token = dataset.data_infos[i]['token']
        condition = get_scene_condition(nusc, sample_token)
        if len(condition_indices[condition]) < args.max_samples:
            condition_indices[condition].append(i)
    
    print("\n" + "="*80)
    print("INTERVENTION ANALYSIS")
    print("="*80)
    print(f"Samples per condition: day={len(condition_indices['day'])}, "
          f"night={len(condition_indices['night'])}, rain={len(condition_indices['rain'])}")
    
    # Define interventions to test
    interventions = [
        ('baseline', {'img_scale': 1.0, 'radar_scale': 1.0, 'lss_scale': 1.0}),
        ('boost_radar_1.5x', {'img_scale': 1.0, 'radar_scale': 1.5, 'lss_scale': 1.0}),
        ('boost_radar_2x', {'img_scale': 1.0, 'radar_scale': 2.0, 'lss_scale': 1.0}),
        ('suppress_img_0.5x', {'img_scale': 0.5, 'radar_scale': 1.0, 'lss_scale': 1.0}),
        ('radar_heavy', {'img_scale': 0.5, 'radar_scale': 1.5, 'lss_scale': 0.5}),
        ('zero_radar', {'img_scale': 1.0, 'radar_scale': 0.0, 'lss_scale': 1.0}),
        ('zero_img', {'img_scale': 0.0, 'radar_scale': 1.0, 'lss_scale': 1.0}),
    ]
    
    # Store results
    all_results = defaultdict(dict)
    
    for condition in ['day', 'night', 'rain']:
        if len(condition_indices[condition]) == 0:
            print(f"Skipping {condition} - no samples")
            continue
            
        print(f"\n{'='*80}")
        print(f"Evaluating {condition.upper()} ({len(condition_indices[condition])} samples)")
        print("="*80)
        
        for interv_name, scales in interventions:
            print(f"  Running {interv_name}...", end=" ", flush=True)
            
            # Set intervention
            global INTERVENTION
            INTERVENTION.update(scales)
            
            # Build fresh model for each intervention
            model = build_model(cfg.model)
            load_checkpoint(model, args.checkpoint, map_location='cpu')
            model.cuda(args.gpu_id)
            model = MMDataParallel(model, device_ids=[args.gpu_id])
            model.eval()
            
            # Patch forward method with proper binding
            import types
            decoder_layer = model.module.pts_bbox_head.transformer.decoder.decoder_layer
            original_forward = decoder_layer.forward
            decoder_layer.forward = types.MethodType(patched_forward, decoder_layer)
            
            try:
                metrics = evaluate_subset(model, dataset, condition_indices[condition], condition)
                all_results[condition][interv_name] = metrics
                print(f"avg_dets={metrics['avg_detections']:.1f}, avg_score={metrics['avg_score']:.4f}")
            except Exception as e:
                import traceback
                print(f"Error: {e}")
                traceback.print_exc()
                all_results[condition][interv_name] = {'avg_score': 0, 'avg_detections': 0, 'error': str(e)}
            finally:
                decoder_layer.forward = original_forward
            
            del model
            torch.cuda.empty_cache()
    
    # Print summary table
    print("\n" + "="*100)
    print("INTERVENTION SUMMARY")
    print("="*100)
    
    # Header
    header = f"{'Intervention':<20}"
    for condition in ['day', 'night', 'rain']:
        header += f" | {condition:^12} | {'Δ':^8}"
    print(header)
    print("-"*100)
    
    # Get baseline scores
    baseline_scores = {}
    for condition in ['day', 'night', 'rain']:
        if 'baseline' in all_results[condition]:
            baseline_scores[condition] = all_results[condition]['baseline'].get('avg_score', 0)
    
    for interv_name, _ in interventions:
        row = f"{interv_name:<20}"
        for condition in ['day', 'night', 'rain']:
            if interv_name in all_results[condition]:
                score = all_results[condition][interv_name].get('avg_score', 0)
                baseline = baseline_scores.get(condition, 0)
                delta = ((score / baseline) - 1) * 100 if baseline > 0 else 0
                row += f" | {score:12.4f} | {delta:+7.1f}%"
            else:
                row += f" | {'N/A':^12} | {'N/A':^8}"
        print(row)
    
    print("="*100)
    print("\nINTERPRETATION:")
    print("- If 'boost_radar' helps at NIGHT but hurts at DAY → Adaptive fusion is justified")
    print("- If 'zero_radar' hurts more at NIGHT than DAY → Radar is more important at night")
    print("- If patterns are similar across conditions → Static fusion may be optimal")
    print("="*100)


if __name__ == '__main__':
    main()
