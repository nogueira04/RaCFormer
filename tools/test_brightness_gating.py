#!/usr/bin/env python
"""
Brightness-Gated Fusion Test V2

Fixed version with correct threshold based on actual brightness distributions:
- Day brightness: 0.34-0.41 (mean 0.37)
- Night brightness: 0.04-0.12 (mean 0.07)
- Correct threshold: ~0.15-0.20 (separates day from night)

Usage:
    python tools/test_brightness_gating_v2.py \
        configs/racformer_r50_nuimg_704x256_f8.py \
        checkpoints/racformer_r50_f8.pth \
        --max-samples 200
"""

import argparse
import os
import sys
import copy
import types
from collections import defaultdict

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

# Global gate settings
GATE_CONFIG = {
    'enabled': False,
    'mode': 'none',  # 'none', 'step', 'linear'
    'threshold': 0.20,
    'low_gate': 0.2,
    'current_gate': 1.0,
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


def compute_image_brightness(img_tensor):
    """Compute normalized brightness from image tensor."""
    if isinstance(img_tensor, np.ndarray):
        img_tensor = torch.from_numpy(img_tensor)
    mean_val = img_tensor.float().mean().item()
    if mean_val > 1:  # Raw [0, 255] range
        return mean_val / 255.0
    elif mean_val < 0:  # Normalized
        return (mean_val + 2.5) / 5.0  # Approximate denormalization
    else:
        return mean_val


def compute_gate(brightness, config):
    """Compute gate value based on brightness and config."""
    if config['mode'] == 'none':
        return 1.0
    elif config['mode'] == 'step':
        # Step function: low_gate if dark, 1.0 if bright
        if brightness < config['threshold']:
            return config['low_gate']
        else:
            return 1.0
    elif config['mode'] == 'linear':
        # Linear ramp below threshold
        if brightness >= config['threshold']:
            return 1.0
        else:
            t = brightness / config['threshold']
            return config['low_gate'] + (1.0 - config['low_gate']) * t
    else:
        return 1.0


def gated_forward(self, query_bbox, query_feat, mlvl_feats, lss_bev_feats,
                  radar_bev_feats, attn_mask, img_metas, layer=0):
    """Patched forward with brightness-based gating."""
    query_pos = self.position_encoder(query_bbox[..., :3])
    query_feat = query_feat + query_pos
    query_feat = self.norm1(self.self_attn(query_bbox, query_feat, attn_mask))

    query_radar_feat = self.sampling_radar_bev(query_bbox, query_feat, radar_bev_feats,
                                                img_metas, d_region=self.d_region_list[layer])
    query_radar_feat = self.norm_radar_bev(query_radar_feat)
    query_lss_feat = self.sampling_lss_bev(query_bbox, query_feat, lss_bev_feats,
                                           img_metas, d_region=self.d_region_list[layer])
    query_lss_feat = self.norm_lss_bev(query_lss_feat)

    sampled_feat = self.sampling(query_bbox, query_feat, mlvl_feats, img_metas,
                                  d_region=self.d_region_list[layer])
    query_feat = self.norm2(self.mixing(sampled_feat, query_feat))

    # =========== BRIGHTNESS GATING ===========
    global GATE_CONFIG
    if GATE_CONFIG['enabled']:
        query_feat = query_feat * GATE_CONFIG['current_gate']
    # ==========================================

    query_feat = self.norm_fusion(self.fusion(
        torch.cat((query_feat, query_radar_feat, query_lss_feat), dim=-1)))
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


def evaluate_condition(model, dataset, indices, gate_config):
    """Evaluate with specific gate configuration."""
    subset_dataset = copy.copy(dataset)
    subset_dataset.data_infos = [dataset.data_infos[i] for i in indices]

    data_loader = build_dataloader(
        subset_dataset, samples_per_gpu=1, workers_per_gpu=4,
        num_gpus=1, dist=False, shuffle=False, seed=0,
    )

    results = []
    gate_values = []
    brightness_values = []

    global GATE_CONFIG
    GATE_CONFIG.update(gate_config)
    GATE_CONFIG['enabled'] = gate_config.get('mode', 'none') != 'none'

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, leave=False):
            # Compute brightness from raw image
            if 'img' in batch:
                img = batch['img'][0].data[0]
                brightness = compute_image_brightness(img)
            else:
                brightness = 0.5

            brightness_values.append(brightness)

            # Compute and apply gate
            gate = compute_gate(brightness, GATE_CONFIG)
            gate_values.append(gate)
            GATE_CONFIG['current_gate'] = gate

            # Forward
            result = model(return_loss=False, rescale=True, **batch)
            results.append(result)

    # Compute metrics
    total_dets = 0
    total_score = 0
    for res in results:
        if isinstance(res, list):
            res = res[0]
        if 'pts_bbox' in res:
            res = res['pts_bbox']
        if 'scores_3d' in res and len(res['scores_3d']) > 0:
            total_dets += len(res['scores_3d'])
            total_score += res['scores_3d'].sum().item()

    return {
        'num_samples': len(results),
        'avg_detections': total_dets / len(results) if results else 0,
        'avg_score': total_score / total_dets if total_dets > 0 else 0,
        'avg_brightness': np.mean(brightness_values),
        'avg_gate': np.mean(gate_values),
        'gate_std': np.std(gate_values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--max-samples', type=int, default=200)
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val)
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes/', verbose=False)

    # Index by condition
    print("Indexing samples...")
    condition_indices = defaultdict(list)
    for i in tqdm(range(len(dataset))):
        condition = get_scene_condition(nusc, dataset.data_infos[i]['token'])
        if len(condition_indices[condition]) < args.max_samples:
            condition_indices[condition].append(i)

    print(f"Samples: day={len(condition_indices['day'])}, "
          f"night={len(condition_indices['night'])}, rain={len(condition_indices['rain'])}")

    # Test configurations with CORRECT thresholds
    # Based on brightness analysis: day=0.34-0.41, night=0.04-0.12
    configs = [
        # Baseline
        ('baseline', {'mode': 'none'}),

        # Step function tests (cleaner separation)
        ('step_t0.20_g0.0', {'mode': 'step', 'threshold': 0.20, 'low_gate': 0.0}),   # Zero night
        ('step_t0.20_g0.2', {'mode': 'step', 'threshold': 0.20, 'low_gate': 0.2}),   # 80% suppress night
        ('step_t0.20_g0.3', {'mode': 'step', 'threshold': 0.20, 'low_gate': 0.3}),   # 70% suppress night
        ('step_t0.15_g0.0', {'mode': 'step', 'threshold': 0.15, 'low_gate': 0.0}),   # Zero night (tighter)
        ('step_t0.15_g0.2', {'mode': 'step', 'threshold': 0.15, 'low_gate': 0.2}),   # 80% suppress night

        # Linear ramp tests
        ('linear_t0.20_g0.0', {'mode': 'linear', 'threshold': 0.20, 'low_gate': 0.0}),
        ('linear_t0.20_g0.2', {'mode': 'linear', 'threshold': 0.20, 'low_gate': 0.2}),
    ]

    all_results = defaultdict(dict)

    for condition in ['day', 'night', 'rain']:
        if not condition_indices[condition]:
            continue

        print(f"\n{'='*80}")
        print(f"{condition.upper()} ({len(condition_indices[condition])} samples)")
        print("="*80)

        for config_name, config in configs:
            print(f"  {config_name}...", end=" ", flush=True)

            # Fresh model each time
            model = build_model(cfg.model)
            load_checkpoint(model, args.checkpoint, map_location='cpu')
            model.cuda(args.gpu_id)
            model = MMDataParallel(model, device_ids=[args.gpu_id])
            model.eval()

            # Patch forward
            decoder = model.module.pts_bbox_head.transformer.decoder.decoder_layer
            decoder.forward = types.MethodType(gated_forward, decoder)

            try:
                metrics = evaluate_condition(model, dataset, condition_indices[condition], config)
                all_results[condition][config_name] = metrics
                print(f"score={metrics['avg_score']:.4f}, "
                      f"gate={metrics['avg_gate']:.2f}±{metrics['gate_std']:.2f}, "
                      f"bright={metrics['avg_brightness']:.3f}")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                all_results[condition][config_name] = {'avg_score': 0, 'error': str(e)}

            del model
            torch.cuda.empty_cache()

    # Summary table
    print("\n" + "="*120)
    print("BRIGHTNESS GATING V2 SUMMARY (Corrected Thresholds)")
    print("="*120)

    # Header
    header = f"{'Config':<22} |"
    for cond in ['day', 'night', 'rain']:
        header += f" {'Score':>8} {'Gate':>6} {'Δ%':>7} |"
    print(header)
    print("-"*120)

    baseline = {c: all_results[c].get('baseline', {}).get('avg_score', 0)
                for c in ['day', 'night', 'rain']}

    for config_name, _ in configs:
        row = f"{config_name:<22} |"
        for cond in ['day', 'night', 'rain']:
            if config_name in all_results[cond]:
                m = all_results[cond][config_name]
                score = m.get('avg_score', 0)
                gate = m.get('avg_gate', 1.0)
                delta = ((score / baseline[cond]) - 1) * 100 if baseline[cond] > 0 else 0
                row += f" {score:8.4f} {gate:6.2f} {delta:+6.1f}% |"
            else:
                row += f" {'N/A':>8} {'N/A':>6} {'N/A':>7} |"
        print(row)

    print("="*120)

    # Analysis
    print("\nKEY QUESTIONS:")
    print("-"*60)

    # Check if gating correctly identifies conditions
    for cond in ['day', 'night']:
        if 'step_t0.20_g0.0' in all_results[cond]:
            gate = all_results[cond]['step_t0.20_g0.0'].get('avg_gate', -1)
            expected = 1.0 if cond == 'day' else 0.0
            match = "✓" if abs(gate - expected) < 0.1 else "✗"
            print(f"  {cond}: avg_gate={gate:.2f} (expected {expected}) {match}")

    print("\nINTERPRETATION:")
    print("-"*60)

    # Check if night improves with suppression
    night_baseline = baseline.get('night', 0)
    if 'step_t0.20_g0.0' in all_results['night']:
        night_zeroed = all_results['night']['step_t0.20_g0.0'].get('avg_score', 0)
        delta = ((night_zeroed / night_baseline) - 1) * 100 if night_baseline > 0 else 0

        if delta > 5:
            print(f"  ✓ Night improves {delta:+.1f}% with image suppression")
            print("  → Brightness-based gating VALIDATED")
        elif delta > 0:
            print(f"  ~ Night slightly improves {delta:+.1f}% with suppression")
            print("  → Weak positive signal")
        else:
            print(f"  ✗ Night degrades {delta:+.1f}% with suppression")
            print("  → Brightness gating may not work OR threshold still wrong")

    print("="*120)


if __name__ == '__main__':
    main()
