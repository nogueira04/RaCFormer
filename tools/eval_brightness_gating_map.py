#!/usr/bin/env python
"""
Evaluate Brightness Gating - Gated Run Only

Reuses baseline results from previous run, only runs gated inference.
Saves ~50% of compute time.

Usage:
    python tools/eval_brightness_gating_gated_only.py \
        configs/racformer_r50_nuimg_704x256_f8.py \
        checkpoints/racformer_r50_f8.pth \
        --baseline-pkl results/brightness_gating_map.pkl \
        --out results/brightness_gating_final.pkl
"""

import argparse
import os
import sys
import types
import pickle
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

# Global configuration
GATING_CONFIG = {
    'enabled': False,
    'threshold': 0.20,
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


def brightness_gated_forward(self, query_bbox, query_feat, mlvl_feats, lss_bev_feats,
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
    global GATING_CONFIG
    if GATING_CONFIG['enabled']:
        brightness = img_metas[0].get('_brightness', 1.0)
        threshold = GATING_CONFIG['threshold']
        gate = 0.0 if brightness < threshold else 1.0
        query_feat = query_feat * gate
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


def inference_with_brightness(model, data_loader, threshold=0.20):
    """Custom inference that computes brightness in main process."""
    model.eval()
    results = []
    brightness_values = []
    gate_values = []

    prog_bar = tqdm(total=len(data_loader.dataset))

    for data in data_loader:
        # Compute brightness in main process
        if 'img' in data:
            img_tensor = data['img'][0]
            if hasattr(img_tensor, 'data'):
                img_tensor = img_tensor.data[0]

            brightness = img_tensor.float().mean().item()
            if brightness > 1:
                brightness = brightness / 255.0
            elif brightness < 0:
                brightness = (brightness + 2.5) / 5.0

            brightness_values.append(brightness)

            # Store in img_metas
            if 'img_metas' in data:
                img_metas = data['img_metas'][0]
                if hasattr(img_metas, 'data'):
                    for meta in img_metas.data[0]:
                        meta['_brightness'] = brightness

            gate = 0.0 if brightness < threshold else 1.0
            gate_values.append(gate)

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        results.extend(result)
        prog_bar.update(len(result))

    prog_bar.close()

    print(f"\n  Brightness stats: mean={np.mean(brightness_values):.4f}, "
          f"min={np.min(brightness_values):.4f}, max={np.max(brightness_values):.4f}")
    print(f"  Gating: {sum(g == 0 for g in gate_values)} zeroed, "
          f"{sum(g == 1 for g in gate_values)} kept")

    return results, brightness_values, gate_values


def compute_condition_metrics(results, condition_indices):
    """Compute per-condition metrics."""
    metrics = {}
    for condition, indices in condition_indices.items():
        total_dets = 0
        total_score = 0
        for idx in indices:
            res = results[idx]
            if isinstance(res, list):
                res = res[0]
            if 'pts_bbox' in res:
                res = res['pts_bbox']
            if 'scores_3d' in res and len(res['scores_3d']) > 0:
                scores = res['scores_3d']
                total_dets += len(scores)
                total_score += scores.sum().item()

        n = len(indices)
        metrics[condition] = {
            'n_samples': n,
            'avg_detections': total_dets / n if n > 0 else 0,
            'avg_score': total_score / total_dets if total_dets > 0 else 0,
        }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--baseline-pkl', required=True,
                        help='pickle file from previous run with baseline results')
    parser.add_argument('--out', default='results/brightness_gating_final.pkl')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.20)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)

    # Load baseline results
    print(f"Loading baseline results from {args.baseline_pkl}...")
    with open(args.baseline_pkl, 'rb') as f:
        prev_results = pickle.load(f)

    # Extract baseline metrics
    baseline_map = prev_results['official_metrics']['baseline']['mAP']
    baseline_nds = prev_results['official_metrics']['baseline']['NDS']
    baseline_condition = prev_results['condition_metrics']['baseline']

    print(f"  Baseline mAP: {baseline_map:.4f}")
    print(f"  Baseline NDS: {baseline_nds:.4f}")

    # Load config and dataset
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val)

    # Load nuScenes for condition indexing
    print("\nLoading nuScenes...")
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes/', verbose=False)

    print("Indexing samples by condition...")
    condition_indices = defaultdict(list)
    for i in tqdm(range(len(dataset))):
        sample_token = dataset.data_infos[i]['token']
        condition = get_scene_condition(nusc, sample_token)
        condition_indices[condition].append(i)

    print(f"\nDataset: day={len(condition_indices['day'])}, "
          f"night={len(condition_indices['night'])}, rain={len(condition_indices['rain'])}")

    # Build dataloader with workers=0
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0,
        num_gpus=1, dist=False, shuffle=False, seed=0,
    )

    # ========== GATED EVALUATION ONLY ==========
    print("\n" + "="*80)
    print(f"BRIGHTNESS GATING EVALUATION (threshold={args.threshold})")
    print("="*80)

    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda(args.gpu_id)
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    model.eval()

    # Patch forward
    decoder = model.module.pts_bbox_head.transformer.decoder.decoder_layer
    decoder.forward = types.MethodType(brightness_gated_forward, decoder)

    GATING_CONFIG['enabled'] = True
    GATING_CONFIG['threshold'] = args.threshold

    print("\n  Running gated inference...")
    gated_results, brightness_values, gate_values = inference_with_brightness(
        model, data_loader, threshold=args.threshold
    )

    # Gating breakdown
    print("\n  Gating by condition:")
    for cond, indices in condition_indices.items():
        n_gated = sum(1 for i in indices if gate_values[i] == 0)
        avg_bright = np.mean([brightness_values[i] for i in indices])
        print(f"    {cond}: {n_gated}/{len(indices)} gated (avg brightness: {avg_bright:.4f})")

    print("\n  Running nuScenes evaluation...")
    gated_eval = dataset.evaluate(gated_results, metric='bbox')
    gated_condition = compute_condition_metrics(gated_results, condition_indices)

    del model
    torch.cuda.empty_cache()

    # ========== RESULTS ==========
    def get_metric(results, key):
        for k in [key, f'pts_bbox_{key}', f'pts_bbox_NuScenes/{key}']:
            if k in results:
                return results[k]
        return 0

    gated_map = get_metric(gated_eval, 'mAP')
    gated_nds = get_metric(gated_eval, 'NDS')

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    map_delta = gated_map - baseline_map
    map_pct = ((gated_map / baseline_map) - 1) * 100 if baseline_map > 0 else 0
    nds_delta = gated_nds - baseline_nds
    nds_pct = ((gated_nds / baseline_nds) - 1) * 100 if baseline_nds > 0 else 0

    print(f"\n{'Metric':<12} | {'Baseline':>12} | {'Gated':>12} | {'Δ':>10} | {'Δ %':>8}")
    print("-"*60)
    print(f"{'mAP':<12} | {baseline_map:>12.4f} | {gated_map:>12.4f} | {map_delta:>+10.4f} | {map_pct:>+7.2f}%")
    print(f"{'NDS':<12} | {baseline_nds:>12.4f} | {gated_nds:>12.4f} | {nds_delta:>+10.4f} | {nds_pct:>+7.2f}%")

    print("\n" + "-"*60)
    print("Per-condition avg_score:")
    print(f"{'Condition':<10} | {'Baseline':>12} | {'Gated':>12} | {'Δ %':>8}")
    print("-"*50)

    for cond in ['day', 'night', 'rain']:
        b = baseline_condition.get(cond, {}).get('avg_score', 0)
        g = gated_condition.get(cond, {}).get('avg_score', 0)
        pct = ((g / b) - 1) * 100 if b > 0 else 0
        print(f"{cond:<10} | {b:>12.4f} | {g:>12.4f} | {pct:>+7.1f}%")

    # Save
    print(f"\nSaving to {args.out}...")
    with open(args.out, 'wb') as f:
        pickle.dump({
            'official_metrics': {
                'baseline': {'mAP': baseline_map, 'NDS': baseline_nds},
                'gated': {'mAP': gated_map, 'NDS': gated_nds},
            },
            'condition_metrics': {
                'baseline': baseline_condition,
                'gated': gated_condition,
            },
            'brightness': {
                'values': brightness_values,
                'gates': gate_values,
            },
            'config': {'threshold': args.threshold},
        }, f)

    # Summary
    night_b = baseline_condition.get('night', {}).get('avg_score', 0)
    night_g = gated_condition.get('night', {}).get('avg_score', 0)
    night_pct = ((night_g / night_b) - 1) * 100 if night_b > 0 else 0

    n_night_gated = sum(1 for i in condition_indices['night'] if gate_values[i] == 0)

    print(f"""
┌────────────────────────────────────────────────────────────────┐
│  BRIGHTNESS GATING RESULTS                                     │
├────────────────────────────────────────────────────────────────┤
│  mAP: {baseline_map:.4f} → {gated_map:.4f}  ({map_pct:+.2f}%)
│  NDS: {baseline_nds:.4f} → {gated_nds:.4f}  ({nds_pct:+.2f}%)
├────────────────────────────────────────────────────────────────┤
│  Night avg_score: {night_b:.4f} → {night_g:.4f}  ({night_pct:+.1f}%)
│  Night samples gated: {n_night_gated}/{len(condition_indices['night'])}
├────────────────────────────────────────────────────────────────┤
│  Threshold: {args.threshold}                                            │
└────────────────────────────────────────────────────────────────┘
""")


if __name__ == '__main__':
    main()
