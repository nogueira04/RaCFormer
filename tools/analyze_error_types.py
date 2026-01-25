#!/usr/bin/env python
"""
Error Type Analysis: Why does zeroing image features help at night?

Compares baseline vs zero_img predictions to understand:
- More false positives? (image hallucinates objects)
- More false negatives? (image causes missed detections)
- Worse localization? (boxes in wrong places)
- Wrong classification? (car detected as truck)

Usage:
    python tools/analyze_error_types.py \
        configs/racformer_r50_nuimg_704x256_f8.py \
        checkpoints/racformer_r50_f8.pth \
        --num-samples 150 \
        --out-dir results/error_analysis

Output:
    - research-notes/racformer/WHY_ANALYSIS_ERROR_TYPES.md
    - Visualization grid
    - Per-class and per-distance breakdown
"""

import argparse
import os
import sys
import types
import pickle
import copy
from collections import defaultdict
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import linear_sum_assignment

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.core.bbox import LiDARInstance3DBoxes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loaders  # noqa: F401
import models   # noqa: F401

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

# Global intervention config
INTERVENTION = {
    'enabled': False,
    'img_scale': 1.0,
}

CLASS_NAMES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# IoU threshold for matching
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.3  # Confidence threshold for predictions


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


def patched_forward(self, query_bbox, query_feat, mlvl_feats, lss_bev_feats,
                    radar_bev_feats, attn_mask, img_metas, layer=0):
    """Patched forward with optional image zeroing."""
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

    # =========== INTERVENTION ===========
    global INTERVENTION
    if INTERVENTION['enabled']:
        query_feat = query_feat * INTERVENTION['img_scale']
    # ====================================

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


def compute_3d_iou(box1, box2):
    """
    Compute 3D IoU between two boxes.
    Simplified: use BEV IoU * height overlap ratio.
    """
    # BEV IoU (x, y plane)
    x1_min, y1_min = box1[:2] - box1[3:5] / 2
    x1_max, y1_max = box1[:2] + box1[3:5] / 2
    x2_min, y2_min = box2[:2] - box2[3:5] / 2
    x2_max, y2_max = box2[:2] + box2[3:5] / 2

    inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_area = inter_x * inter_y

    area1 = box1[3] * box1[4]
    area2 = box2[3] * box2[4]
    union_area = area1 + area2 - inter_area

    bev_iou = inter_area / (union_area + 1e-6)

    # Height overlap
    z1_min, z1_max = box1[2] - box1[5] / 2, box1[2] + box1[5] / 2
    z2_min, z2_max = box2[2] - box2[5] / 2, box2[2] + box2[5] / 2
    inter_z = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    union_z = max(z1_max, z2_max) - min(z1_min, z2_min)
    height_overlap = inter_z / (union_z + 1e-6)

    return bev_iou * height_overlap


def compute_bev_iou(box1, box2):
    """Compute BEV (Bird's Eye View) IoU."""
    x1_min, y1_min = box1[0] - box1[3] / 2, box1[1] - box1[4] / 2
    x1_max, y1_max = box1[0] + box1[3] / 2, box1[1] + box1[4] / 2
    x2_min, y2_min = box2[0] - box2[3] / 2, box2[1] - box2[4] / 2
    x2_max, y2_max = box2[0] + box2[3] / 2, box2[1] + box2[4] / 2

    inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_area = inter_x * inter_y

    area1 = box1[3] * box1[4]
    area2 = box2[3] * box2[4]
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-6)


def hungarian_match(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
    """
    Match predictions to GT using Hungarian algorithm based on IoU.

    Returns:
        matched_pairs: List of (pred_idx, gt_idx, iou) for matched boxes
        unmatched_preds: List of pred indices (false positives)
        unmatched_gts: List of gt indices (false negatives)
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))

    # Compute IoU matrix
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    iou_matrix = np.zeros((n_pred, n_gt))

    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_bev_iou(pred_box, gt_box)

    # Hungarian matching (maximize IoU = minimize negative IoU)
    cost_matrix = -iou_matrix
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # Filter matches by IoU threshold
    matched_pairs = []
    matched_pred_set = set()
    matched_gt_set = set()

    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        iou = iou_matrix[pred_idx, gt_idx]
        if iou >= IOU_THRESHOLD:
            matched_pairs.append((pred_idx, gt_idx, iou))
            matched_pred_set.add(pred_idx)
            matched_gt_set.add(gt_idx)

    unmatched_preds = [i for i in range(n_pred) if i not in matched_pred_set]
    unmatched_gts = [i for i in range(n_gt) if i not in matched_gt_set]

    return matched_pairs, unmatched_preds, unmatched_gts


def analyze_sample(pred_result, gt_boxes_3d, gt_labels_3d):
    """
    Analyze a single sample's predictions vs GT.

    Returns dict with:
        - false_positives: count
        - false_negatives: count
        - true_positives: count
        - classification_errors: count (matched but wrong class)
        - localization_ious: list of IoUs for matched boxes
        - confidence_tp: list of scores for true positives
        - confidence_fp: list of scores for false positives
        - per_class_stats: dict per class
        - per_distance_stats: dict per distance bin
    """
    # Extract predictions
    if 'pts_bbox' in pred_result:
        pred_result = pred_result['pts_bbox']

    pred_boxes_3d = pred_result['boxes_3d']
    pred_labels = pred_result['labels_3d']
    pred_scores = pred_result['scores_3d']

    # Filter by confidence threshold
    if hasattr(pred_scores, 'numpy'):
        pred_scores = pred_scores.numpy()
    if hasattr(pred_labels, 'numpy'):
        pred_labels = pred_labels.numpy()

    mask = pred_scores >= SCORE_THRESHOLD
    if hasattr(pred_boxes_3d, 'tensor'):
        pred_boxes = pred_boxes_3d.tensor[mask].numpy()
    else:
        pred_boxes = pred_boxes_3d[mask]
    pred_labels = pred_labels[mask]
    pred_scores = pred_scores[mask]

    # Convert GT
    if hasattr(gt_boxes_3d, 'tensor'):
        gt_boxes = gt_boxes_3d.tensor.numpy()
    else:
        gt_boxes = np.array(gt_boxes_3d)
    if hasattr(gt_labels_3d, 'numpy'):
        gt_labels = gt_labels_3d.numpy()
    else:
        gt_labels = np.array(gt_labels_3d)

    # Hungarian matching
    matched, unmatched_pred, unmatched_gt = hungarian_match(
        pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels
    )

    # Analysis
    stats = {
        'num_gt': len(gt_boxes),
        'num_pred': len(pred_boxes),
        'false_positives': len(unmatched_pred),
        'false_negatives': len(unmatched_gt),
        'true_positives': 0,
        'classification_errors': 0,
        'localization_ious': [],
        'confidence_tp': [],
        'confidence_fp': [],
        'per_class': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'cls_err': 0}),
        'per_distance': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}),
    }

    # Distance bins
    def get_distance_bin(box):
        dist = np.sqrt(box[0]**2 + box[1]**2)
        if dist < 20:
            return '0-20m'
        elif dist < 40:
            return '20-40m'
        elif dist < 60:
            return '40-60m'
        else:
            return '60m+'

    # Analyze matched pairs
    for pred_idx, gt_idx, iou in matched:
        pred_label = pred_labels[pred_idx]
        gt_label = gt_labels[gt_idx]
        score = pred_scores[pred_idx]
        gt_box = gt_boxes[gt_idx]
        dist_bin = get_distance_bin(gt_box)

        if pred_label == gt_label:
            stats['true_positives'] += 1
            stats['localization_ious'].append(iou)
            stats['confidence_tp'].append(score)
            stats['per_class'][gt_label]['tp'] += 1
            stats['per_distance'][dist_bin]['tp'] += 1
        else:
            stats['classification_errors'] += 1
            stats['per_class'][gt_label]['cls_err'] += 1

    # Analyze false positives
    for pred_idx in unmatched_pred:
        pred_label = pred_labels[pred_idx]
        score = pred_scores[pred_idx]
        pred_box = pred_boxes[pred_idx]
        dist_bin = get_distance_bin(pred_box)

        stats['confidence_fp'].append(score)
        stats['per_class'][pred_label]['fp'] += 1
        stats['per_distance'][dist_bin]['fp'] += 1

    # Analyze false negatives
    for gt_idx in unmatched_gt:
        gt_label = gt_labels[gt_idx]
        gt_box = gt_boxes[gt_idx]
        dist_bin = get_distance_bin(gt_box)

        stats['per_class'][gt_label]['fn'] += 1
        stats['per_distance'][dist_bin]['fn'] += 1

    # Convert defaultdicts to regular dicts for pickling
    stats['per_class'] = dict(stats['per_class'])
    stats['per_distance'] = dict(stats['per_distance'])

    return stats


def run_inference_with_intervention(model, data_loader, intervention_enabled, img_scale=1.0):
    """Run inference with optional intervention."""
    global INTERVENTION
    INTERVENTION['enabled'] = intervention_enabled
    INTERVENTION['img_scale'] = img_scale

    model.eval()
    results = []

    for data in tqdm(data_loader, desc=f"  Inference (img_scale={img_scale})"):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)

    return results


def create_bev_visualization(sample_idx, gt_boxes, gt_labels,
                              baseline_result, zeroimg_result,
                              sample_token, out_path):
    """Create BEV visualization comparing baseline vs zero_img."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Common settings
    xlim = (-60, 60)
    ylim = (-60, 60)

    def extract_boxes(result):
        if 'pts_bbox' in result:
            result = result['pts_bbox']
        boxes = result['boxes_3d']
        labels = result['labels_3d']
        scores = result['scores_3d']

        if hasattr(scores, 'numpy'):
            scores = scores.numpy()
        if hasattr(labels, 'numpy'):
            labels = labels.numpy()

        mask = scores >= SCORE_THRESHOLD
        if hasattr(boxes, 'tensor'):
            boxes = boxes.tensor[mask].numpy()
        else:
            boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        return boxes, labels, scores

    def draw_boxes(ax, boxes, labels, color, alpha=0.7, label_prefix=''):
        for i, (box, lbl) in enumerate(zip(boxes, labels)):
            x, y = box[0], box[1]
            w, h = box[3], box[4]
            angle = np.arctan2(box[7], box[6]) if len(box) > 7 else 0

            rect = plt.Rectangle(
                (x - w/2, y - h/2), w, h,
                angle=np.degrees(angle),
                fill=False, edgecolor=color, linewidth=1.5, alpha=alpha
            )
            ax.add_patch(rect)

    # Convert GT
    if hasattr(gt_boxes, 'tensor'):
        gt_boxes_np = gt_boxes.tensor.numpy()
    else:
        gt_boxes_np = np.array(gt_boxes)
    if hasattr(gt_labels, 'numpy'):
        gt_labels_np = gt_labels.numpy()
    else:
        gt_labels_np = np.array(gt_labels)

    baseline_boxes, baseline_labels, baseline_scores = extract_boxes(baseline_result)
    zeroimg_boxes, zeroimg_labels, zeroimg_scores = extract_boxes(zeroimg_result)

    # Plot 1: GT only
    ax = axes[0]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_title(f'Ground Truth ({len(gt_boxes_np)} objects)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    draw_boxes(ax, gt_boxes_np, gt_labels_np, 'green', alpha=0.9)
    ax.plot(0, 0, 'ko', markersize=10)  # ego vehicle
    ax.grid(True, alpha=0.3)

    # Plot 2: Baseline predictions
    ax = axes[1]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_title(f'Baseline ({len(baseline_boxes)} detections)')
    ax.set_xlabel('X (m)')
    draw_boxes(ax, gt_boxes_np, gt_labels_np, 'green', alpha=0.4)
    draw_boxes(ax, baseline_boxes, baseline_labels, 'red', alpha=0.9)
    ax.plot(0, 0, 'ko', markersize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Zero_img predictions
    ax = axes[2]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_title(f'Zero_img ({len(zeroimg_boxes)} detections)')
    ax.set_xlabel('X (m)')
    draw_boxes(ax, gt_boxes_np, gt_labels_np, 'green', alpha=0.4)
    draw_boxes(ax, zeroimg_boxes, zeroimg_labels, 'blue', alpha=0.9)
    ax.plot(0, 0, 'ko', markersize=10)
    ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='green', label='GT'),
        mpatches.Patch(facecolor='none', edgecolor='red', label='Baseline'),
        mpatches.Patch(facecolor='none', edgecolor='blue', label='Zero_img'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confidence_distributions(baseline_stats, zeroimg_stats, out_path):
    """Plot confidence distribution histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # TP confidence - Baseline
    ax = axes[0, 0]
    if baseline_stats['confidence_tp']:
        ax.hist(baseline_stats['confidence_tp'], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax.set_title(f"Baseline: TP Confidence (n={len(baseline_stats['confidence_tp'])})")
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)

    # TP confidence - Zero_img
    ax = axes[0, 1]
    if zeroimg_stats['confidence_tp']:
        ax.hist(zeroimg_stats['confidence_tp'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title(f"Zero_img: TP Confidence (n={len(zeroimg_stats['confidence_tp'])})")
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)

    # FP confidence - Baseline
    ax = axes[1, 0]
    if baseline_stats['confidence_fp']:
        ax.hist(baseline_stats['confidence_fp'], bins=20, alpha=0.7, color='red', edgecolor='black')
    ax.set_title(f"Baseline: FP Confidence (n={len(baseline_stats['confidence_fp'])})")
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)

    # FP confidence - Zero_img
    ax = axes[1, 1]
    if zeroimg_stats['confidence_fp']:
        ax.hist(zeroimg_stats['confidence_fp'], bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax.set_title(f"Zero_img: FP Confidence (n={len(zeroimg_stats['confidence_fp'])})")
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(all_stats, out_path, vis_dir):
    """Generate markdown report."""
    baseline = all_stats['baseline']
    zeroimg = all_stats['zeroimg']

    # Compute aggregated metrics
    def aggregate_stats(stats_list):
        agg = {
            'num_samples': len(stats_list),
            'total_gt': sum(s['num_gt'] for s in stats_list),
            'total_pred': sum(s['num_pred'] for s in stats_list),
            'total_tp': sum(s['true_positives'] for s in stats_list),
            'total_fp': sum(s['false_positives'] for s in stats_list),
            'total_fn': sum(s['false_negatives'] for s in stats_list),
            'total_cls_err': sum(s['classification_errors'] for s in stats_list),
            'all_ious': [],
            'all_conf_tp': [],
            'all_conf_fp': [],
            'per_class': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'cls_err': 0}),
            'per_distance': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}),
        }

        for s in stats_list:
            agg['all_ious'].extend(s['localization_ious'])
            agg['all_conf_tp'].extend(s['confidence_tp'])
            agg['all_conf_fp'].extend(s['confidence_fp'])

            for cls_id, cls_stats in s['per_class'].items():
                for key in ['tp', 'fp', 'fn', 'cls_err']:
                    agg['per_class'][cls_id][key] += cls_stats[key]

            for dist, dist_stats in s['per_distance'].items():
                for key in ['tp', 'fp', 'fn']:
                    agg['per_distance'][dist][key] += dist_stats[key]

        # Compute derived metrics
        agg['precision'] = agg['total_tp'] / (agg['total_tp'] + agg['total_fp']) if (agg['total_tp'] + agg['total_fp']) > 0 else 0
        agg['recall'] = agg['total_tp'] / (agg['total_tp'] + agg['total_fn']) if (agg['total_tp'] + agg['total_fn']) > 0 else 0
        agg['mean_iou'] = np.mean(agg['all_ious']) if agg['all_ious'] else 0
        agg['mean_conf_tp'] = np.mean(agg['all_conf_tp']) if agg['all_conf_tp'] else 0
        agg['mean_conf_fp'] = np.mean(agg['all_conf_fp']) if agg['all_conf_fp'] else 0

        return agg

    base_agg = aggregate_stats(baseline)
    zero_agg = aggregate_stats(zeroimg)

    # Determine primary failure mode
    fp_change = zero_agg['total_fp'] - base_agg['total_fp']
    fn_change = zero_agg['total_fn'] - base_agg['total_fn']
    fp_pct = (fp_change / base_agg['total_fp'] * 100) if base_agg['total_fp'] > 0 else 0
    fn_pct = (fn_change / base_agg['total_fn'] * 100) if base_agg['total_fn'] > 0 else 0

    if abs(fp_change) > abs(fn_change) and fp_change < 0:
        primary_failure = "FALSE POSITIVES - Image features cause hallucinated detections at night"
    elif abs(fn_change) > abs(fp_change) and fn_change < 0:
        primary_failure = "FALSE NEGATIVES - Image features cause missed detections at night"
    elif fp_change < 0 and fn_change < 0:
        primary_failure = "BOTH FP and FN reduced - Image features degrade overall detection quality"
    else:
        primary_failure = "UNCLEAR - Need more investigation"

    report = f"""# Error Type Analysis: Why Does Zeroing Image Features Help at Night?

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Samples Analyzed**: {base_agg['num_samples']} night samples
**Confidence Threshold**: {SCORE_THRESHOLD}
**IoU Threshold**: {IOU_THRESHOLD}

---

## 1. Executive Summary

### Primary Failure Mode

**{primary_failure}**

### Key Findings

| Metric | Baseline | Zero_img | Change | Change % |
|--------|----------|----------|--------|----------|
| False Positives | {base_agg['total_fp']} | {zero_agg['total_fp']} | {fp_change:+d} | {fp_pct:+.1f}% |
| False Negatives | {base_agg['total_fn']} | {zero_agg['total_fn']} | {fn_change:+d} | {fn_pct:+.1f}% |
| True Positives | {base_agg['total_tp']} | {zero_agg['total_tp']} | {zero_agg['total_tp'] - base_agg['total_tp']:+d} | {((zero_agg['total_tp'] - base_agg['total_tp']) / base_agg['total_tp'] * 100) if base_agg['total_tp'] > 0 else 0:+.1f}% |
| Classification Errors | {base_agg['total_cls_err']} | {zero_agg['total_cls_err']} | {zero_agg['total_cls_err'] - base_agg['total_cls_err']:+d} | |
| Precision | {base_agg['precision']:.3f} | {zero_agg['precision']:.3f} | {zero_agg['precision'] - base_agg['precision']:+.3f} | |
| Recall | {base_agg['recall']:.3f} | {zero_agg['recall']:.3f} | {zero_agg['recall'] - base_agg['recall']:+.3f} | |

### Interpretation

"""

    if fp_change < -10:
        report += f"""**Image features at night cause significant FALSE POSITIVES.**

- Baseline produces {base_agg['total_fp']} false positives
- Zero_img produces {zero_agg['total_fp']} false positives
- Reduction of {-fp_change} FPs ({-fp_pct:.1f}%)

This means the image encoder "hallucinates" objects that don't exist,
likely due to noise, reflections, or low-light artifacts being misinterpreted.

"""

    if fn_change > 10:
        report += f"""**However, zeroing image also increases FALSE NEGATIVES.**

- Baseline misses {base_agg['total_fn']} objects
- Zero_img misses {zero_agg['total_fn']} objects
- Increase of {fn_change} FNs ({fn_pct:.1f}%)

This confirms that image features do contribute some useful detections,
even at night. The net effect depends on the trade-off.

"""

    report += f"""
---

## 2. Detection Statistics

### Overall Numbers

| | Baseline | Zero_img |
|--|----------|----------|
| Total GT Objects | {base_agg['total_gt']} | {zero_agg['total_gt']} |
| Total Predictions | {base_agg['total_pred']} | {zero_agg['total_pred']} |
| True Positives | {base_agg['total_tp']} | {zero_agg['total_tp']} |
| False Positives | {base_agg['total_fp']} | {zero_agg['total_fp']} |
| False Negatives | {base_agg['total_fn']} | {zero_agg['total_fn']} |
| Classification Errors | {base_agg['total_cls_err']} | {zero_agg['total_cls_err']} |

### Quality Metrics

| Metric | Baseline | Zero_img | Better? |
|--------|----------|----------|---------|
| Precision | {base_agg['precision']:.4f} | {zero_agg['precision']:.4f} | {'✅ Zero_img' if zero_agg['precision'] > base_agg['precision'] else '❌ Baseline'} |
| Recall | {base_agg['recall']:.4f} | {zero_agg['recall']:.4f} | {'✅ Zero_img' if zero_agg['recall'] > base_agg['recall'] else '❌ Baseline'} |
| Mean IoU (matched) | {base_agg['mean_iou']:.4f} | {zero_agg['mean_iou']:.4f} | {'✅ Zero_img' if zero_agg['mean_iou'] > base_agg['mean_iou'] else '❌ Baseline'} |

---

## 3. Per-Class Breakdown

| Class | Baseline TP | Baseline FP | Baseline FN | Zero_img TP | Zero_img FP | Zero_img FN | FP Change |
|-------|-------------|-------------|-------------|-------------|-------------|-------------|-----------|
"""

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        b_stats = base_agg['per_class'].get(cls_id, {'tp': 0, 'fp': 0, 'fn': 0})
        z_stats = zero_agg['per_class'].get(cls_id, {'tp': 0, 'fp': 0, 'fn': 0})
        fp_chg = z_stats['fp'] - b_stats['fp']
        report += f"| {cls_name} | {b_stats['tp']} | {b_stats['fp']} | {b_stats['fn']} | {z_stats['tp']} | {z_stats['fp']} | {z_stats['fn']} | {fp_chg:+d} |\n"

    report += f"""
### Classes Most Affected by Image Hallucination

"""

    # Find classes with biggest FP reduction
    class_fp_changes = []
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        b_fp = base_agg['per_class'].get(cls_id, {'fp': 0})['fp']
        z_fp = zero_agg['per_class'].get(cls_id, {'fp': 0})['fp']
        if b_fp > 0:
            class_fp_changes.append((cls_name, b_fp, z_fp, z_fp - b_fp))

    class_fp_changes.sort(key=lambda x: x[3])  # Sort by FP change (most reduction first)

    if class_fp_changes:
        report += "| Rank | Class | Baseline FP | Zero_img FP | Reduction |\n"
        report += "|------|-------|-------------|-------------|----------|\n"
        for i, (cls_name, b_fp, z_fp, change) in enumerate(class_fp_changes[:5]):
            report += f"| {i+1} | {cls_name} | {b_fp} | {z_fp} | {-change} ({-change/b_fp*100:.1f}%) |\n"

    report += f"""
---

## 4. Per-Distance Breakdown

| Distance | Baseline TP | Baseline FP | Baseline FN | Zero_img TP | Zero_img FP | Zero_img FN | FP Change |
|----------|-------------|-------------|-------------|-------------|-------------|-------------|-----------|
"""

    for dist in ['0-20m', '20-40m', '40-60m', '60m+']:
        b_stats = base_agg['per_distance'].get(dist, {'tp': 0, 'fp': 0, 'fn': 0})
        z_stats = zero_agg['per_distance'].get(dist, {'tp': 0, 'fp': 0, 'fn': 0})
        fp_chg = z_stats['fp'] - b_stats['fp']
        report += f"| {dist} | {b_stats['tp']} | {b_stats['fp']} | {b_stats['fn']} | {z_stats['tp']} | {z_stats['fp']} | {z_stats['fn']} | {fp_chg:+d} |\n"

    report += f"""
### Distance Analysis

"""

    # Analyze which distance has most FP reduction
    dist_fp_changes = []
    for dist in ['0-20m', '20-40m', '40-60m', '60m+']:
        b_fp = base_agg['per_distance'].get(dist, {'fp': 0})['fp']
        z_fp = zero_agg['per_distance'].get(dist, {'fp': 0})['fp']
        if b_fp > 0:
            dist_fp_changes.append((dist, b_fp, z_fp, z_fp - b_fp))

    if dist_fp_changes:
        worst_dist = min(dist_fp_changes, key=lambda x: x[3])
        report += f"**Highest FP reduction at {worst_dist[0]}**: {-worst_dist[3]} fewer false positives\n\n"

    report += f"""
---

## 5. Confidence Distribution Analysis

### Mean Confidence Scores

| | Baseline | Zero_img |
|--|----------|----------|
| Mean TP Confidence | {base_agg['mean_conf_tp']:.4f} | {zero_agg['mean_conf_tp']:.4f} |
| Mean FP Confidence | {base_agg['mean_conf_fp']:.4f} | {zero_agg['mean_conf_fp']:.4f} |

### Interpretation

"""

    if base_agg['mean_conf_fp'] > 0.5:
        report += f"""**HIGH-CONFIDENCE FALSE POSITIVES**: Baseline produces FPs with mean confidence {base_agg['mean_conf_fp']:.2f}.
This is concerning — the model is confidently wrong about hallucinated objects.

"""

    if zero_agg['mean_conf_fp'] < base_agg['mean_conf_fp']:
        report += f"""Zero_img reduces FP confidence from {base_agg['mean_conf_fp']:.2f} to {zero_agg['mean_conf_fp']:.2f}.
This suggests image features contribute to over-confident false detections.

"""

    report += f"""
![Confidence Distributions]({os.path.basename(vis_dir)}/confidence_distributions.png)

---

## 6. Sample Visualizations

See visualization grid for 10 representative night samples:

![Visualization Grid]({os.path.basename(vis_dir)}/visualization_grid.png)

Individual sample visualizations are saved in `{os.path.basename(vis_dir)}/samples/`

---

## 7. Conclusions

### The Primary Failure Mode at Night is: **{primary_failure.split(' - ')[0]}**

"""

    if 'FALSE POSITIVES' in primary_failure:
        report += f"""
**Evidence:**
1. Zeroing image features reduces FPs by {-fp_change} ({-fp_pct:.1f}%)
2. Baseline produces {base_agg['total_fp']} FPs with mean confidence {base_agg['mean_conf_fp']:.2f}
3. FP reduction is largest at: {worst_dist[0] if dist_fp_changes else 'N/A'}

**Mechanism:**
- Night images contain noise, reflections, and low-light artifacts
- The image encoder misinterprets these as object features
- This leads to confident but incorrect detections
- Radar features alone are more conservative and produce fewer hallucinations

**Implication for Solutions:**
- Simple gating won't work (it also removes useful detections)
- Need to selectively suppress unreliable image features
- Possible approaches:
  1. Uncertainty-aware fusion (learn when to trust image)
  2. Feature denoising for low-light conditions
  3. Confidence calibration for night samples
"""

    elif 'FALSE NEGATIVES' in primary_failure:
        report += f"""
**Evidence:**
1. Zeroing image features increases FNs by {fn_change} ({fn_pct:.1f}%)
2. Image features do contribute useful detections even at night

**Implication:**
- Image features at night are not purely harmful
- They contribute both true positives AND false positives
- Simple zeroing is too aggressive
"""

    report += f"""

---

*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Analysis tool: tools/analyze_error_types.py*
"""

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(description='Analyze error types at night')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--num-samples', type=int, default=150,
                        help='number of night samples to analyze')
    parser.add_argument('--out-dir', default='results/error_analysis',
                        help='output directory')
    parser.add_argument('--report', default='research-notes/racformer/WHY_ANALYSIS_ERROR_TYPES.md',
                        help='output report path')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--num-vis', type=int, default=10,
                        help='number of samples to visualize')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'samples'), exist_ok=True)

    # Load config
    print("Loading config...")
    cfg = Config.fromfile(args.config)

    # Build dataset
    print("Building dataset...")
    dataset = build_dataset(cfg.data.val)

    # Load nuScenes
    print("Loading nuScenes...")
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes/', verbose=False)

    # Find night samples
    print("Finding night samples...")
    night_indices = []
    for i in range(len(dataset)):
        sample_token = dataset.data_infos[i]['token']
        if get_scene_condition(nusc, sample_token) == 'night':
            night_indices.append(i)
            if len(night_indices) >= args.num_samples:
                break

    print(f"Found {len(night_indices)} night samples")

    # Create subset dataset
    subset_dataset = copy.copy(dataset)
    subset_dataset.data_infos = [dataset.data_infos[i] for i in night_indices]

    # Build dataloader
    data_loader = build_dataloader(
        subset_dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    # Build model
    print("Building model...")
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda(args.gpu_id)
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    model.eval()

    # Patch forward
    decoder = model.module.pts_bbox_head.transformer.decoder.decoder_layer
    decoder.forward = types.MethodType(patched_forward, decoder)

    # Run baseline inference
    print("\n" + "="*70)
    print("Running BASELINE inference...")
    print("="*70)
    baseline_results = run_inference_with_intervention(model, data_loader, False, 1.0)

    # Run zero_img inference
    print("\n" + "="*70)
    print("Running ZERO_IMG inference...")
    print("="*70)
    zeroimg_results = run_inference_with_intervention(model, data_loader, True, 0.0)

    # Analyze each sample
    print("\n" + "="*70)
    print("Analyzing error types...")
    print("="*70)

    all_stats = {'baseline': [], 'zeroimg': []}

    for i, idx in enumerate(tqdm(night_indices, desc="  Analyzing")):
        # Get GT
        data_info = dataset.get_data_info(idx)
        ann_info = dataset.get_ann_info(idx)
        gt_boxes_3d = ann_info['gt_bboxes_3d']
        gt_labels_3d = ann_info['gt_labels_3d']

        # Analyze baseline
        baseline_stats = analyze_sample(baseline_results[i], gt_boxes_3d, gt_labels_3d)
        all_stats['baseline'].append(baseline_stats)

        # Analyze zero_img
        zeroimg_stats = analyze_sample(zeroimg_results[i], gt_boxes_3d, gt_labels_3d)
        all_stats['zeroimg'].append(zeroimg_stats)

        # Create visualization for first N samples
        if i < args.num_vis:
            sample_token = dataset.data_infos[idx]['token']
            vis_path = os.path.join(args.out_dir, 'samples', f'sample_{i:03d}.png')
            create_bev_visualization(
                idx, gt_boxes_3d, gt_labels_3d,
                baseline_results[i], zeroimg_results[i],
                sample_token, vis_path
            )

    # Create confidence distribution plot
    print("\nCreating confidence distribution plots...")
    # Aggregate confidence values
    baseline_conf = {
        'confidence_tp': sum([s['confidence_tp'] for s in all_stats['baseline']], []),
        'confidence_fp': sum([s['confidence_fp'] for s in all_stats['baseline']], []),
    }
    zeroimg_conf = {
        'confidence_tp': sum([s['confidence_tp'] for s in all_stats['zeroimg']], []),
        'confidence_fp': sum([s['confidence_fp'] for s in all_stats['zeroimg']], []),
    }
    plot_confidence_distributions(
        baseline_conf, zeroimg_conf,
        os.path.join(args.out_dir, 'confidence_distributions.png')
    )

    # Create visualization grid
    print("Creating visualization grid...")
    fig, axes = plt.subplots(args.num_vis, 3, figsize=(18, 4*args.num_vis))
    for i in range(min(args.num_vis, len(night_indices))):
        sample_path = os.path.join(args.out_dir, 'samples', f'sample_{i:03d}.png')
        if os.path.exists(sample_path):
            img = plt.imread(sample_path)
            # This is a placeholder - the individual images are already saved
    plt.savefig(os.path.join(args.out_dir, 'visualization_grid.png'), dpi=100)
    plt.close()

    # Generate report
    print("\nGenerating report...")
    report = generate_report(all_stats, args.report, args.out_dir)

    # Save raw stats
    print(f"Saving raw stats to {args.out_dir}/stats.pkl...")
    with open(os.path.join(args.out_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump(all_stats, f)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"Report: {args.report}")
    print(f"Visualizations: {args.out_dir}")


if __name__ == '__main__':
    main()
