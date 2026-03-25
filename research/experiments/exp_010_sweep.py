"""Experiment 010: Offline sweep of DINOv3 post-processing strategies.

Runs in racformerfix env. Loads precomputed DINOv3 scores and original predictions,
applies different strategies, evaluates each.

Strategies:
  A) FP filter: remove predictions with max_dino_sim < threshold
  B) Score fusion: new_score = score * (1 + alpha * class_sim)
  C) FP filter + score fusion combined
"""
import os
import sys
import pickle
import logging
import argparse
import importlib
import torch
import numpy as np

sys.path.insert(0, '/srv/nfs/shared/gnmp/RaCFormer')
importlib.import_module('models')
importlib.import_module('loaders')

from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes

PROJECT_ROOT = '/srv/nfs/shared/gnmp/RaCFormer'

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def build_results(preds, dino_scores, strategy, **kwargs):
    """Apply a strategy and return mmdet3d-compatible results."""
    results = []
    for si in range(len(preds)):
        p = preds[si]
        ds = dino_scores[si]
        boxes = torch.tensor(p['boxes_3d'], dtype=torch.float32)
        scores = torch.tensor(p['scores_3d'], dtype=torch.float32)
        labels = torch.tensor(p['labels_3d'], dtype=torch.long)
        sims = ds['sims']
        proj = ds['projected']
        n = len(boxes)

        if strategy == 'fp_filter':
            thresh = kwargs['threshold']
            keep = torch.ones(n, dtype=torch.bool)
            for i in range(n):
                if proj[i]:
                    max_sim = sims[i].max().item()
                    if max_sim < thresh:
                        keep[i] = False
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        elif strategy == 'score_fusion':
            alpha = kwargs['alpha']
            for i in range(n):
                if proj[i]:
                    cls_sim = sims[i, labels[i]].item()
                    scores[i] = scores[i] * (1 + alpha * cls_sim)

        elif strategy == 'fp_filter_score_fusion':
            thresh = kwargs['threshold']
            alpha = kwargs['alpha']
            keep = torch.ones(n, dtype=torch.bool)
            for i in range(n):
                if proj[i]:
                    max_sim = sims[i].max().item()
                    if max_sim < thresh:
                        keep[i] = False
                    else:
                        cls_sim = sims[i, labels[i]].item()
                        scores[i] = scores[i] * (1 + alpha * cls_sim)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        results.append({
            'pts_bbox': {
                'boxes_3d': LiDARInstance3DBoxes(boxes, box_dim=boxes.shape[-1]),
                'scores_3d': scores,
                'labels_3d': labels,
            }
        })
    return results


def evaluate(val_dataset, results):
    metrics = val_dataset.evaluate(results, jsonfile_prefix='submission')
    if not metrics:
        return None, None
    return metrics['pts_bbox_NuScenes/mAP'], metrics['pts_bbox_NuScenes/NDS']


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger('x', logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger('x', logging.WARNING)

    config = os.path.join(PROJECT_ROOT, 'configs/racformer_mini_research.py')
    cfgs = Config.fromfile(config)
    val_dataset = build_dataset(cfgs.data.val)

    logging.info('Loading predictions...')
    with open(os.path.join(PROJECT_ROOT, 'research/outputs/mini_preds/predictions_simple.pkl'), 'rb') as f:
        preds = pickle.load(f)

    logging.info('Loading DINOv3 scores...')
    dino_scores = torch.load(os.path.join(PROJECT_ROOT, 'research/outputs/exp_010/dino_scores.pt'),
                             map_location='cpu')

    logging.info(f'{len(preds)} samples loaded')

    # Baseline
    logging.info('\n=== Baseline ===')
    base_results = build_results(preds, dino_scores, 'score_fusion', alpha=0.0)
    mAP, NDS = evaluate(val_dataset, base_results)
    logging.info(f'Baseline: mAP={mAP:.4f}, NDS={NDS:.4f}')

    # Strategy A: FP filter
    logging.info('\n=== Strategy A: FP Filter ===')
    for thresh in [0.1, 0.15, 0.2, 0.25, 0.3]:
        results = build_results(preds, dino_scores, 'fp_filter', threshold=thresh)
        n_removed = sum(len(preds[i]['boxes_3d']) - len(results[i]['pts_bbox']['scores_3d']) for i in range(len(preds)))
        mAP, NDS = evaluate(val_dataset, results)
        logging.info(f'  thresh={thresh:.2f}: mAP={mAP:.4f}, NDS={NDS:.4f} (removed {n_removed})')

    # Strategy B: Score fusion
    logging.info('\n=== Strategy B: Score Fusion ===')
    for alpha in [0.1, 0.3, 0.5, 1.0, -0.3, -0.5]:
        results = build_results(preds, dino_scores, 'score_fusion', alpha=alpha)
        mAP, NDS = evaluate(val_dataset, results)
        logging.info(f'  alpha={alpha:.1f}: mAP={mAP:.4f}, NDS={NDS:.4f}')

    logging.info('\n=== DONE ===')


if __name__ == '__main__':
    main()
