"""Experiment 004: Radar-CLOCs — post-hoc radar verification of predictions.

Runs inference once, then sweeps boost/penalty parameters using radar
proximity as a verification signal for predicted boxes.
"""
import os
import sys
import copy
import pickle
import numpy as np
import torch
import importlib

PROJECT_ROOT = '/srv/nfs/shared/gnmp/RaCFormer'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'research/outputs')
sys.path.insert(0, PROJECT_ROOT)

importlib.import_module('models')
importlib.import_module('loaders')

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] - %(message)s')

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from loaders.nuscenes_dataset import get_nu_radar


def run_inference_and_collect(cfg_path, weights_path):
    cfg = Config.fromfile(cfg_path)
    dataset = build_dataset(cfg.data.val)
    dataloader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False
    )
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, weights_path, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    results = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            result = model(return_loss=False, rescale=True, **data)
            results.extend(result)
            if (i + 1) % 50 == 0:
                logging.info(f'Inference: {i+1}/{len(dataloader)}')
    logging.info(f'Inference done: {len(results)} samples')
    return results, dataset


def get_radar_points_for_sample(sample_token):
    points, _, _ = get_nu_radar(sample_token, mutil_sweep=True, num_sweeps=6, filter=True)
    if points.shape[1] == 0:
        return np.zeros((0, 3))
    return points[:3, :].numpy().T  # (N, 3)


def compute_radar_evidence(pred_boxes, radar_xyz, radius=3.0):
    M = len(pred_boxes)
    has_radar = np.zeros(M, dtype=bool)
    min_dist = np.full(M, 999.0)
    radar_count = np.zeros(M, dtype=int)

    if len(radar_xyz) == 0 or M == 0:
        return has_radar, min_dist, radar_count

    pred_xy = pred_boxes[:, :2]
    radar_xy = radar_xyz[:, :2]
    diff = pred_xy[:, None, :] - radar_xy[None, :, :]
    bev_dist = np.sqrt((diff ** 2).sum(axis=-1))

    min_dist = bev_dist.min(axis=1)
    has_radar = min_dist < radius
    radar_count = (bev_dist < radius).sum(axis=1)
    return has_radar, min_dist, radar_count


def apply_radar_clocs(results, dataset, boost=0.1, penalty=0.0, radius=3.0):
    adjusted_results = []
    for i, result in enumerate(results):
        r = result['pts_bbox'] if 'pts_bbox' in result else result
        scores = r['scores_3d']
        labels = r['labels_3d']
        boxes = r['boxes_3d']

        sample_token = dataset.data_infos[i]['token']
        radar_xyz = get_radar_points_for_sample(sample_token)

        if hasattr(boxes, 'tensor'):
            boxes_np = boxes.tensor.cpu().numpy()
        else:
            boxes_np = np.array(boxes)

        if torch.is_tensor(scores):
            scores_np = scores.cpu().numpy().copy()
        else:
            scores_np = np.array(scores, dtype=float).copy()

        if len(scores_np) > 0 and len(boxes_np) > 0:
            has_radar, min_dist, radar_count = compute_radar_evidence(boxes_np, radar_xyz, radius)
            scores_np[has_radar] *= (1.0 + boost)
            if penalty > 0:
                scores_np[~has_radar] *= (1.0 - penalty)
            scores_np = np.clip(scores_np, 0.0, 1.0)

        if torch.is_tensor(scores):
            new_scores = torch.tensor(scores_np, dtype=scores.dtype, device=scores.device)
        else:
            new_scores = scores_np

        adjusted_results.append({
            'pts_bbox': {
                'scores_3d': new_scores,
                'labels_3d': labels,
                'boxes_3d': boxes,
            }
        })
    return adjusted_results


def evaluate_config(results, dataset, boost, penalty, radius):
    adjusted = apply_radar_clocs(results, dataset, boost=boost, penalty=penalty, radius=radius)
    metrics = dataset.evaluate(adjusted, jsonfile_prefix='/tmp/clocs_eval')
    mAP = metrics.get('pts_bbox_NuScenes/mAP', 0.0)
    NDS = metrics.get('pts_bbox_NuScenes/NDS', 0.0)
    return mAP, NDS


if __name__ == '__main__':
    logging.info('=== Experiment 004: Radar-CLOCs ===')

    cfg_path = os.path.join(PROJECT_ROOT, 'configs/racformer_mini_research.py')
    weights_path = os.path.join(PROJECT_ROOT, 'checkpoints/racformer_r50_f8.pth')

    results, dataset = run_inference_and_collect(cfg_path, weights_path)

    base_mAP, base_NDS = evaluate_config(results, dataset, boost=0.0, penalty=0.0, radius=3.0)
    logging.info(f'Baseline: mAP={base_mAP:.4f}, NDS={base_NDS:.4f}')

    configs = [
        (0.1, 0.0, 3.0, 'boost_0.1_r3'),
        (0.2, 0.0, 3.0, 'boost_0.2_r3'),
        (0.3, 0.0, 3.0, 'boost_0.3_r3'),
        (0.5, 0.0, 3.0, 'boost_0.5_r3'),
        (0.0, 0.1, 3.0, 'penalty_0.1_r3'),
        (0.0, 0.2, 3.0, 'penalty_0.2_r3'),
        (0.0, 0.3, 3.0, 'penalty_0.3_r3'),
        (0.1, 0.1, 3.0, 'boost_0.1_pen_0.1_r3'),
        (0.2, 0.1, 3.0, 'boost_0.2_pen_0.1_r3'),
        (0.2, 0.2, 3.0, 'boost_0.2_pen_0.2_r3'),
        (0.2, 0.0, 2.0, 'boost_0.2_r2'),
        (0.2, 0.0, 5.0, 'boost_0.2_r5'),
        (0.0, 0.2, 2.0, 'penalty_0.2_r2'),
        (0.0, 0.2, 5.0, 'penalty_0.2_r5'),
    ]

    best_mAP = base_mAP
    best_config = None

    for boost, penalty, radius, desc in configs:
        mAP, NDS = evaluate_config(results, dataset, boost, penalty, radius)
        delta = (mAP - base_mAP) * 100
        marker = ' ***' if mAP > best_mAP else ''
        logging.info(f'{desc}: mAP={mAP:.4f} ({delta:+.2f}%), NDS={NDS:.4f}{marker}')
        if mAP > best_mAP:
            best_mAP = mAP
            best_config = (boost, penalty, radius, desc)

    if best_config:
        logging.info(f'Best: {best_config[3]}, mAP={best_mAP:.4f} (+{(best_mAP-base_mAP)*100:.2f}%)')
    else:
        logging.info(f'No improvement over baseline mAP={base_mAP:.4f}')
