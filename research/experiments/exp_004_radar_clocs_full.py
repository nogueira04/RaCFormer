"""Experiment 004 Full Validation: Radar-CLOCs best configs on full 6019 samples."""
import os
import sys
import numpy as np
import torch
import importlib

PROJECT_ROOT = '/srv/nfs/shared/gnmp/RaCFormer'
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


def run_inference(cfg_path, weights_path):
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
            if (i + 1) % 500 == 0:
                logging.info(f'Inference: {i+1}/{len(dataloader)}')
    logging.info(f'Inference done: {len(results)} samples')
    return results, dataset


def apply_radar_clocs(results, dataset, boost=0.2, penalty=0.0, radius=3.0):
    adjusted = []
    for i, result in enumerate(results):
        r = result['pts_bbox'] if 'pts_bbox' in result else result
        scores = r['scores_3d']
        labels = r['labels_3d']
        boxes = r['boxes_3d']

        sample_token = dataset.data_infos[i]['token']
        points, _, _ = get_nu_radar(sample_token, mutil_sweep=True, num_sweeps=6, filter=True)
        radar_xyz = points[:3, :].numpy().T if points.shape[1] > 0 else np.zeros((0, 3))

        if hasattr(boxes, 'tensor'):
            boxes_np = boxes.tensor.cpu().numpy()
        else:
            boxes_np = np.array(boxes)
        scores_np = scores.cpu().numpy().copy() if torch.is_tensor(scores) else np.array(scores, dtype=float).copy()

        if len(scores_np) > 0 and len(boxes_np) > 0 and len(radar_xyz) > 0:
            pred_xy = boxes_np[:, :2]
            radar_xy = radar_xyz[:, :2]
            diff = pred_xy[:, None, :] - radar_xy[None, :, :]
            bev_dist = np.sqrt((diff ** 2).sum(axis=-1))
            has_radar = bev_dist.min(axis=1) < radius
            scores_np[has_radar] *= (1.0 + boost)
            if penalty > 0:
                scores_np[~has_radar] *= (1.0 - penalty)
            scores_np = np.clip(scores_np, 0.0, 1.0)

        new_scores = torch.tensor(scores_np, dtype=scores.dtype, device=scores.device) if torch.is_tensor(scores) else scores_np
        adjusted.append({'pts_bbox': {'scores_3d': new_scores, 'labels_3d': labels, 'boxes_3d': boxes}})

        if (i + 1) % 500 == 0:
            logging.info(f'Radar-CLOCs: {i+1}/{len(results)}')
    return adjusted


if __name__ == '__main__':
    logging.info('=== Exp 004 Full Validation: Radar-CLOCs ===')
    cfg_path = os.path.join(PROJECT_ROOT, 'configs/racformer_r50_nuimg_704x256_f8.py')
    weights_path = os.path.join(PROJECT_ROOT, 'checkpoints/racformer_r50_f8.pth')

    results, dataset = run_inference(cfg_path, weights_path)

    # Baseline
    metrics = dataset.evaluate(results, jsonfile_prefix='/tmp/full_base')
    logging.info(f"Baseline: mAP={metrics['pts_bbox_NuScenes/mAP']:.4f}, NDS={metrics['pts_bbox_NuScenes/NDS']:.4f}")

    # Config 1: boost=0.2, r=3 (best mAP from screening)
    adj1 = apply_radar_clocs(results, dataset, boost=0.2, penalty=0.0, radius=3.0)
    m1 = dataset.evaluate(adj1, jsonfile_prefix='/tmp/full_clocs1')
    logging.info(f"boost_0.2_r3: mAP={m1['pts_bbox_NuScenes/mAP']:.4f}, NDS={m1['pts_bbox_NuScenes/NDS']:.4f}")

    # Config 2: boost=0.1, r=3 (balanced mAP/NDS)
    adj2 = apply_radar_clocs(results, dataset, boost=0.1, penalty=0.0, radius=3.0)
    m2 = dataset.evaluate(adj2, jsonfile_prefix='/tmp/full_clocs2')
    logging.info(f"boost_0.1_r3: mAP={m2['pts_bbox_NuScenes/mAP']:.4f}, NDS={m2['pts_bbox_NuScenes/NDS']:.4f}")

    # Config 3: penalty=0.1, r=3 (NDS-friendly)
    adj3 = apply_radar_clocs(results, dataset, boost=0.0, penalty=0.1, radius=3.0)
    m3 = dataset.evaluate(adj3, jsonfile_prefix='/tmp/full_clocs3')
    logging.info(f"penalty_0.1_r3: mAP={m3['pts_bbox_NuScenes/mAP']:.4f}, NDS={m3['pts_bbox_NuScenes/NDS']:.4f}")

    # Config 4: boost=0.3, r=3 (Pareto: good mAP, stable NDS in screening)
    adj4 = apply_radar_clocs(results, dataset, boost=0.3, penalty=0.0, radius=3.0)
    m4 = dataset.evaluate(adj4, jsonfile_prefix='/tmp/full_clocs4')
    logging.info(f"boost_0.3_r3: mAP={m4['pts_bbox_NuScenes/mAP']:.4f}, NDS={m4['pts_bbox_NuScenes/NDS']:.4f}")

    logging.info('=== Full Validation Complete ===')
