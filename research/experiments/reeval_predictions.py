"""Re-evaluate modified predictions using nuScenes metrics.

Runs in racformerfix env (Python 3.8).
Loads modified predictions (simple numpy format), reconstructs mmdet3d objects,
and runs nuScenes evaluation.
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


def to_mmdet3d(simple_result):
    """Convert simple numpy result back to mmdet3d format."""
    boxes_np = simple_result['boxes_3d']
    scores_np = simple_result['scores_3d']
    labels_np = simple_result['labels_3d']
    return {
        'pts_bbox': {
            'boxes_3d': LiDARInstance3DBoxes(torch.tensor(boxes_np, dtype=torch.float32)),
            'scores_3d': torch.tensor(scores_np, dtype=torch.float32),
            'labels_3d': torch.tensor(labels_np, dtype=torch.long),
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--config', type=str,
                        default='/srv/nfs/shared/gnmp/RaCFormer/configs/racformer_mini_research.py')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger('x', logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger('x', logging.WARNING)

    cfgs = Config.fromfile(args.config)
    logging.info('Building dataset...')
    val_dataset = build_dataset(cfgs.data.val)

    logging.info(f'Loading predictions from {args.predictions}')
    with open(args.predictions, 'rb') as f:
        results = pickle.load(f)
    logging.info(f'  {len(results)} samples')

    # Convert simple format to mmdet3d if needed
    if len(results) > 0:
        r0 = results[0]
        if 'pts_bbox' not in r0:
            logging.info('Converting simple format to mmdet3d...')
            results = [to_mmdet3d(r) for r in results]

    logging.info('Running evaluation...')
    metrics = val_dataset.evaluate(results, jsonfile_prefix='submission')

    if not metrics:
        logging.error('No metrics returned!')
        return

    logging.info('--- Evaluation Results ---')
    logging.info('mAP: %.4f' % metrics['pts_bbox_NuScenes/mAP'])
    logging.info('NDS: %.4f' % metrics['pts_bbox_NuScenes/NDS'])
    logging.info('mATE: %.4f' % metrics['pts_bbox_NuScenes/mATE'])
    logging.info('mASE: %.4f' % metrics['pts_bbox_NuScenes/mASE'])
    logging.info('mAOE: %.4f' % metrics['pts_bbox_NuScenes/mAOE'])
    logging.info('mAVE: %.4f' % metrics['pts_bbox_NuScenes/mAVE'])
    logging.info('mAAE: %.4f' % metrics['pts_bbox_NuScenes/mAAE'])

    # Per-class AP
    for key, val in sorted(metrics.items()):
        if 'AP_dist' not in key and 'NuScenes' in key and key not in [
            'pts_bbox_NuScenes/mAP', 'pts_bbox_NuScenes/NDS',
            'pts_bbox_NuScenes/mATE', 'pts_bbox_NuScenes/mASE',
            'pts_bbox_NuScenes/mAOE', 'pts_bbox_NuScenes/mAVE',
            'pts_bbox_NuScenes/mAAE']:
            logging.info(f'  {key}: {val:.4f}')


if __name__ == '__main__':
    main()
