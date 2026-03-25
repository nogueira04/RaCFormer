"""Re-evaluate modified predictions saved as torch format.
Runs in racformerfix env (Python 3.8).
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
    data = torch.load(args.predictions, map_location='cpu')
    logging.info(f'  {len(data)} samples')

    # Convert to mmdet3d format
    results = []
    for d in data:
        results.append({
            'pts_bbox': {
                'boxes_3d': LiDARInstance3DBoxes(d['boxes_3d'].float(), box_dim=d['boxes_3d'].shape[-1]),
                'scores_3d': d['scores_3d'].float(),
                'labels_3d': d['labels_3d'].long(),
            }
        })

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

    for key, val in sorted(metrics.items()):
        if 'AP_dist' not in key and 'NuScenes' in key and key not in [
            'pts_bbox_NuScenes/mAP', 'pts_bbox_NuScenes/NDS',
            'pts_bbox_NuScenes/mATE', 'pts_bbox_NuScenes/mASE',
            'pts_bbox_NuScenes/mAOE', 'pts_bbox_NuScenes/mAVE',
            'pts_bbox_NuScenes/mAAE']:
            logging.info(f'  {key}: {val:.4f}')


if __name__ == '__main__':
    main()
