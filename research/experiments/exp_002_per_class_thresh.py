"""Experiment 002: Per-class score threshold optimization.

Saves raw predictions from the model (with a very low threshold),
then sweeps per-class thresholds to maximize mAP on the mini set.
"""
import os
import sys
import pickle
import numpy as np
import torch
import logging
import itertools
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] - %(message)s')

PROJECT_ROOT = '/srv/nfs/shared/gnmp/RaCFormer'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'research/outputs')

# Step 1: Run inference with very low threshold to keep all predictions
def run_inference():
    cfg = Config.fromfile(os.path.join(PROJECT_ROOT, 'configs/racformer_mini_research.py'))
    
    # Override to very low threshold to keep more predictions
    cfg.model.pts_bbox_head.bbox_coder.score_threshold = 0.01
    cfg.model.pts_bbox_head.bbox_coder.max_num = 300
    
    dataset = build_dataset(cfg.data.val)
    dataloader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False
    )
    
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(
        model, os.path.join(PROJECT_ROOT, 'checkpoints/racformer_r50_f8.pth'), map_location='cpu'
    )
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    results = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            result = model(return_loss=False, rescale=True, **data)
            results.extend(result)
            if (i + 1) % 50 == 0:
                logging.info(f'Inference: {i+1}/{len(dataloader)}')
    
    # Save raw results
    save_path = os.path.join(RESULTS_DIR, 'exp_002_raw_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f'Saved {len(results)} results to {save_path}')
    
    return results, dataset


def apply_threshold(results, class_thresholds, num_classes=10):
    """Apply per-class score thresholds to results."""
    filtered = []
    for result in results:
        r = result['pts_bbox'] if 'pts_bbox' in result else result
        scores = r['scores_3d']
        labels = r['labels_3d']
        boxes = r['boxes_3d']
        
        if torch.is_tensor(scores):
            scores_np = scores.cpu().numpy()
            labels_np = labels.cpu().numpy()
        else:
            scores_np = np.array(scores)
            labels_np = np.array(labels)
        
        mask = np.zeros(len(scores_np), dtype=bool)
        for cls_id in range(num_classes):
            cls_mask = labels_np == cls_id
            thresh = class_thresholds[cls_id]
            mask |= (cls_mask & (scores_np > thresh))
        
        if torch.is_tensor(scores):
            mask_t = torch.tensor(mask, device=scores.device)
            filtered_result = {
                'pts_bbox': {
                    'scores_3d': scores[mask_t],
                    'labels_3d': labels[mask_t],
                    'boxes_3d': boxes[mask_t],
                }
            }
        else:
            filtered_result = {
                'pts_bbox': {
                    'scores_3d': scores_np[mask],
                    'labels_3d': labels_np[mask],
                    'boxes_3d': boxes[mask],
                }
            }
        filtered.append(filtered_result)
    return filtered


def evaluate_thresholds(results, dataset, class_thresholds):
    """Evaluate a set of per-class thresholds."""
    filtered = apply_threshold(results, class_thresholds)
    metrics = dataset.evaluate(filtered, jsonfile_prefix='/tmp/thresh_eval')
    return metrics.get('pts_bbox_NuScenes/mAP', 0.0), metrics.get('pts_bbox_NuScenes/NDS', 0.0)


def sweep_per_class(results, dataset):
    """Sweep threshold for each class independently (greedy optimization)."""
    num_classes = 10
    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    
    # Start with baseline threshold
    best_thresholds = [0.05] * num_classes
    
    # First evaluate baseline
    base_mAP, base_NDS = evaluate_thresholds(results, dataset, best_thresholds)
    logging.info(f'Baseline (all 0.05): mAP={base_mAP:.4f}, NDS={base_NDS:.4f}')
    
    # Candidate thresholds to try
    candidates = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    # Greedy: optimize one class at a time, repeat until convergence
    best_mAP = base_mAP
    for iteration in range(3):  # max 3 rounds of greedy optimization
        improved = False
        for cls_id in range(num_classes):
            cls_best_mAP = best_mAP
            cls_best_thresh = best_thresholds[cls_id]
            
            for thresh in candidates:
                trial = best_thresholds.copy()
                trial[cls_id] = thresh
                mAP, NDS = evaluate_thresholds(results, dataset, trial)
                
                if mAP > cls_best_mAP:
                    cls_best_mAP = mAP
                    cls_best_thresh = thresh
                    logging.info(f'  {class_names[cls_id]}: thresh={thresh:.2f} -> mAP={mAP:.4f} (+{(mAP-best_mAP)*100:.2f}%)')
            
            if cls_best_thresh != best_thresholds[cls_id]:
                best_thresholds[cls_id] = cls_best_thresh
                best_mAP = cls_best_mAP
                improved = True
                logging.info(f'Round {iteration+1}: Updated {class_names[cls_id]} threshold to {cls_best_thresh:.2f}')
        
        if not improved:
            logging.info(f'Converged after {iteration+1} rounds')
            break
    
    # Final evaluation
    final_mAP, final_NDS = evaluate_thresholds(results, dataset, best_thresholds)
    logging.info(f'\n--- Per-class Threshold Results ---')
    for i, name in enumerate(class_names):
        logging.info(f'{name}: {best_thresholds[i]:.2f}')
    logging.info(f'Optimized: mAP={final_mAP:.4f}, NDS={final_NDS:.4f}')
    logging.info(f'Improvement: mAP +{(final_mAP - base_mAP)*100:.2f}%, NDS +{(final_NDS - base_NDS)*100:.2f}%')
    
    return best_thresholds, final_mAP, final_NDS


if __name__ == '__main__':
    logging.info('=== Experiment 002: Per-class Score Threshold Optimization ===')
    
    pkl_path = os.path.join(RESULTS_DIR, 'exp_002_raw_results.pkl')
    
    if os.path.exists(pkl_path):
        logging.info('Loading cached predictions...')
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        cfg = Config.fromfile(os.path.join(PROJECT_ROOT, 'configs/racformer_mini_research.py'))
        dataset = build_dataset(cfg.data.val)
    else:
        logging.info('Running inference...')
        results, dataset = run_inference()
    
    logging.info(f'Loaded {len(results)} predictions')
    best_thresholds, best_mAP, best_NDS = sweep_per_class(results, dataset)
    
    # Save optimized thresholds
    import json
    thresh_path = os.path.join(RESULTS_DIR, 'exp_002_best_thresholds.json')
    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    with open(thresh_path, 'w') as f:
        json.dump(dict(zip(class_names, best_thresholds)), f, indent=2)
    logging.info(f'Saved thresholds to {thresh_path}')
