"""
Evaluate model performance broken down by scene condition (day/night/rain).

Runs inference on the full validation set, then evaluates each condition
subset separately using the nuScenes evaluation protocol.

Usage:
    python tools/eval_night_gap.py --config <config> --weights <checkpoint>
    python tools/eval_night_gap.py --config <config> --weights <checkpoint> --output_dir outputs/night_eval
"""

import os
import sys
import json
import copy
import logging
import argparse
import importlib
import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from datetime import datetime
from collections import defaultdict

# Add project root to path so imports work when run as `python tools/eval_night_gap.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed, single_gpu_test, multi_gpu_test
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_gt as original_load_gt

import utils
from models.utils import VERSION


def get_condition_indices(dataset):
    """Classify each validation sample by scene condition.

    Returns:
        dict mapping condition name -> list of sample indices
    """
    conditions = defaultdict(list)
    for i, info in enumerate(dataset.data_infos):
        condition = dataset._get_scene_condition(info['token'])
        conditions[condition].append(i)
    return dict(conditions)


def evaluate_condition(dataset, results, condition_name, indices, tmpdir):
    """Evaluate results for a specific condition subset.

    Uses the same monkeypatch pattern as the front_only evaluation:
    filters both predictions and GT to the subset's sample tokens.
    """
    if len(indices) == 0:
        logging.warning(f'No samples for condition: {condition_name}')
        return {}

    # Collect sample tokens for this condition
    condition_tokens = set(dataset.data_infos[i]['token'] for i in indices)

    # Filter results and data_infos to this condition
    filtered_results = [results[i] for i in indices]
    original_data_infos = dataset.data_infos
    dataset.data_infos = [original_data_infos[i] for i in indices]

    # Monkeypatch load_gt to only include this condition's GT
    import nuscenes.eval.detection.evaluate as eval_module
    saved_load_gt = eval_module.load_gt

    def filtered_load_gt(nusc, eval_split, box_cls, verbose=False):
        gt_boxes = original_load_gt(nusc, eval_split, box_cls, verbose)
        filtered_boxes = EvalBoxes()
        for sample_token in gt_boxes.sample_tokens:
            if sample_token in condition_tokens:
                filtered_boxes.add_boxes(sample_token, gt_boxes[sample_token])
        return filtered_boxes

    eval_module.load_gt = filtered_load_gt

    try:
        prefix = os.path.join(tmpdir, f'submission_{condition_name}')
        metrics = dataset.evaluate(filtered_results, jsonfile_prefix=prefix)
    finally:
        eval_module.load_gt = saved_load_gt
        dataset.data_infos = original_data_infos

    return metrics


def extract_metrics(raw_metrics):
    """Extract the key metrics from the raw nuScenes metrics dict."""
    if not raw_metrics:
        return {}
    prefix = 'pts_bbox_NuScenes/'
    result = {}
    for key in ['mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']:
        result[key] = raw_metrics.get(prefix + key, 0.0)

    # Per-class AP
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    per_class = {}
    for cls in class_names:
        # nuScenes stores per-class AP under various key patterns
        for k, v in raw_metrics.items():
            if cls in k and 'AP' in k and 'dist' not in k:
                per_class[cls] = v
                break
    result['per_class_AP'] = per_class
    return result


def print_comparison_table(all_metrics, condition_counts):
    """Print a formatted comparison table across conditions."""
    conditions = list(all_metrics.keys())

    # Header
    header = f"{'Metric':<12}"
    for cond in conditions:
        header += f"  {cond} ({condition_counts[cond]:,})"
        header = header.ljust(len(header) + (20 - len(f"{cond} ({condition_counts[cond]:,})")))
    print("\n" + "=" * 80)
    print("CONDITION-BASED EVALUATION RESULTS")
    print("=" * 80)

    # Print sample counts
    print(f"\n{'Condition':<12}", end="")
    for cond in conditions:
        print(f"  {cond:<18}", end="")
    print()
    print(f"{'Samples':<12}", end="")
    for cond in conditions:
        print(f"  {condition_counts[cond]:<18}", end="")
    print()

    # Print main metrics
    print("\n" + "-" * 80)
    for metric in ['mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']:
        print(f"{metric:<12}", end="")
        for cond in conditions:
            val = all_metrics[cond].get(metric, 0.0)
            print(f"  {val:<18.4f}", end="")
        print()

    # Day-night gap
    if 'day' in all_metrics and 'night' in all_metrics:
        print("\n" + "-" * 80)
        print("DAY-NIGHT GAP:")
        day_map = all_metrics['day'].get('mAP', 0)
        night_map = all_metrics['night'].get('mAP', 0)
        gap = day_map - night_map
        rel_gap = (gap / day_map * 100) if day_map > 0 else 0
        print(f"  mAP gap:      {gap:.4f} ({rel_gap:.1f}% relative)")
        day_nds = all_metrics['day'].get('NDS', 0)
        night_nds = all_metrics['night'].get('NDS', 0)
        nds_gap = day_nds - night_nds
        print(f"  NDS gap:      {nds_gap:.4f}")

    # Per-class AP
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    print("\n" + "-" * 80)
    print("PER-CLASS AP:")
    print(f"{'Class':<24}", end="")
    for cond in conditions:
        print(f"  {cond:<18}", end="")
    if 'day' in all_metrics and 'night' in all_metrics:
        print(f"  {'gap':<18}", end="")
    print()
    print("-" * 80)

    for cls in class_names:
        print(f"  {cls:<22}", end="")
        day_val = None
        night_val = None
        for cond in conditions:
            val = all_metrics[cond].get('per_class_AP', {}).get(cls, None)
            if val is not None:
                print(f"  {val:<18.4f}", end="")
                if cond == 'day':
                    day_val = val
                elif cond == 'night':
                    night_val = val
            else:
                print(f"  {'N/A':<18}", end="")
        if day_val is not None and night_val is not None:
            print(f"  {day_val - night_val:<+18.4f}", end="")
        print()

    print("=" * 80)


def save_results(output_dir, all_metrics, condition_counts, args):
    """Save evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        'timestamp': timestamp,
        'config': args.config,
        'weights': args.weights,
        'condition_counts': condition_counts,
        'metrics_by_condition': {},
    }

    for cond, metrics in all_metrics.items():
        output['metrics_by_condition'][cond] = {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in metrics.items()
        }

    # Add day-night gap summary
    if 'day' in all_metrics and 'night' in all_metrics:
        day_map = all_metrics['day'].get('mAP', 0)
        night_map = all_metrics['night'].get('mAP', 0)
        output['day_night_gap'] = {
            'mAP_gap': float(day_map - night_map),
            'mAP_relative_gap': float((day_map - night_map) / day_map * 100) if day_map > 0 else 0,
            'NDS_gap': float(all_metrics['day'].get('NDS', 0) - all_metrics['night'].get('NDS', 0)),
        }

    filepath = os.path.join(output_dir, 'night_gap_results.json')
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logging.info(f'Results saved to {filepath}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate by scene condition (day/night/rain)')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--weights', required=True, help='Checkpoint file path')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results JSON')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--conditions', type=str, default='all,day,night,rain',
                        help='Comma-separated conditions to evaluate (default: all,day,night,rain)')
    args = parser.parse_args()

    conditions_to_eval = [c.strip() for c in args.conditions.split(',')]

    # Parse config
    cfgs = Config.fromfile(args.config)

    # Register custom modules
    importlib.import_module('models')
    importlib.import_module('loaders')

    # Suppress noisy loggers
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)

    assert torch.cuda.is_available(), 'GPU required'
    utils.init_logging(None, cfgs.debug if hasattr(cfgs, 'debug') else False)

    torch.cuda.set_device(0)
    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True

    # Build dataset and model
    logging.info(f'Loading validation set from {cfgs.data.val.data_root}')
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    logging.info(f'Creating model: {cfgs.model.type}')
    model = build_model(cfgs.model)
    model.cuda()
    model = MMDataParallel(model, [0])

    logging.info(f'Loading checkpoint from {args.weights}')
    checkpoint = load_checkpoint(
        model, args.weights, map_location='cuda', strict=True,
        logger=logging.Logger(__name__, logging.ERROR)
    )
    if 'version' in checkpoint:
        VERSION.name = checkpoint['version']

    # Run inference on full validation set
    logging.info('Running inference on full validation set...')
    results = single_gpu_test(model, val_loader)
    logging.info(f'Inference complete: {len(results)} samples')

    # Classify samples by condition
    condition_indices = get_condition_indices(val_dataset)
    condition_counts = {k: len(v) for k, v in condition_indices.items()}
    condition_counts['all'] = len(results)
    logging.info(f'Sample counts by condition: {condition_counts}')

    # Create tmpdir for submission files
    tmpdir = os.path.join(
        args.output_dir or 'outputs/night_eval',
        'tmp_submissions'
    )
    os.makedirs(tmpdir, exist_ok=True)

    # Evaluate each condition
    all_metrics = {}

    if 'all' in conditions_to_eval:
        logging.info('Evaluating: ALL samples')
        raw = val_dataset.evaluate(results, jsonfile_prefix=os.path.join(tmpdir, 'submission_all'))
        all_metrics['all'] = extract_metrics(raw)

    for condition in ['day', 'night', 'rain']:
        if condition not in conditions_to_eval:
            continue
        if condition not in condition_indices:
            logging.warning(f'No samples found for condition: {condition}')
            continue
        logging.info(f'Evaluating: {condition} ({len(condition_indices[condition])} samples)')
        raw = evaluate_condition(
            val_dataset, results, condition,
            condition_indices[condition], tmpdir
        )
        all_metrics[condition] = extract_metrics(raw)

    # Print comparison
    print_comparison_table(all_metrics, condition_counts)

    # Save results
    output_dir = args.output_dir or 'outputs/night_eval'
    save_results(output_dir, all_metrics, condition_counts, args)

    logging.info('Done.')


if __name__ == '__main__':
    main()
