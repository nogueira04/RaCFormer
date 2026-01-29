import os
import json
import pickle
import time
import numpy as np
import utils
import logging
import argparse
import importlib
import torch
import torch.distributed
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from datetime import datetime
from mmengine.config import Config
from mmengine.model import MMDistributedDataParallel
from mmengine.runner import load_checkpoint
# Compatibility: MMDataParallel deprecated, use simple wrapper
from torch.nn.parallel import DataParallel as MMDataParallel
from mmengine.runner import set_random_seed
from tqdm import tqdm

def single_gpu_test(model, data_loader, show=False, out_dir=None):
    """Test model with single GPU, with timing metrics."""
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = tqdm(total=len(dataset))

    # Timing metrics
    inference_times = []

    # Warmup run
    warmup_done = False

    for i, batch in enumerate(data_loader):
        batch_results = []
        for sample in batch:
            # Handle different data formats
            if isinstance(sample, dict):
                data = sample
            elif isinstance(sample, (list, tuple)):
                while isinstance(sample, (list, tuple)) and len(sample) == 1:
                    sample = sample[0]
                if isinstance(sample, dict):
                    data = sample
                elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
                    data = sample[0] if isinstance(sample[0], dict) else {'inputs': sample[0], 'data_samples': sample[1]}
                else:
                    continue
            else:
                continue

            # Wrap data in batch format expected by model
            for key in ['img_metas', 'img', 'radar_points', 'radar_depth', 'radar_rcs', 'gt_depth']:
                if key in data and not isinstance(data[key], list):
                    data[key] = [data[key]]

            # Warmup for first sample (GPU initialization)
            if not warmup_done:
                with torch.no_grad():
                    _ = model(return_loss=False, rescale=True, **data)
                torch.cuda.synchronize()
                warmup_done = True

            # Timed inference
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

            torch.cuda.synchronize()
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms

            if isinstance(result, list):
                batch_results.extend(result)
            else:
                batch_results.append(result)
        results.extend(batch_results)
        prog_bar.update(len(batch))
    prog_bar.close()

    # Calculate timing statistics
    timing_stats = {}
    if inference_times:
        timing_stats = {
            'mean_inference_ms': np.mean(inference_times),
            'std_inference_ms': np.std(inference_times),
            'min_inference_ms': np.min(inference_times),
            'max_inference_ms': np.max(inference_times),
            'median_inference_ms': np.median(inference_times),
            'fps': 1000.0 / np.mean(inference_times),
            'num_samples': len(inference_times),
        }

    return results, timing_stats

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple GPUs, with timing metrics."""
    model.eval()
    results = []
    inference_times = []
    dataset = data_loader.dataset
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    if rank == 0:
        prog_bar = tqdm(total=len(dataset))

    warmup_done = False

    for i, batch in enumerate(data_loader):
        batch_results = []
        for sample in batch:
            if isinstance(sample, dict):
                data = sample
            elif isinstance(sample, (list, tuple)):
                while isinstance(sample, (list, tuple)) and len(sample) == 1:
                    sample = sample[0]
                if isinstance(sample, dict):
                    data = sample
                else:
                    continue
            else:
                continue

            for key in ['img_metas', 'img', 'radar_points', 'radar_depth', 'radar_rcs', 'gt_depth']:
                if key in data and not isinstance(data[key], list):
                    data[key] = [data[key]]

            # Warmup
            if not warmup_done:
                with torch.no_grad():
                    _ = model(return_loss=False, rescale=True, **data)
                torch.cuda.synchronize()
                warmup_done = True

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

            torch.cuda.synchronize()
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)

            if isinstance(result, list):
                batch_results.extend(result)
            else:
                batch_results.append(result)
        results.extend(batch_results)
        if rank == 0:
            prog_bar.update(len(batch) * world_size)
    if rank == 0:
        prog_bar.close()

    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)

    timing_stats = {}
    if inference_times:
        timing_stats = {
            'mean_inference_ms': np.mean(inference_times),
            'std_inference_ms': np.std(inference_times),
            'min_inference_ms': np.min(inference_times),
            'max_inference_ms': np.max(inference_times),
            'median_inference_ms': np.median(inference_times),
            'fps': 1000.0 / np.mean(inference_times),
            'num_samples': len(inference_times),
        }

    return results, timing_stats

def collect_results_cpu(results, size, tmpdir=None):
    """Collect results under cpu mode (simplified)."""
    return results[:size]

def collect_results_gpu(results, size):
    """Collect results under gpu mode (simplified)."""
    return results[:size]
from mmdet3d.registry import DATASETS
from mmengine.dataset import worker_init_fn
from torch.utils.data import DataLoader

# Compatibility wrapper for build_dataset
def build_dataset(cfg, default_args=None):
    return DATASETS.build(cfg, default_args=default_args)

# Custom collate function that handles class types and non-tensor data
def pseudo_collate(batch):
    """Collate function that doesn't stack tensors - just returns a list.

    This handles cases where the batch contains non-collatable types like
    class types, None values, or complex nested structures.
    """
    return batch

# Compatibility wrapper for build_dataloader
def build_dataloader(dataset, samples_per_gpu, workers_per_gpu, num_gpus=1, dist=True, shuffle=True, seed=None, **kwargs):
    from functools import partial
    from mmengine.dist import get_dist_info
    rank, world_size = get_dist_info()
    if dist:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    else:
        sampler = None
    dataloader = DataLoader(
        dataset,
        batch_size=samples_per_gpu,
        sampler=sampler,
        num_workers=workers_per_gpu,
        shuffle=shuffle if sampler is None else False,
        collate_fn=pseudo_collate,  # Use custom collate to handle class types
        **kwargs
    )
    return dataloader
from mmdet3d.registry import MODELS

# Compatibility wrapper for build_model
def build_model(cfg, train_cfg=None, test_cfg=None):
    return MODELS.build(cfg)
from models.utils import VERSION

# Default distance bins for analysis (in meters)
DEFAULT_DISTANCE_BINS = [0, 20, 40, 60, 80, 100]

# Color mapping for different classes in BEV visualization
CLASS_COLORS = {
    'car': '#1f77b4',
    'truck': '#ff7f0e',
    'bus': '#2ca02c',
    'trailer': '#d62728',
    'construction_vehicle': '#9467bd',
    'pedestrian': '#8c564b',
    'motorcycle': '#e377c2',
    'bicycle': '#7f7f7f',
    'traffic_cone': '#bcbd22',
    'barrier': '#17becf',
}


def evaluate(dataset, results, epoch):
    metrics = dataset.evaluate(results, jsonfile_prefix='submission')

    if not metrics or 'error' in metrics:
        logging.warning('No metrics returned or evaluation failed. Skipping evaluation report.')
        return metrics if metrics else {}

    # Handle both old format (pts_bbox_NuScenes/mAP) and new format (mAP)
    def get_metric(key):
        old_key = f'pts_bbox_NuScenes/{key}'
        if old_key in metrics:
            return metrics[old_key]
        return metrics.get(key, 0.0)

    mAP = get_metric('mAP')
    mATE = get_metric('mATE')
    mASE = get_metric('mASE')
    mAOE = get_metric('mAOE')
    mAVE = get_metric('mAVE')
    mAAE = get_metric('mAAE')
    NDS = get_metric('NDS')

    logging.info('--- Evaluation Results (Epoch %d) ---' % epoch)
    logging.info('mAP: %.4f' % mAP)
    logging.info('mATE: %.4f' % mATE)
    logging.info('mASE: %.4f' % mASE)
    logging.info('mAOE: %.4f' % mAOE)
    logging.info('mAVE: %.4f' % mAVE)
    logging.info('mAAE: %.4f' % mAAE)
    logging.info('NDS: %.4f' % NDS)

    return {
        'mAP': mAP,
        'mATE': mATE,
        'mASE': mASE,
        'mAOE': mAOE,
        'mAVE': mAVE,
        'mAAE': mAAE,
        'NDS': NDS,
        'all_metrics': metrics,
    }


def compute_distance(boxes):
    """Compute distance from ego vehicle (origin) for each box.
    
    Args:
        boxes: LiDARInstance3DBoxes or tensor with shape (N, 7+) where first 3 are x, y, z
    
    Returns:
        distances: numpy array of distances
    """
    if hasattr(boxes, 'center'):
        centers = boxes.center.cpu().numpy()
    elif hasattr(boxes, 'tensor'):
        centers = boxes.tensor[:, :3].cpu().numpy()
    else:
        centers = np.array(boxes)[:, :3]
    
    # Distance in XY plane (2D Euclidean distance)
    distances = np.sqrt(centers[:, 0]**2 + centers[:, 1]**2)
    return distances


def analyze_by_distance(results, val_dataset, distance_bins):
    """Analyze detection results by distance bins.
    
    Returns dict with counts and statistics per distance bin.
    """
    analysis = {
        'distance_bins': distance_bins,
        'bin_labels': [],
        'prediction_counts': [],
        'avg_scores_per_bin': [],
    }
    
    for i in range(len(distance_bins) - 1):
        analysis['bin_labels'].append(f"{distance_bins[i]}-{distance_bins[i+1]}m")
    analysis['bin_labels'].append(f"{distance_bins[-1]}m+")
    
    # Initialize counters
    bin_counts = [0] * (len(distance_bins))
    bin_scores = [[] for _ in range(len(distance_bins))]
    
    for result in results:
        if 'pts_bbox' in result:
            result = result['pts_bbox']
        
        boxes = result['boxes_3d']
        scores = result['scores_3d']
        
        if len(boxes) == 0:
            continue
            
        distances = compute_distance(boxes)
        scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else np.array(scores)
        
        for dist, score in zip(distances, scores_np):
            bin_idx = np.searchsorted(distance_bins, dist, side='right') - 1
            bin_idx = max(0, min(bin_idx, len(distance_bins) - 1))
            bin_counts[bin_idx] += 1
            bin_scores[bin_idx].append(score)
    
    analysis['prediction_counts'] = bin_counts
    analysis['avg_scores_per_bin'] = [
        float(np.mean(scores)) if len(scores) > 0 else 0.0 
        for scores in bin_scores
    ]
    analysis['total_predictions'] = sum(bin_counts)
    
    return analysis


def save_bev_visualization(results, val_dataset, output_dir, max_samples=50):
    """Generate and save BEV visualizations of detections.
    
    Args:
        results: List of detection results
        val_dataset: Validation dataset
        output_dir: Directory to save visualizations
        max_samples: Maximum number of samples to visualize
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyBboxPatch
        import matplotlib.patches as mpatches
    except ImportError:
        logging.warning("matplotlib not available, skipping BEV visualization")
        return
    
    vis_dir = os.path.join(output_dir, 'bev_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Class names for this dataset
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    
    num_to_visualize = min(max_samples, len(results))
    logging.info(f"Generating BEV visualizations for {num_to_visualize} samples...")
    
    for idx in range(num_to_visualize):
        result = results[idx]
        if 'pts_bbox' in result:
            result = result['pts_bbox']
        
        boxes = result['boxes_3d']
        scores = result['scores_3d']
        labels = result['labels_3d']
        
        if len(boxes) == 0:
            continue
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Set up BEV plot
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m) - Forward')
        ax.set_ylabel('Y (m) - Left')
        
        # Draw ego vehicle at origin
        ego_rect = Rectangle((-1, -0.5), 2, 1, linewidth=2, 
                            edgecolor='black', facecolor='yellow', zorder=10)
        ax.add_patch(ego_rect)
        ax.annotate('EGO', (0, 0), ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Draw distance circles
        for r in [20, 40, 60]:
            circle = plt.Circle((0, 0), r, fill=False, linestyle='--', 
                               color='gray', alpha=0.5)
            ax.add_patch(circle)
            ax.annotate(f'{r}m', (r, 0), fontsize=8, color='gray')
        
        # Get box parameters
        if hasattr(boxes, 'tensor'):
            boxes_np = boxes.tensor.cpu().numpy()
        else:
            boxes_np = np.array(boxes)
        
        scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else np.array(scores)
        labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
        
        # Draw each detection
        for i, (box, score, label) in enumerate(zip(boxes_np, scores_np, labels_np)):
            if score < 0.3:  # Skip low confidence detections
                continue
                
            x, y, z, l, w, h, yaw = box[:7]
            
            # Get class name and color
            class_name = class_names[int(label)] if int(label) < len(class_names) else 'unknown'
            color = CLASS_COLORS.get(class_name, '#000000')
            
            # Create rotated rectangle
            # Box corners before rotation
            corners = np.array([
                [-l/2, -w/2],
                [l/2, -w/2],
                [l/2, w/2],
                [-l/2, w/2],
                [-l/2, -w/2]
            ])
            
            # Rotation matrix
            rot = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            
            # Rotate and translate
            corners_rot = corners @ rot.T + np.array([x, y])
            
            # Draw box
            ax.plot(corners_rot[:, 0], corners_rot[:, 1], color=color, linewidth=2)
            
            # Add label
            ax.annotate(f'{class_name[:3]}\n{score:.2f}', 
                       (x, y), fontsize=6, ha='center', va='center',
                       color=color, fontweight='bold')
        
        # Create legend
        legend_patches = []
        for cls_name, cls_color in CLASS_COLORS.items():
            legend_patches.append(mpatches.Patch(color=cls_color, label=cls_name))
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8)
        
        # Get sample info if available
        try:
            sample_info = val_dataset.data_infos[idx]
            sample_token = sample_info.get('token', f'sample_{idx}')
            title = f'Sample: {sample_token[:16]}... | Detections: {len(boxes)}'
        except:
            title = f'Sample {idx} | Detections: {len(boxes)}'
        
        ax.set_title(title, fontsize=12)
        
        # Save figure
        save_path = os.path.join(vis_dir, f'bev_sample_{idx:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    logging.info(f"BEV visualizations saved to {vis_dir}")


def save_camera_visualization(results, val_dataset, output_dir, max_samples=20):
    """Generate and save camera images with projected 3D detections.

    Args:
        results: List of detection results
        val_dataset: Validation dataset
        output_dir: Directory to save visualizations
        max_samples: Maximum number of samples to visualize
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        logging.warning("matplotlib/PIL not available, skipping camera visualization")
        return

    vis_dir = os.path.join(output_dir, 'camera_visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    class_names = list(CLASS_COLORS.keys())
    num_to_visualize = min(max_samples, len(results))
    logging.info(f"Generating camera visualizations for {num_to_visualize} samples...")

    for idx in range(num_to_visualize):
        try:
            result = results[idx]
            if 'pts_bbox' in result:
                result = result['pts_bbox']

            boxes = result['boxes_3d']
            scores = result['scores_3d']
            labels = result['labels_3d']

            # Get dataset info
            info = val_dataset.data_infos[idx]

            # Get image paths
            if 'cams' in info:
                cam_names = list(info['cams'].keys())[:1]  # Just front camera
                for cam_name in cam_names:
                    cam_info = info['cams'][cam_name]
                    img_path = cam_info['data_path']

                    if not os.path.exists(img_path):
                        continue

                    # Load image
                    img = Image.open(img_path)
                    img_array = np.array(img)

                    # Create figure
                    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
                    ax.imshow(img_array)

                    # Get camera intrinsics and lidar2cam transform
                    intrinsic = cam_info['cam_intrinsic']
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                    # Project boxes to image
                    if hasattr(boxes, 'tensor'):
                        box_tensor = boxes.tensor.cpu().numpy()
                    else:
                        box_tensor = np.array(boxes)

                    scores_np = scores.cpu().numpy() if hasattr(scores, 'cpu') else np.array(scores)
                    labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.array(labels)

                    for j in range(len(box_tensor)):
                        if scores_np[j] < 0.3:  # Score threshold
                            continue

                        box = box_tensor[j]
                        center = box[:3]

                        # Transform to camera frame
                        center_cam = lidar2cam_r.T @ center - lidar2cam_t

                        # Skip if behind camera
                        if center_cam[2] <= 0:
                            continue

                        # Project to image
                        center_img = intrinsic @ center_cam
                        center_img = center_img[:2] / center_img[2]

                        # Check if in image bounds
                        if 0 <= center_img[0] < img_array.shape[1] and 0 <= center_img[1] < img_array.shape[0]:
                            label_idx = int(labels_np[j])
                            if label_idx < len(class_names):
                                class_name = class_names[label_idx]
                                color = CLASS_COLORS.get(class_name, '#ffffff')
                            else:
                                class_name = f'class_{label_idx}'
                                color = '#ffffff'

                            # Draw detection marker
                            ax.plot(center_img[0], center_img[1], 'o', color=color, markersize=10)
                            ax.text(center_img[0], center_img[1] - 15,
                                   f'{class_name[:3]} {scores_np[j]:.2f}',
                                   fontsize=8, color=color, ha='center',
                                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

                    ax.set_xlim(0, img_array.shape[1])
                    ax.set_ylim(img_array.shape[0], 0)
                    ax.axis('off')
                    ax.set_title(f'Sample {idx} - {cam_name}', fontsize=12)

                    save_path = os.path.join(vis_dir, f'cam_sample_{idx:04d}_{cam_name}.png')
                    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig)

        except Exception as e:
            logging.warning(f"Failed to visualize sample {idx}: {e}")
            continue

    logging.info(f"Camera visualizations saved to {vis_dir}")


def save_evaluation_outputs(output_dir, metrics, results, val_dataset, args, distance_bins):
    """Save all evaluation outputs to the specified directory.
    
    Args:
        output_dir: Directory to save outputs
        metrics: Evaluation metrics dict
        results: Raw detection results
        val_dataset: Validation dataset
        args: Command line arguments
        distance_bins: Distance bins for analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.info(f"Saving evaluation outputs to {output_dir}")
    
    # 1. Save overall metrics as JSON
    metrics_file = os.path.join(output_dir, 'metrics.json')
    metrics_to_save = {
        'timestamp': timestamp,
        'config': args.config,
        'weights': args.weights,
        'mAP': float(metrics.get('mAP', 0)),
        'NDS': float(metrics.get('NDS', 0)),
        'mATE': float(metrics.get('mATE', 0)),
        'mASE': float(metrics.get('mASE', 0)),
        'mAOE': float(metrics.get('mAOE', 0)),
        'mAVE': float(metrics.get('mAVE', 0)),
        'mAAE': float(metrics.get('mAAE', 0)),
    }

    # Add timing stats if available
    if 'timing' in metrics:
        metrics_to_save['timing'] = metrics['timing']

    # Extract per-class AP if available
    all_metrics = metrics.get('all_metrics', {})
    per_class_ap = {}
    for key, value in all_metrics.items():
        if 'AP_dist' in key or '/AP/' in key:
            per_class_ap[key] = float(value) if value is not None else None
    metrics_to_save['per_class_details'] = per_class_ap

    with open(metrics_file, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    logging.info(f"Saved metrics to {metrics_file}")
    
    # 2. Save distance analysis
    distance_analysis = analyze_by_distance(results, val_dataset, distance_bins)
    distance_file = os.path.join(output_dir, 'distance_analysis.json')
    with open(distance_file, 'w') as f:
        json.dump(distance_analysis, f, indent=2)
    logging.info(f"Saved distance analysis to {distance_file}")
    
    # 3. Save predictions pickle (for detailed analysis)
    predictions_file = os.path.join(output_dir, 'predictions.pkl')
    with open(predictions_file, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f"Saved predictions to {predictions_file}")
    
    # 4. Generate BEV visualizations
    save_bev_visualization(results, val_dataset, output_dir,
                          max_samples=getattr(args, 'max_vis_samples', 50))

    # 5. Generate camera visualizations with projected detections
    save_camera_visualization(results, val_dataset, output_dir,
                             max_samples=getattr(args, 'max_vis_samples', 20))

    # 6. Generate text summary report
    report_file = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Weights: {args.weights}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"mAP:  {metrics.get('mAP', 0):.4f}\n")
        f.write(f"NDS:  {metrics.get('NDS', 0):.4f}\n")
        f.write(f"mATE: {metrics.get('mATE', 0):.4f}\n")
        f.write(f"mASE: {metrics.get('mASE', 0):.4f}\n")
        f.write(f"mAOE: {metrics.get('mAOE', 0):.4f}\n")
        f.write(f"mAVE: {metrics.get('mAVE', 0):.4f}\n")
        f.write(f"mAAE: {metrics.get('mAAE', 0):.4f}\n\n")

        # Add timing stats if available
        if 'timing' in metrics:
            timing = metrics['timing']
            f.write("-" * 40 + "\n")
            f.write("LATENCY METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean inference time: {timing.get('mean_inference_ms', 0):.2f} ms\n")
            f.write(f"Std inference time:  {timing.get('std_inference_ms', 0):.2f} ms\n")
            f.write(f"Min inference time:  {timing.get('min_inference_ms', 0):.2f} ms\n")
            f.write(f"Max inference time:  {timing.get('max_inference_ms', 0):.2f} ms\n")
            f.write(f"FPS: {timing.get('fps', 0):.2f}\n")
            f.write(f"Samples processed: {timing.get('num_samples', 0)}\n\n")

        f.write("-" * 40 + "\n")
        f.write("DISTANCE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total predictions: {distance_analysis['total_predictions']}\n\n")
        f.write(f"{'Distance Bin':<15} {'Count':<10} {'Avg Score':<10}\n")
        f.write("-" * 35 + "\n")
        for label, count, avg_score in zip(
            distance_analysis['bin_labels'],
            distance_analysis['prediction_counts'],
            distance_analysis['avg_scores_per_bin']
        ):
            f.write(f"{label:<15} {count:<10} {avg_score:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 60 + "\n")
    
    logging.info(f"Saved evaluation report to {report_file}")
    logging.info(f"All evaluation outputs saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Validate a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation outputs (metrics, visualizations, etc.)')
    parser.add_argument('--distance_bins', type=str, default=None,
                        help='Comma-separated distance bins in meters, e.g., "0,20,40,60,80"')
    parser.add_argument('--max_vis_samples', type=int, default=50,
                        help='Maximum number of samples to visualize in BEV')
    args = parser.parse_args()

    # Parse distance bins
    if args.distance_bins:
        distance_bins = [float(x) for x in args.distance_bins.split(',')]
    else:
        distance_bins = DEFAULT_DISTANCE_BINS

    # parse configs
    cfgs = Config.fromfile(args.config)

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')
    # Ensure mmdet3d transforms are registered
    import mmdet3d.datasets.transforms

    # Copy mmdet3d transforms to mmengine registry for compatibility
    from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
    from mmdet3d.registry import TRANSFORMS as MMDET3D_TRANSFORMS
    for name, module in MMDET3D_TRANSFORMS.module_dict.items():
        if name not in MMENGINE_TRANSFORMS.module_dict:
            MMENGINE_TRANSFORMS.register_module(name=name, module=module)

    # Copy mmdet models to mmdet3d registry for compatibility (e.g., ResNet backbone)
    from mmdet.registry import MODELS as MMDET_MODELS
    from mmdet3d.registry import MODELS as MMDET3D_MODELS
    for name, module in MMDET_MODELS.module_dict.items():
        if name not in MMDET3D_MODELS.module_dict:
            MMDET3D_MODELS.register_module(name=name, module=module)

    # Copy mmdet task utils to mmdet3d registry for compatibility
    from mmdet.registry import TASK_UTILS as MMDET_TASK_UTILS
    from mmdet3d.registry import TASK_UTILS as MMDET3D_TASK_UTILS
    for name, module in MMDET_TASK_UTILS.module_dict.items():
        if name not in MMDET3D_TASK_UTILS.module_dict:
            MMDET3D_TASK_UTILS.register_module(name=name, module=module)
    # Also copy mmdet3d task utils to mmdet registry (for DETRHead which uses mmdet's registry)
    for name, module in MMDET3D_TASK_UTILS.module_dict.items():
        if name not in MMDET_TASK_UTILS.module_dict:
            MMDET_TASK_UTILS.register_module(name=name, module=module)

    # Suppress verbose logging from mmengine/mmcv
    import logging
    logging.getLogger('mmengine').setLevel(logging.WARNING)
    logging.getLogger('mmcv').setLevel(logging.WARNING)

    # you need GPUs
    assert torch.cuda.is_available()

    # determine local_rank and world_size
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(args.world_size)

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if local_rank == 0:
        utils.init_logging(None, cfgs.debug)
    else:
        logging.root.disabled = True

    logging.info('Using GPU: %s' % torch.cuda.get_device_name(local_rank))
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        logging.info('Initializing DDP with %d GPUs...' % world_size)
        dist.init_process_group('nccl', init_method='env://')

    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=False)  # Disable deterministic for CUDA compatibility
    cudnn.benchmark = True

    logging.info('Loading validation set from %s' % cfgs.data.val.data_root)
    val_dataset = build_dataset(cfgs.data.val)
    # Use num_workers=0 to avoid multiprocessing serialization issues with custom dataset
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=0,  # Disabled to avoid serialization issues
        num_gpus=world_size,
        dist=world_size > 1,
        shuffle=False,
        seed=0,
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.cuda()

    if world_size > 1:
        model = MMDistributedDataParallel(model, [local_rank], broadcast_buffers=False)
    else:
        model = MMDataParallel(model, [0])

    logging.info('Loading checkpoint from %s' % args.weights)
    checkpoint = load_checkpoint(
        model, args.weights, map_location='cuda', strict=False,
        logger=logging.Logger(__name__, logging.ERROR)
    )

    if 'version' in checkpoint:
        VERSION.name = checkpoint['version']

    if world_size > 1:
        results, timing_stats = multi_gpu_test(model, val_loader, gpu_collect=False)
    else:
        results, timing_stats = single_gpu_test(model, val_loader)

    if local_rank == 0:
        # Print timing statistics
        if timing_stats:
            logging.info('')
            logging.info('=' * 50)
            logging.info('Latency Metrics:')
            logging.info('=' * 50)
            logging.info('  Mean inference time: %.2f ms' % timing_stats['mean_inference_ms'])
            logging.info('  Std inference time:  %.2f ms' % timing_stats['std_inference_ms'])
            logging.info('  Min inference time:  %.2f ms' % timing_stats['min_inference_ms'])
            logging.info('  Max inference time:  %.2f ms' % timing_stats['max_inference_ms'])
            logging.info('  Median inference time: %.2f ms' % timing_stats['median_inference_ms'])
            logging.info('  FPS: %.2f' % timing_stats['fps'])
            logging.info('  Samples processed: %d' % timing_stats['num_samples'])
            logging.info('=' * 50)
            logging.info('')

        metrics = evaluate(val_dataset, results, -1)

        # Add timing stats to metrics
        if timing_stats:
            metrics['timing'] = timing_stats

        # Save evaluation outputs if output_dir is specified
        if args.output_dir:
            save_evaluation_outputs(
                args.output_dir,
                metrics,
                results,
                val_dataset,
                args,
                distance_bins
            )


if __name__ == '__main__':
    main()
