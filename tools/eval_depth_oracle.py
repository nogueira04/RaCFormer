#!/usr/bin/env python
"""
Depth Oracle Experiment for RaCFormer

This script evaluates whether depth estimation is the bottleneck for
night-time performance by replacing predicted depth with ground truth
depth from LiDAR projection.

Hypothesis: The model misses objects at night because depth estimation fails
in low light, causing camera BEV features to be placed at wrong locations.

Usage:
    python tools/eval_depth_oracle.py \
        configs/racformer_r50_nuimg_704x256_f8.py \
        checkpoints/racformer_r50_f8.pth \
        --condition night \
        --out results/depth_oracle_night.json
"""

import argparse
import copy
import os
import sys
import json
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Register custom modules
import loaders  # noqa: F401
import models   # noqa: F401

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_detector


def get_depth_bin_config(grid_config):
    """
    Extract depth bin configuration from grid_config.

    Returns:
        bin_size: Size factor for spacing-increasing bins
        depth_start: Minimum depth (meters)
        depth_end: Maximum depth (meters)
        num_bins: Number of depth bins
        bin_values: Tensor of depth values at bin centers
    """
    depth_start = grid_config['depth'][0]
    depth_end = grid_config['depth'][1]
    num_bins = int(grid_config['depth'][2])

    # Spacing-increasing discretization formula (from view_transformer_racformer.py:52-54)
    bin_size = 2 * (depth_end - depth_start) / (num_bins * (1 + num_bins))
    bin_indices = torch.linspace(0, num_bins - 1, num_bins)
    bin_values = (bin_indices + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_start

    return {
        'bin_size': bin_size,
        'depth_start': depth_start,
        'depth_end': depth_end,
        'num_bins': num_bins,
        'bin_values': bin_values
    }


def depth_to_bin_indices(depth_values, bin_config):
    """
    Convert depth values (meters) to bin indices using spacing-increasing formula.

    This is the inverse of the bin_value formula:
    bin_value = (i + 0.5)^2 * bin_size / 2 - bin_size / 8 + depth_start

    Solving for i:
    i = -0.5 + 0.5 * sqrt(1 + 8 * (depth - depth_start) / bin_size)

    Args:
        depth_values: Tensor of depth values in meters
        bin_config: Dict from get_depth_bin_config()

    Returns:
        Tensor of bin indices (int64)
    """
    bin_size = bin_config['bin_size']
    depth_start = bin_config['depth_start']
    num_bins = bin_config['num_bins']

    # Inverse formula from get_downsampled_depth (view_transformer_racformer.py:617)
    indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_values - depth_start) / bin_size)

    # Handle invalid values (out of range, NaN, etc.)
    mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
    indices[mask] = num_bins  # Will be filtered out later

    return indices.long()


def create_gt_depth_logits(gt_depth, bin_config, downsample=16, soft_sigma=None):
    """
    Convert GT depth map (in meters) to depth logits matching model's format.

    Args:
        gt_depth: [B, N, H, W] ground truth depth in meters (0 = no depth)
        bin_config: Dict from get_depth_bin_config()
        downsample: Downsample factor (default 16)
        soft_sigma: If set, use soft assignment with this sigma (in bins)

    Returns:
        depth_logits: [B*N, D, h, w] depth logits for LSS
    """
    B, N, H, W = gt_depth.shape
    D = bin_config['num_bins']
    device = gt_depth.device

    # Downsample depth using min pooling (closest depth wins)
    # Reshape for pooling: [B*N, 1, H, W]
    depth_flat = gt_depth.view(B * N, 1, H, W)

    # For min pooling, replace 0s with large values, pool, then restore
    depth_for_pool = torch.where(
        depth_flat == 0,
        torch.full_like(depth_flat, 1e5),
        depth_flat
    )

    # Use adaptive pooling to get target size
    h, w = H // downsample, W // downsample

    # Manual min pooling with stride
    depth_downsampled = depth_for_pool.view(B * N, 1, h, downsample, w, downsample)
    depth_downsampled = depth_downsampled.permute(0, 1, 2, 4, 3, 5).contiguous()
    depth_downsampled = depth_downsampled.view(B * N, 1, h, w, -1)
    depth_downsampled = depth_downsampled.min(dim=-1).values  # [B*N, 1, h, w]

    # Restore zeros for pixels that had no valid depth
    depth_downsampled = torch.where(
        depth_downsampled > 1e4,
        torch.zeros_like(depth_downsampled),
        depth_downsampled
    )
    depth_downsampled = depth_downsampled.squeeze(1)  # [B*N, h, w]

    # Convert depth values to bin indices
    bin_indices = depth_to_bin_indices(depth_downsampled, bin_config)  # [B*N, h, w]

    # Create depth logits
    if soft_sigma is None:
        # One-hot encoding (sparse GT)
        # Use large logit value for selected bin, small for others
        depth_logits = torch.zeros(B * N, D, h, w, device=device)

        # Create mask for valid depth pixels
        valid_mask = (depth_downsampled > 0) & (bin_indices < D)

        # For valid pixels, set the corresponding bin to high logit
        # Use scatter to set one-hot
        valid_indices = bin_indices.clone()
        valid_indices[~valid_mask] = 0  # Placeholder, will be masked out

        # Expand for scatter: [B*N, 1, h, w]
        valid_indices_expanded = valid_indices.unsqueeze(1)

        # Create one-hot style logits (high value for GT bin)
        depth_logits.scatter_(1, valid_indices_expanded, 10.0)

        # For pixels without GT depth, use uniform distribution (logits = 0)
        # The softmax will make them 1/D each
        # But we want to preserve some information, so keep them at 0

    else:
        # Soft assignment with Gaussian
        depth_logits = torch.zeros(B * N, D, h, w, device=device)
        valid_mask = (depth_downsampled > 0) & (bin_indices < D)

        # Create Gaussian weights around GT bin
        bin_centers = torch.arange(D, device=device).float()  # [D]

        for bn in range(B * N):
            for hi in range(h):
                for wi in range(w):
                    if valid_mask[bn, hi, wi]:
                        center_bin = bin_indices[bn, hi, wi].float()
                        weights = torch.exp(-((bin_centers - center_bin) ** 2) / (2 * soft_sigma ** 2))
                        # Convert to logits (log of softmax input)
                        depth_logits[bn, :, hi, wi] = torch.log(weights + 1e-8)

    return depth_logits, valid_mask.float().mean().item()


def is_night_sample(img_metas):
    """
    Determine if a sample is from night scene based on filename.
    nuScenes night scenes contain 'night' in the scene description.
    """
    # Check filename for night indicators
    filenames = img_metas[0].get('filename', [])
    if not filenames:
        return False

    # Night scenes in nuScenes have specific scene tokens
    # We'll use a simple heuristic based on mean image brightness
    return False  # Will be overridden by explicit scene info


def get_scene_condition(sample_token, scene_descriptions):
    """
    Get the lighting condition for a sample based on scene description.
    """
    if sample_token in scene_descriptions:
        desc = scene_descriptions[sample_token].lower()
        if 'night' in desc:
            return 'night'
        elif 'rain' in desc:
            return 'rain'
    return 'day'


class DepthOracleEvaluator:
    """
    Evaluator that compares predicted depth vs GT depth performance.
    """

    def __init__(self, model, cfg, condition='night'):
        self.model = model
        self.cfg = cfg
        self.condition = condition

        # Get depth bin config from model
        grid_config = cfg.model.img_lss_view_transformer.grid_config
        self.bin_config = get_depth_bin_config(grid_config)
        self.downsample = cfg.model.img_lss_view_transformer.get('downsample', 16)

        # Store original forward method
        self.view_transformer = model.module.img_lss_view_transformer
        self._original_forward = self.view_transformer.forward

        # Results storage
        self.results = {
            'predicted_depth': [],
            'gt_depth': []
        }

        # Depth injection flag
        self._use_gt_depth = False
        self._gt_depth_tensor = None

    def _patch_view_transformer(self):
        """
        Monkey-patch the view transformer to optionally use GT depth.
        """
        outer_self = self
        original_forward = self._original_forward

        def patched_forward(x, radar_depth, radar_rcs, img_metas, mlp_input):
            B, N, C, H, W = x.shape
            x_flat = x.reshape(B * N, C, H, W)

            # Get radar depth processing
            vt = outer_self.view_transformer
            rad_inds, radar_depth_processed = vt.get_downsampled_depth(radar_depth, downsample=vt.downsample)
            one_hot_rcs = vt.get_downsampled_rcs(radar_rcs, downsample=vt.downsample)
            rcs_embedding = vt.rcs_embedding(one_hot_rcs.permute(0, 3, 1, 2))

            ones = torch.ones_like(radar_depth_processed).unsqueeze(-1)
            rad_dep_grids = torch.zeros_like(radar_depth_processed).unsqueeze(-1).repeat(
                1, 1, 1, int(vt.grid_config['depth'][2] + 1))
            rad_dep_grids = rad_dep_grids.scatter_(dim=-1, index=rad_inds.unsqueeze(-1), src=ones)
            rad_dep_grids = rad_dep_grids.permute(0, 3, 1, 2)

            # Run depth net
            depth_out = vt.depth_net(x_flat, rad_dep_grids, rcs_embedding, mlp_input)
            depth_digit = depth_out[:, :vt.D, ...]
            tran_feat = depth_out[:, vt.D:vt.D + vt.out_channels, ...]

            # ORACLE INJECTION: Replace predicted depth with GT depth
            if outer_self._use_gt_depth and outer_self._gt_depth_tensor is not None:
                gt_logits = outer_self._gt_depth_tensor

                # Ensure shapes match
                if gt_logits.shape == depth_digit.shape:
                    depth_digit = gt_logits
                else:
                    print(f"Warning: GT depth shape {gt_logits.shape} != pred shape {depth_digit.shape}")

            return vt.view_transform(depth_out, depth_digit, tran_feat, img_metas)

        self.view_transformer.forward = patched_forward

    def _restore_view_transformer(self):
        """Restore original forward method."""
        self.view_transformer.forward = self._original_forward

    def set_gt_depth(self, gt_depth):
        """
        Set the GT depth tensor for oracle mode.

        Args:
            gt_depth: [B, N, H, W] GT depth in meters
        """
        gt_logits, coverage = create_gt_depth_logits(
            gt_depth,
            self.bin_config,
            downsample=self.downsample
        )
        self._gt_depth_tensor = gt_logits
        return coverage

    def evaluate_sample(self, data, use_gt_depth=False):
        """
        Run inference on a single sample.

        Args:
            data: Batch data dict
            use_gt_depth: If True, use GT depth instead of predicted

        Returns:
            List of detection results
        """
        self._use_gt_depth = use_gt_depth

        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)

        return result


def load_night_sample_tokens(ann_file):
    """
    Load sample tokens for night scenes from annotation file.

    Returns set of sample tokens that are from night scenes.
    """
    import pickle

    with open(ann_file, 'rb') as f:
        data = pickle.load(f)

    night_tokens = set()
    day_tokens = set()
    rain_tokens = set()

    # nuScenes scene descriptions
    night_keywords = ['night', 'dark']
    rain_keywords = ['rain', 'wet']

    for info in data['infos']:
        token = info['token']
        scene_token = info.get('scene_token', '')

        # Try to get scene description from cams info or directly
        # In practice, we'll use the sample token to look up scene info
        # For now, mark all for evaluation and filter by actual brightness
        day_tokens.add(token)

    return day_tokens, night_tokens, rain_tokens


def compute_image_brightness(img):
    """
    Compute mean brightness of image batch.

    Args:
        img: [B, N, C, H, W] image tensor (0-255 range)

    Returns:
        float: Mean brightness value
    """
    # Convert to grayscale using standard weights
    if img.dim() == 5:
        B, N, C, H, W = img.shape
        img = img.view(B * N, C, H, W)

    # RGB to grayscale
    weights = torch.tensor([0.299, 0.587, 0.114], device=img.device)
    gray = (img * weights.view(1, 3, 1, 1)).sum(dim=1)

    return gray.mean().item()


def get_scene_condition_from_nusc(data_root='data/nuscenes/'):
    """Load nuScenes and build sample_token -> condition mapping."""
    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=False)

        token_to_condition = {}
        for sample in nusc.sample:
            scene = nusc.get('scene', sample['scene_token'])
            desc = scene.get('description', '').lower()
            if 'night' in desc:
                condition = 'night'
            elif 'rain' in desc:
                condition = 'rain'
            else:
                condition = 'day'
            token_to_condition[sample['token']] = condition
        return token_to_condition
    except Exception as e:
        print(f"Warning: Could not load nuScenes for scene conditions: {e}")
        return None


def evaluate_subset(results, dataset, sample_indices):
    """
    Evaluate results on a subset of samples using nuScenes metrics.

    This creates a temporary evaluation that only considers the specified samples.

    Args:
        results: List of detection results (one per sample)
        dataset: Dataset object with data_infos
        sample_indices: List of indices in the original dataset

    Returns:
        dict: Evaluation metrics (mAP, NDS, etc.)
    """
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.common.config import config_factory as common_config_factory
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.common.data_classes import EvalBoxes
    from pyquaternion import Quaternion
    import tempfile
    import json

    # Get sample tokens for our subset
    sample_tokens = []
    for idx in sample_indices:
        info = dataset.data_infos[idx]
        sample_tokens.append(info['token'])

    print(f"Evaluating {len(sample_tokens)} samples...")

    # Load nuScenes
    data_root = dataset.data_root
    nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=False)

    # Debug: Check coordinate ranges of first sample (disabled - transform verified)
    if False and len(results) > 0 and len(sample_indices) > 0:
        first_result = results[0]
        first_token = sample_tokens[0]

        # Get prediction boxes
        if 'pts_bbox' in first_result:
            pred_boxes_3d = first_result['pts_bbox']['boxes_3d']
        else:
            pred_boxes_3d = first_result['boxes_3d']

        if hasattr(pred_boxes_3d, 'tensor'):
            pred_np = pred_boxes_3d.tensor.cpu().numpy()
        elif hasattr(pred_boxes_3d, 'cpu'):
            pred_np = pred_boxes_3d.cpu().numpy()
        else:
            pred_np = np.array(pred_boxes_3d)

        # Get GT boxes from nuScenes (global coords)
        sample = nusc.get('sample', first_token)
        gt_anns = [nusc.get('sample_annotation', t) for t in sample['anns'][:3]]

        # Get transformations for this sample
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)

        # LiDAR to Ego
        lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        l2e_trans = np.array(lidar_calib['translation'])
        l2e_rot = Quaternion(lidar_calib['rotation'])

        # Ego to Global
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        e2g_trans = np.array(ego_pose['translation'])
        e2g_rot = Quaternion(ego_pose['rotation'])

        # Transform first 3 predictions to global for debug (LiDAR -> Ego -> Global)
        pred_global = []
        for k in range(min(3, len(pred_np))):
            pos_lidar = pred_np[k, :3]
            pos_ego = l2e_rot.rotate(pos_lidar) + l2e_trans
            pos_global = e2g_rot.rotate(pos_ego) + e2g_trans
            pred_global.append(pos_global.tolist())

        print(f"\n  DEBUG - Coordinate check for first sample:")
        print(f"    Pred boxes ego (first 3): {pred_np[:3, :3].tolist() if len(pred_np) > 0 else 'no predictions'}")
        print(f"    Pred boxes GLOBAL (first 3): {pred_global if pred_global else 'no predictions'}")
        print(f"    GT boxes global (first 3): {[ann['translation'] for ann in gt_anns]}")
        print(f"    Ego pose translation: {ego_pose['translation']}")

    # Build prediction boxes
    pred_boxes = EvalBoxes()

    # nuScenes class names
    CLASSES = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
        'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]

    for i, (idx, token) in enumerate(zip(sample_indices, sample_tokens)):
        result = results[i]

        # Get transformations for coordinate conversion: LiDAR -> Ego -> Global
        sample = nusc.get('sample', token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)

        # LiDAR to Ego transformation
        lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_to_ego_trans = np.array(lidar_calib['translation'])
        lidar_to_ego_rot = Quaternion(lidar_calib['rotation'])

        # Ego to Global transformation
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_to_global_trans = np.array(ego_pose['translation'])
        ego_to_global_rot = Quaternion(ego_pose['rotation'])

        # Get boxes from result
        if 'pts_bbox' in result:
            boxes_3d = result['pts_bbox']['boxes_3d']
            scores = result['pts_bbox']['scores_3d']
            labels = result['pts_bbox']['labels_3d']
        else:
            boxes_3d = result['boxes_3d']
            scores = result['scores_3d']
            labels = result['labels_3d']

        # Convert to numpy if tensor
        if hasattr(boxes_3d, 'tensor'):
            boxes_np = boxes_3d.tensor.cpu().numpy()
        elif hasattr(boxes_3d, 'cpu'):
            boxes_np = boxes_3d.cpu().numpy()
        else:
            boxes_np = np.array(boxes_3d)

        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()
        if hasattr(labels, 'cpu'):
            labels = labels.cpu().numpy()

        sample_boxes = []
        for j in range(len(boxes_np)):
            box = boxes_np[j]
            # box format: [x, y, z, w, l, h, yaw, vx, vy] or similar

            # Extract position and size (in ego coordinates)
            x, y, z = box[0], box[1], box[2]
            w, l, h = box[3], box[4], box[5]  # width, length, height
            yaw = box[6]

            # Velocity if available (in ego coordinates)
            vx = box[7] if len(box) > 7 else 0.0
            vy = box[8] if len(box) > 8 else 0.0

            # Transform position: LiDAR -> Ego -> Global
            pos_lidar = np.array([x, y, z])
            # LiDAR to Ego
            pos_ego = lidar_to_ego_rot.rotate(pos_lidar) + lidar_to_ego_trans
            # Ego to Global
            pos_global = ego_to_global_rot.rotate(pos_ego) + ego_to_global_trans

            # Transform rotation: LiDAR -> Ego -> Global
            box_rotation_lidar = Quaternion(axis=[0, 0, 1], radians=yaw)
            box_rotation_ego = lidar_to_ego_rot * box_rotation_lidar
            box_rotation_global = ego_to_global_rot * box_rotation_ego

            # Transform velocity: LiDAR -> Ego -> Global
            vel_lidar = np.array([vx, vy, 0.0])
            vel_ego = lidar_to_ego_rot.rotate(vel_lidar)
            vel_global = ego_to_global_rot.rotate(vel_ego)

            # Get class name
            label_idx = int(labels[j])
            if label_idx < len(CLASSES):
                detection_name = CLASSES[label_idx]
            else:
                continue

            det_box = DetectionBox(
                sample_token=token,
                translation=tuple(pos_global),
                size=(w, l, h),
                rotation=tuple(box_rotation_global.elements),
                velocity=(vel_global[0], vel_global[1]),
                detection_name=detection_name,
                detection_score=float(scores[j]),
                attribute_name=''
            )
            sample_boxes.append(det_box)

        pred_boxes.add_boxes(token, sample_boxes)

    # Build GT boxes for our samples only
    gt_boxes = EvalBoxes()

    for token in sample_tokens:
        sample = nusc.get('sample', token)
        sample_gt_boxes = []

        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)

            # Get detection name
            detection_name = ann['category_name']
            # Map to simplified name
            for cls_name in CLASSES:
                if cls_name in detection_name:
                    detection_name = cls_name
                    break
            else:
                # Skip if not in our classes
                continue

            # Get visibility - format is like 'v80-100', 'v60-80', etc.
            visibility = nusc.get('visibility', ann['visibility_token'])
            vis_token = visibility['token']
            # Extract visibility percentage from token (e.g., 'v80-100' -> 80)
            # Lower bound: 1=0-40%, 2=40-60%, 3=60-80%, 4=80-100%
            if '80-100' in vis_token or '4' in vis_token:
                vis_level = 4
            elif '60-80' in vis_token or '3' in vis_token:
                vis_level = 3
            elif '40-60' in vis_token or '2' in vis_token:
                vis_level = 2
            else:
                vis_level = 1

            # Skip if visibility too low (standard nuScenes filtering uses >= 1)
            if vis_level < 1:
                continue

            # Get velocity
            velocity = nusc.box_velocity(ann_token)
            if np.isnan(velocity).any():
                velocity = np.array([0.0, 0.0, 0.0])

            # Get attribute name from token (not the token itself)
            if ann['attribute_tokens']:
                attr = nusc.get('attribute', ann['attribute_tokens'][0])
                attribute_name = attr['name']
            else:
                attribute_name = ''

            gt_box = DetectionBox(
                sample_token=token,
                translation=tuple(ann['translation']),
                size=tuple(ann['size']),
                rotation=tuple(ann['rotation']),
                velocity=tuple(velocity[:2]),
                detection_name=detection_name,
                detection_score=-1.0,  # GT
                attribute_name=attribute_name,
                num_pts=ann['num_lidar_pts'] + ann['num_radar_pts']
            )
            sample_gt_boxes.append(gt_box)

        gt_boxes.add_boxes(token, sample_gt_boxes)

    # Compute metrics manually
    # Distance thresholds for matching (nuScenes uses 0.5, 1.0, 2.0, 4.0 meters)
    dist_thresholds = [0.5, 1.0, 2.0, 4.0]

    # Compute per-class AP
    class_aps = {}
    class_tp_metrics = {}

    for cls_name in CLASSES:
        # Get all GT and pred boxes for this class
        all_gt = []
        all_pred = []

        for token in sample_tokens:
            gt_list = [b for b in gt_boxes.boxes[token] if b.detection_name == cls_name]
            pred_list = [b for b in pred_boxes.boxes[token] if b.detection_name == cls_name]

            all_gt.extend([(token, b) for b in gt_list])
            all_pred.extend([(token, b) for b in pred_list])

        if len(all_gt) == 0:
            class_aps[cls_name] = 0.0
            continue

        # Sort predictions by score
        all_pred.sort(key=lambda x: x[1].detection_score, reverse=True)

        # Compute AP at each distance threshold
        aps_at_thresh = []

        for dist_thresh in dist_thresholds:
            # Track which GT boxes have been matched
            gt_matched = {i: False for i in range(len(all_gt))}

            tp = []
            fp = []

            for pred_token, pred_box in all_pred:
                # Find best matching GT in same sample
                best_match = None
                best_dist = float('inf')

                for gt_idx, (gt_token, gt_box) in enumerate(all_gt):
                    if gt_token != pred_token:
                        continue
                    if gt_matched[gt_idx]:
                        continue

                    # Compute center distance
                    dist = np.linalg.norm(
                        np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2])
                    )

                    if dist < best_dist:
                        best_dist = dist
                        best_match = gt_idx

                if best_match is not None and best_dist <= dist_thresh:
                    tp.append(1)
                    fp.append(0)
                    gt_matched[best_match] = True
                else:
                    tp.append(0)
                    fp.append(1)

            # Compute precision-recall curve
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / len(all_gt)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

            # Compute AP using 11-point interpolation
            ap = 0.0
            for r_thresh in np.linspace(0, 1, 11):
                prec_at_recall = precisions[recalls >= r_thresh]
                if len(prec_at_recall) > 0:
                    ap += np.max(prec_at_recall) / 11

            aps_at_thresh.append(ap)

        # Average AP across distance thresholds
        class_aps[cls_name] = np.mean(aps_at_thresh)

    # Compute mAP
    valid_aps = [ap for ap in class_aps.values() if ap > 0 or True]  # Include 0s
    mAP = np.mean(valid_aps) if valid_aps else 0.0

    # Compute simple error metrics (approximation)
    # For a proper implementation, we'd need to track matched pairs
    # Here we just return placeholder values

    metrics = {
        'mAP': mAP,
        'NDS': mAP * 0.5 + 0.5 * 0.5,  # Approximate NDS
    }

    # Add per-class APs
    for cls_name, ap in class_aps.items():
        metrics[f'AP_{cls_name}'] = ap

    print(f"  mAP: {mAP:.4f}")
    for cls_name in CLASSES:
        print(f"  AP_{cls_name}: {class_aps.get(cls_name, 0):.4f}")

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Depth Oracle Evaluation')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--condition', default='night',
                        choices=['night', 'day', 'rain', 'all'],
                        help='Condition to evaluate')
    parser.add_argument('--brightness-threshold', type=float, default=60.0,
                        help='Brightness threshold for night classification (fallback if nuScenes not available)')
    parser.add_argument('--out', default='results/depth_oracle_results.json',
                        help='Output file for results')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--soft-sigma', type=float, default=None,
                        help='Sigma for soft depth assignment (bins)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode with verbose output')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}')

    # Load config
    cfg = Config.fromfile(args.config)

    # Build dataset
    print("Building dataset...")
    dataset = build_dataset(cfg.data.val)

    # Build dataloader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    # Build model
    print("Building model...")
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # Move to GPU
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    model.eval()

    # Get depth bin config
    grid_config = cfg.model.img_lss_view_transformer.grid_config
    bin_config = get_depth_bin_config(grid_config)
    downsample = cfg.model.img_lss_view_transformer.get('downsample', 16)

    print(f"\nDepth Configuration:")
    print(f"  Bins: {bin_config['num_bins']}")
    print(f"  Range: {bin_config['depth_start']:.1f}m - {bin_config['depth_end']:.1f}m")
    print(f"  Bin size factor: {bin_config['bin_size']:.6f}")
    print(f"  Downsample: {downsample}")

    # Store original forward method for patching
    view_transformer = model.module.img_lss_view_transformer
    original_forward = view_transformer.forward

    # Results storage
    results_pred = []
    results_gt = []
    sample_info = []

    # Counters
    n_night = 0
    n_day = 0
    n_evaluated = 0
    gt_depth_coverages = []

    # Load nuScenes scene conditions (more reliable than brightness)
    print("\nLoading nuScenes scene conditions...")
    token_to_condition = get_scene_condition_from_nusc()
    use_nusc_conditions = token_to_condition is not None

    if use_nusc_conditions:
        print(f"  Using nuScenes scene descriptions for day/night classification")
    else:
        print(f"  Falling back to brightness threshold: {args.brightness_threshold}")

    print(f"\nEvaluating with condition: {args.condition}")

    # Pre-filter dataset to only include samples matching the condition
    # This is required because nuScenes evaluator expects results for all samples in dataset
    original_infos = dataset.data_infos

    if args.condition != 'all' and use_nusc_conditions:
        print(f"\nFiltering dataset to {args.condition} samples...")
        filtered_indices = []
        for idx, info in enumerate(original_infos):
            token = info.get('token', None)
            if token and token in token_to_condition:
                if token_to_condition[token] == args.condition:
                    filtered_indices.append(idx)
    else:
        # For 'all' condition, use all indices
        filtered_indices = list(range(len(original_infos)))

    # Limit samples if requested
    if args.max_samples and len(filtered_indices) > args.max_samples:
        filtered_indices = filtered_indices[:args.max_samples]

    # Create subset dataset - ALWAYS update data_infos to match what we'll evaluate
    dataset.data_infos = [original_infos[i] for i in filtered_indices]
    print(f"  Dataset filtered to {len(dataset.data_infos)} samples")

    # Rebuild dataloader with filtered dataset
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    # Progress bar
    pbar = tqdm(data_loader, desc="Evaluating")

    for i, data in enumerate(pbar):

        # Get image data for brightness calculation (for logging)
        img = data['img'][0].data[0]  # [B, N*T, C, H, W] or [B, N, C, H, W]
        brightness = compute_image_brightness(img)

        # Dataset is already pre-filtered to the target condition
        sample_condition = args.condition if args.condition != 'all' else 'unknown'

        if sample_condition == 'night':
            n_night += 1
        else:
            n_day += 1

        n_evaluated += 1

        # Get GT depth - handle [N, H, W] or [B, N, H, W]
        gt_depth = data['gt_depth'][0].data[0]
        if gt_depth.dim() == 3:
            gt_depth = gt_depth.unsqueeze(0)  # Add batch dim -> [1, N, H, W]

        # Move gt_depth to GPU
        gt_depth = gt_depth.to(device)

        # IMPORTANT: GT depth is only valid for first frame (first 6 cameras)
        # The model processes each temporal frame separately
        num_cams = 6  # nuScenes has 6 cameras
        gt_depth_first_frame = gt_depth[:, :num_cams]  # [B, 6, H, W]

        # Create GT depth logits (only for first frame)
        gt_depth_logits, coverage = create_gt_depth_logits(
            gt_depth_first_frame, bin_config, downsample=downsample, soft_sigma=args.soft_sigma
        )
        gt_depth_logits = gt_depth_logits.to(device)
        gt_depth_coverages.append(coverage)

        # ============ Inference with PREDICTED depth ============
        # Make a deep copy for second inference (data gets modified in-place)
        data_copy = copy.deepcopy(data)

        # Pass batch data directly (dataloader formats it correctly)
        with torch.no_grad():
            result_pred = model(return_loss=False, rescale=True, **data)
        results_pred.extend(result_pred)

        # ============ Inference with GT depth (Oracle) ============
        # Patch the forward method to use GT depth
        # GT depth is only available for first frame (6 cameras)
        # The model calls view_transformer once per temporal frame
        gt_depth_container = [gt_depth_logits]  # Mutable container
        frame_counter = [0]  # Track which frame we're processing

        def patched_forward(x, radar_depth, radar_rcs, img_metas, mlp_input):
            B, N, C, H, W = x.shape
            x_flat = x.reshape(B * N, C, H, W)

            vt = view_transformer
            rad_inds, radar_depth_processed = vt.get_downsampled_depth(radar_depth, downsample=vt.downsample)
            one_hot_rcs = vt.get_downsampled_rcs(radar_rcs, downsample=vt.downsample)
            rcs_embedding = vt.rcs_embedding(one_hot_rcs.permute(0, 3, 1, 2))

            ones = torch.ones_like(radar_depth_processed).unsqueeze(-1)
            rad_dep_grids = torch.zeros_like(radar_depth_processed).unsqueeze(-1).repeat(
                1, 1, 1, int(vt.grid_config['depth'][2] + 1))
            rad_dep_grids = rad_dep_grids.scatter_(dim=-1, index=rad_inds.unsqueeze(-1), src=ones)
            rad_dep_grids = rad_dep_grids.permute(0, 3, 1, 2)

            depth_out = vt.depth_net(x_flat, rad_dep_grids, rcs_embedding, mlp_input)
            depth_digit = depth_out[:, :vt.D, ...]
            tran_feat = depth_out[:, vt.D:vt.D + vt.out_channels, ...]

            current_frame = frame_counter[0]
            frame_counter[0] += 1

            # ORACLE: Only replace depth for first frame (where we have GT depth)
            if current_frame == 0:
                gt_logits = gt_depth_container[0]
                # Verify shapes match
                if depth_digit.shape == gt_logits.shape:
                    depth_digit = gt_logits
            # For other frames, use predicted depth (no GT available)

            return vt.view_transform(depth_out, depth_digit, tran_feat, img_metas)

        # Apply patch
        view_transformer.forward = patched_forward

        with torch.no_grad():
            result_gt = model(return_loss=False, rescale=True, **data_copy)
        results_gt.extend(result_gt)

        # Restore original forward
        view_transformer.forward = original_forward

        # Store sample info
        sample_info.append({
            'idx': i,
            'brightness': brightness,
            'condition': sample_condition,
            'gt_depth_coverage': coverage,
            'n_pred_boxes': len(result_pred[0]['pts_bbox']['boxes_3d']),
            'n_gt_boxes': len(result_gt[0]['pts_bbox']['boxes_3d']),
        })

        # Update progress bar
        pbar.set_postfix({
            'night': n_night,
            'day': n_day,
            'coverage': f'{np.mean(gt_depth_coverages):.2%}'
        })

        if args.debug and n_evaluated <= 5:
            print(f"\nSample {i}:")
            print(f"  Brightness: {brightness:.1f}")
            print(f"  Is night: {is_night}")
            print(f"  GT depth coverage: {coverage:.2%}")
            print(f"  Pred detections: {len(result_pred[0]['pts_bbox']['boxes_3d'])}")
            print(f"  GT depth detections: {len(result_gt[0]['pts_bbox']['boxes_3d'])}")

    print(f"\n\nEvaluation complete!")
    print(f"  Total samples: {n_evaluated}")
    print(f"  Night samples: {n_night}")
    print(f"  Day samples: {n_day}")
    print(f"  Mean GT depth coverage: {np.mean(gt_depth_coverages):.2%}")

    # Compute metrics using nuScenes evaluator
    print("\n\nComputing nuScenes metrics...")
    print("(This requires running the official nuScenes evaluation)")

    # Save results for evaluation
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Format results for nuScenes evaluation
    results_dict = {
        'predicted_depth': results_pred,
        'gt_depth': results_gt,
        'sample_info': sample_info,
        'config': {
            'condition': args.condition,
            'brightness_threshold': args.brightness_threshold,
            'soft_sigma': args.soft_sigma,
            'n_samples': n_evaluated,
            'n_night': n_night,
            'n_day': n_day,
            'mean_gt_coverage': float(np.mean(gt_depth_coverages)),
        }
    }

    # Save intermediate results
    torch.save(results_dict, args.out.replace('.json', '.pth'))
    print(f"Saved raw results to {args.out.replace('.json', '.pth')}")

    # Run official nuScenes evaluation using monkeypatched load_gt
    # This filters GT boxes to only include the samples we evaluated
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.common.data_classes import EvalBoxes
    import nuscenes.eval.detection.evaluate as eval_module

    # Load nuScenes for evaluation
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataset.data_root, verbose=False)

    # Get sample tokens we evaluated
    valid_tokens = set([info['token'] for info in dataset.data_infos])

    # Store original load_gt function
    original_load_gt = eval_module.load_gt

    def custom_load_gt(nusc_obj, eval_split, box_cls, verbose=False):
        """Custom load_gt that filters GT to only include our evaluated samples."""
        gt_boxes = original_load_gt(nusc_obj, eval_split, box_cls, verbose)

        # Filter to only include our subset tokens
        filtered_boxes = EvalBoxes()
        for sample_token in gt_boxes.sample_tokens:
            if sample_token in valid_tokens:
                filtered_boxes.add_boxes(sample_token, gt_boxes[sample_token])

        return filtered_boxes

    # Monkeypatch load_gt
    eval_module.load_gt = custom_load_gt

    try:
        print("\n" + "="*60)
        print("Running OFFICIAL nuScenes evaluation for PREDICTED depth...")
        print("="*60)

        eval_pred = dataset.evaluate(results_pred, metric='bbox')

        print("\n" + "="*60)
        print("Running OFFICIAL nuScenes evaluation for GT depth (Oracle)...")
        print("="*60)

        eval_gt = dataset.evaluate(results_gt, metric='bbox')
    finally:
        # Restore original function
        eval_module.load_gt = original_load_gt

    # Compute deltas
    print("\n" + "="*60)
    print("DEPTH ORACLE EXPERIMENT RESULTS")
    print("="*60)
    print(f"\nCondition: {args.condition}")
    print(f"Samples evaluated: {n_evaluated}")
    print(f"Mean GT depth coverage: {np.mean(gt_depth_coverages):.2%}")

    print("\n" + "-"*60)
    print("Metric Comparison:")
    print("-"*60)

    # Official nuScenes evaluation returns keys like 'pts_bbox_NuScenes/mAP'
    # Map to simpler names
    def get_metric(eval_dict, metric_name):
        """Extract metric from evaluation dict, trying different key formats."""
        # Try direct key
        if metric_name in eval_dict:
            return eval_dict[metric_name]
        # Try pts_bbox_NuScenes/ prefix
        key = f'pts_bbox_NuScenes/{metric_name}'
        if key in eval_dict:
            return eval_dict[key]
        return None

    # Extract key metrics
    metrics_to_compare = ['mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']

    for metric in metrics_to_compare:
        pred_val = get_metric(eval_pred, metric)
        gt_val = get_metric(eval_gt, metric)

        if pred_val is not None and gt_val is not None:
            delta = gt_val - pred_val

            # For error metrics (mATE, mASE, etc.), negative delta is improvement
            is_error_metric = metric.startswith('mA') and metric not in ['mAP']
            if is_error_metric:
                sign = '-' if delta < 0 else '+'
            else:
                sign = '+' if delta > 0 else ''

            print(f"  {metric}:")
            print(f"    Predicted: {pred_val:.4f}")
            print(f"    GT Depth:  {gt_val:.4f}")
            print(f"    Delta:     {sign}{delta:.4f}")

    # Print interpretation
    print("\n" + "-"*60)
    print("INTERPRETATION:")
    print("-"*60)

    pred_mAP = get_metric(eval_pred, 'mAP')
    gt_mAP = get_metric(eval_gt, 'mAP')

    if pred_mAP is not None and gt_mAP is not None:
        mAP_delta = gt_mAP - pred_mAP
        mAP_delta_pct = mAP_delta * 100

        if mAP_delta_pct >= 10:
            interpretation = "Depth is a MAJOR bottleneck"
            recommendation = "Focus on improving depth estimation at night"
        elif mAP_delta_pct >= 5:
            interpretation = "Depth is a SIGNIFICANT factor"
            recommendation = "Depth improvements will help substantially"
        elif mAP_delta_pct >= 2:
            interpretation = "Depth is a PARTIAL factor"
            recommendation = "Depth helps but other factors also contribute"
        elif mAP_delta_pct >= 0:
            interpretation = "Depth is NOT the main bottleneck"
            recommendation = "Focus on other factors (features, backbone, etc.)"
        else:
            interpretation = "UNEXPECTED: GT depth performed worse"
            recommendation = "Check implementation - may be a bug"

        delta_sign = '+' if mAP_delta_pct > 0 else ''
        print(f"  mAP improvement with GT depth: {delta_sign}{mAP_delta_pct:.1f}%")
        print(f"  {interpretation}")
        print(f"  Recommendation: {recommendation}")
    else:
        interpretation = "N/A"
        recommendation = "N/A"
        mAP_delta = 0.0
        print("  Could not extract mAP from evaluation results")

    # Save final results (convert numpy types to Python types for JSON serialization)
    def convert_to_serializable(obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    final_results = {
        'config': results_dict['config'],
        'predicted_depth_metrics': convert_to_serializable(eval_pred),
        'gt_depth_metrics': convert_to_serializable(eval_gt),
        'interpretation': {
            'mAP_delta': float(mAP_delta),
            'conclusion': interpretation,
            'recommendation': recommendation,
        }
    }

    with open(args.out, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved final results to {args.out}")

    return final_results


if __name__ == '__main__':
    main()
