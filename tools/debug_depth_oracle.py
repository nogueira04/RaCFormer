#!/usr/bin/env python
"""
Debug script for Depth Oracle Experiment

This script validates the depth format and conversion before running
the full experiment. Run this first to ensure everything is correct.

Usage:
    python tools/debug_depth_oracle.py \
        configs/racformer_r50_nuimg_704x256_f8.py \
        checkpoints/racformer_r50_f8.pth \
        --n-samples 5
"""

import argparse
import copy
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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
    """Extract depth bin configuration."""
    depth_start = grid_config['depth'][0]
    depth_end = grid_config['depth'][1]
    num_bins = int(grid_config['depth'][2])

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
    """Convert depth values to bin indices."""
    bin_size = bin_config['bin_size']
    depth_start = bin_config['depth_start']
    num_bins = bin_config['num_bins']

    indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_values - depth_start) / bin_size)
    mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
    indices[mask] = num_bins

    return indices.long()


def bin_indices_to_depth(bin_indices, bin_config):
    """Convert bin indices back to depth values (for verification)."""
    return bin_config['bin_values'][bin_indices.clamp(0, bin_config['num_bins'] - 1)]


def parse_args():
    parser = argparse.ArgumentParser(description='Debug Depth Oracle')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--n-samples', type=int, default=5, help='Number of samples')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--save-vis', action='store_true', help='Save visualizations')
    return parser.parse_args()


def main():
    args = parse_args()

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}')

    # Load config
    print("Loading config...")
    cfg = Config.fromfile(args.config)

    # Get depth bin config
    grid_config = cfg.model.img_lss_view_transformer.grid_config
    bin_config = get_depth_bin_config(grid_config)
    downsample = cfg.model.img_lss_view_transformer.get('downsample', 16)

    print("\n" + "="*60)
    print("DEPTH BIN CONFIGURATION")
    print("="*60)
    print(f"Number of bins (D): {bin_config['num_bins']}")
    print(f"Depth range: {bin_config['depth_start']:.1f}m - {bin_config['depth_end']:.1f}m")
    print(f"Bin size factor: {bin_config['bin_size']:.6f}")
    print(f"Downsample factor: {downsample}")
    print(f"\nFirst 10 bin values (meters):")
    for i in range(min(10, bin_config['num_bins'])):
        print(f"  Bin {i}: {bin_config['bin_values'][i]:.3f}m")
    print(f"\nLast 10 bin values (meters):")
    for i in range(max(0, bin_config['num_bins']-10), bin_config['num_bins']):
        print(f"  Bin {i}: {bin_config['bin_values'][i]:.3f}m")

    # Verify inverse conversion
    print("\n" + "="*60)
    print("DEPTH CONVERSION VERIFICATION")
    print("="*60)

    test_depths = torch.tensor([1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 65.0])
    test_indices = depth_to_bin_indices(test_depths, bin_config)
    recovered_depths = bin_indices_to_depth(test_indices, bin_config)

    print("Depth -> Bin -> Depth roundtrip:")
    for d, i, rd in zip(test_depths, test_indices, recovered_depths):
        error = abs(rd - d)
        print(f"  {d:.1f}m -> bin {i.item():2d} -> {rd:.2f}m (error: {error:.2f}m)")

    # Build dataset
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False
    )

    # Build model
    print("Building model...")
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    model.eval()

    # Get view transformer reference
    view_transformer = model.module.img_lss_view_transformer

    print("\n" + "="*60)
    print("ANALYZING SAMPLES")
    print("="*60)

    for i, data in enumerate(data_loader):
        if i >= args.n_samples:
            break

        print(f"\n--- Sample {i} ---")

        # Get data
        img = data['img'][0].data[0]  # [B, N, C, H, W]
        gt_depth = data['gt_depth'][0].data[0]  # [N, H, W] or [B, N, H, W]
        radar_depth = data['radar_depth'][0].data[0]

        # Handle different image shapes
        if img.dim() == 5:
            B, N, C, H, W = img.shape
        elif img.dim() == 4:
            N, C, H, W = img.shape
            B = 1
            img = img.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected img shape: {img.shape}")

        # Handle different gt_depth shapes
        if gt_depth.dim() == 3:
            N_depth, H_depth, W_depth = gt_depth.shape
            gt_depth = gt_depth.unsqueeze(0)  # Add batch dim
        elif gt_depth.dim() == 4:
            _, N_depth, H_depth, W_depth = gt_depth.shape
        else:
            raise ValueError(f"Unexpected gt_depth shape: {gt_depth.shape}")

        print(f"Image shape: {img.shape}")
        print(f"GT depth shape: {gt_depth.shape}")
        print(f"Radar depth shape: {radar_depth.shape}")

        # Compute image brightness
        weights = torch.tensor([0.299, 0.587, 0.114])
        gray = (img.view(-1, C, H, W) * weights.view(1, 3, 1, 1)).sum(dim=1)
        brightness = gray.mean().item()
        print(f"Mean brightness: {brightness:.1f}")

        # Analyze GT depth
        gt_depth_flat = gt_depth.view(-1)
        valid_depths = gt_depth_flat[gt_depth_flat > 0]

        if len(valid_depths) > 0:
            print(f"GT depth statistics:")
            print(f"  Valid pixels: {len(valid_depths)} / {len(gt_depth_flat)} ({100*len(valid_depths)/len(gt_depth_flat):.2f}%)")
            print(f"  Min depth: {valid_depths.min():.2f}m")
            print(f"  Max depth: {valid_depths.max():.2f}m")
            print(f"  Mean depth: {valid_depths.mean():.2f}m")

            # Check depth range
            in_range = ((valid_depths >= bin_config['depth_start']) &
                       (valid_depths <= bin_config['depth_end'])).sum()
            print(f"  In valid range [{bin_config['depth_start']}-{bin_config['depth_end']}m]: {in_range}/{len(valid_depths)}")
        else:
            print("  No valid GT depth values!")

        # Analyze radar depth
        radar_depth_flat = radar_depth.view(-1)
        valid_radar = radar_depth_flat[radar_depth_flat > 0]
        if len(valid_radar) > 0:
            print(f"Radar depth statistics:")
            print(f"  Valid pixels: {len(valid_radar)} / {len(radar_depth_flat)} ({100*len(valid_radar)/len(radar_depth_flat):.2f}%)")
            print(f"  Min depth: {valid_radar.min():.2f}m")
            print(f"  Max depth: {valid_radar.max():.2f}m")

        # Move to GPU and run model
        gt_depth = gt_depth.to(device)

        # IMPORTANT: GT depth is only valid for first frame (first 6 cameras)
        # The model processes each temporal frame separately, calling view_transformer per frame
        # We need to create GT depth logits only for first 6 cameras
        num_cams = 6  # nuScenes has 6 cameras
        gt_depth_first_frame = gt_depth[:, :num_cams]  # [B, 6, H, W]
        N_first = num_cams

        # Create GT depth logits
        h, w = H_depth // downsample, W_depth // downsample

        # Downsample GT depth (only first frame)
        depth_flat = gt_depth_first_frame.view(B * N_first, 1, H_depth, W_depth)
        depth_for_pool = torch.where(
            depth_flat == 0,
            torch.full_like(depth_flat, 1e5),
            depth_flat
        )

        depth_ds = depth_for_pool.view(B * N_first, 1, h, downsample, w, downsample)
        depth_ds = depth_ds.permute(0, 1, 2, 4, 3, 5).contiguous()
        depth_ds = depth_ds.view(B * N_first, 1, h, w, -1)
        depth_ds = depth_ds.min(dim=-1).values
        depth_ds = torch.where(depth_ds > 1e4, torch.zeros_like(depth_ds), depth_ds)
        depth_ds = depth_ds.squeeze(1)  # [B*N_first, h, w]

        print(f"\nDownsampled depth shape (first frame only): {depth_ds.shape}")
        print(f"Expected shape: [B*N_first={B*N_first}, h={h}, w={w}]")

        valid_ds = depth_ds[depth_ds > 0]
        if len(valid_ds) > 0:
            print(f"Downsampled GT depth coverage: {len(valid_ds)}/{depth_ds.numel()} = {100*len(valid_ds)/depth_ds.numel():.2f}%")

        # Convert to bin indices
        bin_indices = depth_to_bin_indices(depth_ds, bin_config)

        # Create one-hot logits (only for first frame)
        D = bin_config['num_bins']
        gt_logits = torch.zeros(B * N_first, D, h, w, device=device)
        valid_mask = (depth_ds > 0) & (bin_indices < D)
        valid_indices = bin_indices.clone()
        valid_indices[~valid_mask] = 0
        gt_logits.scatter_(1, valid_indices.unsqueeze(1), 10.0)

        print(f"GT depth logits shape (first frame): {gt_logits.shape}")
        print(f"Expected model depth shape per frame: [B*N_cams={B}*{num_cams}={B*num_cams}, D={D}, h={h}, w={w}]")

        # Verify by checking softmax output
        gt_probs = F.softmax(gt_logits, dim=1)
        print(f"GT depth probabilities (softmax) - sum check: {gt_probs.sum(dim=1).mean():.4f} (should be ~1.0)")

        # Run model to get predicted depth
        print("\n--- Running model inference ---")

        # The dataloader already produces correctly formatted data
        # Just pass **data directly to the model (like test_brightness_gating.py does)

        # Make a deep copy of data for the second inference (data gets modified in-place)
        data_copy = copy.deepcopy(data)

        # Run with normal depth - pass batch directly (dataloader formats it correctly)
        with torch.no_grad():
            result_pred = model(return_loss=False, rescale=True, **data)

        n_pred_boxes = len(result_pred[0]['pts_bbox']['boxes_3d'])
        print(f"Predictions with normal depth: {n_pred_boxes} boxes")

        # Now run with GT depth (patched)
        # GT depth is only available for first frame (6 cameras)
        # The model calls view_transformer once per temporal frame
        # We use a frame counter to only replace depth for first frame
        original_forward = view_transformer.forward
        gt_logits_container = [gt_logits]
        frame_counter = [0]  # Mutable counter to track which frame we're processing

        def patched_forward(x, radar_depth, radar_rcs, img_metas, mlp_input):
            B_x, N_x, C_x, H_x, W_x = x.shape
            x_flat = x.reshape(B_x * N_x, C_x, H_x, W_x)

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

            # Only replace depth for first frame (where we have GT depth)
            if current_frame == 0:
                gt_logits_to_use = gt_logits_container[0]

                print(f"    Frame {current_frame}: Replacing with GT depth")
                print(f"    Predicted depth shape: {depth_digit.shape}")
                print(f"    GT depth logits shape: {gt_logits_to_use.shape}")

                # Verify shapes match
                if depth_digit.shape == gt_logits_to_use.shape:
                    # Compare distributions
                    pred_probs = F.softmax(depth_digit, dim=1)
                    gt_probs_check = F.softmax(gt_logits_to_use, dim=1)

                    # Mean depth (weighted average of bin values)
                    bin_vals = bin_config['bin_values'].to(depth_digit.device)
                    pred_mean_depth = (pred_probs * bin_vals.view(1, -1, 1, 1)).sum(dim=1).mean()
                    gt_mean_depth_check = (gt_probs_check * bin_vals.view(1, -1, 1, 1)).sum(dim=1).mean()
                    print(f"    Predicted mean depth: {pred_mean_depth:.2f}m")
                    print(f"    GT mean depth: {gt_mean_depth_check:.2f}m")

                    depth_digit = gt_logits_to_use
                else:
                    print(f"    WARNING: Shape mismatch! Using predicted depth.")
            else:
                # For other frames, use predicted depth (no GT available)
                pass

            return vt.view_transform(depth_out, depth_digit, tran_feat, img_metas)

        view_transformer.forward = patched_forward

        with torch.no_grad():
            result_gt = model(return_loss=False, rescale=True, **data_copy)

        view_transformer.forward = original_forward

        n_gt_boxes = len(result_gt[0]['pts_bbox']['boxes_3d'])
        print(f"Predictions with GT depth: {n_gt_boxes} boxes")
        print(f"Difference: {n_gt_boxes - n_pred_boxes:+d} boxes")

        # Save visualization if requested
        if args.save_vis:
            os.makedirs('debug_vis', exist_ok=True)

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Show first camera image
            img_vis = img[0, 0].permute(1, 2, 0).cpu().numpy()
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)
            axes[0, 0].imshow(img_vis)
            axes[0, 0].set_title('Camera 0 Image')

            # Show GT depth for first camera
            gt_depth_vis = gt_depth[0, 0].cpu().numpy()
            im = axes[0, 1].imshow(gt_depth_vis, cmap='viridis')
            axes[0, 1].set_title('GT Depth (Camera 0)')
            plt.colorbar(im, ax=axes[0, 1])

            # Show downsampled GT depth
            gt_depth_ds_vis = depth_ds[0].cpu().numpy()
            im = axes[0, 2].imshow(gt_depth_ds_vis, cmap='viridis')
            axes[0, 2].set_title(f'GT Depth Downsampled ({h}x{w})')
            plt.colorbar(im, ax=axes[0, 2])

            # Show radar depth (handle different shapes)
            if radar_depth.dim() == 3:
                radar_vis = radar_depth[0].cpu().numpy()  # [N, H, W] -> first camera
            else:
                radar_vis = radar_depth[0, 0].cpu().numpy()  # [B, N, H, W]
            im = axes[1, 0].imshow(radar_vis, cmap='plasma')
            axes[1, 0].set_title('Radar Depth (Camera 0)')
            plt.colorbar(im, ax=axes[1, 0])

            # Show GT depth probability (argmax)
            gt_argmax = gt_logits[0].argmax(dim=0).cpu().numpy()
            im = axes[1, 1].imshow(gt_argmax, cmap='viridis')
            axes[1, 1].set_title('GT Depth Bins (argmax)')
            plt.colorbar(im, ax=axes[1, 1])

            # Show valid depth mask
            valid_vis = valid_mask[0].cpu().numpy().astype(float)
            axes[1, 2].imshow(valid_vis, cmap='gray')
            axes[1, 2].set_title(f'Valid Depth Mask ({valid_vis.mean()*100:.1f}% coverage)')

            plt.tight_layout()
            plt.savefig(f'debug_vis/sample_{i}_depth_analysis.png', dpi=150)
            plt.close()
            print(f"Saved visualization to debug_vis/sample_{i}_depth_analysis.png")

    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)
    print("\nIf all checks passed, you can run the full experiment with:")
    print(f"  python tools/eval_depth_oracle.py {args.config} {args.checkpoint} --condition night")


if __name__ == '__main__':
    main()
