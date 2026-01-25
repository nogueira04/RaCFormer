#!/usr/bin/env python
"""
Analyze the trained fusion layer weights to understand modality importance.

Usage:
    python tools/analyze_fusion_weights.py checkpoints/racformer_r50_f8.pth
"""

import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models  # noqa: F401

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model


def analyze_fusion_weights(checkpoint_path, config_path=None):
    """Analyze fusion layer weights to understand modality contributions."""
    
    # Build model
    if config_path is None:
        config_path = 'configs/racformer_r50_nuimg_704x256_f8.py'
    
    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    # Get the fusion layer
    # The decoder_layer is shared, so we access it directly
    decoder_layer = model.pts_bbox_head.transformer.decoder.decoder_layer
    fusion_layer = decoder_layer.fusion
    
    print("\n" + "="*70)
    print("FUSION LAYER ANALYSIS")
    print("="*70)
    print(f"Fusion layer shape: {fusion_layer.weight.shape}")  # [256, 768]
    print(f"  - Output dims: {fusion_layer.weight.shape[0]}")
    print(f"  - Input dims: {fusion_layer.weight.shape[1]} = [img_256 + radar_256 + lss_256]")
    
    # Split weights by modality
    # The concatenation order is: (query_feat, query_radar_feat, query_lss_feat)
    # So: [0:256] = img, [256:512] = radar, [512:768] = lss
    weight = fusion_layer.weight.data  # [256, 768]
    bias = fusion_layer.bias.data if fusion_layer.bias is not None else None
    
    img_weights = weight[:, :256]       # [256, 256]
    radar_weights = weight[:, 256:512]  # [256, 256]
    lss_weights = weight[:, 512:]       # [256, 256]
    
    # Compute various metrics
    print("\n" + "-"*70)
    print("MODALITY IMPORTANCE (L2 Norm)")
    print("-"*70)
    
    img_norm = img_weights.norm().item()
    radar_norm = radar_weights.norm().item()
    lss_norm = lss_weights.norm().item()
    total_norm = img_norm + radar_norm + lss_norm
    
    print(f"  Image (query_feat):  {img_norm:8.4f}  ({100*img_norm/total_norm:5.1f}%)")
    print(f"  Radar:               {radar_norm:8.4f}  ({100*radar_norm/total_norm:5.1f}%)")
    print(f"  LSS (BEV):           {lss_norm:8.4f}  ({100*lss_norm/total_norm:5.1f}%)")
    print(f"  Total:               {total_norm:8.4f}")
    
    # Frobenius norm (same as L2 for matrices)
    print("\n" + "-"*70)
    print("MODALITY IMPORTANCE (Frobenius Norm)")
    print("-"*70)
    
    img_frob = img_weights.norm(p='fro').item()
    radar_frob = radar_weights.norm(p='fro').item()
    lss_frob = lss_weights.norm(p='fro').item()
    total_frob = img_frob + radar_frob + lss_frob
    
    print(f"  Image:  {img_frob:8.4f}  ({100*img_frob/total_frob:5.1f}%)")
    print(f"  Radar:  {radar_frob:8.4f}  ({100*radar_frob/total_frob:5.1f}%)")
    print(f"  LSS:    {lss_frob:8.4f}  ({100*lss_frob/total_frob:5.1f}%)")
    
    # Mean absolute weight
    print("\n" + "-"*70)
    print("MODALITY IMPORTANCE (Mean Absolute Weight)")
    print("-"*70)
    
    img_mean = img_weights.abs().mean().item()
    radar_mean = radar_weights.abs().mean().item()
    lss_mean = lss_weights.abs().mean().item()
    total_mean = img_mean + radar_mean + lss_mean
    
    print(f"  Image:  {img_mean:8.6f}  ({100*img_mean/total_mean:5.1f}%)")
    print(f"  Radar:  {radar_mean:8.6f}  ({100*radar_mean/total_mean:5.1f}%)")
    print(f"  LSS:    {lss_mean:8.6f}  ({100*lss_mean/total_mean:5.1f}%)")
    
    # Spectral norm (largest singular value) - indicates max amplification
    print("\n" + "-"*70)
    print("MODALITY IMPORTANCE (Spectral Norm - Max Amplification)")
    print("-"*70)
    
    img_spec = torch.linalg.svdvals(img_weights)[0].item()
    radar_spec = torch.linalg.svdvals(radar_weights)[0].item()
    lss_spec = torch.linalg.svdvals(lss_weights)[0].item()
    total_spec = img_spec + radar_spec + lss_spec
    
    print(f"  Image:  {img_spec:8.4f}  ({100*img_spec/total_spec:5.1f}%)")
    print(f"  Radar:  {radar_spec:8.4f}  ({100*radar_spec/total_spec:5.1f}%)")
    print(f"  LSS:    {lss_spec:8.4f}  ({100*lss_spec/total_spec:5.1f}%)")
    
    # Statistics
    print("\n" + "-"*70)
    print("WEIGHT STATISTICS")
    print("-"*70)
    print(f"{'Modality':<10} | {'Min':>10} | {'Max':>10} | {'Mean':>10} | {'Std':>10}")
    print("-"*70)
    
    for name, w in [('Image', img_weights), ('Radar', radar_weights), ('LSS', lss_weights)]:
        print(f"{name:<10} | {w.min().item():10.6f} | {w.max().item():10.6f} | {w.mean().item():10.6f} | {w.std().item():10.6f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Find which modality has highest contribution
    norms = {'Image': img_norm, 'Radar': radar_norm, 'LSS': lss_norm}
    sorted_norms = sorted(norms.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Modality ranking by L2 norm: {sorted_norms[0][0]} > {sorted_norms[1][0]} > {sorted_norms[2][0]}")
    
    max_diff = max(norms.values()) / min(norms.values())
    if max_diff < 1.2:
        print("→ Modalities are roughly BALANCED (max/min ratio < 1.2)")
        print("→ This suggests adaptive fusion may still have room to improve")
    elif max_diff < 1.5:
        print("→ Modalities show MODERATE imbalance (1.2 < ratio < 1.5)")
        print(f"→ {sorted_norms[0][0]} is favored, but not dominantly")
    else:
        print(f"→ Modalities show STRONG imbalance (ratio = {max_diff:.2f})")
        print(f"→ {sorted_norms[0][0]} is strongly favored")
        print("→ Model has already learned to weight modalities differently")
    
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze fusion layer weights')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--config', default='configs/racformer_r50_nuimg_704x256_f8.py',
                        help='config file path')
    args = parser.parse_args()
    
    analyze_fusion_weights(args.checkpoint, args.config)
