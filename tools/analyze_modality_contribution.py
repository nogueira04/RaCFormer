#!/usr/bin/env python
"""
Modality Contribution Analysis for RaCFormer.

This script measures the actual contribution of each modality (Image, Radar, LSS)
to the fused representation, accounting for both learned weights AND input feature
magnitudes.

Usage:
    python tools/analyze_modality_contribution.py \
        configs/racformer_r50_nuimg_704x256_f8.py \
        checkpoints/racformer_r50_f8.pth \
        --num-samples 200 \
        --output outputs/modality_analysis/

Output:
    - Console statistics for day/night/overall contributions
    - JSON file with detailed results
    - Visualization plots
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules to register them BEFORE building the model
import loaders  # noqa: F401
import models  # noqa: F401

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model


def get_night_sample_tokens(nusc_infos_path):
    """
    Load night sample tokens from nuScenes info file.

    Args:
        nusc_infos_path: Path to nuscenes_infos_val_sweep.pkl

    Returns:
        set: Set of sample tokens that are night samples
    """
    import pickle

    night_tokens = set()

    try:
        with open(nusc_infos_path, 'rb') as f:
            infos = pickle.load(f)

        # Check if infos have scene descriptions or we need to load from nuScenes
        # For now, we'll use a simpler heuristic based on filename patterns
        # or load from pre-computed night sample list

    except Exception as e:
        print(f"Warning: Could not load infos file: {e}")

    return night_tokens


def identify_condition_from_nuscenes(dataroot='data/nuscenes/', version='v1.0-trainval'):
    """
    Identify day/night samples from nuScenes scene descriptions.

    Returns:
        dict: Mapping from sample token to condition ('day' or 'night')
    """
    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        sample_conditions = {}
        night_keywords = ['night', 'dark']

        for scene in nusc.scene:
            description = scene['description'].lower()
            is_night = any(kw in description for kw in night_keywords)
            condition = 'night' if is_night else 'day'

            # Get all samples in this scene
            sample_token = scene['first_sample_token']
            while sample_token:
                sample_conditions[sample_token] = condition
                sample = nusc.get('sample', sample_token)
                sample_token = sample['next']

        return sample_conditions

    except Exception as e:
        print(f"Warning: Could not load nuScenes for condition identification: {e}")
        return {}


class FusionHook:
    """Hook to capture fusion layer inputs and outputs."""

    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.handles = []

    def hook_fn(self, module, input, output):
        """Capture input to fusion layer."""
        if len(input) > 0:
            self.inputs.append(input[0].detach().cpu())
        self.outputs.append(output.detach().cpu())

    def register(self, module):
        """Register hook on module."""
        handle = module.register_forward_hook(self.hook_fn)
        self.handles.append(handle)

    def clear(self):
        """Clear captured data."""
        self.inputs.clear()
        self.outputs.clear()

    def remove(self):
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def compute_contributions(fusion_input, fusion_weights):
    """
    Compute the contribution of each modality to the fusion output.

    Args:
        fusion_input: Tensor [B, N, 768] - concatenated features
        fusion_weights: Tensor [256, 768] - fusion layer weights

    Returns:
        dict: Contribution statistics for each modality
    """
    # Split weights into modality blocks
    W_img = fusion_weights[:, :256]      # [256, 256]
    W_radar = fusion_weights[:, 256:512]  # [256, 256]
    W_lss = fusion_weights[:, 512:]       # [256, 256]

    # Split input features
    img_feat = fusion_input[..., :256]      # [B, N, 256]
    radar_feat = fusion_input[..., 256:512]  # [B, N, 256]
    lss_feat = fusion_input[..., 512:]       # [B, N, 256]

    # Compute each modality's contribution to output
    # contribution = W_mod @ feat^T -> we want norm of this
    img_contrib = torch.matmul(img_feat, W_img.T)      # [B, N, 256]
    radar_contrib = torch.matmul(radar_feat, W_radar.T)  # [B, N, 256]
    lss_contrib = torch.matmul(lss_feat, W_lss.T)      # [B, N, 256]

    # Compute various statistics
    results = {}

    for name, contrib, feat in [('img', img_contrib, img_feat),
                                 ('radar', radar_contrib, radar_feat),
                                 ('lss', lss_contrib, lss_feat)]:
        results[name] = {
            # Contribution magnitude (L2 norm)
            'contrib_norm': contrib.norm(dim=-1).mean().item(),
            'contrib_norm_std': contrib.norm(dim=-1).std().item(),
            # Feature magnitude
            'feat_norm': feat.norm(dim=-1).mean().item(),
            'feat_norm_std': feat.norm(dim=-1).std().item(),
            # Mean absolute contribution
            'contrib_abs_mean': contrib.abs().mean().item(),
            # Max contribution
            'contrib_max': contrib.abs().max().item(),
        }

    # Compute percentages
    total_contrib = sum(results[m]['contrib_norm'] for m in ['img', 'radar', 'lss'])
    total_feat = sum(results[m]['feat_norm'] for m in ['img', 'radar', 'lss'])

    for name in ['img', 'radar', 'lss']:
        results[name]['contrib_pct'] = 100 * results[name]['contrib_norm'] / total_contrib
        results[name]['feat_pct'] = 100 * results[name]['feat_norm'] / total_feat

    return results


def analyze_model(config_path, checkpoint_path, num_samples=200, output_dir=None,
                  dataroot='data/nuscenes/', use_gpu=True):
    """
    Analyze modality contributions in RaCFormer.

    Args:
        config_path: Path to config file
        checkpoint_path: Path to checkpoint
        num_samples: Number of samples to analyze
        output_dir: Directory to save results
        dataroot: Path to nuScenes data
        use_gpu: Whether to use GPU
    """
    # Load config and build model
    print("Loading config and model...")
    cfg = Config.fromfile(config_path)

    # Build model
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    if use_gpu:
        model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # Get fusion layer weights
    # The fusion layer is inside the decoder layer:
    # model.pts_bbox_head.transformer.decoder.decoder_layer.fusion
    if hasattr(model, 'module'):
        fusion_layer = model.module.pts_bbox_head.transformer.decoder.decoder_layer.fusion
    else:
        fusion_layer = model.pts_bbox_head.transformer.decoder.decoder_layer.fusion

    fusion_weights = fusion_layer.weight.data.cpu()  # [256, 768]
    fusion_bias = fusion_layer.bias.data.cpu() if fusion_layer.bias is not None else None

    print(f"Fusion layer: input={fusion_weights.shape[1]}, output={fusion_weights.shape[0]}")

    # Analyze weight norms (static analysis)
    W_img = fusion_weights[:, :256]
    W_radar = fusion_weights[:, 256:512]
    W_lss = fusion_weights[:, 512:]

    weight_norms = {
        'img': W_img.norm().item(),
        'radar': W_radar.norm().item(),
        'lss': W_lss.norm().item(),
    }
    total_weight_norm = sum(weight_norms.values())
    weight_pcts = {k: 100 * v / total_weight_norm for k, v in weight_norms.items()}

    print("\n" + "=" * 60)
    print("STATIC WEIGHT ANALYSIS (L2 Norm)")
    print("=" * 60)
    for name in ['img', 'radar', 'lss']:
        print(f"  {name:8s}: {weight_norms[name]:8.4f} ({weight_pcts[name]:5.1f}%)")

    # Build dataset
    print("\nBuilding dataset...")
    cfg.data.val.test_mode = True
    dataset = build_dataset(cfg.data.val)

    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=False,
        shuffle=False
    )

    # Get day/night sample mapping
    print("Identifying day/night samples...")
    sample_conditions = identify_condition_from_nuscenes(dataroot)

    # Register hook on fusion layer
    hook = FusionHook()
    if hasattr(model, 'module'):
        hook.register(model.module.pts_bbox_head.transformer.decoder.decoder_layer.fusion)
    else:
        hook.register(model.pts_bbox_head.transformer.decoder.decoder_layer.fusion)

    # Collect contributions
    results_by_condition = defaultdict(lambda: defaultdict(list))
    all_results = defaultdict(list)

    print(f"\nAnalyzing {num_samples} samples...")

    for i, data in enumerate(tqdm(dataloader, total=min(num_samples, len(dataloader)))):
        if i >= num_samples:
            break

        # Determine condition
        try:
            # Try to get sample token from data
            img_metas = data['img_metas'][0].data[0][0]
            sample_token = img_metas.get('sample_idx', img_metas.get('token', ''))
            condition = sample_conditions.get(sample_token, 'unknown')
        except:
            condition = 'unknown'

        # Forward pass
        hook.clear()
        with torch.no_grad():
            if use_gpu:
                result = model(return_loss=False, rescale=True, **data)
            else:
                result = model(return_loss=False, rescale=True, **data)

        # Compute contributions from captured fusion input
        if hook.inputs:
            fusion_input = hook.inputs[0]

            # Handle different input shapes
            if fusion_input.dim() == 4:  # [B, num_layers, N, 768]
                # Average across layers or take last layer
                fusion_input = fusion_input[:, -1, :, :]  # Take last layer

            contrib = compute_contributions(fusion_input, fusion_weights)

            # Store results
            for modality in ['img', 'radar', 'lss']:
                for metric, value in contrib[modality].items():
                    all_results[f'{modality}_{metric}'].append(value)
                    results_by_condition[condition][f'{modality}_{metric}'].append(value)

    hook.remove()

    # Aggregate results
    print("\n" + "=" * 60)
    print("ACTIVATION CONTRIBUTION ANALYSIS")
    print("=" * 60)

    # Overall statistics
    print("\n--- Overall (all samples) ---")
    overall_stats = {}
    for modality in ['img', 'radar', 'lss']:
        contrib_norms = all_results[f'{modality}_contrib_norm']
        feat_norms = all_results[f'{modality}_feat_norm']
        contrib_pcts = all_results[f'{modality}_contrib_pct']

        overall_stats[modality] = {
            'contrib_norm_mean': np.mean(contrib_norms),
            'contrib_norm_std': np.std(contrib_norms),
            'feat_norm_mean': np.mean(feat_norms),
            'contrib_pct_mean': np.mean(contrib_pcts),
        }

        print(f"  {modality:8s}: contrib={np.mean(contrib_norms):8.2f} "
              f"(±{np.std(contrib_norms):.2f}), "
              f"feat={np.mean(feat_norms):6.2f}, "
              f"pct={np.mean(contrib_pcts):5.1f}%")

    # Per-condition statistics
    condition_stats = {}
    for condition in ['day', 'night', 'unknown']:
        if condition not in results_by_condition:
            continue

        cond_data = results_by_condition[condition]
        n_samples = len(cond_data.get('img_contrib_norm', []))

        if n_samples == 0:
            continue

        print(f"\n--- {condition.upper()} ({n_samples} samples) ---")
        condition_stats[condition] = {'n_samples': n_samples}

        for modality in ['img', 'radar', 'lss']:
            contrib_norms = cond_data[f'{modality}_contrib_norm']
            feat_norms = cond_data[f'{modality}_feat_norm']
            contrib_pcts = cond_data[f'{modality}_contrib_pct']

            condition_stats[condition][modality] = {
                'contrib_norm_mean': np.mean(contrib_norms),
                'contrib_norm_std': np.std(contrib_norms),
                'feat_norm_mean': np.mean(feat_norms),
                'contrib_pct_mean': np.mean(contrib_pcts),
            }

            print(f"  {modality:8s}: contrib={np.mean(contrib_norms):8.2f} "
                  f"(±{np.std(contrib_norms):.2f}), "
                  f"feat={np.mean(feat_norms):6.2f}, "
                  f"pct={np.mean(contrib_pcts):5.1f}%")

    # Compare day vs night
    if 'day' in condition_stats and 'night' in condition_stats:
        print("\n--- DAY vs NIGHT Comparison ---")
        print(f"{'Modality':<10} {'Day Contrib':<15} {'Night Contrib':<15} {'Change':<10}")
        print("-" * 50)

        for modality in ['img', 'radar', 'lss']:
            day_contrib = condition_stats['day'][modality]['contrib_pct_mean']
            night_contrib = condition_stats['night'][modality]['contrib_pct_mean']
            change = night_contrib - day_contrib

            print(f"{modality:<10} {day_contrib:>10.1f}%     {night_contrib:>10.1f}%     "
                  f"{change:>+6.1f}%")

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON results
        results_json = {
            'weight_analysis': {
                'norms': weight_norms,
                'percentages': weight_pcts,
            },
            'overall': overall_stats,
            'by_condition': condition_stats,
            'num_samples': num_samples,
            'config': config_path,
            'checkpoint': checkpoint_path,
        }

        json_path = os.path.join(output_dir, 'modality_contributions.json')
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to: {json_path}")

        # Create visualization
        create_visualizations(results_json, output_dir)

    return results_json


def create_visualizations(results, output_dir):
    """Create visualization plots for the analysis."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    modalities = ['img', 'radar', 'lss']
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red

    # Plot 1: Weight norm comparison
    ax = axes[0, 0]
    weight_pcts = [results['weight_analysis']['percentages'][m] for m in modalities]
    bars = ax.bar(modalities, weight_pcts, color=colors)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Static Weight Analysis (L2 Norm)')
    ax.set_ylim(0, 50)
    for bar, pct in zip(bars, weight_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', fontsize=10)

    # Plot 2: Activation contribution comparison
    ax = axes[0, 1]
    contrib_pcts = [results['overall'][m]['contrib_pct_mean'] for m in modalities]
    bars = ax.bar(modalities, contrib_pcts, color=colors)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Activation Contribution Analysis')
    ax.set_ylim(0, 50)
    for bar, pct in zip(bars, contrib_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', fontsize=10)

    # Plot 3: Day vs Night contribution (if available)
    ax = axes[1, 0]
    if 'day' in results['by_condition'] and 'night' in results['by_condition']:
        x = np.arange(len(modalities))
        width = 0.35

        day_pcts = [results['by_condition']['day'][m]['contrib_pct_mean'] for m in modalities]
        night_pcts = [results['by_condition']['night'][m]['contrib_pct_mean'] for m in modalities]

        bars1 = ax.bar(x - width/2, day_pcts, width, label='Day', color='gold')
        bars2 = ax.bar(x + width/2, night_pcts, width, label='Night', color='darkblue')

        ax.set_ylabel('Contribution (%)')
        ax.set_title('Day vs Night Contribution')
        ax.set_xticks(x)
        ax.set_xticklabels(modalities)
        ax.legend()
        ax.set_ylim(0, 50)

        for bars in [bars1, bars2]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{bar.get_height():.1f}', ha='center', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Day/Night data not available', ha='center', va='center')
        ax.set_title('Day vs Night Contribution')

    # Plot 4: Feature magnitude comparison
    ax = axes[1, 1]
    if 'day' in results['by_condition'] and 'night' in results['by_condition']:
        x = np.arange(len(modalities))
        width = 0.35

        day_feats = [results['by_condition']['day'][m]['feat_norm_mean'] for m in modalities]
        night_feats = [results['by_condition']['night'][m]['feat_norm_mean'] for m in modalities]

        bars1 = ax.bar(x - width/2, day_feats, width, label='Day', color='gold')
        bars2 = ax.bar(x + width/2, night_feats, width, label='Night', color='darkblue')

        ax.set_ylabel('Feature Norm')
        ax.set_title('Feature Magnitude: Day vs Night')
        ax.set_xticks(x)
        ax.set_xticklabels(modalities)
        ax.legend()
    else:
        feat_norms = [results['overall'][m]['feat_norm_mean'] for m in modalities]
        ax.bar(modalities, feat_norms, color=colors)
        ax.set_ylabel('Feature Norm')
        ax.set_title('Feature Magnitude (Overall)')

    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'modality_contribution_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze modality contributions in RaCFormer fusion'
    )
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--num-samples', '-n', type=int, default=200,
                        help='Number of samples to analyze (default: 200)')
    parser.add_argument('--output', '-o', type=str, default='outputs/modality_analysis/',
                        help='Output directory for results')
    parser.add_argument('--dataroot', type=str, default='data/nuscenes/',
                        help='Path to nuScenes data root')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')

    args = parser.parse_args()

    print("=" * 60)
    print("RaCFormer Modality Contribution Analysis")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {args.output}")
    print("=" * 60)

    results = analyze_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        output_dir=args.output,
        dataroot=args.dataroot,
        use_gpu=not args.cpu
    )

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
