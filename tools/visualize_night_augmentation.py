#!/usr/bin/env python
"""
Visual validation tool for night augmentation.

This script generates side-by-side comparisons of:
- Original daytime images
- Synthetic night images (light, medium, heavy augmentation)
- Real nighttime images from nuScenes

Usage:
    python tools/visualize_night_augmentation.py --output outputs/night_aug_validation/
    python tools/visualize_night_augmentation.py --output outputs/night_aug_validation/ --num_samples 20 --seed 42
"""

import os
import sys
import argparse
import json
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.pipelines.night_augmentation import apply_simulate_night


# Camera channels in nuScenes
CAMERA_CHANNELS = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]

# Augmentation presets - Basic (original)
AUGMENTATION_PRESETS = {
    'light': {
        'gamma': 1.5,
        'brightness': 0.5,
        'contrast': 0.75,
        'noise_std': 8,
        'color_shift_strength': 0.1
    },
    'medium': {
        'gamma': 2.0,
        'brightness': 0.35,
        'contrast': 0.65,
        'noise_std': 12,
        'color_shift_strength': 0.15
    },
    'heavy': {
        'gamma': 2.5,
        'brightness': 0.25,
        'contrast': 0.55,
        'noise_std': 18,
        'color_shift_strength': 0.2
    }
}

# Augmentation presets - Enhanced (with new features)
AUGMENTATION_PRESETS_ENHANCED = {
    'enhanced_light': {
        'gamma': 1.5,
        'brightness': 0.5,
        'contrast': 0.75,
        'noise_std': 8,
        'color_shift_strength': 0.1,
        'vignette_strength': 0.3,
        'headlight_strength': 0.3,
        'headlight_height': 0.4,
        'num_bright_spots': 4,
        'spot_brightness': (150, 220),
        'spot_size_range': (15, 35),
        'preserve_bright': False,
    },
    'enhanced_medium': {
        'gamma': 1.8,
        'brightness': 0.45,
        'contrast': 0.7,
        'noise_std': 10,
        'color_shift_strength': 0.15,
        'vignette_strength': 0.4,
        'headlight_strength': 0.4,
        'headlight_height': 0.4,
        'num_bright_spots': 5,
        'spot_brightness': (160, 240),
        'spot_size_range': (12, 40),
        'preserve_bright': True,
        'bright_threshold': 200,
        'bright_preserve_factor': 0.6,
    },
    'enhanced_heavy': {
        'gamma': 2.0,
        'brightness': 0.4,
        'contrast': 0.65,
        'noise_std': 12,
        'color_shift_strength': 0.2,
        'vignette_strength': 0.5,
        'headlight_strength': 0.5,
        'headlight_height': 0.45,
        'num_bright_spots': 6,
        'spot_brightness': (180, 255),
        'spot_size_range': (10, 45),
        'preserve_bright': True,
        'bright_threshold': 180,
        'bright_preserve_factor': 0.7,
    }
}


def load_nuscenes(dataroot, version='v1.0-trainval'):
    """
    Load nuScenes dataset.

    Args:
        dataroot (str): Path to nuScenes data root
        version (str): Dataset version

    Returns:
        NuScenes: nuScenes dataset object or None if unavailable
    """
    try:
        from nuscenes.nuscenes import NuScenes
        print(f"Loading nuScenes {version} from {dataroot}...")
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        print(f"Loaded {len(nusc.scene)} scenes, {len(nusc.sample)} samples")
        return nusc
    except ImportError:
        print("ERROR: nuscenes-devkit not installed. Install with: pip install nuscenes-devkit")
        return None
    except Exception as e:
        print(f"ERROR: Could not load nuScenes: {e}")
        return None


def identify_day_night_samples(nusc):
    """
    Identify day and night samples based on scene descriptions.

    Args:
        nusc: nuScenes dataset object

    Returns:
        tuple: (day_samples, night_samples) - lists of sample tokens
    """
    night_samples = []
    day_samples = []

    night_keywords = ['night', 'dark']

    for scene in nusc.scene:
        description = scene['description'].lower()
        is_night = any(keyword in description for keyword in night_keywords)

        # Get all samples from this scene
        sample_token = scene['first_sample_token']
        while sample_token:
            sample = nusc.get('sample', sample_token)
            if is_night:
                night_samples.append(sample_token)
            else:
                day_samples.append(sample_token)
            sample_token = sample['next']

    print(f"Found {len(day_samples)} day samples and {len(night_samples)} night samples")
    return day_samples, night_samples


def load_sample_images(nusc, sample_token, cameras=None):
    """
    Load camera images for a given sample.

    Args:
        nusc: nuScenes dataset object
        sample_token (str): Sample token
        cameras (list): List of camera channels to load. None = all cameras.

    Returns:
        dict: Camera channel -> image (BGR numpy array)
    """
    if cameras is None:
        cameras = CAMERA_CHANNELS

    sample = nusc.get('sample', sample_token)
    images = {}

    for cam in cameras:
        if cam not in sample['data']:
            continue
        try:
            cam_data = nusc.get('sample_data', sample['data'][cam])
            img_path = os.path.join(nusc.dataroot, cam_data['filename'])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    images[cam] = img
        except Exception as e:
            print(f"Warning: Could not load {cam} for sample {sample_token}: {e}")

    return images


def compute_image_stats(img):
    """
    Compute statistics for an image.

    Args:
        img (np.ndarray): Image in BGR format [H, W, 3]

    Returns:
        dict: Dictionary of image statistics
    """
    return {
        'mean_brightness': float(img.mean()),
        'std_brightness': float(img.std()),
        'mean_b': float(img[..., 0].mean()),
        'mean_g': float(img[..., 1].mean()),
        'mean_r': float(img[..., 2].mean()),
        'std_b': float(img[..., 0].std()),
        'std_g': float(img[..., 1].std()),
        'std_r': float(img[..., 2].std()),
        'min': float(img.min()),
        'max': float(img.max()),
        'histogram': np.histogram(img.flatten(), bins=256, range=(0, 255))[0]
    }


def create_comparison_figure(day_img, synthetic_images, real_night_img=None, title=''):
    """
    Create a comparison figure with day, synthetic night variants, and real night.

    Args:
        day_img (np.ndarray): Original day image (BGR)
        synthetic_images (dict): Dict mapping preset name to synthetic image
        real_night_img (np.ndarray): Real night image (BGR), optional
        title (str): Figure title

    Returns:
        matplotlib.figure.Figure: The comparison figure
    """
    num_rows = 1 + len(synthetic_images) + (1 if real_night_img is not None else 0)

    fig, axes = plt.subplots(num_rows, 1, figsize=(12, 4 * num_rows))
    if num_rows == 1:
        axes = [axes]

    # Convert BGR to RGB for display
    day_rgb = cv2.cvtColor(day_img, cv2.COLOR_BGR2RGB)

    # Row 0: Original day
    axes[0].imshow(day_rgb)
    axes[0].set_title('Original Day Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Rows 1-N: Synthetic night variants
    for idx, (preset_name, synth_img) in enumerate(synthetic_images.items()):
        synth_rgb = cv2.cvtColor(synth_img, cv2.COLOR_BGR2RGB)
        axes[idx + 1].imshow(synth_rgb)
        preset = AUGMENTATION_PRESETS[preset_name]
        axes[idx + 1].set_title(
            f'Synthetic Night ({preset_name.capitalize()}) - '
            f'γ={preset["gamma"]}, br={preset["brightness"]}, '
            f'cont={preset["contrast"]}, noise={preset["noise_std"]}',
            fontsize=10
        )
        axes[idx + 1].axis('off')

    # Last row: Real night (if provided)
    if real_night_img is not None:
        real_rgb = cv2.cvtColor(real_night_img, cv2.COLOR_BGR2RGB)
        axes[-1].imshow(real_rgb)
        axes[-1].set_title('Real Night Image (from nuScenes)', fontsize=12, fontweight='bold')
        axes[-1].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def create_multicam_comparison(day_images, synthetic_images, real_night_images=None, title=''):
    """
    Create a multi-camera comparison grid.

    Layout:
    Row 1: Day images (all 6 cameras)
    Row 2: Synthetic night - medium (all 6 cameras)
    Row 3: Real night images (all 6 cameras) - if provided

    Args:
        day_images (dict): Camera -> day image
        synthetic_images (dict): Camera -> synthetic image
        real_night_images (dict): Camera -> real night image (optional)
        title (str): Figure title

    Returns:
        matplotlib.figure.Figure: The comparison figure
    """
    num_rows = 2 if real_night_images is None else 3
    num_cols = 3  # Left, Center, Right columns

    fig = plt.figure(figsize=(16, 5 * num_rows))

    # Camera layout matches typical surround-view setup
    # Row 1: FRONT_LEFT, FRONT, FRONT_RIGHT
    # Row 2: BACK_LEFT, BACK, BACK_RIGHT
    camera_layout = [
        ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
        ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    ]

    gs = GridSpec(num_rows * 2, 3, figure=fig, hspace=0.3, wspace=0.05)

    row_labels = ['Day', 'Synthetic Night (Medium)']
    if real_night_images is not None:
        row_labels.append('Real Night')

    image_sets = [day_images, synthetic_images]
    if real_night_images is not None:
        image_sets.append(real_night_images)

    for set_idx, (images, label) in enumerate(zip(image_sets, row_labels)):
        base_row = set_idx * 2

        for cam_row_idx, cam_row in enumerate(camera_layout):
            for cam_col_idx, cam_name in enumerate(cam_row):
                ax = fig.add_subplot(gs[base_row + cam_row_idx, cam_col_idx])

                if cam_name in images and images[cam_name] is not None:
                    img_rgb = cv2.cvtColor(images[cam_name], cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)

                if cam_row_idx == 0:
                    ax.set_title(cam_name.replace('CAM_', ''), fontsize=9)

                ax.axis('off')

        # Add row label
        fig.text(0.01, 1 - (set_idx + 0.5) / num_rows, label,
                 fontsize=12, fontweight='bold', rotation=90, va='center')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    return fig


def create_histogram_comparison(day_stats, synthetic_stats, real_night_stats=None, title=''):
    """
    Create histogram comparison plot.

    Args:
        day_stats (dict): Statistics from day image
        synthetic_stats (dict): Dict mapping preset name to stats
        real_night_stats (dict): Statistics from real night image (optional)
        title (str): Figure title

    Returns:
        matplotlib.figure.Figure: The histogram figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(256)

    # Plot 1: Overall brightness histogram
    ax = axes[0, 0]
    ax.plot(x, day_stats['histogram'], label='Day', linewidth=2, alpha=0.8)
    for preset_name, stats in synthetic_stats.items():
        ax.plot(x, stats['histogram'], label=f'Synthetic ({preset_name})', linestyle='--', alpha=0.7)
    if real_night_stats is not None:
        ax.plot(x, real_night_stats['histogram'], label='Real Night', linewidth=2, color='black')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Overall Brightness Distribution')
    ax.legend()
    ax.set_xlim(0, 255)

    # Plot 2: Mean brightness bar chart
    ax = axes[0, 1]
    categories = ['Day'] + [f'Synth ({k})' for k in synthetic_stats.keys()]
    means = [day_stats['mean_brightness']] + [s['mean_brightness'] for s in synthetic_stats.values()]
    if real_night_stats is not None:
        categories.append('Real Night')
        means.append(real_night_stats['mean_brightness'])

    colors = ['gold'] + ['steelblue'] * len(synthetic_stats) + (['darkblue'] if real_night_stats else [])
    bars = ax.bar(categories, means, color=colors)
    ax.set_ylabel('Mean Brightness')
    ax.set_title('Mean Brightness Comparison')
    ax.tick_params(axis='x', rotation=45)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    # Plot 3: Standard deviation bar chart
    ax = axes[1, 0]
    stds = [day_stats['std_brightness']] + [s['std_brightness'] for s in synthetic_stats.values()]
    if real_night_stats is not None:
        stds.append(real_night_stats['std_brightness'])
    bars = ax.bar(categories, stds, color=colors)
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Brightness Std Dev Comparison (Contrast Proxy)')
    ax.tick_params(axis='x', rotation=45)
    for bar, std in zip(bars, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{std:.1f}', ha='center', va='bottom', fontsize=9)

    # Plot 4: Per-channel mean
    ax = axes[1, 1]
    x_pos = np.arange(len(categories))
    width = 0.25

    r_means = [day_stats['mean_r']] + [s['mean_r'] for s in synthetic_stats.values()]
    g_means = [day_stats['mean_g']] + [s['mean_g'] for s in synthetic_stats.values()]
    b_means = [day_stats['mean_b']] + [s['mean_b'] for s in synthetic_stats.values()]
    if real_night_stats is not None:
        r_means.append(real_night_stats['mean_r'])
        g_means.append(real_night_stats['mean_g'])
        b_means.append(real_night_stats['mean_b'])

    ax.bar(x_pos - width, r_means, width, label='Red', color='red', alpha=0.7)
    ax.bar(x_pos, g_means, width, label='Green', color='green', alpha=0.7)
    ax.bar(x_pos + width, b_means, width, label='Blue', color='blue', alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45)
    ax.set_ylabel('Mean Channel Value')
    ax.set_title('Per-Channel Mean Values')
    ax.legend()

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def create_intensity_comparison(day_img, title=''):
    """
    Create comparison of different augmentation intensities.

    Args:
        day_img (np.ndarray): Original day image (BGR)
        title (str): Figure title

    Returns:
        matplotlib.figure.Figure: The comparison figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Convert BGR to RGB for display
    day_rgb = cv2.cvtColor(day_img, cv2.COLOR_BGR2RGB)

    # Original
    axes[0, 0].imshow(day_rgb)
    axes[0, 0].set_title('Original Day', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Light
    light_img = apply_simulate_night(day_img, **AUGMENTATION_PRESETS['light'])
    light_rgb = cv2.cvtColor(light_img, cv2.COLOR_BGR2RGB)
    axes[0, 1].imshow(light_rgb)
    preset = AUGMENTATION_PRESETS['light']
    axes[0, 1].set_title(f'Light: γ={preset["gamma"]}, br={preset["brightness"]}', fontsize=11)
    axes[0, 1].axis('off')

    # Medium
    medium_img = apply_simulate_night(day_img, **AUGMENTATION_PRESETS['medium'])
    medium_rgb = cv2.cvtColor(medium_img, cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(medium_rgb)
    preset = AUGMENTATION_PRESETS['medium']
    axes[1, 0].set_title(f'Medium: γ={preset["gamma"]}, br={preset["brightness"]}', fontsize=11)
    axes[1, 0].axis('off')

    # Heavy
    heavy_img = apply_simulate_night(day_img, **AUGMENTATION_PRESETS['heavy'])
    heavy_rgb = cv2.cvtColor(heavy_img, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(heavy_rgb)
    preset = AUGMENTATION_PRESETS['heavy']
    axes[1, 1].set_title(f'Heavy: γ={preset["gamma"]}, br={preset["brightness"]}', fontsize=11)
    axes[1, 1].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def create_enhanced_comparison(day_img, real_night_img=None, title=''):
    """
    Create comparison of basic vs enhanced augmentation.

    Args:
        day_img (np.ndarray): Original day image (BGR)
        real_night_img (np.ndarray): Real night image for reference (BGR)
        title (str): Figure title

    Returns:
        matplotlib.figure.Figure: The comparison figure
    """
    has_real_night = real_night_img is not None
    num_rows = 4 if has_real_night else 3
    fig, axes = plt.subplots(num_rows, 3, figsize=(16, 4 * num_rows))

    # Convert BGR to RGB for display
    day_rgb = cv2.cvtColor(day_img, cv2.COLOR_BGR2RGB)

    # Row 0: Original day (span all columns via text)
    for col in range(3):
        axes[0, col].imshow(day_rgb)
        axes[0, col].axis('off')
    axes[0, 1].set_title('Original Day Image', fontsize=12, fontweight='bold')

    # Row 1: Basic augmentation (light, medium, heavy)
    for col, preset_name in enumerate(['light', 'medium', 'heavy']):
        preset = AUGMENTATION_PRESETS[preset_name]
        img = apply_simulate_night(day_img, **preset)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[1, col].imshow(img_rgb)
        axes[1, col].set_title(f'Basic {preset_name.capitalize()}', fontsize=10)
        axes[1, col].axis('off')

    # Row 2: Enhanced augmentation
    for col, preset_name in enumerate(['enhanced_light', 'enhanced_medium', 'enhanced_heavy']):
        preset = AUGMENTATION_PRESETS_ENHANCED[preset_name]
        img = apply_simulate_night(day_img, **preset)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[2, col].imshow(img_rgb)
        label = preset_name.replace('enhanced_', '').capitalize()
        axes[2, col].set_title(f'Enhanced {label}\n(+vignette, +headlights, +spots)', fontsize=9)
        axes[2, col].axis('off')

    # Row 3: Real night (if provided)
    if has_real_night:
        real_rgb = cv2.cvtColor(real_night_img, cv2.COLOR_BGR2RGB)
        for col in range(3):
            axes[3, col].imshow(real_rgb)
            axes[3, col].axis('off')
        axes[3, 1].set_title('Real Night Image (Reference)', fontsize=12, fontweight='bold')

    # Add row labels
    fig.text(0.01, 0.85 if has_real_night else 0.82, 'Original', fontsize=11, fontweight='bold', rotation=90, va='center')
    fig.text(0.01, 0.58 if has_real_night else 0.5, 'Basic', fontsize=11, fontweight='bold', rotation=90, va='center')
    fig.text(0.01, 0.32 if has_real_night else 0.18, 'Enhanced', fontsize=11, fontweight='bold', rotation=90, va='center')
    if has_real_night:
        fig.text(0.01, 0.1, 'Real', fontsize=11, fontweight='bold', rotation=90, va='center')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def create_feature_breakdown(day_img, title=''):
    """
    Create a breakdown showing individual enhanced features.

    Args:
        day_img (np.ndarray): Original day image (BGR)
        title (str): Figure title

    Returns:
        matplotlib.figure.Figure: The comparison figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    # Convert BGR to RGB for display
    day_rgb = cv2.cvtColor(day_img, cv2.COLOR_BGR2RGB)

    # Base parameters (light preset)
    base_params = {
        'gamma': 1.5,
        'brightness': 0.5,
        'contrast': 0.75,
        'noise_std': 8,
        'color_shift_strength': 0.1
    }

    # Row 0: Original and base augmentations
    axes[0, 0].imshow(day_rgb)
    axes[0, 0].set_title('Original Day', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    # Base only
    img = apply_simulate_night(day_img, **base_params)
    axes[0, 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Base (gamma+brightness+contrast)', fontsize=10)
    axes[0, 1].axis('off')

    # Base + vignette
    img = apply_simulate_night(day_img, **base_params, vignette_strength=0.4)
    axes[0, 2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('+ Vignette', fontsize=10)
    axes[0, 2].axis('off')

    # Base + headlights
    img = apply_simulate_night(day_img, **base_params, headlight_strength=0.4, headlight_height=0.4)
    axes[0, 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 3].set_title('+ Headlight Gradient', fontsize=10)
    axes[0, 3].axis('off')

    # Row 1: More combinations
    # Base + bright spots
    img = apply_simulate_night(day_img, **base_params, num_bright_spots=5,
                               spot_brightness=(150, 230), spot_size_range=(15, 40))
    axes[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('+ Streetlight Spots', fontsize=10)
    axes[1, 0].axis('off')

    # Base + preserve bright
    img = apply_simulate_night(day_img, **base_params, preserve_bright=True,
                               bright_threshold=200, bright_preserve_factor=0.7)
    axes[1, 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('+ Preserve Bright Regions', fontsize=10)
    axes[1, 1].axis('off')

    # Vignette + headlights
    img = apply_simulate_night(day_img, **base_params, vignette_strength=0.4,
                               headlight_strength=0.4, headlight_height=0.4)
    axes[1, 2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('+ Vignette + Headlights', fontsize=10)
    axes[1, 2].axis('off')

    # All features combined
    img = apply_simulate_night(day_img, **AUGMENTATION_PRESETS_ENHANCED['enhanced_medium'])
    axes[1, 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title('All Enhanced Features', fontsize=10, fontweight='bold')
    axes[1, 3].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def aggregate_statistics(stats_list):
    """
    Aggregate statistics from multiple images.

    Args:
        stats_list (list): List of stat dictionaries

    Returns:
        dict: Aggregated statistics (mean of each metric)
    """
    if not stats_list:
        return {}

    keys = ['mean_brightness', 'std_brightness', 'mean_b', 'mean_g', 'mean_r']
    result = {}
    for key in keys:
        values = [s[key] for s in stats_list if key in s]
        if values:
            result[key] = np.mean(values)
            result[f'{key}_std'] = np.std(values)

    return result


def generate_report(output_dir, day_stats_agg, synthetic_stats_agg, real_night_stats_agg, num_samples,
                    enhanced_stats_agg=None):
    """
    Generate a markdown report with findings.

    Args:
        output_dir (str): Output directory
        day_stats_agg (dict): Aggregated day statistics
        synthetic_stats_agg (dict): Dict mapping preset to aggregated stats
        real_night_stats_agg (dict): Aggregated real night statistics
        num_samples (int): Number of samples processed
        enhanced_stats_agg (dict): Dict mapping enhanced preset to aggregated stats
    """
    report_path = os.path.join(output_dir, 'report.md')

    with open(report_path, 'w') as f:
        f.write("# Night Augmentation Visual Validation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Samples analyzed: {num_samples}\n\n")

        f.write("## Methodology\n\n")
        f.write("The `SimulateNight` augmentation applies the following transformations in sequence:\n\n")
        f.write("1. **Gamma correction**: Darkens shadows more than highlights (`img = 255 * (img/255)^γ`)\n")
        f.write("2. **Brightness reduction**: Multiplies pixel values by a factor < 1\n")
        f.write("3. **Contrast reduction**: Reduces deviation from mean\n")
        f.write("4. **Gaussian noise**: Simulates high-ISO sensor noise\n")
        f.write("5. **Color shift (optional)**: Adds orange tint (streetlight effect)\n\n")

        f.write("### Basic Augmentation Presets\n\n")
        f.write("| Preset | Gamma | Brightness | Contrast | Noise Std | Color Shift |\n")
        f.write("|--------|-------|------------|----------|-----------|-------------|\n")
        for name, preset in AUGMENTATION_PRESETS.items():
            f.write(f"| {name.capitalize()} | {preset['gamma']} | {preset['brightness']} | "
                    f"{preset['contrast']} | {preset['noise_std']} | {preset['color_shift_strength']} |\n")
        f.write("\n")

        f.write("### Enhanced Augmentation Presets\n\n")
        f.write("Enhanced presets add: vignetting, headlight gradient, random bright spots, and bright region preservation.\n\n")
        f.write("| Preset | Gamma | Brightness | Vignette | Headlight | Spots | Preserve Bright |\n")
        f.write("|--------|-------|------------|----------|-----------|-------|----------------|\n")
        for name, preset in AUGMENTATION_PRESETS_ENHANCED.items():
            label = name.replace('enhanced_', '').capitalize()
            f.write(f"| {label} | {preset['gamma']} | {preset['brightness']} | "
                    f"{preset['vignette_strength']} | {preset['headlight_strength']} | "
                    f"{preset['num_bright_spots']} | {preset.get('preserve_bright', False)} |\n")
        f.write("\n")

        f.write("## Statistical Comparison\n\n")
        f.write("### Basic Presets\n\n")
        f.write("| Metric | Day | Synth (Light) | Synth (Medium) | Synth (Heavy) | Real Night |\n")
        f.write("|--------|-----|---------------|----------------|---------------|------------|\n")

        metrics = ['mean_brightness', 'std_brightness', 'mean_r', 'mean_g', 'mean_b']
        metric_labels = ['Mean Brightness', 'Std Dev', 'Mean Red', 'Mean Green', 'Mean Blue']

        for metric, label in zip(metrics, metric_labels):
            row = f"| {label} | {day_stats_agg.get(metric, 'N/A'):.1f} |"
            for preset in ['light', 'medium', 'heavy']:
                if preset in synthetic_stats_agg and metric in synthetic_stats_agg[preset]:
                    row += f" {synthetic_stats_agg[preset][metric]:.1f} |"
                else:
                    row += " N/A |"
            if real_night_stats_agg and metric in real_night_stats_agg:
                row += f" {real_night_stats_agg[metric]:.1f} |"
            else:
                row += " N/A |"
            f.write(row + "\n")

        if enhanced_stats_agg:
            f.write("\n### Enhanced Presets\n\n")
            f.write("| Metric | Day | Enhanced Light | Enhanced Medium | Enhanced Heavy | Real Night |\n")
            f.write("|--------|-----|----------------|-----------------|----------------|------------|\n")

            for metric, label in zip(metrics, metric_labels):
                row = f"| {label} | {day_stats_agg.get(metric, 'N/A'):.1f} |"
                for preset in ['enhanced_light', 'enhanced_medium', 'enhanced_heavy']:
                    if preset in enhanced_stats_agg and metric in enhanced_stats_agg[preset]:
                        row += f" {enhanced_stats_agg[preset][metric]:.1f} |"
                    else:
                        row += " N/A |"
                if real_night_stats_agg and metric in real_night_stats_agg:
                    row += f" {real_night_stats_agg[metric]:.1f} |"
                else:
                    row += " N/A |"
                f.write(row + "\n")

        f.write("\n## Visual Comparisons\n\n")
        f.write("### Side-by-side Comparisons\n\n")
        for i in range(min(5, num_samples)):
            f.write(f"![Comparison {i+1}](comparison_sample_{i+1:03d}.png)\n\n")

        f.write("### Intensity Comparison (Basic)\n\n")
        f.write("![Intensity comparison](intensity_comparison.png)\n\n")

        f.write("### Basic vs Enhanced Comparison\n\n")
        f.write("![Enhanced comparison](enhanced_comparison.png)\n\n")

        f.write("### Feature Breakdown\n\n")
        f.write("Shows individual effect of each enhanced feature:\n")
        f.write("![Feature breakdown](feature_breakdown.png)\n\n")

        f.write("### Multi-camera Comparison\n\n")
        f.write("![Multi-camera comparison](multicam_comparison_001.png)\n\n")

        f.write("### Histogram Comparison\n\n")
        f.write("![Histogram comparison](histogram_comparison.png)\n\n")

        f.write("## Findings\n\n")

        # Analyze which preset is closest to real night
        if real_night_stats_agg and 'mean_brightness' in real_night_stats_agg:
            real_mean = real_night_stats_agg['mean_brightness']
            real_std = real_night_stats_agg.get('std_brightness', 0)

            # Find best basic preset
            best_basic = None
            best_basic_diff = float('inf')
            for preset_name, stats in synthetic_stats_agg.items():
                if 'mean_brightness' in stats:
                    diff = abs(stats['mean_brightness'] - real_mean)
                    if diff < best_basic_diff:
                        best_basic_diff = diff
                        best_basic = preset_name

            f.write(f"### Basic Presets\n\n")
            f.write(f"- **Best matching basic preset**: {best_basic.capitalize() if best_basic else 'N/A'}\n")
            f.write(f"  - Real night mean brightness: {real_mean:.1f}\n")
            if best_basic and best_basic in synthetic_stats_agg:
                f.write(f"  - {best_basic.capitalize()} synthetic mean brightness: "
                        f"{synthetic_stats_agg[best_basic].get('mean_brightness', 'N/A'):.1f}\n")
                f.write(f"  - Difference: {best_basic_diff:.1f}\n")
            f.write("\n")

            # Find best enhanced preset
            if enhanced_stats_agg:
                best_enhanced = None
                best_enhanced_diff = float('inf')
                best_enhanced_std_diff = float('inf')

                for preset_name, stats in enhanced_stats_agg.items():
                    if 'mean_brightness' in stats:
                        diff = abs(stats['mean_brightness'] - real_mean)
                        std_diff = abs(stats.get('std_brightness', 0) - real_std)
                        # Consider both brightness and contrast matching
                        combined_diff = diff + std_diff * 0.5
                        if combined_diff < best_enhanced_diff + best_enhanced_std_diff * 0.5:
                            best_enhanced_diff = diff
                            best_enhanced_std_diff = std_diff
                            best_enhanced = preset_name

                f.write(f"### Enhanced Presets\n\n")
                f.write(f"- **Best matching enhanced preset**: {best_enhanced.replace('enhanced_', '').capitalize() if best_enhanced else 'N/A'}\n")
                f.write(f"  - Real night: mean={real_mean:.1f}, std={real_std:.1f}\n")
                if best_enhanced and best_enhanced in enhanced_stats_agg:
                    enh_mean = enhanced_stats_agg[best_enhanced].get('mean_brightness', 0)
                    enh_std = enhanced_stats_agg[best_enhanced].get('std_brightness', 0)
                    f.write(f"  - {best_enhanced.replace('enhanced_', '').capitalize()} enhanced: "
                            f"mean={enh_mean:.1f}, std={enh_std:.1f}\n")
                    f.write(f"  - Mean diff: {best_enhanced_diff:.1f}, Std diff: {best_enhanced_std_diff:.1f}\n")

                # Compare basic vs enhanced
                f.write(f"\n### Comparison\n\n")
                if best_basic in synthetic_stats_agg and best_enhanced in enhanced_stats_agg:
                    basic_std = synthetic_stats_agg[best_basic].get('std_brightness', 0)
                    enhanced_std = enhanced_stats_agg[best_enhanced].get('std_brightness', 0)
                    f.write(f"- Basic {best_basic} std dev: {basic_std:.1f} (real: {real_std:.1f}, gap: {abs(basic_std - real_std):.1f})\n")
                    f.write(f"- Enhanced {best_enhanced.replace('enhanced_', '')} std dev: {enhanced_std:.1f} (real: {real_std:.1f}, gap: {abs(enhanced_std - real_std):.1f})\n")
                    if abs(enhanced_std - real_std) < abs(basic_std - real_std):
                        f.write(f"- **Enhanced presets better match real night contrast!**\n")
            f.write("\n")

        f.write("## Recommendations\n\n")
        f.write("Based on the analysis:\n\n")
        f.write("1. **For training**: Consider using the preset that best matches real night statistics\n")
        f.write("2. **Augmentation probability**: Start with `prob=0.3` to augment ~30% of training samples\n")
        f.write("3. **Multi-intensity training**: Consider randomly sampling from light/medium/heavy presets\n")
        f.write("4. **Fine-tuning**: Adjust parameters based on visual inspection of comparison images\n\n")

        f.write("## Usage\n\n")
        f.write("### Basic Usage\n\n")
        f.write("```python\n")
        f.write("# Add to training pipeline config (basic):\n")
        f.write("train_pipeline = [\n")
        f.write("    ...\n")
        f.write("    dict(type='SimulateNight',\n")
        f.write("         prob=0.3,\n")
        f.write("         brightness_range=(0.4, 0.6),\n")
        f.write("         contrast_range=(0.65, 0.85),\n")
        f.write("         noise_std_range=(5, 15),\n")
        f.write("         gamma_range=(1.3, 1.8),\n")
        f.write("         color_shift=True),\n")
        f.write("    ...\n")
        f.write("]\n")
        f.write("```\n\n")
        f.write("### Enhanced Usage (Recommended)\n\n")
        f.write("```python\n")
        f.write("# Add to training pipeline config (enhanced realism):\n")
        f.write("train_pipeline = [\n")
        f.write("    ...\n")
        f.write("    dict(type='SimulateNight',\n")
        f.write("         prob=0.3,\n")
        f.write("         brightness_range=(0.4, 0.55),\n")
        f.write("         contrast_range=(0.65, 0.8),\n")
        f.write("         noise_std_range=(5, 12),\n")
        f.write("         gamma_range=(1.4, 1.9),\n")
        f.write("         color_shift=True,\n")
        f.write("         color_shift_strength=(0.1, 0.2),\n")
        f.write("         # Enhanced features\n")
        f.write("         vignette=True,\n")
        f.write("         vignette_strength=(0.3, 0.5),\n")
        f.write("         headlight_gradient=True,\n")
        f.write("         headlight_strength=(0.3, 0.5),\n")
        f.write("         headlight_height=0.4,\n")
        f.write("         random_bright_spots=True,\n")
        f.write("         num_bright_spots=(3, 6),\n")
        f.write("         spot_brightness=(150, 240),\n")
        f.write("         spot_size_range=(12, 40),\n")
        f.write("         preserve_bright=True,\n")
        f.write("         bright_threshold=200,\n")
        f.write("         bright_preserve_factor=0.6),\n")
        f.write("    ...\n")
        f.write("]\n")
        f.write("```\n")

    print(f"Report saved to: {report_path}")


def run_with_sample_images(output_dir, num_samples=10):
    """
    Run visualization using generated sample images (when nuScenes is unavailable).

    This creates synthetic test images to demonstrate the augmentation behavior.
    """
    print("Running with synthetic sample images (nuScenes unavailable)...")

    os.makedirs(output_dir, exist_ok=True)

    # Create synthetic day images with different characteristics
    np.random.seed(42)

    # Generate sample "day" images with gradients and patterns
    sample_images = []
    for i in range(num_samples):
        # Create a gradient + noise image to simulate outdoor scene
        h, w = 256, 704
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Sky gradient (top portion)
        sky_height = h // 3
        for y in range(sky_height):
            brightness = 200 - y * 0.5
            img[y, :, 0] = brightness * 0.8  # Blue
            img[y, :, 1] = brightness * 0.9  # Green
            img[y, :, 2] = brightness * 0.95  # Red

        # Ground gradient (bottom portion)
        for y in range(sky_height, h):
            brightness = 100 + (y - sky_height) * 0.3
            img[y, :, 0] = brightness * 0.6
            img[y, :, 1] = brightness * 0.7
            img[y, :, 2] = brightness * 0.65

        # Add some random variation
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add some "features" (rectangles simulating objects)
        num_features = np.random.randint(3, 8)
        for _ in range(num_features):
            x1 = np.random.randint(0, w - 50)
            y1 = np.random.randint(sky_height, h - 30)
            x2 = x1 + np.random.randint(30, 100)
            y2 = y1 + np.random.randint(20, 50)
            color = np.random.randint(50, 200, 3).tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        sample_images.append(img)

    # Create a "real night" synthetic image for reference
    real_night_img = sample_images[0].copy()
    real_night_img = (real_night_img * 0.25).astype(np.uint8)
    noise = np.random.normal(0, 15, real_night_img.shape)
    real_night_img = np.clip(real_night_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Collect statistics
    day_stats_list = []
    synthetic_stats = {'light': [], 'medium': [], 'heavy': []}
    enhanced_stats = {'enhanced_light': [], 'enhanced_medium': [], 'enhanced_heavy': []}

    print(f"Processing {num_samples} synthetic samples...")

    for i, day_img in enumerate(sample_images):
        print(f"  Processing sample {i + 1}/{num_samples}")

        # Compute day stats
        day_stats = compute_image_stats(day_img)
        day_stats_list.append(day_stats)

        # Generate synthetic night versions (basic)
        synthetic_images = {}
        for preset_name, preset_params in AUGMENTATION_PRESETS.items():
            synth_img = apply_simulate_night(day_img, **preset_params)
            synthetic_images[preset_name] = synth_img
            synthetic_stats[preset_name].append(compute_image_stats(synth_img))

        # Generate synthetic night versions (enhanced)
        for preset_name, preset_params in AUGMENTATION_PRESETS_ENHANCED.items():
            synth_img = apply_simulate_night(day_img, **preset_params)
            enhanced_stats[preset_name].append(compute_image_stats(synth_img))

        # Create comparison figure
        fig = create_comparison_figure(
            day_img, synthetic_images, real_night_img if i == 0 else None,
            title=f'Sample {i + 1}: Day vs Synthetic Night Comparison'
        )
        fig.savefig(os.path.join(output_dir, f'comparison_sample_{i + 1:03d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Create intensity comparison
    print("Creating intensity comparison...")
    fig = create_intensity_comparison(sample_images[0], 'Augmentation Intensity Comparison (Basic)')
    fig.savefig(os.path.join(output_dir, 'intensity_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Create enhanced comparison
    print("Creating enhanced vs basic comparison...")
    fig = create_enhanced_comparison(sample_images[0], real_night_img,
                                     'Basic vs Enhanced Augmentation Comparison')
    fig.savefig(os.path.join(output_dir, 'enhanced_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Create feature breakdown
    print("Creating feature breakdown...")
    fig = create_feature_breakdown(sample_images[0], 'Individual Feature Effects')
    fig.savefig(os.path.join(output_dir, 'feature_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Create multi-camera comparison (simulate with same image for all cameras)
    print("Creating multi-camera comparison...")
    day_multicam = {cam: sample_images[0] for cam in CAMERA_CHANNELS}
    synth_multicam = {cam: apply_simulate_night(sample_images[0], **AUGMENTATION_PRESETS['medium'])
                      for cam in CAMERA_CHANNELS}
    real_night_multicam = {cam: real_night_img for cam in CAMERA_CHANNELS}

    fig = create_multicam_comparison(
        day_multicam, synth_multicam, real_night_multicam,
        title='Multi-Camera View Comparison (Synthetic Data)'
    )
    fig.savefig(os.path.join(output_dir, 'multicam_comparison_001.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Create histogram comparison
    print("Creating histogram comparison...")
    day_stats_agg = aggregate_statistics(day_stats_list)
    synthetic_stats_agg = {k: aggregate_statistics(v) for k, v in synthetic_stats.items()}
    real_night_stats = compute_image_stats(real_night_img)

    fig = create_histogram_comparison(
        day_stats_list[0],
        {k: synthetic_stats[k][0] for k in synthetic_stats},
        real_night_stats,
        title='Brightness Distribution Comparison'
    )
    fig.savefig(os.path.join(output_dir, 'histogram_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Aggregate enhanced stats
    enhanced_stats_agg = {k: aggregate_statistics(v) for k, v in enhanced_stats.items()}

    # Save statistics
    print("Saving statistics...")
    stats_data = {
        'day': day_stats_agg,
        'synthetic_basic': synthetic_stats_agg,
        'synthetic_enhanced': enhanced_stats_agg,
        'real_night': {k: v for k, v in real_night_stats.items() if k != 'histogram'}
    }
    # Convert to serializable format
    stats_serializable = {}
    for category, data in stats_data.items():
        if isinstance(data, dict):
            stats_serializable[category] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in data.items()
                if not isinstance(v, np.ndarray)
            }

    with open(os.path.join(output_dir, 'statistics_summary.json'), 'w') as f:
        json.dump(stats_serializable, f, indent=2)

    # Generate report
    print("Generating report...")
    generate_report(output_dir, day_stats_agg, synthetic_stats_agg,
                    {k: v for k, v in real_night_stats.items() if k != 'histogram'},
                    num_samples, enhanced_stats_agg)

    print(f"\nVisualization complete! Results saved to: {output_dir}")
    print("\nNote: Running with synthetic images since nuScenes data is unavailable.")
    print("Re-run with valid nuScenes data path for accurate real-world comparison.")


def run_with_nuscenes(nusc, output_dir, num_samples=10, seed=42):
    """
    Run visualization using actual nuScenes data.

    Args:
        nusc: nuScenes dataset object
        output_dir (str): Output directory
        num_samples (int): Number of samples to visualize
        seed (int): Random seed for reproducibility
    """
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Identify day/night samples
    day_samples, night_samples = identify_day_night_samples(nusc)

    if len(day_samples) == 0:
        print("ERROR: No day samples found!")
        return

    # Randomly select samples
    selected_day = np.random.choice(day_samples, min(num_samples, len(day_samples)), replace=False)
    selected_night = np.random.choice(night_samples, min(num_samples, len(night_samples)), replace=False) \
        if len(night_samples) > 0 else []

    # Collect statistics
    day_stats_list = []
    night_stats_list = []
    synthetic_stats = {'light': [], 'medium': [], 'heavy': []}
    enhanced_stats = {'enhanced_light': [], 'enhanced_medium': [], 'enhanced_heavy': []}

    print(f"\nProcessing {len(selected_day)} day samples and {len(selected_night)} night samples...")

    for i, day_token in enumerate(selected_day):
        print(f"  Processing day sample {i + 1}/{len(selected_day)}")

        # Load day images
        day_images = load_sample_images(nusc, day_token)
        if 'CAM_FRONT' not in day_images:
            print(f"    Warning: CAM_FRONT not found for sample {day_token}, skipping")
            continue

        day_img = day_images['CAM_FRONT']
        day_stats = compute_image_stats(day_img)
        day_stats_list.append(day_stats)

        # Generate synthetic night versions (basic)
        synthetic_images = {}
        for preset_name, preset_params in AUGMENTATION_PRESETS.items():
            synth_img = apply_simulate_night(day_img, **preset_params)
            synthetic_images[preset_name] = synth_img
            synthetic_stats[preset_name].append(compute_image_stats(synth_img))

        # Generate synthetic night versions (enhanced)
        for preset_name, preset_params in AUGMENTATION_PRESETS_ENHANCED.items():
            synth_img = apply_simulate_night(day_img, **preset_params)
            enhanced_stats[preset_name].append(compute_image_stats(synth_img))

        # Get corresponding real night image if available
        real_night_img = None
        if i < len(selected_night):
            night_images = load_sample_images(nusc, selected_night[i])
            if 'CAM_FRONT' in night_images:
                real_night_img = night_images['CAM_FRONT']
                if i < num_samples:  # Only collect stats for matched pairs
                    night_stats_list.append(compute_image_stats(real_night_img))

        # Create comparison figure
        fig = create_comparison_figure(
            day_img, synthetic_images, real_night_img,
            title=f'Sample {i + 1}: Day vs Synthetic Night vs Real Night'
        )
        fig.savefig(os.path.join(output_dir, f'comparison_sample_{i + 1:03d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Create intensity comparison using first day sample
    if day_stats_list:
        print("Creating intensity comparison...")
        first_day_img = load_sample_images(nusc, selected_day[0]).get('CAM_FRONT')
        first_night_img = None
        if len(selected_night) > 0:
            first_night_img = load_sample_images(nusc, selected_night[0]).get('CAM_FRONT')

        if first_day_img is not None:
            fig = create_intensity_comparison(first_day_img, 'Augmentation Intensity Comparison (Basic)')
            fig.savefig(os.path.join(output_dir, 'intensity_comparison.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Create enhanced comparison
            print("Creating enhanced vs basic comparison...")
            fig = create_enhanced_comparison(first_day_img, first_night_img,
                                             'Basic vs Enhanced Augmentation Comparison')
            fig.savefig(os.path.join(output_dir, 'enhanced_comparison.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Create feature breakdown
            print("Creating feature breakdown...")
            fig = create_feature_breakdown(first_day_img, 'Individual Feature Effects')
            fig.savefig(os.path.join(output_dir, 'feature_breakdown.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)

    # Create multi-camera comparison
    if len(selected_day) > 0:
        print("Creating multi-camera comparison...")
        day_multicam = load_sample_images(nusc, selected_day[0])
        synth_multicam = {cam: apply_simulate_night(img, **AUGMENTATION_PRESETS['medium'])
                         for cam, img in day_multicam.items()}

        real_night_multicam = None
        if len(selected_night) > 0:
            real_night_multicam = load_sample_images(nusc, selected_night[0])

        fig = create_multicam_comparison(
            day_multicam, synth_multicam, real_night_multicam,
            title='Multi-Camera View Comparison'
        )
        fig.savefig(os.path.join(output_dir, 'multicam_comparison_001.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Create histogram comparison
    if day_stats_list:
        print("Creating histogram comparison...")
        real_night_stats_for_hist = night_stats_list[0] if night_stats_list else None

        fig = create_histogram_comparison(
            day_stats_list[0],
            {k: synthetic_stats[k][0] for k in synthetic_stats if synthetic_stats[k]},
            real_night_stats_for_hist,
            title='Brightness Distribution Comparison'
        )
        fig.savefig(os.path.join(output_dir, 'histogram_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Aggregate statistics
    day_stats_agg = aggregate_statistics(day_stats_list)
    synthetic_stats_agg = {k: aggregate_statistics(v) for k, v in synthetic_stats.items()}
    enhanced_stats_agg = {k: aggregate_statistics(v) for k, v in enhanced_stats.items()}
    real_night_stats_agg = aggregate_statistics(night_stats_list) if night_stats_list else None

    # Save statistics
    print("Saving statistics...")
    stats_data = {
        'day': day_stats_agg,
        'synthetic_basic': synthetic_stats_agg,
        'synthetic_enhanced': enhanced_stats_agg,
        'real_night': real_night_stats_agg or {}
    }
    stats_serializable = {}
    for category, data in stats_data.items():
        if isinstance(data, dict):
            stats_serializable[category] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in data.items()
                if not isinstance(v, np.ndarray)
            }

    with open(os.path.join(output_dir, 'statistics_summary.json'), 'w') as f:
        json.dump(stats_serializable, f, indent=2)

    # Generate report
    print("Generating report...")
    generate_report(output_dir, day_stats_agg, synthetic_stats_agg, real_night_stats_agg,
                    len(selected_day), enhanced_stats_agg)

    print(f"\nVisualization complete! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize night augmentation and compare with real nuScenes night images'
    )
    parser.add_argument('--output', '-o', type=str, default='outputs/night_aug_validation/',
                        help='Output directory for visualizations')
    parser.add_argument('--dataroot', type=str, default='data/nuscenes/',
                        help='Path to nuScenes data root')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='nuScenes version (default: v1.0-trainval)')
    parser.add_argument('--num_samples', '-n', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--force_synthetic', action='store_true',
                        help='Force use of synthetic images even if nuScenes is available')

    args = parser.parse_args()

    print("=" * 60)
    print("Night Augmentation Visual Validation Tool")
    print("=" * 60)

    # Try to load nuScenes
    nusc = None
    if not args.force_synthetic:
        nusc = load_nuscenes(args.dataroot, args.version)

    if nusc is not None:
        run_with_nuscenes(nusc, args.output, args.num_samples, args.seed)
    else:
        run_with_sample_images(args.output, args.num_samples)


if __name__ == '__main__':
    main()
