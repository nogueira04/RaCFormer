"""
Inference profiling script for RaCFormer model.
Uses PyTorch hooks to time individual modules during inference.
"""

"""
Inference profiling script for RaCFormer model.
Uses PyTorch hooks to time individual modules during inference.

Optimizations included:
- Tier 1.1: Async DataLoader with num_workers, pin_memory, prefetch_factor
- Tier 1.2: FP16 Inference with torch.cuda.amp.autocast
- Tier 1.4: Non-blocking memory transfers
"""

import os
import sys
import time
import argparse
import importlib
import numpy as np
import torch
import torch.cuda
import torch.backends.cudnn as cudnn
from collections import defaultdict
from torch.utils.data import DataLoader

os.environ['NUSCENES_VERSION'] = os.environ.get('NUSCENES_VERSION', 'v1.0-mini')

from mmengine.config import Config
from mmengine.runner import load_checkpoint, set_random_seed
from torch.nn.parallel import DataParallel as MMDataParallel

from mmdet3d.registry import MODELS, DATASETS


def build_dataset(cfg, default_args=None):
    return DATASETS.build(cfg, default_args=default_args)


def pseudo_collate(batch):
    return batch


def build_model(cfg, train_cfg=None, test_cfg=None):
    return MODELS.build(cfg)


class ModuleProfiler:
    """Profile individual modules using PyTorch hooks."""

    def __init__(self):
        self.timings = defaultdict(list)
        self.hooks = []
        self.start_times = {}

    def register_hooks(self, model):
        """Register timing hooks on key modules."""
        if hasattr(model, 'module'):
            m = model.module
        else:
            m = model

        # Define modules to profile (avoiding nested modules to prevent double-counting)
        modules_to_time = {
            '01. Grid Mask': m.grid_mask if hasattr(m, 'grid_mask') else None,
            '02. Image Backbone': m.img_backbone,
            '03. Image Neck (FPN)': m.img_neck,
            '04. Radar Voxel Layer': m.radar_voxel_layer if hasattr(m, 'radar_voxel_layer') else None,
            '05. Radar Voxel Encoder': m.radar_voxel_encoder,
            '06. Radar Middle Encoder': m.radar_middle_encoder,
            '07. Radar BEV Conv': m.radar_bev_conv,
            '08. View Transformer': m.img_lss_view_transformer,
            # Note: Only profile the transformer inside detection head (not the whole head)
            # to avoid double-counting. The head's other operations are minimal.
            '09. Det Head Transformer': m.pts_bbox_head.transformer if hasattr(m.pts_bbox_head, 'transformer') else None,
        }

        # Add LSS neck if it exists and is different from img_neck
        if hasattr(m, 'img_lss_neck') and m.img_lss_neck is not None:
            if m.img_lss_neck is not m.img_neck:
                modules_to_time['03b. LSS Neck'] = m.img_lss_neck

        # Add pre_process_net if it exists
        if hasattr(m, 'pre_process_net') and m.pre_process_net is not None:
            modules_to_time['08b. Pre-process Net'] = m.pre_process_net

        for name, module in modules_to_time.items():
            if module is not None:
                def make_pre_hook(n):
                    def hook(mod, inp):
                        torch.cuda.synchronize()
                        self.start_times[n] = time.perf_counter()
                    return hook

                def make_post_hook(n):
                    def hook(mod, inp, out):
                        torch.cuda.synchronize()
                        if n in self.start_times:
                            elapsed = (time.perf_counter() - self.start_times[n]) * 1000
                            self.timings[n].append(elapsed)
                    return hook

                h1 = module.register_forward_pre_hook(make_pre_hook(name))
                h2 = module.register_forward_hook(make_post_hook(name))
                self.hooks.extend([h1, h2])

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_stats(self):
        stats = {}
        for name, times in self.timings.items():
            if len(times) > 0:
                stats[name] = {
                    'mean_ms': np.mean(times),
                    'std_ms': np.std(times),
                    'min_ms': np.min(times),
                    'max_ms': np.max(times),
                    'count': len(times),
                }
        return stats

    def print_report(self, num_samples=10):
        stats = self.get_stats()

        print("\n" + "="*80)
        print("MODULE TIMING REPORT")
        print("="*80)
        print(f"{'Module':<35} {'Mean/call':<12} {'Calls/inf':<10} {'Total/inf':<12} {'%':>6}")
        print("-"*80)

        # Calculate total time per inference
        module_totals = {}
        for name, data in stats.items():
            calls_per_inference = data['count'] / num_samples
            total_per_inference = data['mean_ms'] * calls_per_inference
            module_totals[name] = {
                'mean_per_call': data['mean_ms'],
                'std_ms': data['std_ms'],
                'calls_per_inf': calls_per_inference,
                'total_per_inf': total_per_inference,
            }

        grand_total = sum(m['total_per_inf'] for m in module_totals.values())

        sorted_modules = sorted(module_totals.items(), key=lambda x: x[1]['total_per_inf'], reverse=True)
        for name, data in sorted_modules:
            pct = (data['total_per_inf'] / grand_total * 100) if grand_total > 0 else 0
            print(f"{name:<35} {data['mean_per_call']:>10.2f}  {data['calls_per_inf']:>8.0f}x  {data['total_per_inf']:>10.2f}  {pct:>6.1f}%")

        print("-"*80)
        print(f"{'TOTAL (measured modules)':<35} {'':<12} {'':<10} {grand_total:>10.2f} ms")
        print("="*80)

        return stats


class InferenceTimer:
    """Simple timer for full inference with optional mixed precision support."""

    def __init__(self, autocast_dtype=None):
        self.times = []
        self.autocast_dtype = autocast_dtype  # None, torch.float16, or torch.bfloat16

    def time_inference(self, model, data):
        if hasattr(model, 'module'):
            m = model.module
        else:
            m = model

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            # Tier 1.2: Mixed Precision Inference with autocast
            if self.autocast_dtype is not None:
                with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
                    result = m(return_loss=False, rescale=True, **data)
            else:
                result = m(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        self.times.append(elapsed)

        return result

    def get_stats(self):
        if not self.times:
            return {}
        return {
            'mean_ms': np.mean(self.times),
            'std_ms': np.std(self.times),
            'min_ms': np.min(self.times),
            'max_ms': np.max(self.times),
            'count': len(self.times),
        }


def move_to_device(data, device, non_blocking=True):
    """Move data tensors to the specified device.

    Optimization 1.4: Use non_blocking=True for async CPU->GPU transfers.
    Uses single .to(device, dtype) call to avoid double copy.
    """
    for key in ['img', 'radar_depth', 'radar_rcs']:
        if key in data:
            if isinstance(data[key], list):
                data[key] = [d.to(device, dtype=torch.float32, non_blocking=non_blocking) if isinstance(d, torch.Tensor) else d for d in data[key]]
            elif isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device, dtype=torch.float32, non_blocking=non_blocking)
    if 'radar_points' in data:
        if isinstance(data['radar_points'], list):
            new_pts = []
            for pts in data['radar_points']:
                if isinstance(pts, list):
                    new_pts.append([p.to(device, dtype=torch.float32, non_blocking=non_blocking) if isinstance(p, torch.Tensor) else p for p in pts])
                elif isinstance(pts, torch.Tensor):
                    new_pts.append(pts.to(device, dtype=torch.float32, non_blocking=non_blocking))
                else:
                    new_pts.append(pts)
            data['radar_points'] = new_pts
    return data


def run_torch_profiler(model, data_loader, num_samples, device, move_to_device_fn):
    """Run PyTorch's built-in profiler for detailed GPU/CPU analysis."""
    from torch.profiler import profile, record_function, ProfilerActivity

    print("\n" + "="*70)
    print("PYTORCH PROFILER (kernel-level analysis)")
    print("="*70)

    if hasattr(model, 'module'):
        m = model.module
    else:
        m = model

    data_iter = iter(data_loader)

    # Warmup
    print("Warming up...")
    for _ in range(2):
        batch = next(data_iter)
        sample = batch[0]
        if isinstance(sample, (list, tuple)):
            while isinstance(sample, (list, tuple)) and len(sample) == 1:
                sample = sample[0]
        data = sample
        for key in ['img_metas', 'img', 'radar_points', 'radar_depth', 'radar_rcs']:
            if key in data and not isinstance(data[key], list):
                data[key] = [data[key]]
        data = move_to_device_fn(data, device)
        with torch.no_grad():
            _ = m(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize()

    # Profile
    print(f"Profiling {num_samples} samples with PyTorch profiler...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for i in range(num_samples):
            batch = next(data_iter)
            sample = batch[0]
            if isinstance(sample, (list, tuple)):
                while isinstance(sample, (list, tuple)) and len(sample) == 1:
                    sample = sample[0]
            data = sample
            for key in ['img_metas', 'img', 'radar_points', 'radar_depth', 'radar_rcs']:
                if key in data and not isinstance(data[key], list):
                    data[key] = [data[key]]
            data = move_to_device_fn(data, device)

            with record_function("model_inference"):
                with torch.no_grad():
                    _ = m(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            print(f"  Sample {i+1}/{num_samples} done")

    # Print results
    print("\n" + "="*70)
    print("TOP 20 CUDA OPERATIONS (by total GPU time)")
    print("="*70)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("\n" + "="*70)
    print("TOP 20 OPERATIONS (by self CPU time)")
    print("="*70)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    # Export chrome trace
    trace_path = "profile_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace saved to {trace_path}")
    print("Open chrome://tracing in Chrome and load this file for visual timeline")

    # Also save a summary table
    summary_path = "profile_torch_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TOP 50 CUDA OPERATIONS (by total GPU time)\n")
        f.write("="*80 + "\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("TOP 50 OPERATIONS (by self CPU time)\n")
        f.write("="*80 + "\n")
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=50))
    print(f"Full summary saved to {summary_path}")

    return prof


def main():
    parser = argparse.ArgumentParser(description='Profile RaCFormer inference')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to profile')
    parser.add_argument('--warmup', type=int, default=3, help='Number of warmup iterations')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--detailed', action='store_true', help='Run detailed per-module profiling')
    parser.add_argument('--torch-profiler', action='store_true', help='Use PyTorch profiler for kernel-level analysis')
    # Tier 1 optimization flags
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 inference with autocast (Tier 1.2)')
    parser.add_argument('--bf16', action='store_true', help='Enable BF16 inference (requires Ampere+, e.g., Orin)')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader num_workers (Tier 1.1, 0=disabled)')
    parser.add_argument('--no-pin-memory', action='store_true', help='Disable pin_memory for DataLoader')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='DataLoader prefetch_factor (Tier 1.1)')
    parser.add_argument('--no-optimizations', action='store_true', help='Disable all Tier 1 optimizations for baseline')
    parser.add_argument('--trt-backbone', type=str, default=None,
                        help='Path to TensorRT engine for backbone+FPN (Tier 2.1)')
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    cfg = Config.fromfile(args.config)

    # Register custom modules (exactly like val.py)
    importlib.import_module('models')
    importlib.import_module('loaders')
    import mmdet3d.datasets.transforms

    # Copy registries
    from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
    from mmdet3d.registry import TRANSFORMS as MMDET3D_TRANSFORMS
    for name, module in MMDET3D_TRANSFORMS.module_dict.items():
        if name not in MMENGINE_TRANSFORMS.module_dict:
            MMENGINE_TRANSFORMS.register_module(name=name, module=module)

    from mmdet.registry import MODELS as MMDET_MODELS
    from mmdet3d.registry import MODELS as MMDET3D_MODELS
    for name, module in MMDET_MODELS.module_dict.items():
        if name not in MMDET3D_MODELS.module_dict:
            MMDET3D_MODELS.register_module(name=name, module=module)

    from mmdet.registry import TASK_UTILS as MMDET_TASK_UTILS
    from mmdet3d.registry import TASK_UTILS as MMDET3D_TASK_UTILS
    for name, module in MMDET_TASK_UTILS.module_dict.items():
        if name not in MMDET3D_TASK_UTILS.module_dict:
            MMDET3D_TASK_UTILS.register_module(name=name, module=module)
    for name, module in MMDET3D_TASK_UTILS.module_dict.items():
        if name not in MMDET_TASK_UTILS.module_dict:
            MMDET_TASK_UTILS.register_module(name=name, module=module)

    import logging
    logging.getLogger('mmengine').setLevel(logging.WARNING)
    logging.getLogger('mmcv').setLevel(logging.WARNING)

    assert torch.cuda.is_available()
    set_random_seed(0, deterministic=False)
    cudnn.benchmark = True

    # Build model
    print("Building model...")
    model = build_model(cfg.model)

    print(f"Loading checkpoint from {args.checkpoint}")
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = model.to(args.device)
    model.eval()

    # Tier 2.1: TensorRT backbone+FPN
    if args.trt_backbone:
        import os
        if not os.path.exists(args.trt_backbone):
            print(f"ERROR: TRT engine not found: {args.trt_backbone}")
            sys.exit(1)
        from export_tensorrt import TRTBackboneNeck, patch_extract_img_feat
        trt_engine = TRTBackboneNeck(args.trt_backbone, device=args.device)
        model = patch_extract_img_feat(model, trt_engine)
        print(f"  Tier 2.1 TensorRT backbone+FPN: {args.trt_backbone}")

    # Build dataloader with Tier 1.1 optimizations
    print("Building dataloader...")
    val_dataset = build_dataset(cfg.data.val)

    # Tier 1.1: Async DataLoader configuration
    if args.no_optimizations:
        num_workers = 0
        pin_memory = False
        prefetch_factor = None  # Only valid when num_workers > 0
        print("  DataLoader: baseline mode (num_workers=0, no pin_memory)")
    else:
        num_workers = args.num_workers
        pin_memory = not args.no_pin_memory
        prefetch_factor = args.prefetch_factor if num_workers > 0 else None
        print(f"  DataLoader: num_workers={num_workers}, pin_memory={pin_memory}, prefetch_factor={prefetch_factor}")

    loader_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pseudo_collate,
        pin_memory=pin_memory,
    )
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor

    val_loader = DataLoader(val_dataset, **loader_kwargs)

    # Tier 1.2: Mixed precision configuration
    use_fp16 = args.fp16 and not args.no_optimizations
    use_bf16 = args.bf16 and not args.no_optimizations

    # Check BF16 support
    if use_bf16:
        if not torch.cuda.is_bf16_supported():
            print("  WARNING: BF16 not supported on this GPU, falling back to FP16")
            use_bf16 = False
            use_fp16 = True
        else:
            use_fp16 = False  # BF16 takes precedence

    # Determine autocast dtype
    if use_bf16:
        autocast_dtype = torch.bfloat16
        print("  BF16 inference: enabled (torch.cuda.amp.autocast with bfloat16)")
    elif use_fp16:
        autocast_dtype = torch.float16
        print("  FP16 inference: enabled (torch.cuda.amp.autocast with float16)")
    else:
        autocast_dtype = None
        print("  Mixed precision: disabled (FP32)")

    # Print optimization summary
    print("\n" + "="*60)
    print("OPTIMIZATION CONFIGURATION")
    print("="*60)
    if args.no_optimizations:
        print("  Mode: BASELINE (all optimizations disabled)")
    else:
        print("  Mode: OPTIMIZED")
        print(f"  - Tier 1.1 Async DataLoader: num_workers={num_workers}, pin_memory={pin_memory}")
        dtype_str = "BF16" if use_bf16 else ("FP16" if use_fp16 else "FP32")
        print(f"  - Tier 1.2 Mixed Precision: {dtype_str}")
        print(f"  - Tier 1.4 Non-blocking transfers: True")
        if args.trt_backbone:
            print(f"  - Tier 2.1 TRT Backbone+FPN: {args.trt_backbone}")
    print("="*60 + "\n")

    # If using torch profiler, run that and exit
    if args.torch_profiler:
        run_torch_profiler(model, val_loader, args.num_samples, args.device, move_to_device)
        return

    # Create profilers with mixed precision support
    inference_timer = InferenceTimer(autocast_dtype=autocast_dtype)
    module_profiler = ModuleProfiler()

    if args.detailed:
        module_profiler.register_hooks(model)

    # Warmup
    print(f"\nRunning {args.warmup} warmup iterations...")
    data_iter = iter(val_loader)
    non_blocking = not args.no_optimizations  # Tier 1.4: non-blocking transfers
    for i in range(args.warmup):
        batch = next(data_iter)
        sample = batch[0]
        if isinstance(sample, (list, tuple)):
            while isinstance(sample, (list, tuple)) and len(sample) == 1:
                sample = sample[0]
        data = sample
        for key in ['img_metas', 'img', 'radar_points', 'radar_depth', 'radar_rcs']:
            if key in data and not isinstance(data[key], list):
                data[key] = [data[key]]
        data = move_to_device(data, args.device, non_blocking=non_blocking)

        with torch.no_grad():
            # Tier 1.2: Mixed precision inference during warmup
            if autocast_dtype is not None:
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    if hasattr(model, 'module'):
                        _ = model.module(return_loss=False, rescale=True, **data)
                    else:
                        _ = model(return_loss=False, rescale=True, **data)
            else:
                if hasattr(model, 'module'):
                    _ = model.module(return_loss=False, rescale=True, **data)
                else:
                    _ = model(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize()
        print(f"  Warmup {i+1}/{args.warmup} done")

    # Clear warmup timings from module profiler
    if args.detailed:
        module_profiler.timings.clear()
        module_profiler.start_times.clear()

    # Profile
    print(f"\nProfiling {args.num_samples} samples...")
    data_iter = iter(val_loader)
    for i in range(args.num_samples):
        batch = next(data_iter)
        sample = batch[0]
        if isinstance(sample, (list, tuple)):
            while isinstance(sample, (list, tuple)) and len(sample) == 1:
                sample = sample[0]
        data = sample
        for key in ['img_metas', 'img', 'radar_points', 'radar_depth', 'radar_rcs']:
            if key in data and not isinstance(data[key], list):
                data[key] = [data[key]]
        # Tier 1.4: Non-blocking transfers
        data = move_to_device(data, args.device, non_blocking=non_blocking)

        inference_timer.time_inference(model, data)
        print(f"  Sample {i+1}/{args.num_samples} done")

    # Remove hooks
    if args.detailed:
        module_profiler.remove_hooks()

    # Print results
    print("\n" + "="*70)
    print("FULL INFERENCE TIMING")
    print("="*70)
    full_stats = inference_timer.get_stats()
    dtype_str = "BF16" if use_bf16 else ("FP16" if use_fp16 else "FP32")
    opt_mode = "BASELINE" if args.no_optimizations else f"OPTIMIZED ({dtype_str})"
    print(f"Mode: {opt_mode}")
    print(f"Mean: {full_stats.get('mean_ms', 0):.2f} ms")
    print(f"Std:  {full_stats.get('std_ms', 0):.2f} ms")
    print(f"Min:  {full_stats.get('min_ms', 0):.2f} ms")
    print(f"Max:  {full_stats.get('max_ms', 0):.2f} ms")
    print(f"FPS:  {1000/full_stats.get('mean_ms', 1):.2f}")

    if args.detailed:
        module_stats = module_profiler.print_report(num_samples=args.num_samples)

        # Calculate total per inference for bottleneck analysis
        module_totals = {}
        for name, data in module_stats.items():
            calls_per_inf = data['count'] / args.num_samples
            module_totals[name] = data['mean_ms'] * calls_per_inf

        print("\n" + "="*70)
        print("BOTTLENECK ANALYSIS (total time per inference)")
        print("="*70)
        sorted_modules = sorted(module_totals.items(), key=lambda x: x[1], reverse=True)
        print("\nTop bottlenecks:")
        for i, (name, total_ms) in enumerate(sorted_modules[:5]):
            print(f"  {i+1}. {name}: {total_ms:.2f} ms ({total_ms/full_stats['mean_ms']*100:.1f}% of total inference)")

    # Save results
    import json
    results = {
        'full_inference': full_stats,
        'modules': module_profiler.get_stats() if args.detailed else {},
        'optimizations': {
            'dtype': dtype_str,
            'fp16': use_fp16,
            'bf16': use_bf16,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'non_blocking': non_blocking,
            'baseline_mode': args.no_optimizations,
        }
    }
    with open('profile_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to profile_results.json")

    # Generate histogram chart
    if args.detailed and module_stats:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Calculate total per inference for each module
            module_totals = {}
            for name, data in module_stats.items():
                calls_per_inf = data['count'] / args.num_samples
                module_totals[name] = data['mean_ms'] * calls_per_inf

            # Sort by time descending
            sorted_items = sorted(module_totals.items(), key=lambda x: x[1], reverse=True)
            names = [item[0] for item in sorted_items]
            times = [item[1] for item in sorted_items]

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create horizontal bar chart
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
            bars = ax.barh(range(len(names)), times, color=colors)

            # Customize
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.invert_yaxis()  # Largest at top
            ax.set_xlabel('Time (ms)')
            ax.set_title(f'RaCFormer Inference Breakdown\nTotal: {full_stats["mean_ms"]:.1f}ms ({1000/full_stats["mean_ms"]:.2f} FPS)')

            # Add value labels on bars
            for i, (bar, time_val) in enumerate(zip(bars, times)):
                pct = time_val / full_stats['mean_ms'] * 100
                ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                       f'{time_val:.1f}ms ({pct:.1f}%)',
                       va='center', fontsize=9)

            # Add measured total vs actual
            measured_total = sum(times)
            unmeasured = full_stats['mean_ms'] - measured_total
            ax.axvline(x=measured_total, color='red', linestyle='--', alpha=0.7, label=f'Measured: {measured_total:.1f}ms')

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(right=0.75)

            # Add text box with summary
            textstr = f"Full Inference: {full_stats['mean_ms']:.1f}ms\n"
            textstr += f"Measured Total: {measured_total:.1f}ms\n"
            textstr += f"Unmeasured: {unmeasured:.1f}ms\n"
            textstr += f"FPS: {1000/full_stats['mean_ms']:.2f}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(1.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)

            # Save
            chart_path = 'profile_chart.png'
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Chart saved to {chart_path}")

        except ImportError:
            print("matplotlib not available, skipping chart generation")


if __name__ == '__main__':
    main()
