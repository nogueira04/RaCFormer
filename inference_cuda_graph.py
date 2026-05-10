"""
CUDA Graphs inference for RaCFormer model.
Tier 1.3: Reduces kernel launch overhead by capturing and replaying static compute graphs.

Usage:
    python inference_cuda_graph.py configs/racformer_r50_nuimg_704x256_f8.py \
        checkpoints/racformer.pth --warmup 5 --iterations 20

Note: CUDA Graphs require static graph shapes. This script captures the graph
after warmup and replays it for consistent inference. Dynamic operations like
voxelization may need to be excluded from the captured graph.
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

from mmdet3d.registry import MODELS, DATASETS


def build_dataset(cfg, default_args=None):
    return DATASETS.build(cfg, default_args=default_args)


def pseudo_collate(batch):
    return batch


def build_model(cfg, train_cfg=None, test_cfg=None):
    return MODELS.build(cfg)


def move_to_device(data, device, non_blocking=True):
    """Move data tensors to the specified device with non-blocking transfers."""
    for key in ['img', 'radar_depth', 'radar_rcs']:
        if key in data:
            if isinstance(data[key], list):
                data[key] = [d.to(device, non_blocking=non_blocking).float() if isinstance(d, torch.Tensor) else d for d in data[key]]
            elif isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device, non_blocking=non_blocking).float()
    if 'radar_points' in data:
        if isinstance(data['radar_points'], list):
            new_pts = []
            for pts in data['radar_points']:
                if isinstance(pts, list):
                    new_pts.append([p.to(device, non_blocking=non_blocking).float() if isinstance(p, torch.Tensor) else p for p in pts])
                elif isinstance(pts, torch.Tensor):
                    new_pts.append(pts.to(device, non_blocking=non_blocking).float())
                else:
                    new_pts.append(pts)
            data['radar_points'] = new_pts
    return data


class CUDAGraphInference:
    """
    CUDA Graph wrapper for RaCFormer inference.

    CUDA Graphs capture a sequence of GPU operations and replay them with minimal
    CPU overhead. This is especially beneficial when kernel launch overhead is
    significant relative to compute time.

    Limitations:
    - Requires static tensor shapes
    - Cannot capture operations with dynamic control flow
    - Memory allocations during graph capture become fixed
    """

    def __init__(self, model, device, use_fp16=True):
        self.model = model
        self.device = device
        self.use_fp16 = use_fp16
        self.graph = None
        self.static_input = None
        self.static_output = None
        self.captured = False

    def _get_model(self):
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    def warmup_and_capture(self, sample_data, num_warmup=3):
        """
        Warm up the model and capture CUDA graph.

        Args:
            sample_data: Sample input data dictionary
            num_warmup: Number of warmup iterations before capture
        """
        m = self._get_model()

        print(f"Running {num_warmup} warmup iterations...")
        for i in range(num_warmup):
            with torch.no_grad():
                if self.use_fp16:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        _ = m(return_loss=False, rescale=True, **sample_data)
                else:
                    _ = m(return_loss=False, rescale=True, **sample_data)
            torch.cuda.synchronize()
            print(f"  Warmup {i+1}/{num_warmup}")

        # Store static input tensors (we'll copy new data into these)
        self.static_input = {}
        for key in ['img', 'radar_depth', 'radar_rcs']:
            if key in sample_data:
                if isinstance(sample_data[key], list):
                    self.static_input[key] = [t.clone() if isinstance(t, torch.Tensor) else t for t in sample_data[key]]
                elif isinstance(sample_data[key], torch.Tensor):
                    self.static_input[key] = sample_data[key].clone()

        # Copy non-tensor data directly
        for key in ['img_metas', 'radar_points']:
            if key in sample_data:
                self.static_input[key] = sample_data[key]

        # Capture the graph
        print("Capturing CUDA graph...")
        self.graph = torch.cuda.CUDAGraph()

        # Use a CUDA stream for capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            try:
                with torch.cuda.graph(self.graph):
                    with torch.no_grad():
                        if self.use_fp16:
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                self.static_output = m(return_loss=False, rescale=True, **self.static_input)
                        else:
                            self.static_output = m(return_loss=False, rescale=True, **self.static_input)
                self.captured = True
                print("CUDA graph captured successfully!")
            except Exception as e:
                print(f"Warning: Could not capture CUDA graph: {e}")
                print("Falling back to standard inference.")
                self.captured = False

        torch.cuda.current_stream().wait_stream(s)

    def inference(self, data):
        """
        Run inference using CUDA graph if captured, otherwise standard inference.
        """
        m = self._get_model()

        if self.captured:
            # Copy new data into static input tensors
            for key in ['img', 'radar_depth', 'radar_rcs']:
                if key in data and key in self.static_input:
                    if isinstance(data[key], list):
                        for i, t in enumerate(data[key]):
                            if isinstance(t, torch.Tensor) and i < len(self.static_input[key]):
                                self.static_input[key][i].copy_(t)
                    elif isinstance(data[key], torch.Tensor):
                        self.static_input[key].copy_(data[key])

            # Update metadata (these are not captured in graph)
            for key in ['img_metas', 'radar_points']:
                if key in data:
                    self.static_input[key] = data[key]

            # Replay the graph
            self.graph.replay()
            return self.static_output
        else:
            # Fallback to standard inference
            with torch.no_grad():
                if self.use_fp16:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        return m(return_loss=False, rescale=True, **data)
                else:
                    return m(return_loss=False, rescale=True, **data)


def main():
    parser = argparse.ArgumentParser(description='CUDA Graph inference for RaCFormer')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--num-samples', type=int, default=20, help='Number of samples to run')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup iterations')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--fp16', action='store_true', default=True, help='Enable FP16 inference')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false', help='Disable FP16 inference')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader num_workers')
    parser.add_argument('--compare-baseline', action='store_true', help='Also run baseline for comparison')
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    cfg = Config.fromfile(args.config)

    # Register custom modules
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

    # Build dataloader with optimizations
    print("Building dataloader...")
    val_dataset = build_dataset(cfg.data.val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=pseudo_collate,
    )

    # Get first sample for graph capture
    data_iter = iter(val_loader)
    batch = next(data_iter)
    sample = batch[0]
    if isinstance(sample, (list, tuple)):
        while isinstance(sample, (list, tuple)) and len(sample) == 1:
            sample = sample[0]
    data = sample
    for key in ['img_metas', 'img', 'radar_points', 'radar_depth', 'radar_rcs']:
        if key in data and not isinstance(data[key], list):
            data[key] = [data[key]]
    data = move_to_device(data, args.device)

    # Create CUDA graph wrapper
    cuda_graph_inf = CUDAGraphInference(model, args.device, use_fp16=args.fp16)

    # Warmup and capture
    cuda_graph_inf.warmup_and_capture(data, num_warmup=args.warmup)

    # Run benchmarks
    print(f"\nRunning {args.num_samples} samples with CUDA Graph...")

    times_cuda_graph = []
    data_iter = iter(val_loader)

    for i in range(args.num_samples):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(val_loader)
            batch = next(data_iter)

        sample = batch[0]
        if isinstance(sample, (list, tuple)):
            while isinstance(sample, (list, tuple)) and len(sample) == 1:
                sample = sample[0]
        data = sample
        for key in ['img_metas', 'img', 'radar_points', 'radar_depth', 'radar_rcs']:
            if key in data and not isinstance(data[key], list):
                data[key] = [data[key]]
        data = move_to_device(data, args.device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        _ = cuda_graph_inf.inference(data)

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times_cuda_graph.append(elapsed)
        print(f"  Sample {i+1}/{args.num_samples}: {elapsed:.2f} ms")

    # Baseline comparison if requested
    times_baseline = []
    if args.compare_baseline:
        print(f"\nRunning {args.num_samples} samples with standard inference...")
        m = model.module if hasattr(model, 'module') else model
        data_iter = iter(val_loader)

        for i in range(args.num_samples):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(val_loader)
                batch = next(data_iter)

            sample = batch[0]
            if isinstance(sample, (list, tuple)):
                while isinstance(sample, (list, tuple)) and len(sample) == 1:
                    sample = sample[0]
            data = sample
            for key in ['img_metas', 'img', 'radar_points', 'radar_depth', 'radar_rcs']:
                if key in data and not isinstance(data[key], list):
                    data[key] = [data[key]]
            data = move_to_device(data, args.device)

            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                if args.fp16:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        _ = m(return_loss=False, rescale=True, **data)
                else:
                    _ = m(return_loss=False, rescale=True, **data)

            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            times_baseline.append(elapsed)
            print(f"  Sample {i+1}/{args.num_samples}: {elapsed:.2f} ms")

    # Print results
    print("\n" + "="*70)
    print("CUDA GRAPH INFERENCE RESULTS")
    print("="*70)

    print(f"\nCUDA Graph Mode (FP16={args.fp16}):")
    print(f"  Mean: {np.mean(times_cuda_graph):.2f} ms")
    print(f"  Std:  {np.std(times_cuda_graph):.2f} ms")
    print(f"  Min:  {np.min(times_cuda_graph):.2f} ms")
    print(f"  Max:  {np.max(times_cuda_graph):.2f} ms")
    print(f"  FPS:  {1000/np.mean(times_cuda_graph):.2f}")

    if args.compare_baseline and times_baseline:
        print(f"\nBaseline (FP16={args.fp16}):")
        print(f"  Mean: {np.mean(times_baseline):.2f} ms")
        print(f"  Std:  {np.std(times_baseline):.2f} ms")
        print(f"  Min:  {np.min(times_baseline):.2f} ms")
        print(f"  Max:  {np.max(times_baseline):.2f} ms")
        print(f"  FPS:  {1000/np.mean(times_baseline):.2f}")

        speedup = np.mean(times_baseline) / np.mean(times_cuda_graph)
        print(f"\nSpeedup: {speedup:.2f}x")

    print("="*70)


if __name__ == '__main__':
    main()
