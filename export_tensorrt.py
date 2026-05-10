"""
TensorRT Export Script for RaCFormer Backbone + Neck
Tier 2.1: Export ResNet50 backbone and FPN neck to TensorRT for 2-3x speedup.

Usage:
    # Export backbone + neck to ONNX
    python export_tensorrt.py configs/racformer_r50_nuimg_704x256_f8.py \
        checkpoints/racformer.pth --export-onnx

    # After exporting to ONNX, convert to TensorRT engine:
    # trtexec --onnx=racformer_backbone_neck.onnx --fp16 \
    #         --saveEngine=racformer_backbone_neck.trt

    # Run inference with TensorRT backend
    python export_tensorrt.py configs/racformer_r50_nuimg_704x256_f8.py \
        checkpoints/racformer.pth --trt-engine racformer_backbone_neck.trt

Requirements:
    - torch
    - onnx
    - onnxruntime (for validation)
    - tensorrt (for TRT inference)
    - torch2trt (optional, for direct PyTorch->TRT conversion)
"""

import os
import sys
import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn

os.environ['NUSCENES_VERSION'] = os.environ.get('NUSCENES_VERSION', 'v1.0-mini')

from mmengine.config import Config
from mmengine.runner import load_checkpoint, set_random_seed
from mmdet3d.registry import MODELS


def build_model(cfg):
    return MODELS.build(cfg)


class BackboneNeckWrapper(nn.Module):
    """
    Wrapper module that extracts just the backbone + neck from RaCFormer.
    This allows exporting these components to ONNX/TensorRT separately.
    """

    def __init__(self, backbone, neck, grid_mask=None, use_grid_mask=True):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.grid_mask = grid_mask
        self.use_grid_mask = use_grid_mask and grid_mask is not None

    def forward(self, img):
        """
        Args:
            img: [B*N*T, C, H, W] input images

        Returns:
            Tuple of (fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_feat_3,
                       backbone_feat_C4, backbone_feat_C5)
            FPN features (4 levels, 256 channels each) +
            last 2 backbone features (C4: 1024ch, C5: 2048ch) for LSS neck
        """
        if self.use_grid_mask and self.training:
            img = self.grid_mask(img)

        # Backbone forward
        feats = self.backbone(img)

        if isinstance(feats, dict):
            feats = list(feats.values())

        # Neck (FPN) forward
        fpn_feats = self.neck(feats)

        # Return FPN outputs + last 2 backbone features for LSS neck
        return (*fpn_feats, feats[-2], feats[-1])


class LSSNeckWrapper(nn.Module):
    """
    Wrapper for LSS neck that processes higher-level backbone features.
    """

    def __init__(self, lss_neck, num_lss_fpn=2):
        super().__init__()
        self.lss_neck = lss_neck
        self.num_lss_fpn = num_lss_fpn

    def forward(self, backbone_feats):
        """
        Args:
            backbone_feats: List of backbone features

        Returns:
            LSS features for view transformation
        """
        lss_feats = self.lss_neck(backbone_feats[-self.num_lss_fpn:])
        if isinstance(lss_feats, (list, tuple)):
            lss_feats = lss_feats[0]
        return lss_feats


def export_to_onnx(model, output_path, img_size=(256, 704), batch_size=48, opset_version=17):
    """
    Export backbone + neck to ONNX format.

    Args:
        model: BackboneNeckWrapper instance
        output_path: Path to save ONNX model
        img_size: (H, W) input image size
        batch_size: Batch size (typically B*N*T = 1*6*8 = 48)
        opset_version: ONNX opset version
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, img_size[0], img_size[1]).cuda()

    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Opset version: {opset_version}")

    # Run a test forward to determine output names and count
    with torch.no_grad():
        test_outputs = model(dummy_input)
    num_outputs = len(test_outputs)
    output_names = [f'fpn_{i}' for i in range(4)] + ['backbone_C4', 'backbone_C5']
    output_names = output_names[:num_outputs]

    print(f"  Number of outputs: {num_outputs}")
    for i, out in enumerate(test_outputs):
        print(f"    {output_names[i]}: {out.shape}")

    # Dynamic axes for flexible batch size
    dynamic_axes = {
        'input': {0: 'batch_size'},
    }
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size'}

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )

    print(f"ONNX model saved to: {output_path}")

    # Validate ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation: PASSED")
    except ImportError:
        print("Warning: onnx package not installed, skipping validation")
    except Exception as e:
        print(f"Warning: ONNX validation failed: {e}")

    return output_path


def validate_onnx_output(pytorch_model, onnx_path, img_size=(256, 704), batch_size=48):
    """
    Validate ONNX model output matches PyTorch output.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnxruntime not installed, skipping validation")
        return

    pytorch_model.eval()

    # Create test input
    test_input = torch.randn(batch_size, 3, img_size[0], img_size[1]).cuda()

    # PyTorch forward
    with torch.no_grad():
        pytorch_outputs = pytorch_model(test_input)

    # ONNX Runtime forward
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    ort_inputs = {'input': test_input.cpu().numpy()}
    ort_outputs = sess.run(None, ort_inputs)

    # Compare outputs
    print("\nValidating ONNX output vs PyTorch:")
    all_close = True
    for i, (pt_out, ort_out) in enumerate(zip(pytorch_outputs, ort_outputs)):
        pt_np = pt_out.cpu().numpy()
        max_diff = np.max(np.abs(pt_np - ort_out))
        mean_diff = np.mean(np.abs(pt_np - ort_out))
        close = np.allclose(pt_np, ort_out, rtol=1e-3, atol=1e-5)
        print(f"  Level {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, close={close}")
        all_close = all_close and close

    if all_close:
        print("Validation: PASSED")
    else:
        print("Validation: FAILED (outputs differ beyond tolerance)")


class TRTBackboneNeck:
    """
    TensorRT inference wrapper for backbone + FPN neck.

    Loads a TRT engine built from BackboneNeckWrapper ONNX export.
    Returns 6 outputs: 4 FPN features + 2 raw backbone features (C4, C5).
    The backbone features are needed by the LSS neck for view transformation.

    Uses TRT 8.6 API (set_binding_shape / execute_v2) for dynamic batch.
    """

    def __init__(self, engine_path, device='cuda:0'):
        import tensorrt as trt

        self.device = device
        self.engine_path = engine_path
        self.trt = trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(self.engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Discover bindings
        self.input_idx = None
        self.output_indices = []
        self.output_names = []

        print(f"TensorRT engine loaded from: {self.engine_path}")
        print(f"  Number of bindings: {self.engine.num_bindings}")
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = self.engine.get_binding_dtype(i)
            is_input = self.engine.binding_is_input(i)
            print(f"    [{i}] {name}: shape={shape}, dtype={dtype}, input={is_input}")
            if is_input:
                self.input_idx = i
            else:
                self.output_indices.append(i)
                self.output_names.append(name)

        self.num_outputs = len(self.output_indices)
        print(f"  Outputs: {self.num_outputs} ({', '.join(self.output_names)})")

        # Pre-allocate output buffers (will be resized on first call)
        self._output_buffers = None
        self._last_batch_size = None

    def _trt_dtype_to_torch(self, trt_dtype):
        """Convert TensorRT dtype to PyTorch dtype."""
        import tensorrt as trt
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
        }
        return mapping.get(trt_dtype, torch.float32)

    def __call__(self, img):
        """
        Run TensorRT inference.

        Args:
            img: [B*N*T, C, H, W] input tensor (float32, on GPU)

        Returns:
            Tuple of torch tensors: (fpn_0, fpn_1, fpn_2, fpn_3, backbone_C4, backbone_C5)
        """
        batch_size = img.shape[0]
        img_contiguous = img.contiguous()

        # Set dynamic input shape
        self.context.set_binding_shape(self.input_idx, tuple(img_contiguous.shape))

        # Allocate output buffers if batch size changed
        if self._last_batch_size != batch_size:
            self._output_buffers = []
            for idx in self.output_indices:
                out_shape = tuple(self.context.get_binding_shape(idx))
                out_dtype = self._trt_dtype_to_torch(self.engine.get_binding_dtype(idx))
                buf = torch.empty(out_shape, dtype=out_dtype, device=self.device)
                self._output_buffers.append(buf)
            self._last_batch_size = batch_size

        # Build bindings list (input + outputs)
        bindings = [0] * self.engine.num_bindings
        bindings[self.input_idx] = img_contiguous.data_ptr()
        for buf, idx in zip(self._output_buffers, self.output_indices):
            bindings[idx] = buf.data_ptr()

        # Execute
        self.context.execute_v2(bindings)

        return tuple(self._output_buffers)


def benchmark_tensorrt(trt_inference, pytorch_model, img_size=(256, 704), batch_size=48, num_iterations=100):
    """
    Benchmark TensorRT vs PyTorch inference speed.
    """
    import time

    pytorch_model.eval()
    test_input = torch.randn(batch_size, 3, img_size[0], img_size[1]).cuda()

    # Warmup
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = pytorch_model(test_input)
        if trt_inference is not None:
            _ = trt_inference(test_input)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    print(f"Benchmarking PyTorch ({num_iterations} iterations)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = pytorch_model(test_input)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) * 1000 / num_iterations
    print(f"  PyTorch: {pytorch_time:.2f} ms/iter")

    # Benchmark TensorRT
    if trt_inference is not None:
        print(f"Benchmarking TensorRT ({num_iterations} iterations)...")
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = trt_inference(test_input)
        torch.cuda.synchronize()
        trt_time = (time.perf_counter() - start) * 1000 / num_iterations
        print(f"  TensorRT: {trt_time:.2f} ms/iter")
        print(f"  Speedup: {pytorch_time / trt_time:.2f}x")


def patch_extract_img_feat(model, trt_engine):
    """
    Monkey-patch model.extract_img_feat to use TRT for backbone+FPN,
    while keeping PyTorch LSS neck.

    Args:
        model: RaCFormer model instance
        trt_engine: TRTBackboneNeck instance

    Returns:
        The patched model
    """
    import types

    original_lss_neck = model.img_lss_neck
    num_lss_fpn = model.num_lss_fpn
    use_grid_mask = model.use_grid_mask

    # Build name-to-index mapping for output bindings
    name_to_idx = {name: i for i, name in enumerate(trt_engine.output_names)}

    def extract_img_feat_trt(self, img):
        # Grid mask is disabled during inference (only used in training)
        BNT, C, imH, imW = img.shape

        # Run backbone + FPN through TensorRT
        trt_outputs = trt_engine(img)

        # Map outputs by name (binding order may vary)
        fpn_feats = tuple(
            trt_outputs[name_to_idx[f'fpn_{i}']] for i in range(4)
        )
        backbone_C4 = trt_outputs[name_to_idx['backbone_C4']]
        backbone_C5 = trt_outputs[name_to_idx['backbone_C5']]

        # Run LSS neck on raw backbone features (PyTorch)
        lss_input = [backbone_C4, backbone_C5][-num_lss_fpn:]
        img_lss_feats = original_lss_neck(lss_input)
        if type(img_lss_feats) in [list, tuple]:
            img_lss_feats = img_lss_feats[0]
        _, output_dim, ouput_H, output_W = img_lss_feats.shape
        img_lss_feats = img_lss_feats.view(BNT, output_dim, ouput_H, output_W)

        return fpn_feats, img_lss_feats

    model.extract_img_feat = types.MethodType(extract_img_feat_trt, model)
    print("Patched model.extract_img_feat to use TensorRT backbone+FPN")
    return model


def main():
    parser = argparse.ArgumentParser(description='Export RaCFormer backbone+neck to TensorRT')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--export-onnx', action='store_true', help='Export to ONNX format')
    parser.add_argument('--onnx-path', type=str, default='racformer_backbone_neck.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--trt-engine', type=str, help='TensorRT engine path for inference')
    parser.add_argument('--validate', action='store_true', help='Validate ONNX output')
    parser.add_argument('--benchmark', action='store_true', help='Run speed benchmark')
    parser.add_argument('--backbone-only', action='store_true',
                        help='Export backbone only (4 outputs, no FPN)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--img-height', type=int, default=256, help='Input image height')
    parser.add_argument('--img-width', type=int, default=704, help='Input image width')
    parser.add_argument('--batch-size', type=int, default=48, help='Batch size (B*N*T)')
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    cfg = Config.fromfile(args.config)

    # Register custom modules
    importlib.import_module('models')
    importlib.import_module('loaders')

    from mmdet.registry import MODELS as MMDET_MODELS
    from mmdet3d.registry import MODELS as MMDET3D_MODELS
    for name, module in MMDET_MODELS.module_dict.items():
        if name not in MMDET3D_MODELS.module_dict:
            MMDET3D_MODELS.register_module(name=name, module=module)

    import logging
    logging.getLogger('mmengine').setLevel(logging.WARNING)
    logging.getLogger('mmcv').setLevel(logging.WARNING)

    set_random_seed(0, deterministic=False)

    # Build model
    print("Building model...")
    model = build_model(cfg.model)

    print(f"Loading checkpoint from {args.checkpoint}")
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = model.to(args.device)
    model.eval()

    # Create backbone + neck wrapper
    backbone_neck = BackboneNeckWrapper(
        backbone=model.img_backbone,
        neck=model.img_neck,
        grid_mask=model.grid_mask if hasattr(model, 'grid_mask') else None,
        use_grid_mask=False,  # Disable grid_mask for export (data augmentation)
    ).to(args.device)
    backbone_neck.eval()

    img_size = (args.img_height, args.img_width)

    if args.export_onnx:
        export_to_onnx(
            backbone_neck,
            args.onnx_path,
            img_size=img_size,
            batch_size=args.batch_size,
        )

        if args.validate:
            validate_onnx_output(backbone_neck, args.onnx_path, img_size=img_size, batch_size=args.batch_size)

    trt_inference = None
    if args.trt_engine and os.path.exists(args.trt_engine):
        trt_inference = TRTBackboneNeck(args.trt_engine, device=args.device)

    if args.benchmark:
        benchmark_tensorrt(trt_inference, backbone_neck, img_size=img_size, batch_size=args.batch_size)

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    if args.export_onnx:
        print(f"1. Convert ONNX to TensorRT engine:")
        print(f"   /usr/src/tensorrt/bin/trtexec --onnx={args.onnx_path} --fp16 \\")
        print(f"       --saveEngine=racformer_backbone_neck.trt --workspace=4096")
        print()
        print("2. Run inference with TensorRT:")
        print(f"   python export_tensorrt.py {args.config} {args.checkpoint} --trt-engine racformer_backbone_neck.trt --benchmark")


if __name__ == '__main__':
    main()
