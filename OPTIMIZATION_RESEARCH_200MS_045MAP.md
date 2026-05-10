# RaCFormer Orin Research: 200 ms With mAP > 0.45

Date: 2026-05-10
Target: mean inference latency < 200 ms on Jetson AGX Orin while keeping mini-val mAP > 0.45.

## Current Evidence

The target is not met by any tested RaCFormer checkpoint/config pair yet.

| Variant | mAP | NDS | Mean latency | Notes |
|---|---:|---:|---:|---|
| Baseline mini | 0.5195 | 0.4674 | 2859.77 ms | Original accuracy reference |
| TRT backbone, 4 layers | 0.5146 | 0.4581 | 1072.09 ms | Best accuracy-preserving path so far |
| TRT backbone, 3 layers | 0.4956 | 0.4443 | 911.75 ms | Above mAP target, far above latency target |
| TRT backbone, 3 layers, BF16 profile after `jetson_clocks` | n/a | n/a | 854.82 ms | Model timing only; still far above 200 ms |
| latency200 q900 fast-loader | 0.2356 | 0.1972 | 142.13 ms | Meets latency, fails mAP badly |
| f2 1-layer q900 BF16 | 0.2571 | 0.2422 | 197.29 ms | Meets latency, still far below mAP target |
| latency200 q300 BF16 | 0.0910 | 0.1069 | 197.83 ms | Meets latency, severe mAP collapse |

Detailed 3-layer profile after TRT backbone shows the real bottlenecks:

| Module | Total per inference |
|---|---:|
| Detection head transformer | 388.49 ms |
| View transformer | 313.61 ms |
| Radar voxel encoder | 33.93 ms |
| Radar BEV conv | 27.67 ms |
| Radar voxel layer | 16.46 ms |

Conclusion: the remaining gap is not a validation-loop artifact. Keeping mAP > 0.45 requires accelerating or replacing the transformer/view-transformer path, or retraining a smaller student. A 2-frame, 1-layer checkpoint-sliced variant reaches 197.29 ms but only 0.2571 mAP, so temporal slicing alone is not enough.

## Research Findings

### 1. Smaller Student + Distillation

Most plausible first training path: cut decoder layers, temporal frames, and/or query count, then recover accuracy with teacher-student distillation from the full RaCFormer checkpoint.

Evidence:
- StreamPETR reports a lightweight online model at 45.0 mAP and 31.7 FPS, using object-centric temporal modeling instead of heavy raw multi-frame processing.
- Distillation methods for 3D detection report mAP/NDS recovery with no inference-time cost, making them suitable for query/layer/frame compression.

Candidate experiments:
- Train `f4/3-layer/900q` or `f2/2-layer/900q` student with BEV-feature, query-feature, and logits/box distillation from the full 8-frame teacher.
- Preserve radar-guided depth supervision and Doppler cues during training.
- Do not rely on checkpoint slicing alone; prior slicing reached latency but collapsed mAP.

### 2. Replace Raw Temporal Re-encoding With Query/BEV Memory

RaCFormer loses too much accuracy when reduced to one frame. StreamPETR and CRT-Fusion suggest that temporal context should be carried in object-centric or motion-compensated memory rather than by repeatedly processing all raw frames.

Candidate experiments:
- Keep one current image/radar encoding, add a lightweight recurrent BEV/query cache.
- Use radar velocity/Doppler to warp or gate cached BEV features.
- Train with the full teacher to recover temporal cues.

### 3. Move View Transformer And Detection Head Out Of PyTorch

The current high-mAP profile spends roughly 700 ms in the detection head transformer plus view transformer after TRT backbone acceleration. TensorRT work must move beyond the backbone.

Candidate experiments:
- Export static-shape subgraphs for the view transformer and detector MLP/FFN/projection blocks.
- Replace unsupported deformable-attention/ray-sampling subgraphs with TensorRT plugins or custom CUDA kernels.
- Use CUDA graph capture after shapes are fixed.

### 4. Selective INT8 / QAT

INT8 is promising only if applied selectively and validated against mAP. TensorRT recommends explicit Q/DQ quantization. For this target, keep numerically sensitive attention, normalization, and prediction heads in FP16 first.

Candidate experiments:
- PTQ: quantize backbone, radar convs, FFNs, and projection GEMMs; keep attention/logits/box heads FP16.
- If mAP falls below 0.45, use QAT for the failing modules.

### 5. Architecture Fallbacks

If RaCFormer-specific ray sampling remains too expensive to optimize, radar-camera BEV architectures provide better deployment priors:
- CRN reports real-time camera-radar BEV perception, including 20 FPS real-time setting and 57.5 mAP / 62.4 NDS offline.
- CRT-Fusion reports motion-guided temporal fusion with multi-frame mAP above 0.45 and sub-200 ms paper latency in several settings.
- MatrixVT / BEVFusion-style optimized BEV pooling is a better deployment target than Python-heavy view-transform code.

## Recommended Next Backlog

1. Make the benchmark harness official:
   - Use profile-style fast DataLoader and direct model timing.
   - Always report mAP, NDS, mean latency, max latency, and samples.
   - Keep q900 fast-loader result as the low-latency floor and 3-layer TRT as the high-mAP floor.

2. Short-term no-retraining engineering:
   - Keep `jetson_clocks` enabled for all Orin benchmarks.
   - Try CUDA graph capture on the q900 and 3-layer steady-state loop.
   - Export/accelerate detector FFN/projection layers before attempting full deformable attention.

3. Medium-term retraining:
   - Train a 2-3 layer, 2-4 frame student with full-teacher distillation.
   - Track whether mAP crosses 0.45 before investing in lower-level TensorRT plugins.

4. High-effort deployment:
   - Implement TensorRT/custom CUDA for view transformer and deformable/ray attention.
   - Add selective INT8 PTQ, then QAT if required.

## Sources

- RaCFormer paper: https://arxiv.org/abs/2412.12725
- StreamPETR paper: https://arxiv.org/abs/2303.11926
- CRN paper: https://arxiv.org/abs/2304.00670
- CRT-Fusion paper: https://arxiv.org/abs/2411.03013
- TensorRT best practices: https://docs.nvidia.com/deeplearning/tensorrt/10.16.0/performance/best-practices.html
- TensorRT transformer attention: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-transformers.html
- TensorRT custom layers/plugins: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html
- TensorRT quantized types: https://docs.nvidia.com/deeplearning/tensorrt/10.16.0/inference-library/work-quantized-types.html
- Torch-TensorRT CUDA graphs: https://docs.pytorch.org/TensorRT/tutorials/runtime_opt/cuda_graphs.html
