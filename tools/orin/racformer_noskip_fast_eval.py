# pyright: reportMissingImports=false
"""Run no-skip RaCFormer validation with exact runtime-only fast paths.

This wrapper reproduces the first verified no-skip mini-val result below
200 ms with mAP above 0.40 on Jetson Orin. It does not use keyframe
propagation or alter model weights. The patches are implementation-only:

* image `sampling_4d` uses an equivalent einsum projection instead of expanded
  per-point matmul;
* image sampler index tensors are cached by shape;
* BEV sampling caches the per-depth linspace and uses expand where possible;
* BEV attention caches spatial-shape tensors and skips eval-mode dropout calls.

Run from the RaCFormer repository root, for example:

PYTHONPATH=. NUSCENES_VERSION=v1.0-mini \
  python tools/racformer_noskip_fast_eval.py \
    --config configs/orin/racformer_f2_3layer_q900_thrnone_mini.py \
    --weights checkpoints/racformer_r50_f8_f2_1layer_adapted.pth \
    --output_dir eval_results/goal_noskip_fast \
    --max_vis_samples 0 \
    --fp16 \
    --trt-backbone racformer_backbone_neck.trt
"""

from __future__ import annotations

import contextlib
import copy
import sys
import time
from typing import Any

import numpy as np
import torch
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from tqdm import tqdm

import val
import models.racformer_transformer as rt
import models.sparsebev_sampling as sparse_sampling
from models.bev_self_attention import BEVSelfAttention
from models.csrc.wrapper import msmv_sampling, msmv_sampling_v2
from models.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from models.racformer_transformer import BEVSampling


_IMAGE_INDEX_CACHE: dict[tuple[int, int, int, int, str], tuple[torch.Tensor, ...]] = {}
_BEV_LINSPACE_CACHE: dict[tuple[float, int, str, str], torch.Tensor] = {}
_BEV_SHAPE_CACHE: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _unwrap_sample(sample: Any) -> dict | None:
    if isinstance(sample, dict):
        return sample
    if isinstance(sample, (list, tuple)):
        while isinstance(sample, (list, tuple)) and len(sample) == 1:
            sample = sample[0]
        if isinstance(sample, dict):
            return sample
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            return sample[0] if isinstance(sample[0], dict) else {"inputs": sample[0], "data_samples": sample[1]}
    return None


def _cached_image_indices(
    batch: int,
    frames: int,
    queries: int,
    group_points: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    key = (batch, frames, queries, group_points, str(device))
    cached = _IMAGE_INDEX_CACHE.get(key)
    if cached is None:
        cached = (
            torch.arange(batch, dtype=torch.long, device=device)
            .view(batch, 1, 1, 1, 1)
            .expand(batch, frames, queries, group_points, 1),
            torch.arange(frames, dtype=torch.long, device=device)
            .view(1, frames, 1, 1, 1)
            .expand(batch, frames, queries, group_points, 1),
            torch.arange(queries, dtype=torch.long, device=device)
            .view(1, 1, queries, 1, 1)
            .expand(batch, frames, queries, group_points, 1),
            torch.arange(group_points, dtype=torch.long, device=device)
            .view(1, 1, 1, group_points, 1)
            .expand(batch, frames, queries, group_points, 1),
        )
        _IMAGE_INDEX_CACHE[key] = cached
    return cached


def sampling_4d_fast(
    sample_points: torch.Tensor,
    mlvl_feats: list[torch.Tensor],
    scale_weights: torch.Tensor,
    lidar2img: torch.Tensor,
    image_h: float,
    image_w: float,
    aggregate: bool = True,
    eps: float = 1e-5,
):
    batch, queries, frames, groups, points, _ = sample_points.shape
    num_views = lidar2img.shape[1] // frames
    group_points = groups * points

    points_h = sample_points.reshape(batch, queries, frames, group_points, 3)
    points_h = torch.cat([points_h, torch.ones_like(points_h[..., :1])], dim=-1)
    points_h = points_h.permute(0, 2, 1, 3, 4).contiguous()
    l2i = lidar2img.reshape(batch, frames, num_views, 4, 4)
    sample_points_cam = torch.einsum("btnij,btqpj->btnqpi", l2i, points_h)

    homo = sample_points_cam[..., 2:3]
    homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
    sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero
    sample_points_cam[..., 0] /= image_w
    sample_points_cam[..., 1] /= image_h

    valid_mask = (
        (homo > eps)
        & (sample_points_cam[..., 1:2] > 0.0)
        & (sample_points_cam[..., 1:2] < 1.0)
        & (sample_points_cam[..., 0:1] > 0.0)
        & (sample_points_cam[..., 0:1] < 1.0)
    ).squeeze(-1).float()
    valid_mask = valid_mask.permute(0, 1, 3, 4, 2)
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 4, 2, 5)

    i_batch, i_time, i_query, i_point = _cached_image_indices(
        batch, frames, queries, group_points, sample_points.device
    )
    i_view = torch.argmax(valid_mask, dim=-1)[..., None]
    sample_points_cam = sample_points_cam[i_batch, i_time, i_query, i_point, i_view, :]
    sample_points_cam = torch.cat([sample_points_cam, i_view[..., None].float() / (num_views - 1)], dim=-1)
    sample_points_cam = sample_points_cam.reshape(batch, frames, queries, groups, points, 1, 3)
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 2, 4, 5, 6)
    sample_points_cam = sample_points_cam.reshape(batch * frames * groups, queries, points, 3)

    packed_weights = scale_weights.reshape(batch, queries, groups, frames, points, -1)
    packed_weights = packed_weights.permute(0, 2, 3, 1, 4, 5)
    packed_weights = packed_weights.reshape(batch * groups * frames, queries, points, -1)

    if aggregate:
        final = msmv_sampling(mlvl_feats, sample_points_cam.contiguous(), packed_weights.contiguous())
    else:
        final = msmv_sampling_v2(mlvl_feats, sample_points_cam.contiguous(), packed_weights.contiguous())

    channels = final.shape[2]
    final = final.reshape(batch, frames, groups, queries, channels, points)
    final = final.permute(0, 3, 2, 1, 5, 4).flatten(3, 4)
    if not aggregate:
        return final, homo[:, 0], i_view[:, 0]
    return final


def bev_inner_forward_fast(self, query_ray, query_feat, bev_feats, img_metas, d_region=0.1):
    if self.temp_radar:
        bev_feats = self.temporal_encoder(bev_feats)

    batch, queries, _ = query_ray.shape
    bev_h, bev_w = bev_feats.shape[-2:]
    query_bbox = rt.theta_d2xy_coods(query_ray).clone()

    sampling_offset = self.sampling_offset(query_feat)
    sampling_offset = sampling_offset.view(batch, queries, self.num_heads * self.num_points * self.depth_num, 2)
    sampling_offset = torch.cat((sampling_offset, torch.zeros_like(sampling_offset[..., 0:1])), dim=-1)
    sampling_points = rt.make_sample_points(query_bbox, sampling_offset, self.pc_range)
    sampling_points = sampling_points.reshape(
        batch, queries, 1, self.num_heads, self.num_points * self.depth_num, 3
    )
    sampling_points = sampling_points.expand(
        batch, queries, self.num_frames, self.num_heads, self.num_points * self.depth_num, 3
    )

    time_diff = img_metas[0]["time_diff"][:, None, :, None]
    vel = query_ray[..., 8:].detach()[:, :, None, :]
    sampling_points = sampling_points[..., 0:2] - (vel * time_diff)[:, :, :, None, None, :]
    sampling_points[..., 0:1] = (sampling_points[..., 0:1] - self.pc_range[0]) / (
        self.pc_range[3] - self.pc_range[0]
    )
    sampling_points[..., 1:2] = (sampling_points[..., 1:2] - self.pc_range[1]) / (
        self.pc_range[4] - self.pc_range[1]
    )
    sampling_points = rt.xy2theta_d_coods(sampling_points)
    sampling_points = sampling_points.reshape(
        batch, queries, self.num_frames, self.num_heads, self.num_points, self.depth_num, 2
    )

    cache_key = (float(d_region), self.depth_num, str(query_bbox.device), str(query_bbox.dtype))
    base = _BEV_LINSPACE_CACHE.get(cache_key)
    if base is None:
        base = torch.linspace(
            -d_region,
            d_region,
            self.depth_num,
            device=query_bbox.device,
            dtype=query_bbox.dtype,
        ).view(1, 1, self.depth_num)
        _BEV_LINSPACE_CACHE[cache_key] = base

    sampling_points_d = base.expand(batch, queries, self.depth_num)
    sampling_points_d = sampling_points_d + (
        (self.ray_points_offset(query_feat).sigmoid() * 2 - 1) * d_region / self.depth_num / 2
    )
    sampling_points_d = sampling_points_d.view(batch, queries, 1, 1, 1, self.depth_num, 1)
    sampling_points_d = sampling_points_d.expand(
        batch, queries, self.num_frames, self.num_heads, self.num_points, self.depth_num, 1
    )

    sampling_points = torch.cat((sampling_points[..., 0:1], sampling_points[..., 1:2] + sampling_points_d), dim=-1)
    sampling_points = sampling_points.reshape(
        batch, queries, self.num_frames, self.num_heads, self.num_points * self.depth_num, 2
    )
    sampling_points = rt.theta_d2xy_coods(sampling_points)
    sampling_points = sampling_points.permute(0, 1, 3, 2, 4, 5).contiguous()

    scale_weights = self.scale_weights(query_feat).view(
        batch, queries, self.num_heads, 1, self.num_levels, self.depth_num * self.num_points
    )
    scale_weights = torch.softmax(scale_weights, dim=-1)
    scale_weights = scale_weights.expand(
        batch, queries, self.num_heads, self.num_frames, self.num_levels, self.depth_num * self.num_points
    ).contiguous()

    if (
        self._cached_bev_pos is None
        or self._cached_bev_pos.device != bev_feats.device
        or self._cached_bev_pos.dtype != bev_feats.dtype
    ):
        bev_mask = torch.zeros((1, bev_h, bev_w), device=bev_feats.device).to(bev_feats.dtype)
        self._cached_bev_pos = self.positional_encoding(bev_mask).to(bev_feats.dtype)
        self._cached_bev_pos = self._cached_bev_pos.view(1, 1, self.embed_dims, bev_h, bev_w)
    bev_pos = self._cached_bev_pos.expand(batch, self.num_frames, -1, -1, -1)

    return self.attention(query_feat, bev_feats + bev_pos, sampling_points, scale_weights, spatial_shapes=(bev_h, bev_w))


def _cached_bev_shape_tensors(device: torch.device, spatial_shapes: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), int(spatial_shapes[0]), int(spatial_shapes[1]))
    cached = _BEV_SHAPE_CACHE.get(key)
    if cached is None:
        cached = (
            torch.tensor([0], dtype=torch.long, device=device),
            torch.tensor([spatial_shapes], dtype=torch.long, device=device),
        )
        _BEV_SHAPE_CACHE[key] = cached
    return cached


def bev_attention_forward_fast(
    self,
    query,
    value,
    sampling_locations,
    attention_weights,
    key_padding_mask=None,
    identity=None,
    spatial_shapes=None,
    **kwargs,
):
    batch, queries, channels = query.shape
    if identity is None:
        identity = query

    value = value.view(batch * self.num_bev_queue, channels, -1).permute(0, 2, 1)
    if not self.batch_first:
        value = value.permute(1, 0, 2)
    bs, num_query, _ = query.shape
    _, num_value, _ = value.shape

    value = self.value_proj(value)
    if key_padding_mask is not None:
        value = value.masked_fill(key_padding_mask[..., None], 0.0)
    value = value.reshape(bs * self.num_bev_queue, num_value, self.num_heads, -1)

    sampling_locations = sampling_locations.view(
        bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2
    )
    attention_weights = attention_weights.view(
        bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points
    )
    attention_weights = attention_weights.permute(3, 0, 1, 2, 4, 5)
    attention_weights = attention_weights.reshape(
        bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points
    ).contiguous()
    sampling_locations = sampling_locations.permute(3, 0, 1, 2, 4, 5, 6)
    sampling_locations = sampling_locations.reshape(
        bs * self.num_bev_queue,
        num_query,
        self.num_heads,
        self.num_levels,
        self.num_points,
        2,
    ).contiguous()

    if spatial_shapes is None:
        raise ValueError("spatial_shapes is required for BEVSelfAttention fast path")
    level_start_index, spatial_shapes_tensor = _cached_bev_shape_tensors(value.device, spatial_shapes)
    if torch.cuda.is_available() and value.is_cuda:
        output = MultiScaleDeformableAttnFunction_fp32.apply(
            value,
            spatial_shapes_tensor,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
    else:
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes_tensor, sampling_locations, attention_weights
        )

    output = output.permute(1, 2, 0).reshape(num_query, channels, bs, self.num_bev_queue)
    if self.queue_weight:
        queue_weight = self.bev_queue_weight(query).permute(1, 0, 2).reshape(
            num_query, 1, bs, self.num_bev_queue
        )
        queue_weight = torch.softmax(queue_weight, dim=-1)
        output = torch.sum(output * queue_weight, dim=-1, keepdim=False)
    else:
        output = torch.sum(output, dim=-1, keepdim=False) / self.num_bev_queue

    output = self.output_proj(output.permute(2, 0, 1))
    if not self.batch_first:
        output = output.permute(1, 0, 2)
    if self.training:
        output = self.dropout(output)
    return output + identity


def single_gpu_test_inference_mode(model, data_loader, show=False, out_dir=None, autocast_dtype=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = tqdm(total=len(dataset))
    inference_times = []
    warmup_done = False

    for batch in data_loader:
        batch_results = []
        for sample in batch:
            data = _unwrap_sample(sample)
            if data is None:
                continue

            for key in ["img_metas", "img", "radar_points", "radar_depth", "radar_rcs", "gt_depth"]:
                if key in data and not isinstance(data[key], list):
                    data[key] = [data[key]]

            device = next((model.module if hasattr(model, "module") else model).parameters()).device
            data = val.move_to_device(data, device)

            if not warmup_done:
                warmup_data = copy.deepcopy(data)
                with torch.inference_mode():
                    with torch.cuda.amp.autocast(dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext():
                        _ = model(return_loss=False, rescale=True, **warmup_data)
                torch.cuda.synchronize()
                warmup_done = True

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext():
                    result = model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            inference_times.append((time.perf_counter() - start_time) * 1000.0)

            if isinstance(result, list):
                batch_results.extend(result)
            else:
                batch_results.append(result)

        results.extend(batch_results)
        prog_bar.update(len(batch))

    prog_bar.close()
    timing_stats = {}
    if inference_times:
        timing_stats = {
            "mean_inference_ms": float(np.mean(inference_times)),
            "std_inference_ms": float(np.std(inference_times)),
            "min_inference_ms": float(np.min(inference_times)),
            "max_inference_ms": float(np.max(inference_times)),
            "median_inference_ms": float(np.median(inference_times)),
            "fps": float(1000.0 / np.mean(inference_times)),
            "num_samples": len(inference_times),
        }
    return results, timing_stats


def install_patch() -> None:
    sparse_sampling.sampling_4d = sampling_4d_fast
    rt.sampling_4d = sampling_4d_fast
    BEVSampling.inner_forward = bev_inner_forward_fast
    BEVSelfAttention.forward = bev_attention_forward_fast
    val.single_gpu_test = single_gpu_test_inference_mode


def main() -> int:
    install_patch()
    return val.main()


if __name__ == "__main__":
    sys.exit(main())
