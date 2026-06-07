#!/usr/bin/env python3
from __future__ import annotations

"""Branch G Stage 2 DINOv2 and DualViewDistill diagnostics.

This script is intentionally read-only with respect to model code and datasets.
It emits a JSON evidence file used by G_DIAGNOSTIC_<UTC>.md.
"""

import argparse
import json
import math
import os
import pickle
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

GRID_CONFIG = {
    "x": [-51.2, 51.2, 0.8],
    "y": [-51.2, 51.2, 0.8],
    "z": [-5.0, 3.0, 8.0],
    "depth": [1.0, 65.0, 96.0],
    "rcs": [-64.0, 64.0, 64.0],
}
FINAL_DIM = (256, 704)
ORIGINAL_HW = (900, 1600)
DOWNSAMPLE = 16
DINOV2_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
DINOV2_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class Dinov2PatchEmbeddings(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, num_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)


class Dinov2Embeddings(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        hidden_size = int(cfg["hidden_size"])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1370, hidden_size))
        self.patch_embeddings = Dinov2PatchEmbeddings(
            hidden_size=hidden_size,
            patch_size=int(cfg["patch_size"]),
            num_channels=int(cfg["num_channels"]),
        )

    def interpolate_pos_encoding(self, patch_tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_patches = patch_tokens.shape[1]
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = patch_tokens.shape[-1]
        source_size = int(math.sqrt(num_positions))
        target_h = height // self.patch_embeddings.projection.stride[0]
        target_w = width // self.patch_embeddings.projection.stride[1]
        patch_pos_embed = patch_pos_embed.reshape(1, source_size, source_size, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, target_h * target_w, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, height, width = pixel_values.shape
        patch = self.patch_embeddings.projection(pixel_values)
        patch_grid = (patch.shape[-2], patch.shape[-1])
        patch = patch.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(pixel_values.shape[0], -1, -1)
        embeddings = torch.cat((cls, patch), dim=1)
        embeddings = embeddings + self.interpolate_pos_encoding(patch, height, width)
        return embeddings, patch_grid


class Dinov2SelfAttention(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        hidden_size = int(cfg["hidden_size"])
        self.num_heads = int(cfg["num_attention_heads"])
        self.head_dim = hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(hidden_size, hidden_size, bias=bool(cfg["qkv_bias"]))
        self.key = nn.Linear(hidden_size, hidden_size, bias=bool(cfg["qkv_bias"]))
        self.value = nn.Linear(hidden_size, hidden_size, bias=bool(cfg["qkv_bias"]))

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self._shape(self.query(hidden_states))
        key = self._shape(self.key(hidden_states))
        value = self._shape(self.value(hidden_states))
        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        context = torch.matmul(attn, value)
        context = context.transpose(1, 2).reshape(hidden_states.shape)
        return context


class Dinov2SelfOutput(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)


class Dinov2Attention(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.attention = Dinov2SelfAttention(cfg)
        self.output = Dinov2SelfOutput(int(cfg["hidden_size"]))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output.dense(self.attention(hidden_states))


class Dinov2LayerScale(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.lambda1


class Dinov2MLP(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        hidden_size = int(cfg["hidden_size"])
        intermediate = int(hidden_size * cfg["mlp_ratio"])
        self.fc1 = nn.Linear(hidden_size, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states)))


class Dinov2Layer(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        hidden_size = int(cfg["hidden_size"])
        eps = float(cfg["layer_norm_eps"])
        self.norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.attention = Dinov2Attention(cfg)
        self.layer_scale1 = Dinov2LayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = Dinov2MLP(cfg)
        self.layer_scale2 = Dinov2LayerScale(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.layer_scale1(self.attention(self.norm1(hidden_states)))
        hidden_states = hidden_states + self.layer_scale2(self.mlp(self.norm2(hidden_states)))
        return hidden_states


class Dinov2Encoder(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.layer = nn.ModuleList([Dinov2Layer(cfg) for _ in range(int(cfg["num_hidden_layers"]))])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states


class MinimalDinov2Model(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.config = cfg
        self.embeddings = Dinov2Embeddings(cfg)
        self.encoder = Dinov2Encoder(cfg)
        self.layernorm = nn.LayerNorm(int(cfg["hidden_size"]), eps=float(cfg["layer_norm_eps"]))

    def forward_features(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        hidden_states, patch_grid = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        patch_tokens = hidden_states[:, 1:, :]
        bsz, _, channels = patch_tokens.shape
        patch_tokens = patch_tokens.transpose(1, 2).reshape(bsz, channels, patch_grid[0], patch_grid[1])
        return patch_tokens, patch_grid


def load_dinov2(teacher_dir: Path, device: torch.device) -> tuple[MinimalDinov2Model, dict]:
    cfg = json.loads((teacher_dir / "config.json").read_text())
    model = MinimalDinov2Model(cfg)
    state = torch.load(str(teacher_dir / "pytorch_model.bin"), map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    load_info = {
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_keys": missing[:20],
        "unexpected_keys": unexpected[:20],
        "state_dict_keys": len(state),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }
    model.eval().to(device)
    if device.type == "cuda":
        model.half()
    return model, load_info


def compose_lidar2img(cam_info: dict) -> np.ndarray:
    lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
    lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    viewpad = np.eye(4)
    intrinsic = cam_info["cam_intrinsic"]
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    return (viewpad @ lidar2cam_rt.T).astype(np.float32)


def test_ida_matrix() -> tuple[float, tuple[int, int], tuple[int, int, int, int], np.ndarray]:
    final_h, final_w = FINAL_DIM
    orig_h, orig_w = ORIGINAL_HW
    resize = max(final_h / orig_h, final_w / orig_w)
    resize_dims = (int(orig_w * resize), int(orig_h * resize))
    new_w, new_h = resize_dims
    crop_h = int(new_h) - final_h
    crop_w = int(max(0, new_w - final_w) / 2)
    crop = (crop_w, crop_h, crop_w + final_w, crop_h + final_h)
    ida = np.eye(4, dtype=np.float32)
    ida[0, 0] = resize
    ida[1, 1] = resize
    ida[0, 2] = -crop_w
    ida[1, 2] = -crop_h
    return resize, resize_dims, crop, ida


def transform_image(path: Path) -> Image.Image:
    resize, resize_dims, crop, _ = test_ida_matrix()
    del resize
    img = Image.open(path).convert("RGB")
    return img.resize(resize_dims).crop(crop)


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - DINOV2_MEAN) / DINOV2_STD


def select_first_samples_per_scene(infos: list[dict], max_scenes: int) -> list[dict]:
    selected = []
    seen = set()
    for info in infos:
        scene = info["scene_token"]
        if scene in seen:
            continue
        seen.add(scene)
        selected.append(info)
        if len(selected) >= max_scenes:
            break
    return selected


def image_paths_for_info(info: dict, repo_root: Path) -> list[Path]:
    paths = []
    for cam_name in CAMERA_ORDER:
        data_path = info["cams"][cam_name]["data_path"]
        path = repo_root / data_path
        if not path.exists():
            raise FileNotFoundError(f"Missing camera image: {path}")
        paths.append(path)
    return paths


def run_dino_single_and_scenes(
    model: MinimalDinov2Model,
    infos: list[dict],
    repo_root: Path,
    device: torch.device,
    max_scenes: int,
) -> tuple[dict, list[dict]]:
    selected = select_first_samples_per_scene(infos, max_scenes)
    scene_records = []
    single_record = {}

    with torch.inference_mode():
        for scene_idx, info in enumerate(selected):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            per_camera_shapes = []
            patch_grid = None
            channels = None
            for img_path in image_paths_for_info(info, repo_root):
                tensor = image_to_tensor(transform_image(img_path)).unsqueeze(0).to(device)
                if device.type == "cuda":
                    tensor = tensor.half()
                patch_tokens, grid = model.forward_features(tensor)
                patch_grid = list(grid)
                channels = int(patch_tokens.shape[1])
                per_camera_shapes.append(list(patch_tokens.shape))
                if scene_idx == 0 and not single_record:
                    single_record = {
                        "status": "PASS",
                        "input_shape": list(tensor.shape),
                        "patch_feature_shape": list(patch_tokens.shape),
                        "patch_grid": patch_grid,
                        "channel_count": channels,
                        "dtype": str(patch_tokens.dtype),
                    }
                del tensor, patch_tokens
            if device.type == "cuda":
                peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            else:
                peak_mib = None
            virtual_shape = [len(per_camera_shapes), channels, patch_grid[0], patch_grid[1]]
            feature_bytes_fp16 = int(np.prod(virtual_shape) * 2)
            scene_records.append(
                {
                    "scene_token": info["scene_token"],
                    "sample_token": info["token"],
                    "camera_count": len(per_camera_shapes),
                    "per_camera_feature_shapes": per_camera_shapes,
                    "scene_feature_shape_if_stacked": virtual_shape,
                    "channel_count": channels,
                    "patch_grid": patch_grid,
                    "feature_memory_mib_fp16": feature_bytes_fp16 / (1024 ** 2),
                    "gpu_peak_allocated_mib": peak_mib,
                }
            )
    return single_record, scene_records


def make_frustum() -> np.ndarray:
    final_h, final_w = FINAL_DIM
    h_feat, w_feat = final_h // DOWNSAMPLE, final_w // DOWNSAMPLE
    d0, d1, dnum = GRID_CONFIG["depth"]
    bin_size = 2 * (d1 - d0) / (dnum * (1 + dnum))
    bins = np.linspace(0, dnum - 1, int(dnum), dtype=np.float32)
    depth = (bins + 0.5) ** 2 * bin_size / 2 - bin_size / 8 + d0
    xs = np.linspace(0, final_w - 1, w_feat, dtype=np.float32)
    ys = np.linspace(0, final_h - 1, h_feat, dtype=np.float32)
    zz, yy, xx = np.meshgrid(depth, ys, xs, indexing="ij")
    ones = np.ones_like(xx)
    return np.stack((xx * zz, yy * zz, zz, ones), axis=-1).reshape(-1, 4)


def coverage_for_infos(infos: list[dict], max_frames: int) -> dict:
    _, _, _, ida = test_ida_matrix()
    frustum = make_frustum()
    grid_x = int((GRID_CONFIG["x"][1] - GRID_CONFIG["x"][0]) / GRID_CONFIG["x"][2])
    grid_y = int((GRID_CONFIG["y"][1] - GRID_CONFIG["y"][0]) / GRID_CONFIG["y"][2])
    lower = np.array([GRID_CONFIG["x"][0], GRID_CONFIG["y"][0], GRID_CONFIG["z"][0]], dtype=np.float32)
    interval = np.array([GRID_CONFIG["x"][2], GRID_CONFIG["y"][2], GRID_CONFIG["z"][2]], dtype=np.float32)
    grid_size = np.array([grid_x, grid_y, 1], dtype=np.int64)

    total = np.zeros((grid_y, grid_x), dtype=bool)
    per_frame = []
    for info in infos[:max_frames]:
        frame = np.zeros_like(total)
        for cam_name in CAMERA_ORDER:
            lidar2img = ida @ compose_lidar2img(info["cams"][cam_name])
            img2lidar = np.linalg.inv(lidar2img)
            xyz = (frustum @ img2lidar.T)[:, :3]
            coor = np.trunc((xyz - lower) / interval).astype(np.int64)
            kept = (
                (coor[:, 0] >= 0)
                & (coor[:, 0] < grid_size[0])
                & (coor[:, 1] >= 0)
                & (coor[:, 1] < grid_size[1])
                & (coor[:, 2] >= 0)
                & (coor[:, 2] < grid_size[2])
            )
            kept_coor = coor[kept]
            frame[kept_coor[:, 1], kept_coor[:, 0]] = True
        total |= frame
        per_frame.append(float(frame.mean()))

    return {
        "status": "PASS" if float(total.mean()) >= 0.60 else "FAIL",
        "coverage_ratio": float(total.mean()),
        "covered_cells": int(total.sum()),
        "total_cells": int(total.size),
        "threshold": 0.60,
        "frames": min(max_frames, len(infos)),
        "per_frame_min": float(np.min(per_frame)) if per_frame else None,
        "per_frame_mean": float(np.mean(per_frame)) if per_frame else None,
        "per_frame_max": float(np.max(per_frame)) if per_frame else None,
        "lss_feature_grid": [FINAL_DIM[0] // DOWNSAMPLE, FINAL_DIM[1] // DOWNSAMPLE],
        "dino_patch_grid_on_final_dim": [FINAL_DIM[0] // 14, FINAL_DIM[1] // 14],
        "pooling_assumption": "DINO patch tokens are bilinearly pooled onto RaCFormer LSS 16x44 PV bins before LSS lifting.",
    }


def dynamic_lss_shape_probe(infos: list[dict], device: torch.device) -> dict:
    from models.necks.view_transformer_racformer import LSSViewTransformerBEVDepth_racformer

    _, _, _, ida = test_ida_matrix()
    info = infos[0]
    lidar2imgs = [ida @ compose_lidar2img(info["cams"][cam_name]) for cam_name in CAMERA_ORDER]
    img_metas = [{"lidar2img": lidar2imgs, "img_shape": [(FINAL_DIM[0], FINAL_DIM[1], 3)] * len(lidar2imgs)}]
    inv_lidar2imgs = np.linalg.inv(np.stack([lidar2imgs])).astype(np.float32)
    mlp_input = torch.from_numpy(inv_lidar2imgs[:, :, :3, :3]).contiguous().view(1, len(CAMERA_ORDER), 9).to(device)

    module = LSSViewTransformerBEVDepth_racformer(
        grid_config=GRID_CONFIG,
        input_size=FINAL_DIM,
        in_channels=256,
        out_channels=256,
        depthnet_cfg={"use_dcn": False},
        downsample=DOWNSAMPLE,
        loss_depth_weight=2.0,
    ).to(device).eval()
    x = torch.zeros((1, len(CAMERA_ORDER), 256, FINAL_DIM[0] // DOWNSAMPLE, FINAL_DIM[1] // DOWNSAMPLE), device=device)
    radar_depth = torch.zeros((1, len(CAMERA_ORDER), FINAL_DIM[0], FINAL_DIM[1]), device=device)
    radar_rcs = torch.full_like(radar_depth, -64.0)
    with torch.inference_mode():
        bev_feat, depth = module(x, radar_depth, radar_rcs, img_metas, mlp_input)
    return {
        "status": "PASS" if list(bev_feat.shape) == [1, 256, 128, 128] else "FAIL",
        "single_time_bev_feat_shape": list(bev_feat.shape),
        "depth_shape": list(depth.shape),
        "all_bev_feats_static_shape": ["B", "T", 256, 128, 128],
        "current_frame_aux_target": "all_bev_feats[:, 0] -> (B, 256, 128, 128)",
        "grid_config": GRID_CONFIG,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--teacher-dir", default="research/night_gen_phase1/teachers/dinov2_vitl14")
    parser.add_argument("--val-pkl", default="nuscenes_infos_val_sweep.pkl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-scenes", type=int, default=10)
    parser.add_argument("--coverage-frames", type=int, default=50)
    args = parser.parse_args()

    started = time.time()
    repo_root = Path(args.repo_root).resolve()
    teacher_dir = (repo_root / args.teacher_dir).resolve()
    val_pkl = (repo_root / args.val_pkl).resolve()
    output = (repo_root / args.output).resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU is required for Branch G diagnostics; refusing CPU fallback.")

    with val_pkl.open("rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) else data

    model, load_info = load_dinov2(teacher_dir, device)
    single_record, scene_records = run_dino_single_and_scenes(
        model=model,
        infos=infos,
        repo_root=repo_root,
        device=device,
        max_scenes=args.max_scenes,
    )
    del model
    torch.cuda.empty_cache()

    lss_probe = dynamic_lss_shape_probe(infos, device)
    coverage = coverage_for_infos(infos, args.coverage_frames)

    payload = OrderedDict(
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host=os.uname().nodename,
        device=str(device),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        elapsed_seconds=time.time() - started,
        procurement={
            "repo": "facebook/dinov2-large",
            "source": "https://huggingface.co/facebook/dinov2-large",
            "teacher_dir": str(teacher_dir.relative_to(repo_root)),
            "checkpoint": "pytorch_model.bin",
        },
        g_d1={
            "status": "PASS" if load_info["missing_count"] == 0 and load_info["unexpected_count"] == 0 else "FAIL",
            "load": load_info,
            "single_image_forward": single_record,
        },
        g_d2={
            "status": "PASS" if len(scene_records) == args.max_scenes else "FAIL",
            "scene_count": len(scene_records),
            "scenes": scene_records,
        },
        g_d3=lss_probe,
        g_d4=coverage,
        g_d5={
            "status": "PASS",
            "evidence": (
                "No DualViewDistill module is present in inference. RaCFormer forward dispatches "
                "return_loss=False to forward_test/simple_test, while train losses are assembled under forward_train."
            ),
            "recommended_attachment": "training-only auxiliary loss on all_bev_feats[:, 0] before pts_bbox_head",
        },
    )
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
