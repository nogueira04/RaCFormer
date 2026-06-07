from pathlib import Path
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dinov2PatchEmbeddings(nn.Module):
    def __init__(self, hidden_size, patch_size, num_channels):
        super().__init__()
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)


class Dinov2Embeddings(nn.Module):
    def __init__(self, cfg):
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

    def interpolate_pos_encoding(self, patch_tokens, height, width):
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

    def forward(self, pixel_values):
        _, _, height, width = pixel_values.shape
        patch = self.patch_embeddings.projection(pixel_values)
        patch_grid = (patch.shape[-2], patch.shape[-1])
        patch = patch.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(pixel_values.shape[0], -1, -1)
        embeddings = torch.cat((cls, patch), dim=1)
        embeddings = embeddings + self.interpolate_pos_encoding(patch, height, width)
        return embeddings, patch_grid


class Dinov2SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_size = int(cfg["hidden_size"])
        self.num_heads = int(cfg["num_attention_heads"])
        self.head_dim = hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(hidden_size, hidden_size, bias=bool(cfg["qkv_bias"]))
        self.key = nn.Linear(hidden_size, hidden_size, bias=bool(cfg["qkv_bias"]))
        self.value = nn.Linear(hidden_size, hidden_size, bias=bool(cfg["qkv_bias"]))

    def _shape(self, x):
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states):
        query = self._shape(self.query(hidden_states))
        key = self._shape(self.key(hidden_states))
        value = self._shape(self.value(hidden_states))
        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        context = torch.matmul(attn, value)
        return context.transpose(1, 2).reshape(hidden_states.shape)


class Dinov2SelfOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)


class Dinov2Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = Dinov2SelfAttention(cfg)
        self.output = Dinov2SelfOutput(int(cfg["hidden_size"]))

    def forward(self, hidden_states):
        return self.output.dense(self.attention(hidden_states))


class Dinov2LayerScale(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        return hidden_states * self.lambda1


class Dinov2MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_size = int(cfg["hidden_size"])
        intermediate = int(hidden_size * cfg["mlp_ratio"])
        self.fc1 = nn.Linear(hidden_size, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden_size)

    def forward(self, hidden_states):
        return self.fc2(F.gelu(self.fc1(hidden_states)))


class Dinov2Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_size = int(cfg["hidden_size"])
        eps = float(cfg["layer_norm_eps"])
        self.norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.attention = Dinov2Attention(cfg)
        self.layer_scale1 = Dinov2LayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = Dinov2MLP(cfg)
        self.layer_scale2 = Dinov2LayerScale(hidden_size)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.layer_scale1(self.attention(self.norm1(hidden_states)))
        hidden_states = hidden_states + self.layer_scale2(self.mlp(self.norm2(hidden_states)))
        return hidden_states


class Dinov2Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.ModuleList([Dinov2Layer(cfg) for _ in range(int(cfg["num_hidden_layers"]))])

    def forward(self, hidden_states):
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states


class MinimalDinov2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.embeddings = Dinov2Embeddings(cfg)
        self.encoder = Dinov2Encoder(cfg)
        self.layernorm = nn.LayerNorm(int(cfg["hidden_size"]), eps=float(cfg["layer_norm_eps"]))

    def forward_features(self, pixel_values):
        hidden_states, patch_grid = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        patch_tokens = hidden_states[:, 1:, :]
        bsz, _, channels = patch_tokens.shape
        patch_tokens = patch_tokens.transpose(1, 2).reshape(bsz, channels, patch_grid[0], patch_grid[1])
        return patch_tokens, patch_grid


def load_dinov2(teacher_dir):
    teacher_dir = Path(teacher_dir)
    cfg = json.loads((teacher_dir / "config.json").read_text())
    model = MinimalDinov2Model(cfg)
    state = torch.load(str(teacher_dir / "pytorch_model.bin"), map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    load_info = {
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "state_dict_keys": len(state),
        "hidden_size": int(cfg["hidden_size"]),
    }
    return model, load_info


class DualViewDistillLoss(nn.Module):
    def __init__(
        self,
        teacher_dir,
        student_channels=256,
        dino_channels=1024,
        loss_weight=0.05,
        cosine_weight=1.0,
        mse_weight=1.0,
        teacher_half=True,
    ):
        super().__init__()
        self.teacher_dir = teacher_dir
        self.student_channels = int(student_channels)
        self.dino_channels = int(dino_channels)
        self.loss_weight = float(loss_weight)
        self.cosine_weight = float(cosine_weight)
        self.mse_weight = float(mse_weight)
        self.teacher_half = bool(teacher_half)
        self.target_adapter = nn.Conv2d(self.dino_channels, self.student_channels, kernel_size=1)
        self.register_buffer("dino_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("dino_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.last_stats = {}

        self.teacher = None
        self.teacher_load_info = {"status": "not_loaded_loss_weight_zero"}
        if self.loss_weight > 0:
            self.teacher, self.teacher_load_info = load_dinov2(teacher_dir)
            for param in self.teacher.parameters():
                param.requires_grad_(False)
            self.teacher.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def _teacher_dtype(self):
        if self.teacher is None:
            return torch.float32
        return next(self.teacher.parameters()).dtype

    def _ensure_teacher_runtime(self, device):
        if self.teacher is None:
            raise RuntimeError("DualViewDistillLoss teacher is not loaded because loss_weight <= 0")
        if next(self.teacher.parameters()).device != device:
            self.teacher.to(device)
        if device.type == "cuda" and self.teacher_half and self._teacher_dtype() != torch.float16:
            self.teacher.half()
        self.teacher.eval()

    def _current_frame_images(self, img, num_cams):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)
        if img.dim() != 5:
            raise ValueError(f"expected img shape (B, NT, C, H, W), got {tuple(img.shape)}")
        current = img[:, :num_cams].float()
        current = current[:, :, [2, 1, 0], :, :] / 255.0
        return current

    def _dino_pv_features(self, img, num_cams):
        current = self._current_frame_images(img, num_cams)
        batch_size, _, _, _, _ = current.shape
        features = []
        per_camera_shapes = []
        patch_grid = None
        self._ensure_teacher_runtime(current.device)
        dtype = self._teacher_dtype()
        with torch.no_grad():
            for cam_idx in range(num_cams):
                camera_img = current[:, cam_idx]
                camera_img = (camera_img - self.dino_mean) / self.dino_std
                camera_img = camera_img.to(dtype=dtype)
                patch_tokens, grid = self.teacher.forward_features(camera_img)
                patch_grid = list(grid)
                per_camera_shapes.append(list(patch_tokens.shape))
                features.append(patch_tokens.float())
        dino_pv = torch.stack(features, dim=1)
        return dino_pv, {
            "batch_size": int(batch_size),
            "per_camera_dino_shapes": per_camera_shapes,
            "patch_grid": patch_grid,
        }

    @staticmethod
    def _current_frame_metas(img_metas, num_cams):
        current_metas = []
        for meta in img_metas:
            current_meta = dict(meta)
            current_meta["lidar2img"] = meta["lidar2img"][:num_cams]
            current_meta["img_shape"] = meta["img_shape"][:num_cams]
            current_metas.append(current_meta)
        return current_metas

    def _pool_dino_to_bev(self, dino_pv, depth_logits, img_metas, view_transformer, num_cams):
        batch_size, num_cams_seen, channels, _, _ = dino_pv.shape
        if num_cams_seen != num_cams:
            raise ValueError(f"expected {num_cams} cameras, got {num_cams_seen}")
        bn, depth_channels, h_lss, w_lss = depth_logits.shape
        if bn != batch_size * num_cams:
            raise ValueError(f"depth batch {bn} does not match B*N {batch_size * num_cams}")

        dino_lss = F.interpolate(
            dino_pv.flatten(0, 1),
            size=(h_lss, w_lss),
            mode="bilinear",
            align_corners=False,
        ).view(batch_size, num_cams, channels, h_lss, w_lss).contiguous()
        depth_prob = depth_logits.detach().float().softmax(dim=1)
        depth_prob = depth_prob.view(batch_size, num_cams, depth_channels, h_lss, w_lss).contiguous()
        current_metas = self._current_frame_metas(img_metas, num_cams)
        dummy_img = dino_lss.new_zeros((batch_size, num_cams, 1, h_lss, w_lss))
        coor = view_transformer.get_lidar_coor(dummy_img, current_metas)

        dino_bev_sum = view_transformer.voxel_pooling_v2(coor, depth_prob, dino_lss.float())
        weights = view_transformer.voxel_pooling_v2(
            coor,
            depth_prob,
            dino_lss.new_ones((batch_size, num_cams, 1, h_lss, w_lss)),
        )
        mask = weights > 1e-6
        dino_bev = dino_bev_sum / weights.clamp_min(1e-6)
        return dino_bev, mask, {
            "lss_pv_shape": [batch_size, num_cams, channels, h_lss, w_lss],
            "bev_target_shape": list(dino_bev.shape),
            "coverage_ratio": float(mask.float().mean().detach().cpu()),
            "covered_cells": int(mask.sum().detach().cpu()),
            "total_cells": int(mask.numel()),
        }

    def forward(self, student_bev, img, depth_logits, img_metas, view_transformer, num_cams):
        if self.loss_weight <= 0:
            return {}

        dino_pv, dino_stats = self._dino_pv_features(img, num_cams)
        dino_bev, mask, pool_stats = self._pool_dino_to_bev(
            dino_pv=dino_pv,
            depth_logits=depth_logits,
            img_metas=img_metas,
            view_transformer=view_transformer,
            num_cams=num_cams,
        )
        target = self.target_adapter(dino_bev.to(dtype=self.target_adapter.weight.dtype))
        target = target.to(device=student_bev.device, dtype=torch.float32)
        student = student_bev.float()
        mask = mask.to(device=student.device)

        denom = (mask.float().sum() * student.shape[1]).clamp_min(1.0)
        mse_loss = (((student - target) ** 2) * mask.float()).sum() / denom
        cosine = F.cosine_similarity(student, target, dim=1, eps=1e-6)
        cosine_loss = ((1.0 - cosine) * mask.squeeze(1).float()).sum() / mask.float().sum().clamp_min(1.0)
        total = self.loss_weight * (self.cosine_weight * cosine_loss + self.mse_weight * mse_loss)

        self.last_stats = {
            **dino_stats,
            **pool_stats,
            "student_bev_shape": list(student_bev.shape),
            "adapted_target_shape": list(target.shape),
            "loss_dualview_distill": float(total.detach().cpu()),
            "loss_cosine_unweighted": float(cosine_loss.detach().cpu()),
            "loss_mse_unweighted": float(mse_loss.detach().cpu()),
            "teacher_load_info": self.teacher_load_info,
        }
        return {"loss_dualview_distill": total}
