"""
Efficient Attention Implementations for RaCFormer
Tier 2.5: Optimized attention using PyTorch SDPA

Drop-in replacement for ScaleAdaptiveSelfAttention that uses
F.scaled_dot_product_attention for fused kernels on Ampere GPUs.

Checkpoint-compatible: parameter names match MMCV's MultiheadAttention
so pretrained weights load without remapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from .checkpoint import checkpoint as cp
from .bbox.utils import decode_bbox, theta_d2xy_coods
from .utils import DUMP


class SDPAInnerAttention(nn.Module):
    """Inner attention module with parameter names matching nn.MultiheadAttention.

    Uses F.scaled_dot_product_attention for fused kernels while keeping
    parameter names (in_proj_weight, in_proj_bias, out_proj) compatible
    with MMCV's MultiheadAttention for checkpoint loading.
    """

    def __init__(self, embed_dims, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.dropout = dropout

        assert embed_dims % num_heads == 0

        # Use same parameter names as nn.MultiheadAttention / MMCV
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dims, embed_dims))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dims))
        self.out_proj = nn.Linear(embed_dims, embed_dims)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query, attn_mask=None):
        """
        Args:
            query: [B, Q, C] (self-attention: used as Q, K, V)
            attn_mask: [B*num_heads, Q, Q] attention mask

        Returns:
            [B, Q, C]
        """
        B, Q, C = query.shape

        # Fused QKV projection using in_proj_weight/bias
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)  # [B, Q, 3*C]
        qkv = qkv.reshape(B, Q, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, Q, head_dim]
        q, k, v = qkv.unbind(0)  # Each: [B, num_heads, Q, head_dim]

        # Reshape attention mask for SDPA
        if attn_mask is not None:
            attn_mask = attn_mask.reshape(B, self.num_heads, Q, Q)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # [B, num_heads, Q, head_dim]

        out = out.transpose(1, 2).reshape(B, Q, C)
        out = self.out_proj(out)
        return out


class SDPAMultiheadAttention(nn.Module):
    """Wrapper that nests SDPAInnerAttention under 'attn' to match
    MMCV MultiheadAttention checkpoint paths: attention.attn.in_proj_weight, etc."""

    def __init__(self, embed_dims, num_heads, dropout=0.1, batch_first=True):
        super().__init__()
        self.attn = SDPAInnerAttention(embed_dims, num_heads, dropout)

    def forward(self, query, attn_mask=None):
        # Residual connection matching MMCV MultiheadAttention behavior:
        # MMCV returns `identity + dropout(proj_drop(attention_output))`
        # proj_drop and dropout_layer both default to 0.0, so just add residual
        return query + self.attn(query, attn_mask=attn_mask)


class EfficientScaleAdaptiveSelfAttention(BaseModule):
    """
    Scale-adaptive Self Attention with SDPA backend.

    Drop-in replacement for ScaleAdaptiveSelfAttention. Parameter names
    are checkpoint-compatible with the original MMCV-based implementation.
    """

    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1, pc_range=[], init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.pc_range = pc_range

        self.attention = SDPAMultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_bbox, query_feat, pre_attn_mask):
        """
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        """
        query_bbox = theta_d2xy_coods(query_bbox).clone()
        dist = self.calc_bbox_dists(query_bbox)
        tau = self.gen_tau(query_feat)  # [B, Q, num_heads]

        if DUMP.enabled:
            torch.save(tau, '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)  # [B, num_heads, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, num_heads, Q, Q]

        if pre_attn_mask is not None:  # for query denoising
            attn_mask[:, :, pre_attn_mask] = float('-inf')

        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_bbox, query_feat, pre_attn_mask):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_bbox, query_feat, pre_attn_mask, use_reentrant=False)
        else:
            return self.inner_forward(query_bbox, query_feat, pre_attn_mask)

    @torch.no_grad()
    def calc_bbox_dists(self, bboxes):
        centers = decode_bbox(bboxes, self.pc_range)[..., :2]  # [B, Q, 2]

        dist = []
        for b in range(centers.shape[0]):
            dist_b = torch.norm(centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2), dim=-1)
            dist.append(dist_b[None, ...])

        dist = torch.cat(dist, dim=0)  # [B, Q, Q]
        dist = -dist

        return dist
