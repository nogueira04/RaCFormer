"""Run RaCFormer validation with a temporary top-k decoder refinement patch.

This wrapper patches RaCFormerTransformerDecoder.forward before delegating to
val.py. It is intended for inference-only experiments on the remote Jetson.
Configuration is via environment variables so val.py's CLI stays unchanged:

RACFORMER_FULL_LAYERS: number of full-query decoder layers to run first.
RACFORMER_TOPK: number of top-scoring queries to refine in later layers.
RACFORMER_SUBSET_LAYERS: number of subset refinement layers to run.
"""

from __future__ import annotations

import os
import sys

import torch

from models.bbox.utils import theta_d2xy_coods
from models.racformer_transformer import RaCFormerTransformerDecoder


def _gather_queries(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather [B, K, ...] query rows from a [B, Q, ...] tensor."""
    view_shape = [index.shape[0], index.shape[1]] + [1] * (tensor.dim() - 2)
    expand_shape = [index.shape[0], index.shape[1]] + list(tensor.shape[2:])
    gather_index = index.view(*view_shape).expand(*expand_shape)
    return torch.gather(tensor, 1, gather_index)


def _scatter_queries(base: torch.Tensor, index: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Scatter [B, K, ...] updates into a [B, Q, ...] clone."""
    out = base.clone()
    view_shape = [index.shape[0], index.shape[1]] + [1] * (base.dim() - 2)
    expand_shape = [index.shape[0], index.shape[1]] + list(base.shape[2:])
    scatter_index = index.view(*view_shape).expand(*expand_shape)
    return out.scatter(1, scatter_index, updates)


def patched_forward(self, query_bbox, query_feat, mlvl_feats, lss_bev_feats, radar_bev_feats, attn_mask, img_metas):
    cls_scores, bbox_preds = [], []

    timestamps = self.np.array([m["img_timestamp"] for m in img_metas], dtype=self.np.float64)
    timestamps = self.np.reshape(timestamps, [query_bbox.shape[0], -1, self.num_cams])
    time_diff = timestamps[:, :1, :] - timestamps
    time_diff = self.np.mean(time_diff, axis=-1).astype(self.np.float32)
    time_diff = torch.from_numpy(time_diff).to(query_bbox.device)
    img_metas[0]["time_diff"] = time_diff

    lidar2img = self.np.asarray([m["lidar2img"] for m in img_metas]).astype(self.np.float32)
    lidar2img = torch.from_numpy(lidar2img).to(query_bbox.device)
    img_metas[0]["lidar2img"] = lidar2img

    for lvl, feat in enumerate(mlvl_feats):
        batch, tn, gc, height, width = feat.shape
        num_cams = self.num_cams
        frames = tn // num_cams
        groups = 4
        channels = gc // groups
        feat = feat.reshape(batch, frames, num_cams, groups, channels, height, width)
        if self.msmv_cuda:
            feat = feat.permute(0, 1, 3, 2, 5, 6, 4)
            feat = feat.reshape(batch * frames * groups, num_cams, height, width, channels)
        else:
            feat = feat.permute(0, 1, 3, 4, 2, 5, 6)
            feat = feat.reshape(batch * frames * groups, channels, num_cams, height, width)
        mlvl_feats[lvl] = feat.contiguous()

    full_layers = int(os.environ.get("RACFORMER_FULL_LAYERS", "1"))
    subset_layers = int(os.environ.get("RACFORMER_SUBSET_LAYERS", "1"))
    topk = int(os.environ.get("RACFORMER_TOPK", "600"))
    full_layers = max(1, min(full_layers, self.num_layers))
    subset_layers = max(0, min(subset_layers, self.num_layers - full_layers))

    for layer in range(full_layers):
        query_feat, cls_score, bbox_pred_theta = self.decoder_layer(
            query_bbox, query_feat, mlvl_feats, lss_bev_feats, radar_bev_feats, attn_mask, img_metas, layer=layer
        )
        query_bbox = bbox_pred_theta.clone().detach()
        cls_scores.append(cls_score)
        bbox_preds.append(theta_d2xy_coods(bbox_pred_theta))

    if subset_layers == 0:
        return torch.stack(cls_scores), torch.stack(bbox_preds)

    k = min(topk, query_bbox.shape[1])
    scores = torch.sigmoid(cls_scores[-1]).amax(dim=-1)
    topk_index = torch.topk(scores, k=k, dim=1).indices

    for offset in range(subset_layers):
        layer = full_layers + offset
        sub_bbox = _gather_queries(query_bbox, topk_index)
        sub_feat = _gather_queries(query_feat, topk_index)
        sub_feat, sub_cls, sub_bbox_theta = self.decoder_layer(
            sub_bbox, sub_feat, mlvl_feats, lss_bev_feats, radar_bev_feats, None, img_metas, layer=layer
        )

        query_feat = _scatter_queries(query_feat, topk_index, sub_feat)
        query_bbox = _scatter_queries(query_bbox, topk_index, sub_bbox_theta.clone().detach())
        merged_cls = _scatter_queries(cls_scores[-1], topk_index, sub_cls)

        cls_scores.append(merged_cls)
        bbox_preds.append(theta_d2xy_coods(query_bbox))

    return torch.stack(cls_scores), torch.stack(bbox_preds)


def install_patch() -> None:
    import numpy as np
    from models.csrc.wrapper import MSMV_CUDA

    RaCFormerTransformerDecoder.np = np
    RaCFormerTransformerDecoder.msmv_cuda = MSMV_CUDA
    RaCFormerTransformerDecoder.forward = patched_forward


if __name__ == "__main__":
    install_patch()
    import val

    sys.exit(val.main())
