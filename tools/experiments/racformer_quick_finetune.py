# pyright: reportMissingImports=false
"""Short deterministic-ish RaCFormer fine-tune used in the Orin matrix.

This is the versioned replacement for the temporary
``/tmp/racformer_quick_finetune.py`` used for the May 2026 Orin experiments.
It intentionally matches the narrow experimental use case: mini split, a small
number of steps, optional frozen image backbone/neck, and a single output
checkpoint for follow-up validation.
"""

from __future__ import annotations

import argparse
import importlib
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.runner import load_checkpoint, set_random_seed
from mmdet.registry import MODELS as MMDET_MODELS
from mmdet.registry import TASK_UTILS as MMDET_TASK_UTILS
from mmdet3d.datasets.convert_utils import NuScenesNameMapping
from mmdet3d.registry import DATASETS
from mmdet3d.registry import MODELS
from mmdet3d.registry import MODELS as MMDET3D_MODELS
from mmdet3d.registry import TASK_UTILS as MMDET3D_TASK_UTILS
from mmdet3d.registry import TRANSFORMS as MMDET3D_TRANSFORMS
from mmdet3d.structures import LiDARInstance3DBoxes


os.environ.setdefault("NUSCENES_VERSION", "v1.0-mini")


def register_project() -> None:
    importlib.import_module("models")
    importlib.import_module("loaders")
    import mmdet3d.datasets.transforms  # noqa: F401
    import models.efficient_attention as ea
    import models.racformer_transformer as rt

    rt.cp = lambda fn, *args, **kwargs: fn(*args)
    ea.cp = lambda fn, *args, **kwargs: fn(*args)
    for name, module in MMDET3D_TRANSFORMS.module_dict.items():
        if name not in MMENGINE_TRANSFORMS.module_dict:
            MMENGINE_TRANSFORMS.register_module(name=name, module=module)
    for name, module in MMDET_MODELS.module_dict.items():
        if name not in MMDET3D_MODELS.module_dict:
            MMDET3D_MODELS.register_module(name=name, module=module)
    for name, module in MMDET_TASK_UTILS.module_dict.items():
        if name not in MMDET3D_TASK_UTILS.module_dict:
            MMDET3D_TASK_UTILS.register_module(name=name, module=module)
    for name, module in MMDET3D_TASK_UTILS.module_dict.items():
        if name not in MMDET_TASK_UTILS.module_dict:
            MMDET_TASK_UTILS.register_module(name=name, module=module)


def patch_dataset() -> None:
    from loaders.nuscenes_dataset import CustomNuScenesDataset_radar

    def patched_get_ann_info(self, index):
        info = self.data_infos[index]
        mask = info.get("valid_flag", np.ones(len(info["gt_boxes"]), dtype=bool)).astype(bool)
        mapped = np.array([NuScenesNameMapping.get(n, n) for n in info["gt_names"]])
        class_names = self.metainfo.get("classes", self.CLASSES)
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        cls_mask = np.array([name in class_to_idx for name in mapped], dtype=bool)
        mask = mask & cls_mask
        boxes = info["gt_boxes"][mask].astype(np.float32)
        vel = info.get("gt_velocity", np.zeros((len(info["gt_boxes"]), 2), dtype=np.float32))[mask].astype(np.float32)
        vel[np.isnan(vel)] = 0.0
        boxes = np.concatenate([boxes, vel], axis=1) if boxes.size else np.zeros((0, 9), dtype=np.float32)
        labels = np.array([class_to_idx[name] for name in mapped[mask]], dtype=np.int64)
        return {
            "gt_bboxes_3d": LiDARInstance3DBoxes(
                boxes, box_dim=boxes.shape[-1], origin=(0.5, 0.5, 0.5)
            ).convert_to(self.box_mode_3d),
            "gt_labels_3d": labels,
        }

    CustomNuScenesDataset_radar.get_ann_info = patched_get_ann_info


class LegacyFocalLossCost:
    def __init__(self, weight=1.0, alpha=0.25, gamma=2.0, eps=1e-12):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        p = cls_pred.sigmoid()
        neg = -(1 - p + self.eps).log() * (1 - self.alpha) * p.pow(self.gamma)
        pos = -(p + self.eps).log() * self.alpha * (1 - p).pow(self.gamma)
        return (pos[:, gt_labels] - neg[:, gt_labels]) * self.weight


class LegacyPseudoSampler:
    def sample(self, assign_result, bboxes, gt_bboxes):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        if pos_inds.numel():
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds]
        else:
            pos_gt_bboxes = gt_bboxes.new_zeros((0, gt_bboxes.size(-1)))
        return SimpleNamespace(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            pos_assigned_gt_inds=pos_assigned_gt_inds,
            pos_gt_bboxes=pos_gt_bboxes,
        )


def clear_training_caches(model) -> None:
    for module in model.modules():
        if hasattr(module, "_cached_bev_pos"):
            module._cached_bev_pos = None


def sample_to_cuda(sample):
    return {
        "img_metas": [sample["img_metas"]],
        "img": sample["img"].cuda().float(),
        "gt_depth": sample["gt_depth"].unsqueeze(0).cuda().float(),
        "radar_depth": sample["radar_depth"].cuda().float(),
        "radar_rcs": sample["radar_rcs"].cuda().float(),
        "radar_points": [point.cuda().float() for point in sample["radar_points"]],
        "gt_bboxes_3d": [sample["gt_bboxes_3d"].to("cuda")],
        "gt_labels_3d": [torch.as_tensor(sample["gt_labels_3d"], device="cuda", dtype=torch.long)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    register_project()
    patch_dataset()
    set_random_seed(0, deterministic=False)
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(args.config)
    cfg.model.img_backbone.with_cp = False
    cfg.data.train.filter_empty_gt = False
    dataset = DATASETS.build(cfg.data.train)
    model = MODELS.build(cfg.model).cuda().train()
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=False)
    model.pts_bbox_head.assigner.cls_cost = LegacyFocalLossCost(weight=2.0, alpha=0.25, gamma=2.0)
    model.pts_bbox_head.sampler = LegacyPseudoSampler()

    if args.freeze_backbone:
        for module_name in ["img_backbone", "img_neck"]:
            module = getattr(model, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad_(False)

    params = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    indices = list(range(len(dataset)))
    rng = random.Random(0)
    losses = []

    steps = 1 if args.smoke else args.steps
    for step in range(steps):
        if step % len(indices) == 0:
            rng.shuffle(indices)
        idx = indices[step % len(indices)]
        sample = dataset[idx]
        data = sample_to_cuda(sample)
        opt.zero_grad(set_to_none=True)
        clear_training_caches(model)
        loss_dict = model(return_loss=True, **data)
        loss_terms = [value.mean() for value in loss_dict.values() if torch.is_tensor(value)]
        if not loss_terms:
            raise RuntimeError("No tensor losses returned by model")
        loss = torch.stack(loss_terms).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 5 == 0 or step == 0 or args.smoke:
            recent = sum(losses[-5:]) / min(len(losses), 5)
            print(f"step {step + 1}/{steps} loss={losses[-1]:.4f} recent5={recent:.4f}", flush=True)

    if args.smoke:
        print("SMOKE_OK", flush=True)
        return 0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "meta": {"steps": args.steps, "lr": args.lr, "source": args.checkpoint}},
        args.out,
    )
    print(f"SAVED {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
