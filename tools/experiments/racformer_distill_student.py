import argparse
import importlib
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
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


def register_project():
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


def patch_dataset():
    from loaders.nuscenes_dataset import CustomNuScenesDataset_radar

    def patched_get_ann_info(self, index):
        info = self.data_infos[index]
        mask = info.get("valid_flag", np.ones(len(info["gt_boxes"]), dtype=bool)).astype(bool)
        mapped = np.array([NuScenesNameMapping.get(n, n) for n in info["gt_names"]])
        class_names = self.metainfo.get("classes", self.CLASSES)
        class_to_idx = {n: i for i, n in enumerate(class_names)}
        cls_mask = np.array([n in class_to_idx for n in mapped], dtype=bool)
        mask = mask & cls_mask
        boxes = info["gt_boxes"][mask].astype(np.float32)
        vel = info.get("gt_velocity", np.zeros((len(info["gt_boxes"]), 2), dtype=np.float32))[mask].astype(np.float32)
        vel[np.isnan(vel)] = 0.0
        boxes = np.concatenate([boxes, vel], axis=1) if boxes.size else np.zeros((0, 9), dtype=np.float32)
        labels = np.array([class_to_idx[n] for n in mapped[mask]], dtype=np.int64)
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


def make_deterministic_train_cfg(cfg):
    cfg.model.img_backbone.with_cp = False
    cfg.data.train.filter_empty_gt = False
    for step in cfg.data.train.pipeline:
        if step.get("type") == "RandomTransformImage":
            step["training"] = False
        elif step.get("type") == "RaCGlobalRotScaleTransImage":
            step["rot_range"] = [0.0, 0.0]
            step["scale_ratio_range"] = [1.0, 1.0]
    return cfg


def clear_training_caches(model):
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
        "radar_points": [p.cuda().float() for p in sample["radar_points"]],
        "gt_bboxes_3d": [sample["gt_bboxes_3d"].to("cuda")],
        "gt_labels_3d": [torch.as_tensor(sample["gt_labels_3d"], device="cuda", dtype=torch.long)],
    }


def patch_trt_backbone(model, engine_path):
    from export_tensorrt import TRTBackboneNeck, patch_extract_img_feat

    trt_engine = TRTBackboneNeck(engine_path, device="cuda:0")
    return patch_extract_img_feat(model, trt_engine)


def configure_trainable(model, scope):
    if scope == "all":
        return
    for param in model.parameters():
        param.requires_grad_(False)
    prefixes = {
        "head": ("pts_bbox_head",),
        "head_radar": ("pts_bbox_head", "radar_voxel_encoder", "radar_middle_encoder"),
        "head_view": ("pts_bbox_head", "img_lss_neck", "img_lss_view_transformer"),
    }[scope]
    for name, param in model.named_parameters():
        if name.startswith(prefixes):
            param.requires_grad_(True)


def forward_raw(model, data):
    clear_training_caches(model)
    img_feats, bev_feats, radar_bev_feats, _ = model.extract_feat(
        data["img"],
        data["radar_points"],
        data["radar_depth"],
        data["radar_rcs"],
        data["img_metas"],
    )
    return model.pts_bbox_head(img_feats, bev_feats, radar_bev_feats, data["img_metas"])


def forward_student_losses(model, data):
    clear_training_caches(model)
    img_feats, bev_feats, radar_bev_feats, depth = model.extract_feat(
        data["img"],
        data["radar_points"],
        data["radar_depth"],
        data["radar_rcs"],
        data["img_metas"],
    )
    for i, meta in enumerate(data["img_metas"]):
        meta["gt_bboxes_3d"] = data["gt_bboxes_3d"][i]
        meta["gt_labels_3d"] = data["gt_labels_3d"][i]
    outs = model.pts_bbox_head(img_feats, bev_feats, radar_bev_feats, data["img_metas"])
    gt_depth_first_frame = data["gt_depth"][:, : model.num_cams].contiguous()
    loss_depth = model.img_lss_view_transformer.get_depth_loss(gt_depth_first_frame, depth)
    losses = {"loss_depth": loss_depth}
    losses.update(model.pts_bbox_head.loss(data["gt_bboxes_3d"], data["gt_labels_3d"], outs))
    return losses, outs


def gather_queries(x, indices):
    expand = indices.unsqueeze(-1).expand(*indices.shape, x.size(-1))
    return x.gather(1, expand)


def distill_loss(student_outs, teacher_outs, topk, temperature):
    student_logits = student_outs["all_cls_scores"][-1]
    teacher_logits = teacher_outs["all_cls_scores"][-1].detach()
    student_boxes = student_outs["all_bbox_preds"][-1]
    teacher_boxes = teacher_outs["all_bbox_preds"][-1].detach()

    conf = teacher_logits.sigmoid().amax(dim=-1)
    k = min(topk, conf.size(1))
    top_idx = conf.topk(k, dim=1).indices

    s_logits = gather_queries(student_logits, top_idx)
    t_logits = gather_queries(teacher_logits, top_idx)
    s_boxes = gather_queries(student_boxes, top_idx)
    t_boxes = gather_queries(teacher_boxes, top_idx)
    weights = gather_queries(conf.unsqueeze(-1), top_idx).clamp_min(0.05).detach()

    cls_targets = torch.sigmoid(t_logits / temperature)
    cls_loss = F.binary_cross_entropy_with_logits(s_logits / temperature, cls_targets, reduction="none")
    cls_loss = (cls_loss * weights).sum() / (weights.sum() * cls_loss.size(-1)).clamp_min(1.0)
    cls_loss = cls_loss * (temperature * temperature)

    code_weights = torch.as_tensor([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2], device=s_boxes.device)
    box_loss = F.smooth_l1_loss(s_boxes, t_boxes, reduction="none", beta=1.0)
    box_loss = (box_loss * code_weights * weights).sum() / (weights.sum() * s_boxes.size(-1)).clamp_min(1.0)
    return cls_loss, box_loss


def scalar_loss(loss_dict):
    return sum(v.mean() for v in loss_dict.values() if torch.is_tensor(v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-config", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--teacher-config", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--gt-weight", type=float, default=0.1)
    parser.add_argument("--kd-cls-weight", type=float, default=2.0)
    parser.add_argument("--kd-box-weight", type=float, default=0.5)
    parser.add_argument("--train-scope", choices=["all", "head", "head_radar", "head_view"], default="head")
    parser.add_argument("--trt-backbone")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    register_project()
    patch_dataset()
    set_random_seed(0, deterministic=False)
    torch.backends.cudnn.benchmark = True

    student_cfg = make_deterministic_train_cfg(Config.fromfile(args.student_config))
    teacher_cfg = make_deterministic_train_cfg(Config.fromfile(args.teacher_config))
    student_ds = DATASETS.build(student_cfg.data.train)
    teacher_ds = DATASETS.build(teacher_cfg.data.train)

    teacher = MODELS.build(teacher_cfg.model).cuda().eval()
    load_checkpoint(teacher, args.teacher_checkpoint, map_location="cpu", strict=False)
    teacher.eval()

    student = MODELS.build(student_cfg.model).cuda().train()
    load_checkpoint(student, args.student_checkpoint, map_location="cpu", strict=False)
    student.pts_bbox_head.assigner.cls_cost = LegacyFocalLossCost(weight=2.0, alpha=0.25, gamma=2.0)
    student.pts_bbox_head.sampler = LegacyPseudoSampler()
    configure_trainable(student, args.train_scope)
    student.train()

    if args.trt_backbone:
        teacher = patch_trt_backbone(teacher, args.trt_backbone)
        student = patch_trt_backbone(student, args.trt_backbone)

    params = [p for p in student.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(f"No trainable parameters for scope {args.train_scope}")
    print(f"trainable_params={sum(p.numel() for p in params)} scope={args.train_scope}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    indices = list(range(min(len(student_ds), len(teacher_ds))))
    rng = random.Random(0)
    totals = []

    steps = 1 if args.smoke else args.steps
    for step in range(steps):
        if step % len(indices) == 0:
            rng.shuffle(indices)
        idx = indices[step % len(indices)]
        np.random.seed(idx)
        random.seed(idx)
        torch.manual_seed(idx)
        student_data = sample_to_cuda(student_ds[idx])
        np.random.seed(idx)
        random.seed(idx)
        torch.manual_seed(idx)
        teacher_data = sample_to_cuda(teacher_ds[idx])

        opt.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            teacher_outs = forward_raw(teacher, teacher_data)
        gt_losses, student_outs = forward_student_losses(student, student_data)
        gt_loss = scalar_loss(gt_losses)
        kd_cls, kd_box = distill_loss(student_outs, teacher_outs, args.topk, args.temperature)
        loss = args.gt_weight * gt_loss + args.kd_cls_weight * kd_cls + args.kd_box_weight * kd_box

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        row = (
            float(loss.detach().cpu()),
            float(gt_loss.detach().cpu()),
            float(kd_cls.detach().cpu()),
            float(kd_box.detach().cpu()),
        )
        totals.append(row)
        if (step + 1) % 5 == 0 or step == 0 or args.smoke:
            recent = totals[-5:]
            mean = [sum(r[i] for r in recent) / len(recent) for i in range(4)]
            print(
                f"step {step + 1}/{steps} loss={row[0]:.4f} gt={row[1]:.4f} "
                f"kd_cls={row[2]:.4f} kd_box={row[3]:.4f} "
                f"recent5={mean[0]:.4f}/{mean[1]:.4f}/{mean[2]:.4f}/{mean[3]:.4f}",
                flush=True,
            )

    if args.smoke:
        print("SMOKE_OK", flush=True)
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": student.state_dict(),
            "meta": {
                "steps": args.steps,
                "lr": args.lr,
                "student": args.student_checkpoint,
                "teacher": args.teacher_checkpoint,
                "topk": args.topk,
                "temperature": args.temperature,
                "gt_weight": args.gt_weight,
                "kd_cls_weight": args.kd_cls_weight,
                "kd_box_weight": args.kd_box_weight,
                "train_scope": args.train_scope,
            },
        },
        args.out,
    )
    print(f"SAVED {args.out}", flush=True)


if __name__ == "__main__":
    main()
