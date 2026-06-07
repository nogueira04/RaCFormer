import argparse
import json
import os
import sys
from collections import defaultdict

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def coerce_full_val(cfg):
    cfg.data.val.ann_file = "/srv/nfs/shared/gnmp/RaCFormer/nuscenes_infos_val_sweep.pkl"
    cfg.data.val.max_samples = None


def extract_metas(data):
    img_metas = data["img_metas"]
    if isinstance(img_metas, list):
        img_metas = img_metas[0]
    if hasattr(img_metas, "data"):
        img_metas = img_metas.data
    while isinstance(img_metas, (list, tuple)) and len(img_metas) == 1:
        img_metas = img_metas[0]
    if isinstance(img_metas, dict):
        return [img_metas]
    return list(img_metas)


def new_stats():
    return {"sum": [0.0, 0.0, 0.0], "count": 0}


def add_stats(bucket, values):
    bucket["count"] += int(values.shape[0])
    sums = values.sum(dim=0).detach().cpu().tolist()
    for i in range(3):
        bucket["sum"][i] += float(sums[i])


def finish_stats(bucket):
    if bucket["count"] == 0:
        return {"count": 0, "mean": [None, None, None]}
    return {
        "count": bucket["count"],
        "mean": [value / bucket["count"] for value in bucket["sum"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--full-val", action="store_true")
    args = parser.parse_args()

    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_model

    import loaders  # noqa: F401
    import models  # noqa: F401

    cfg = Config.fromfile(args.config)
    if args.full_val:
        coerce_full_val(cfg)
    elif getattr(cfg.data.val, "max_samples", None) is None:
        cfg.data.val.max_samples = args.max_samples

    dataset = build_dataset(cfg.data.val)
    if args.max_samples and len(dataset.data_infos) > args.max_samples:
        dataset.data_infos = dataset.data_infos[: args.max_samples]

    loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    model = build_model(cfg.model)
    assert hasattr(model.pts_bbox_head.transformer.decoder.decoder_layer, "reliability_fusion")
    model = MMDataParallel(model.cuda(), [0])
    load_checkpoint(model, args.weights, map_location="cuda", strict=True)
    model.eval()

    by_condition_stage = defaultdict(new_stats)
    by_range_stage = defaultdict(new_stats)
    calls = []

    def hook(_module, inputs, output):
        query_bbox = inputs[0].detach()
        gates = output.detach()
        calls.append((query_bbox, gates))

    handle = model.module.pts_bbox_head.transformer.decoder.decoder_layer.reliability_fusion.register_forward_hook(hook)
    try:
        for idx, data in enumerate(loader):
            if idx * args.batch_size >= args.max_samples:
                break
            calls.clear()
            metas = extract_metas(data)
            conditions = [str(meta.get("scene_condition", "unknown")) for meta in metas]
            with torch.no_grad():
                model(return_loss=False, rescale=True, **data)
            for stage, (query_bbox, gates) in enumerate(calls):
                for sample_idx, condition in enumerate(conditions):
                    sample_gates = gates[sample_idx].float()
                    add_stats(by_condition_stage[(condition, stage)], sample_gates)

                    sample_query = query_bbox[sample_idx].float()
                    query_range = sample_query[..., 1]
                    for label, lo, hi in [
                        ("near", 0.0, 0.33),
                        ("mid", 0.33, 0.66),
                        ("far", 0.66, 1.01),
                    ]:
                        mask = (query_range >= lo) & (query_range < hi)
                        if mask.any():
                            add_stats(by_range_stage[(label, stage)], sample_gates[mask])
    finally:
        handle.remove()

    payload = {
        "config": args.config,
        "weights": args.weights,
        "max_samples": args.max_samples,
        "gate_order": ["image_query", "radar_bev", "lss_bev"],
        "by_condition_stage": {
            f"{condition}_stage{stage}": finish_stats(bucket)
            for (condition, stage), bucket in sorted(by_condition_stage.items())
        },
        "by_range_stage": {
            f"{range_label}_stage{stage}": finish_stats(bucket)
            for (range_label, stage), bucket in sorted(by_range_stage.items())
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "gate_stats.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    md_path = os.path.join(args.out_dir, "gate_stats.md")
    lines = [
        "# Continuous Reliability Gate Stats",
        "",
        f"- Config: `{args.config}`",
        f"- Weights: `{args.weights}`",
        f"- Max samples: `{args.max_samples}`",
        "- Gate order: image query, radar BEV, LSS BEV",
        "",
        "## By Condition And Stage",
        "",
        "| Bucket | Count | Image | Radar | LSS |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, stats in payload["by_condition_stage"].items():
        mean = stats["mean"]
        lines.append(f"| {key} | {stats['count']} | {mean[0]:.4f} | {mean[1]:.4f} | {mean[2]:.4f} |")
    lines.extend(["", "## By Query Range And Stage", "", "| Bucket | Count | Image | Radar | LSS |", "|---|---:|---:|---:|---:|"])
    for key, stats in payload["by_range_stage"].items():
        mean = stats["mean"]
        lines.append(f"| {key} | {stats['count']} | {mean[0]:.4f} | {mean[1]:.4f} | {mean[2]:.4f} |")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(md_path)


if __name__ == "__main__":
    main()
