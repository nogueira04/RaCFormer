import torch
from mmcv import Config
from mmcv.parallel import scatter
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from loaders.builder import build_dataloader

import loaders  # noqa: F401
import models  # noqa: F401


CONFIG = "configs/racformer_train2k_day_radarquery_topk90_research.py"


def _tensor_shape(value):
    if torch.is_tensor(value):
        return tuple(value.shape)
    return None


def _describe_frame(frame_points):
    shapes = []
    valid_counts = []
    for points in frame_points:
        assert torch.is_tensor(points), type(points)
        shapes.append(tuple(points.shape))
        if points.numel() == 0:
            valid_counts.append(0)
            continue
        finite = torch.isfinite(points[:, :6]).all(dim=-1)
        in_range = (
            (points[:, 0] >= -51.2)
            & (points[:, 0] <= 51.2)
            & (points[:, 1] >= -51.2)
            & (points[:, 1] <= 51.2)
        )
        valid_counts.append(int((finite & in_range).sum().item()))
    return shapes, valid_counts


def main():
    assert torch.cuda.is_available(), "Branch D topk90 smoke requires compute-node CUDA"

    cfg = Config.fromfile(CONFIG)
    dataset = build_dataset(cfg.data.train)
    print("dataset_len", len(dataset))

    loader = build_dataloader(
        dataset,
        samples_per_gpu=2,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )
    batch = next(iter(loader))
    print("batch_keys", sorted(batch.keys()))

    batch_gpu = scatter(batch, [0])[0]
    img = batch_gpu["img"]
    img_metas = batch_gpu["img_metas"]
    radar_points = batch_gpu["radar_points"]
    print("img_shape", _tensor_shape(img))
    print("img_metas_len", len(img_metas))
    print("radar_frames", len(radar_points))

    batch_size = len(img_metas)
    assert batch_size == 2, batch_size
    current_frame_points = radar_points[0]
    assert len(current_frame_points) == batch_size, (
        len(current_frame_points),
        batch_size,
    )

    frame_shapes, valid_counts = _describe_frame(current_frame_points)
    print("current_frame_shapes", frame_shapes)
    print("current_frame_valid_counts", valid_counts)
    assert any(count > 0 for count in valid_counts), valid_counts

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg")).cuda().eval()
    head = model.pts_bbox_head
    assert head.radar_query_init is True
    assert head.radar_query_topk == 90
    assert head.radar_query_score == "rcs_speed"
    assert head.radar_query_use_velocity is False

    query_bbox = head.init_query_bbox.weight.detach().view(1, head.num_query, 10)
    query_bbox = query_bbox.repeat(batch_size, 1, 1).cuda()
    updated = head._radar_points_to_query_bbox(query_bbox, current_frame_points)
    assert updated.shape == query_bbox.shape

    changed = (updated[:, :, :2] - query_bbox[:, :, :2]).abs().sum(dim=-1) > 1e-6
    changed_counts = changed.sum(dim=1).detach().cpu().tolist()
    print("changed_query_counts", changed_counts)
    assert all(count > 0 for count in changed_counts), changed_counts
    assert all(count <= head.radar_query_topk for count in changed_counts), changed_counts

    unchanged_tail = torch.allclose(updated[:, head.radar_query_topk :, :], query_bbox[:, head.radar_query_topk :, :])
    print("unchanged_tail", bool(unchanged_tail))
    assert unchanged_tail

    velocity_unchanged = torch.allclose(
        updated[:, : head.radar_query_topk, 8:10],
        query_bbox[:, : head.radar_query_topk, 8:10],
    )
    print("velocity_unchanged", bool(velocity_unchanged))
    assert velocity_unchanged

    print("model_head", type(head).__name__)
    print("radar_query_topk90_smoke", "PASS")


if __name__ == "__main__":
    main()
