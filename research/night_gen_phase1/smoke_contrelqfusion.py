import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import loaders  # noqa: F401
import models  # noqa: F401


CONFIG = "configs/racformer_train2k_mixed_contrelqfusion_research.py"


def main():
    cfg = Config.fromfile(CONFIG)

    dataset = build_dataset(cfg.data.train)
    print("dataset_len", len(dataset))
    info = dataset.get_data_info(0)
    print("first_info_condition", info.get("scene_condition"))
    print("first_info_views", len(info["lidar2img"]), len(info["img_timestamp"]))

    sample = dataset[0]
    meta = sample["img_metas"].data
    print("meta_has_scene_condition", "scene_condition" in meta)
    print("meta_scene_condition", meta.get("scene_condition"))
    print("image_shape", tuple(sample["img"].data.shape))

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    layer = model.pts_bbox_head.transformer.decoder.decoder_layer
    assert layer.continuous_reliability_query_fusion
    assert layer.reliability_fusion.use_pairwise_cosine
    assert layer.reliability_fusion.use_query_geometry

    model.init_weights()
    final_linear = layer.reliability_fusion.gate[-1]
    assert torch.count_nonzero(final_linear.weight).item() == 0
    assert torch.count_nonzero(final_linear.bias).item() == 0

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    gate = layer.reliability_fusion.to(device)
    query_bbox = torch.zeros(2, 8, 10, device=device)
    query_bbox[..., 1] = torch.linspace(0.05, 0.95, 8, device=device).view(1, 8)
    query_feat = torch.randn(2, 8, 256, device=device)
    query_radar_feat = torch.randn(2, 8, 256, device=device)
    query_lss_feat = torch.randn(2, 8, 256, device=device)

    with torch.no_grad():
        reliability_gate = gate(query_bbox, query_feat, query_radar_feat, query_lss_feat)
    assert reliability_gate.shape == (2, 8, 3)
    assert torch.allclose(reliability_gate, torch.ones_like(reliability_gate))

    print("model_layer", type(layer).__name__)
    print("reliability_module", type(layer.reliability_fusion).__name__)
    print("identity_gate", True)


if __name__ == "__main__":
    main()
