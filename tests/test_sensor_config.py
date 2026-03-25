
import torch
import numpy as np
from models.racformer import RaCFormer
from models.racformer_transformer import RaCFormerTransformer
from models.racformer_head import RaCFormer_head

def test_racformer_initialization():
    print("Testing RaCFormer initialization with custom sensor counts...")
    
    # Mock config parts
    grid_config = {
        'x': [-51.2, 51.2, 0.8],
        'y': [-51.2, 51.2, 0.8],
        'z': [-5, 3, 8],
        'depth': [1.0, 65.0, 96.0],
        'rcs': [-64, 64, 64]
    }
    
    img_backbone = dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        style='pytorch')
    
    img_neck = dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4)
        
    img_lss_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0])

    img_lss_view_transformer=dict(
        type='LSSViewTransformerBEVDepth_racformer',
        grid_config=grid_config,
        input_size=(256, 704),
        in_channels=256,
        out_channels=256,
        depthnet_cfg=dict(use_dcn=False),
        downsample=16,
        loss_depth_weight=2.0)
        
    radar_voxel_layer=dict(
        max_num_points=10,
        voxel_size=[0.8, 0.8, 8],
        max_voxels=(30000, 40000),
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        deterministic=False)

    radar_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=7,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.8, 0.8, 8],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False)

    radar_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(128, 128))
        
    pts_bbox_head=dict(
        type='RaCFormer_head',
        num_classes=10,
        num_clusters=6,
        in_channels=256,
        num_query=900,
        transformer=dict(
            type='RaCFormerTransformer',
            embed_dims=256,
            num_cams=3 # Testing with 3 cameras
        ),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            score_threshold=0.05,
            num_classes=10)
    )

    # Initialize model with 3 cameras
    model = RaCFormer(
        img_backbone=img_backbone,
        img_neck=img_neck,
        img_lss_neck=img_lss_neck,
        img_lss_view_transformer=img_lss_view_transformer,
        radar_voxel_layer=radar_voxel_layer,
        radar_voxel_encoder=radar_voxel_encoder,
        radar_middle_encoder=radar_middle_encoder,
        pts_bbox_head=pts_bbox_head,
        num_cams=3
    )
    
    print("Model initialized successfully.")
    print(f"Model num_cams: {model.num_cams}")
    
    # Verify transformer num_cams
    print(f"Transformer type: {type(model.pts_bbox_head.transformer)}")
    print(f"Transformer attributes: {dir(model.pts_bbox_head.transformer)}")
    print(f"Transformer num_cams: {model.pts_bbox_head.transformer.num_cams}")
    
    assert model.num_cams == 3
    assert model.pts_bbox_head.transformer.num_cams == 3
    
    print("Verification passed!")

if __name__ == "__main__":
    test_racformer_initialization()
