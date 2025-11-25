_base_ = ['./racformer_r50_nuimg_704x256_f8.py']

# Sensor configurations
camera_types = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'
]
radar_types = [
    'RADAR_FRONT_LEFT', 'RADAR_FRONT', 'RADAR_FRONT_RIGHT'
]

# Model settings
model = dict(
    num_cams=3,
    pts_bbox_head=dict(
        transformer=dict(
            num_cams=3
        )
    )
)

# Pipeline settings
ida_aug_conf = {
    "resize_lim": (0.386, 0.55),
    "final_dim": (256, 704),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}
num_frames = 8
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 65.0, 96.0],
    'rcs': [-64, 64, 64]
}
file_client_args = dict(backend='disk')
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, test_mode=True, cam_types=camera_types),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='Loadnuradarpoints', coord_type='RADAR', num_sweeps=5, file_client_args=file_client_args, radar_types=radar_types),
    dict(type='LoadradarpointsFromMultiSweeps', sweeps_num=num_frames-1, num_aggr_sweeps=5, test_mode=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='RadarPointToMultiViewDepth', downsample=1, grid_config=grid_config, test_mode=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RaCFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img', 'gt_depth', 'radar_points', 'radar_depth', 'radar_rcs'], meta_keys=(
                'filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape',
                'lidar2img', 'img_timestamp', 'intrinsics'))
        ])
]

# Data settings
data = dict(
    val=dict(
        camera_types=camera_types,
        radar_types=radar_types,
        pipeline=test_pipeline
    ),
    test=dict(
        camera_types=camera_types,
        radar_types=radar_types,
        pipeline=test_pipeline
    )
)
