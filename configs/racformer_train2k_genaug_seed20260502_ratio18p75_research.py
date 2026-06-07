"""Phase 1 S3 — 2K day-only training subset + generated-night manifest loader.

Variant C / Interpretation B: keyframe night via manifest, all history frames remain
day. The manifest defines exactly which (sample_token, camera) pairs swap to a generated
PNG; misses fall back to the original day JPG.

The manifest_path placeholder MUST be overridden at training time (per seed):
    --override data.train.pipeline.0.manifest_path=research/night_gen_phase1/manifests/phase1_genaug_seed20260425_manifest.json
or by writing a thin per-seed config that inherits this one and pins the path.
"""

_base_ = ["./racformer_r50_nuimg_704x256_f8.py"]

num_frames = 8
file_client_args = dict(backend="disk")
class_names = [
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
ida_aug_conf = {
    "resize_lim": (0.38, 0.55),
    "final_dim": (256, 704),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}
grid_config = {
    "x": [-51.2, 51.2, 0.8],
    "y": [-51.2, 51.2, 0.8],
    "z": [-5, 3, 8],
    "depth": [1.0, 65.0, 96.0],
    "rcs": [-64, 64, 64],
}

# Critical: replace the standard LoadMultiViewImageFromFiles with the manifest variant.
# The history sweep loader (LoadMultiViewImageFromMultiSweeps) is unchanged → Variant C
# (keyframe night, day history).
train_pipeline = [
    dict(
        type="LoadMultiViewImageFromManifest",
        manifest_path="research/night_gen_phase1/manifests/phase1_t10_seed20260502_ratio18p75_generated.json",
        to_float32=False,
        color_type="color",
    ),
    dict(type="LoadMultiViewImageFromMultiSweeps", sweeps_num=num_frames - 1),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_label=False,
        with_bbox_depth=False,
    ),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="RandomTransformImage", ida_aug_conf=ida_aug_conf, training=True),
    dict(
        type="Loadnuradarpoints",
        coord_type="RADAR",
        num_sweeps=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadradarpointsFromMultiSweeps",
        sweeps_num=num_frames - 1,
        num_aggr_sweeps=5,
        test_mode=False,
    ),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="RaCGlobalRotScaleTransImage",
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
    ),
    dict(type="PointToMultiViewDepth", downsample=1, grid_config=grid_config),
    dict(
        type="RadarPointToMultiViewDepth",
        downsample=1,
        grid_config=grid_config,
        test_mode=False,
    ),
    dict(type="RaCFormatBundle3D", class_names=class_names),
    dict(
        type="Collect3D",
        keys=[
            "gt_bboxes_3d",
            "gt_labels_3d",
            "img",
            "gt_depth",
            "radar_depth",
            "radar_rcs",
            "radar_points",
        ],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "lidar2img",
            "img_timestamp",
            "intrinsics",
        ),
    ),
]

data = dict(
    train=dict(
        ann_file="/srv/nfs/shared/gnmp/RaCFormer/nuscenes_infos_train_2k_day.pkl",
        max_samples=2000,
        pipeline=train_pipeline,
    ),
    val=dict(max_samples=300),
)

total_epochs = 12
eval_config = dict(interval=total_epochs)
