_base_ = ['./racformer_r50_nuimg_704x256_f8.py']

dataset_root = './data/nuscenes/'

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=True
)

# Note: In mmengine, ann_file is relative to data_root, so don't prepend dataset_root
data = dict(
    workers_per_gpu=2,
    train=dict(
        data_root=dataset_root,
        ann_file='nuscenes_infos_train_mini_sweep.pkl',
        classes=class_names,
        modality=input_modality,
    ),
    val=dict(
        data_root=dataset_root,
        ann_file='nuscenes_infos_val_mini_sweep.pkl',
        classes=class_names,
        modality=input_modality,
    ),
    test=dict(
        data_root=dataset_root,
        ann_file='nuscenes_infos_val_mini_sweep.pkl',
        classes=class_names,
        modality=input_modality,
    )
)
