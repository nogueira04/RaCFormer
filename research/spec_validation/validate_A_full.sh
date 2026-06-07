#!/bin/bash
. /srv/nfs/shared/gnmp/miniconda3/etc/profile.d/conda.sh
conda activate racformerfix
cd /srv/nfs/shared/gnmp/RaCFormer
python3 << 'PYEOF'
from mmcv import Config
c = Config.fromfile("configs/racformer_r50_nuimg_704x256_f8_3cam_3rad.py")
print("num_cams     :", c.model.num_cams)
print("num_frames   :", c.num_frames)
print("pcr          :", c.point_cloud_range)
print("classes      :", c.class_names)
print("test cams    :", c.data.test.camera_types)
print("test rads    :", c.data.test.radar_types)
print("input final  :", c.ida_aug_conf["final_dim"])
print("native H,W   :", c.ida_aug_conf["H"], c.ida_aug_conf["W"])
print("grid x       :", c.grid_config["x"])
print("grid rcs     :", c.grid_config["rcs"])
print("norm mean    :", c.img_norm_cfg["mean"])
print("norm std     :", c.img_norm_cfg["std"])
print("--- radar encoder ---")
print("radar in_ch  :", c.model.radar_voxel_encoder.in_channels)
print("voxel size   :", c.model.radar_voxel_encoder.voxel_size)
print("max pts/vox  :", c.model.radar_voxel_encoder.max_num_points)
print("max voxels   :", c.model.radar_voxel_encoder.max_voxels)
PYEOF
