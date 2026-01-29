"""
Prepare nuScenes mini dataset annotation files for RaCFormer.

This script:
1. Creates base info pickle files matching the original format
2. Adds sweep information required by RaCFormer

Usage:
    python prepare_nuscenes_mini.py --data-root ./data/nuscenes
"""
import os
import argparse
import pickle
import numpy as np
import tqdm
from nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion


def get_sensor2lidar_transformation(nusc, cam_token, lidar_token):
    """Compute sensor to lidar transformation."""
    cam_data = nusc.get('sample_data', cam_token)
    lidar_data = nusc.get('sample_data', lidar_token)

    cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    lidar_cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

    cam_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    lidar_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    # Build 4x4 transformation matrices
    # Camera to ego
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(cam_cs['rotation']).rotation_matrix
    cam2ego[:3, 3] = np.array(cam_cs['translation'])

    # Ego to global (at camera timestamp)
    ego2global_cam = np.eye(4)
    ego2global_cam[:3, :3] = Quaternion(cam_pose['rotation']).rotation_matrix
    ego2global_cam[:3, 3] = np.array(cam_pose['translation'])

    # Global to ego (at lidar timestamp)
    ego2global_lidar = np.eye(4)
    ego2global_lidar[:3, :3] = Quaternion(lidar_pose['rotation']).rotation_matrix
    ego2global_lidar[:3, 3] = np.array(lidar_pose['translation'])
    global2ego_lidar = np.linalg.inv(ego2global_lidar)

    # Ego to lidar
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(lidar_cs['rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.array(lidar_cs['translation'])
    ego2lidar = np.linalg.inv(lidar2ego)

    # Chain: cam -> ego -> global -> ego -> lidar
    cam2lidar = ego2lidar @ global2ego_lidar @ ego2global_cam @ cam2ego

    sensor2lidar_rot = cam2lidar[:3, :3]
    sensor2lidar_trans = cam2lidar[:3, 3]

    return sensor2lidar_rot, sensor2lidar_trans


def create_nuscenes_infos_mini(root_path, version='v1.0-mini'):
    """Create info file for nuScenes mini dataset matching original format."""
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    train_scenes = set(splits.mini_train)
    val_scenes = set(splits.mini_val)

    train_infos = []
    val_infos = []

    for sample in tqdm.tqdm(nusc.sample, desc='Processing samples'):
        scene = nusc.get('scene', sample['scene_token'])
        scene_name = scene['name']

        if scene_name not in train_scenes and scene_name not in val_scenes:
            continue

        info = get_sample_info(nusc, sample, root_path)

        if scene_name in train_scenes:
            train_infos.append(info)
        elif scene_name in val_scenes:
            val_infos.append(info)

    # Save info files
    train_info = {'infos': train_infos, 'metadata': {'version': version}}
    val_info = {'infos': val_infos, 'metadata': {'version': version}}

    train_path = os.path.join(root_path, 'nuscenes_infos_train_mini.pkl')
    val_path = os.path.join(root_path, 'nuscenes_infos_val_mini.pkl')

    with open(train_path, 'wb') as f:
        pickle.dump(train_info, f)
    print(f'Saved {len(train_infos)} training samples to {train_path}')

    with open(val_path, 'wb') as f:
        pickle.dump(val_info, f)
    print(f'Saved {len(val_infos)} validation samples to {val_path}')

    return train_path, val_path, nusc


def get_sample_info(nusc, sample, root_path):
    """Extract info for a single sample matching original format."""
    lidar_token = sample['data']['LIDAR_TOP']
    sd_rec = nusc.get('sample_data', lidar_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])

    lidar_path = os.path.join(root_path, sd_rec['filename'])

    # Get annotations
    boxes = []
    names = []
    velocities = []
    num_lidar_pts = []
    num_radar_pts = []
    valid_flags = []

    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)

        # Skip if no points
        if ann['num_lidar_pts'] + ann['num_radar_pts'] == 0:
            valid_flags.append(False)
        else:
            valid_flags.append(True)

        # Get box in global frame
        box = nusc.get_box(ann_token)

        # Transform to lidar frame
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        # Get velocity
        velocity = nusc.box_velocity(ann_token)
        if np.any(np.isnan(velocity)):
            velocity = np.array([0.0, 0.0, 0.0])
        # Transform velocity to lidar frame
        velocity = Quaternion(pose_record['rotation']).inverse.rotate(velocity)
        velocity = Quaternion(cs_record['rotation']).inverse.rotate(velocity)

        boxes.append([
            box.center[0], box.center[1], box.center[2],
            box.wlh[1], box.wlh[0], box.wlh[2],  # l, w, h (swap wlh order)
            box.orientation.yaw_pitch_roll[0],
        ])
        names.append(ann['category_name'])
        velocities.append(velocity[:2])
        num_lidar_pts.append(ann['num_lidar_pts'])
        num_radar_pts.append(ann['num_radar_pts'])

    # Get camera info
    cam_types = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]

    cams = {}
    for cam in cam_types:
        cam_token = sample['data'][cam]
        cam_data = nusc.get('sample_data', cam_token)
        cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])

        # Compute sensor2lidar transformation
        sensor2lidar_rot, sensor2lidar_trans = get_sensor2lidar_transformation(
            nusc, cam_token, lidar_token
        )

        cams[cam] = {
            'data_path': os.path.join(root_path, cam_data['filename']),
            'type': cam,
            'sample_data_token': cam_token,
            'sensor2ego_translation': cam_cs['translation'],
            'sensor2ego_rotation': cam_cs['rotation'],
            'ego2global_translation': cam_pose['translation'],
            'ego2global_rotation': cam_pose['rotation'],
            'timestamp': cam_data['timestamp'],
            'cam_intrinsic': np.array(cam_cs['camera_intrinsic']),
            'sensor2lidar_rotation': sensor2lidar_rot,
            'sensor2lidar_translation': sensor2lidar_trans,
        }

    info = {
        'token': sample['token'],
        'lidar_path': lidar_path,
        'sweeps': [],
        'cams': cams,
        'lidar2ego_translation': cs_record['translation'],
        'lidar2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sample['timestamp'],
        'gt_boxes': np.array(boxes) if boxes else np.zeros((0, 7)),
        'gt_names': np.array(names),
        'gt_velocity': np.array(velocities) if velocities else np.zeros((0, 2)),
        'num_lidar_pts': np.array(num_lidar_pts) if num_lidar_pts else np.array([]),
        'num_radar_pts': np.array(num_radar_pts) if num_radar_pts else np.array([]),
        'valid_flag': np.array(valid_flags) if valid_flags else np.array([]),
        'scene_token': sample['scene_token'],
        'prev': sample['prev'],
        'next': sample['next'],
    }

    return info


def get_cam_info_for_sweep(nusc, sample_data, root_path, lidar_token):
    """Get camera info for sweep frames."""
    pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

    sensor2ego_translation = cs_record['translation']
    ego2global_translation = pose_record['translation']
    sensor2ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
    ego2global_rotation = Quaternion(pose_record['rotation']).rotation_matrix
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    sensor2global_rotation = sensor2ego_rotation.T @ ego2global_rotation.T
    sensor2global_translation = sensor2ego_translation @ ego2global_rotation.T + ego2global_translation

    # Compute sensor2lidar
    sensor2lidar_rot, sensor2lidar_trans = get_sensor2lidar_transformation(
        nusc, sample_data['token'], lidar_token
    )

    return {
        'data_path': os.path.join(root_path, sample_data['filename']),
        'sensor2global_rotation': sensor2global_rotation,
        'sensor2global_translation': sensor2global_translation,
        'cam_intrinsic': cam_intrinsic,
        'timestamp': sample_data['timestamp'],
        'sensor2lidar_rotation': sensor2lidar_rot,
        'sensor2lidar_translation': sensor2lidar_trans,
    }


def add_sweep_info(nusc, sample_infos, root_path):
    """Add sweep information to sample infos (required by RaCFormer)."""
    cam_types = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]
    rad_types = [
        'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
        'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'
    ]

    for curr_id in tqdm.tqdm(range(len(sample_infos['infos'])), desc='Adding sweep info'):
        info = sample_infos['infos'][curr_id]
        sample = nusc.get('sample', info['token'])
        lidar_token = sample['data']['LIDAR_TOP']

        curr_cams = {}
        for cam in cam_types:
            curr_cams[cam] = nusc.get('sample_data', sample['data'][cam])

        # Update camera info with sweep-compatible fields
        for cam in cam_types:
            sample_data = nusc.get('sample_data', sample['data'][cam])
            sweep_cam = get_cam_info_for_sweep(nusc, sample_data, root_path, lidar_token)
            info['cams'][cam].update(sweep_cam)

        # Add radar info
        curr_rads = {}
        for rad in rad_types:
            curr_rads[rad] = nusc.get('sample_data', sample['data'][rad])
        info['rads'] = curr_rads

        # Remove fields not needed after sweep processing
        for cam in cam_types:
            for key in ['sample_data_token', 'sensor2ego_translation',
                       'sensor2ego_rotation', 'ego2global_translation',
                       'ego2global_rotation']:
                if key in info['cams'][cam]:
                    del info['cams'][cam][key]

        # Add sweep frames
        sweep_infos = []
        if sample['prev'] != '':
            for _ in range(5):
                jump = False
                sweep_info = {}

                for cam in cam_types:
                    if curr_cams[cam]['prev'] == '':
                        sweep_info = sweep_infos[-1] if sweep_infos else {}
                        jump = True
                        break
                    sample_data = nusc.get('sample_data', curr_cams[cam]['prev'])
                    sweep_cam = get_cam_info_for_sweep(nusc, sample_data, root_path, lidar_token)
                    curr_cams[cam] = sample_data
                    sweep_info[cam] = sweep_cam

                for rad in rad_types:
                    if jump:
                        break
                    if curr_rads[rad]['prev'] == '':
                        sweep_info[rad] = None
                    else:
                        radar_sample_data = nusc.get('sample_data', curr_rads[rad]['prev'])
                        curr_rads[rad] = radar_sample_data
                        sweep_info[rad] = radar_sample_data

                sweep_infos.append(sweep_info)

        info['sweeps'] = sweep_infos

    return sample_infos


def main():
    parser = argparse.ArgumentParser(description='Prepare nuScenes mini dataset')
    parser.add_argument('--data-root', default='./data/nuscenes',
                        help='Path to nuScenes dataset root')
    args = parser.parse_args()

    root_path = args.data_root
    version = 'v1.0-mini'

    print(f'Preparing nuScenes mini dataset from {root_path}')
    print('=' * 60)

    # Step 1: Create base info files
    print('\nStep 1: Creating base info files...')
    train_path, val_path, nusc = create_nuscenes_infos_mini(root_path, version)

    # Step 2: Add sweep info
    print('\nStep 2: Adding sweep information...')

    # Process training set
    print('\nProcessing training set...')
    with open(train_path, 'rb') as f:
        train_infos = pickle.load(f)
    train_infos = add_sweep_info(nusc, train_infos, root_path)
    train_sweep_path = os.path.join(root_path, 'nuscenes_infos_train_mini_sweep.pkl')
    with open(train_sweep_path, 'wb') as f:
        pickle.dump(train_infos, f)
    print(f'Saved to {train_sweep_path}')

    # Process validation set
    print('\nProcessing validation set...')
    with open(val_path, 'rb') as f:
        val_infos = pickle.load(f)
    val_infos = add_sweep_info(nusc, val_infos, root_path)
    val_sweep_path = os.path.join(root_path, 'nuscenes_infos_val_mini_sweep.pkl')
    with open(val_sweep_path, 'wb') as f:
        pickle.dump(val_infos, f)
    print(f'Saved to {val_sweep_path}')

    print('\n' + '=' * 60)
    print('Done! Generated files:')
    print(f'  - {train_sweep_path}')
    print(f'  - {val_sweep_path}')
    print('\nTo run validation, set the environment variable and run:')
    print('  export NUSCENES_VERSION=v1.0-mini')
    print(f'  python val.py --config configs/racformer_r50_nuimg_704x256_f8_mini.py --weights <checkpoint>')


if __name__ == '__main__':
    main()
