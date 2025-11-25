import os
import os.path as osp
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from pyquaternion import Quaternion
import torch
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud, PointCloud
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.common.loaders import load_gt as original_load_gt
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.data_classes import EvalBoxes
from typing import Tuple, List, Dict
from nuscenes.utils.geometry_utils import transform_matrix
from functools import reduce
import random


nu_version = 'v1.0-trainval'
renusc = NuScenes(version=nu_version, dataroot=str('data/nuscenes/'), verbose=False)

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    def __init__(self, 
                 camera_types=None,
                 radar_types=None,
                 max_samples=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.camera_types = camera_types
        self.radar_types = radar_types
        
        if max_samples is not None:
            self.data_infos = self.data_infos[:max_samples]
            self.max_samples = max_samples
        else:
            self.max_samples = None

    def evaluate(self, results, logger=None, **kwargs):
        # Filter predictions to front only (x > 0)
        # RaCFormer output is in LiDAR coordinates (x-forward, y-left, z-up)
        # We want to keep boxes with x > 0 (approx front 180)
        # Actually, let's be more precise if possible, but x > 0 is a good start for "front".
        # The user mentioned "front ~190 degrees", which implies slightly more than 180.
        # But x > 0 is safe.
        
        # However, results is a list of dicts or similar.
        # mmdet3d results are usually list of dicts per sample.
        # We need to filter them before passing to super().evaluate?
        # super().evaluate calls self._evaluate_single which converts results to NuScenes format.
        # It's better to let super().evaluate do the conversion, then intercept inside NuScenesEval?
        # No, super().evaluate creates NuScenesEval and runs it.
        
        # We can override _evaluate_single or just copy-paste the logic of evaluate from mmdet3d.
        # Since we can't easily copy-paste due to dependencies, let's try to monkeypatch NuScenesEval.
        
        # Actually, we can filter `results` before passing to super().evaluate?
        # `results` contains 'boxes_3d'.
        # If we filter `results`, the `NuScenesEval` will still load ALL GT and complain about missing predictions if we don't predict for all samples (in mini mode).
        # And if we predict for all but filter some boxes, it's fine.
        
        # So we MUST monkeypatch load_gt to fix the GT side.
        
        # Define the custom load_gt
        def custom_load_gt(nusc, eval_split, box_cls, verbose=False):
            gt_boxes = original_load_gt(nusc, eval_split, box_cls, verbose)
            
            # Filter by sample tokens (for mini evaluation)
            if self.max_samples is not None:
                valid_tokens = set([info['token'] for info in self.data_infos])
                new_boxes = EvalBoxes()
                for sample_token in gt_boxes.sample_tokens:
                    if sample_token in valid_tokens:
                        new_boxes.add_boxes(sample_token, gt_boxes[sample_token])
                gt_boxes = new_boxes
            
            # Filter by FOV (Front only, x > 0 in ego frame)
            # GT boxes are in global frame. We need to convert to ego frame to check x > 0.
            # This is expensive.
            # Alternatively, we can check if the box is within the FOV of the sensors?
            # But we don't have easy access to sensor calibs here for every box.
            
            # Wait, NuScenesEval usually evaluates in Global frame.
            # But to filter "front", we need relation to Ego.
            # We can use nusc to get ego pose.
            
            filtered_boxes = EvalBoxes()
            for sample_token in gt_boxes.sample_tokens:
                sample = nusc.get('sample', sample_token)
                # Get ego pose at timestamp
                # Actually, sample data has ego pose.
                # Let's use the first lidar sample data for ego pose? Or any.
                # sample['data']['LIDAR_TOP'] gives lidar token.
                sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                
                # Global to Ego
                # We want to filter boxes that are "in front" of the ego vehicle.
                # Box frame: Global.
                # Ego frame: x-forward.
                
                boxes = gt_boxes[sample_token]
                valid_boxes = []
                for box in boxes:
                    # box.translation is [x, y, z] in global
                    # Convert to ego
                    # translation - pose_translation
                    # rotate by inverse pose_rotation
                    trans = np.array(box.translation) - np.array(pose_record['translation'])
                    rot = Quaternion(pose_record['rotation']).inverse
                    trans = rot.rotate(trans)
                    
                    # Check x > 0 (Front)
                    # Also maybe check y range?
                    # User said "front ~190 degrees". x > -epsilon?
                    # Let's use x > 0 for now.
                    if trans[0] > 0:
                        valid_boxes.append(box)
                
                filtered_boxes.add_boxes(sample_token, valid_boxes)
            
            return filtered_boxes

        # Monkeypatch
        import nuscenes.eval.detection.evaluate as eval_module
        original_load_gt_func = eval_module.load_gt
        eval_module.load_gt = custom_load_gt
        
        try:
            # Also filter predictions in results?
            # mmdet3d results are in LiDAR frame (usually).
            # If we filter GT, we should also filter predictions to be fair?
            # If we don't filter predictions, false positives in the back will be penalized (as there is no GT there).
            # So yes, we must filter predictions too.
            
            # Filter results (list of dicts)
            # Each result dict has 'boxes_3d' (LiDARInstance3DBoxes) and 'scores_3d', 'labels_3d'.
            # LiDARInstance3DBoxes are in LiDAR frame (x-forward).
            # So we can just filter by x > 0.
            
            import copy
            results = copy.deepcopy(results)
            print(f"Filtering predictions to Front-Only (Ego X > 0)...")
            for i, res in enumerate(results):
                if 'pts_bbox' in res:
                    res = res['pts_bbox']
                
                boxes_3d = res['boxes_3d']
                scores_3d = res['scores_3d']
                labels_3d = res['labels_3d']
                
                # Get calibration for this sample
                info = self.data_infos[i]
                lidar2ego_translation = info['lidar2ego_translation']
                lidar2ego_rotation = info['lidar2ego_rotation']
                
                # Transform centers to Ego frame
                centers = boxes_3d.center # [N, 3]
                # Rotate
                rot = Quaternion(lidar2ego_rotation).rotation_matrix
                centers_ego = centers @ rot.T + np.array(lidar2ego_translation)
                
                # Filter Ego X > 0
                mask = centers_ego[:, 0] > 0
                
                res['boxes_3d'] = boxes_3d[mask]
                res['scores_3d'] = scores_3d[mask]
                res['labels_3d'] = labels_3d[mask]
                
                if 'pts_bbox' in results[i]:
                    results[i]['pts_bbox'] = res
                else:
                    results[i] = res

            print(f"Starting evaluation...")
            return super().evaluate(results, logger=logger, **kwargs)
        finally:
            # Restore
            eval_module.load_gt = original_load_gt_func


    def collect_sweeps(self, index, into_past=60, into_future=60):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append({**self.data_infos[curr_index - 1]['cams'], **self.data_infos[curr_index - 1]['rads']})
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append({**self.data_infos[curr_index]['cams'], **self.data_infos[curr_index]['rads']})
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation = Quaternion(lidar2ego_rotation).rotation_matrix

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            pts_filename=info['lidar_path'],
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation,
        )

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []
            cam_intrinsics = []
            if self.camera_types is None:
                # Default to all cameras if not specified
                cams_to_use = info['cams'].items()
            else:
                cams_to_use = [(k, v) for k, v in info['cams'].items() if k in self.camera_types]

            for _, cam_info in cams_to_use:

                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(intrinsic)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
                intrinsics=cam_intrinsics, #here
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

@DATASETS.register_module()
class CustomNuScenesDataset_radar(CustomNuScenesDataset):

    def get_data_info(self, index):
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation = Quaternion(lidar2ego_rotation).rotation_matrix

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            pts_filename=info['lidar_path'],
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation,
        )

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []
            cam_intrinsics = []
            if self.camera_types is None:
                # Default to all cameras if not specified
                cams_to_use = info['cams'].items()
            else:
                cams_to_use = [(k, v) for k, v in info['cams'].items() if k in self.camera_types]

            for _, cam_info in cams_to_use:

                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
                intrinsics=cam_intrinsics, #here
            ))


        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict


drop=False

def get_nu_radar(sam_idx, mutil_sweep=True, num_sweeps=6, filter=True, radar_sample_rec=None, drop=drop, radar_types=None):
    ref_sample_rec = renusc.get('sample', sam_idx)
    datas = ref_sample_rec['data']
    radar_tokens = []
    points = np.zeros((18,0))
    new_times = np.zeros((1,0))

    ref_chan = 'LIDAR_TOP'
    ref_sd_record = renusc.get('sample_data', datas[ref_chan])

    if radar_types is None:
        rad_types = [
            'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
            'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'
        ]
    else:
        rad_types = radar_types
    
    if drop:
        max_drop = 5
        view_drop = random.randint(0, max_drop-2)
        random_indices = torch.randperm(max_drop)[:view_drop]
        for ind in list(random_indices):
            rad_types[ind] = ''

    for token in datas.keys():
        if token in rad_types:
            radar_tokens.append(token)
        else:
            continue
        if radar_sample_rec is not None:
            sd_record = radar_sample_rec[token]
            if sd_record is None:
                continue
        else:
            sd_record = renusc.get('sample_data', datas[token])
        sample_rec = renusc.get('sample', sd_record['sample_token'])
        chan = sd_record['channel']

        if mutil_sweep:
            if filter:
                pc, times = RadarPointCloud_v2.from_file_multisweep(renusc,
                                sample_rec, ref_sample_rec, chan, ref_chan, nsweeps=num_sweeps)
            else:
                RadarPointCloud_v2.disable_filters()
                
                pc, times = RadarPointCloud_v2.from_file_multisweep(renusc,
                                sample_rec, ref_sample_rec, chan, ref_chan, nsweeps=num_sweeps)

                RadarPointCloud_v2.default_filters()


            radar_cs_record = renusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            ref_cs_record = renusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            velocities = pc.points[8:10, :]  # Compensated velocity
            velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
            velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
            velocities[2, :] = np.zeros(pc.points.shape[1])
            
            pc.points[8:10, :] = velocities[:2, :]
            points = np.concatenate([points, pc.points], axis=1 )
            new_times = np.concatenate([new_times, times], axis=1 )
        else:
            pc = RadarPointCloud.from_file()
            pass

    return torch.from_numpy(points).type(torch.float32), radar_tokens, torch.from_numpy(new_times).type(torch.float32)

class RadarPointCloud_v2(RadarPointCloud):

    @classmethod
    def from_file_multisweep(cls,
                             nusc: 'NuScenes',
                             sample_rec: Dict,
                             ref_sample_rec: Dict,
                             chan: str,
                             ref_chan: str,
                             nsweeps: int = 5,
                             min_distance: float = 1.0) -> Tuple['PointCloud', np.ndarray]:
        """
        Return a point cloud that aggregates multiple sweeps.
        As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
        As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
        :param nusc: A NuScenes instance.
        :param sample_rec: The current sample.
        :param chan: The lidar/radar channel from which we track back n sweeps to aggregate the point cloud.
        :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
        :param nsweeps: Number of sweeps to aggregated.
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """
        # Init.
        points = np.zeros((cls.nbr_dims(), 0), dtype=np.float32 if cls == LidarPointCloud else np.float64)
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp.
        ref_sd_token = ref_sample_rec['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec['data'][chan]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
            times = time_lag * np.ones((1, current_pc.nbr_points()))
            all_times = np.hstack((all_times, times))

            # Merge with key pc.
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

        return all_pc, all_times