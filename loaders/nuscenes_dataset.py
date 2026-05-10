import os
import os.path as osp
import numpy as np
import pickle
from mmdet3d.registry import DATASETS
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


nu_version = os.environ.get('NUSCENES_VERSION', 'v1.0-trainval')
renusc = NuScenes(version=nu_version, dataroot=str('data/nuscenes/'), verbose=False)

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    # Default METAINFO with nuScenes classes - will be overridden by metainfo kwarg
    METAINFO = {
        'classes': (
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ),
    }

    def __init__(self,
                 camera_types=None,
                 radar_types=None,
                 max_samples=None,
                 # Compatibility: old-style args that need to be converted for mmdet3d 1.4+
                 classes=None,
                 modality=None,
                 box_type_3d=None,
                 use_valid_flag=None,
                 **kwargs):
        # Convert old-style 'classes' to new-style 'metainfo'
        if classes is not None:
            if 'metainfo' not in kwargs:
                kwargs['metainfo'] = dict(classes=tuple(classes))
            elif 'classes' not in kwargs['metainfo']:
                kwargs['metainfo']['classes'] = tuple(classes)

        # Store modality for later - we'll set it after super().__init__() which overwrites it
        _modality = modality if modality is not None else dict(use_camera=True, use_lidar=True)

        # Filter out deprecated arguments that shouldn't go to parent
        kwargs.pop('modality', None)
        kwargs.pop('box_type_3d', None)
        kwargs.pop('use_valid_flag', None)
        kwargs.pop('classes', None)  # Already converted to metainfo

        # Store max_samples before super().__init__ (which calls full_init -> load_data_list)
        self._max_samples = max_samples

        # Use lazy_init to avoid Det3DDataset's problematic statistics code
        # which accesses self.metainfo['classes'] before it's properly set
        kwargs['lazy_init'] = True
        # Disable serialization to keep data in _data_list directly
        # This avoids pickle serialization issues with complex numpy arrays
        kwargs['serialize_data'] = False

        super().__init__(**kwargs)

        # Re-set modality AFTER super().__init__() because parent class overwrites it
        self.modality = _modality
        # mmdet3d 1.4+ Det3DDataset expects this attribute
        self.show_ins_var = False

        # Now manually complete initialization
        self.full_init()

        self.camera_types = camera_types
        self.radar_types = radar_types

        # Store for compatibility
        self.CLASSES = classes
        self.max_samples = max_samples

    def load_data_list(self):
        """Load annotations from an annotation file (old format compatibility).

        The old mmdet3d format uses 'infos' key, while new mmengine format
        uses 'data_list' and 'metainfo' keys.

        Returns:
            list[dict]: A list of annotation dicts.
        """
        with open(self.ann_file, 'rb') as f:
            data = pickle.load(f)

        # Handle old format (infos key)
        if 'infos' in data:
            data_list = data['infos']
            if 'metadata' in data:
                self._metainfo = data['metadata']
        # Handle new format (data_list key)
        elif 'data_list' in data:
            data_list = data['data_list']
            if 'metainfo' in data:
                self._metainfo.update(data['metainfo'])
        else:
            raise ValueError('Annotation file must have "infos" or "data_list" key')

        # Apply max_samples limit if set
        if hasattr(self, '_max_samples') and self._max_samples is not None:
            data_list = data_list[:self._max_samples]

        # Store our own copy for backward compatibility (data_infos property)
        self._custom_data_list = data_list
        return data_list

    @property
    def data_infos(self):
        """Property for backward compatibility - returns the internal data list."""
        # Use our custom stored list first
        if hasattr(self, '_custom_data_list') and self._custom_data_list is not None:
            return self._custom_data_list
        # Fallback to mmengine's internal storage
        if hasattr(self, '_data_list') and self._data_list is not None:
            return self._data_list
        # Last resort: use the property
        return self.data_list

    def get_data_info(self, index):
        """Get data info by index - override to use data_infos directly."""
        return self._get_data_info_impl(index)

    def _get_data_info_impl(self, index):
        """Implementation of get_data_info using data_infos."""
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            pts_filename=info['lidar_path'],
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
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
                intrinsics=cam_intrinsics,
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def _to_numpy(self, tensor):
        """Convert tensor to numpy, handling CUDA and low-precision tensors."""
        if tensor is None:
            return None
        if isinstance(tensor, np.ndarray):
            return tensor
        if hasattr(tensor, 'detach'):
            tensor = tensor.detach()
        if hasattr(tensor, 'is_floating_point') and tensor.is_floating_point():
            tensor = tensor.float()
        if hasattr(tensor, 'cpu'):
            tensor = tensor.cpu()
        if hasattr(tensor, 'numpy'):
            return tensor.numpy()
        return np.array(tensor)

    def evaluate(self, results, logger=None, front_only=False, jsonfile_prefix=None, **kwargs):
        """Evaluate detection results using NuScenes evaluation.

        Args:
            results: Detection results
            logger: Logger for output
            front_only: If True, filter GT and predictions to front-only (x > 0).
            jsonfile_prefix: Prefix for output JSON files
            **kwargs: Additional arguments

        Returns:
            dict: Evaluation metrics
        """
        import json
        from nuscenes.eval.detection.config import config_factory

        # Get class names
        class_names = self.metainfo.get('classes', self.METAINFO['classes'])

        # Convert results to NuScenes submission format
        nusc_submissions = {'meta': {'use_camera': True, 'use_lidar': False,
                                      'use_radar': True, 'use_map': False,
                                      'use_external': False},
                           'results': {}}

        print(f"Converting {len(results)} results to NuScenes format...")
        for i, res in enumerate(results):
            if 'pts_bbox' in res:
                res = res['pts_bbox']

            info = self.data_infos[i]
            sample_token = info['token']

            boxes_3d = res['boxes_3d']
            scores_3d = self._to_numpy(res['scores_3d'])
            labels_3d = self._to_numpy(res['labels_3d'])

            # Get box tensor (handles both LiDARInstance3DBoxes and raw tensors)
            if hasattr(boxes_3d, 'tensor'):
                box_tensor = self._to_numpy(boxes_3d.tensor)  # [N, 9] cx,cy,cz,l,w,h,rot,vx,vy
            else:
                box_tensor = self._to_numpy(boxes_3d)

            num_boxes = len(box_tensor) if box_tensor is not None else 0

            # Apply front-only filtering if requested
            if front_only and num_boxes > 0:
                lidar2ego_translation = np.array(info['lidar2ego_translation'])
                lidar2ego_rotation = Quaternion(info['lidar2ego_rotation']).rotation_matrix
                centers = box_tensor[:, :3]
                centers_ego = centers @ lidar2ego_rotation.T + lidar2ego_translation
                mask = centers_ego[:, 0] > 0
                box_tensor = box_tensor[mask]
                scores_3d = scores_3d[mask]
                labels_3d = labels_3d[mask]
                num_boxes = len(box_tensor)

            # Convert to NuScenes format (global frame)
            sample_results = []

            if num_boxes > 0:
                # Get transforms for lidar to global
                lidar2ego = np.eye(4)
                lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
                lidar2ego[:3, 3] = np.array(info['lidar2ego_translation'])

                ego2global = np.eye(4)
                ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
                ego2global[:3, 3] = np.array(info['ego2global_translation'])

                lidar2global = ego2global @ lidar2ego

                for j in range(num_boxes):
                    # Box format: cx, cy, cz, l, w, h, rot, [vx, vy]
                    box = box_tensor[j]
                    center = box[:3]
                    dims = box[3:6]  # l, w, h
                    yaw = float(box[6])

                    # Transform center to global frame
                    center_homo = np.array([center[0], center[1], center[2], 1.0])
                    center_global = (lidar2global @ center_homo)[:3]

                    # Compute global rotation (lidar yaw + ego rotation + global rotation)
                    lidar_rot = Quaternion(axis=[0, 0, 1], angle=yaw)
                    ego_rot = Quaternion(info['lidar2ego_rotation'])
                    global_rot = Quaternion(info['ego2global_rotation'])
                    final_rot = global_rot * ego_rot * lidar_rot

                    # Get velocity if available (columns 7,8)
                    if box.shape[0] >= 9:
                        vel = np.array([box[7], box[8], 0.0])
                        vel_global = lidar2global[:3, :3] @ vel
                    else:
                        vel_global = np.array([0.0, 0.0, 0.0])

                    label_idx = int(labels_3d[j])
                    if label_idx >= len(class_names):
                        print(f"Warning: label {label_idx} >= num_classes {len(class_names)}, skipping")
                        continue

                    detection_name = class_names[label_idx]

                    sample_results.append({
                        'sample_token': sample_token,
                        'translation': [float(center_global[0]), float(center_global[1]), float(center_global[2])],
                        'size': [float(dims[1]), float(dims[0]), float(dims[2])],  # l,w,h -> w,l,h (nusc format)
                        'rotation': [float(x) for x in final_rot.elements.tolist()],
                        'velocity': [float(vel_global[0]), float(vel_global[1])],
                        'detection_name': detection_name,
                        'detection_score': float(scores_3d[j]),
                        'attribute_name': ''
                    })

            nusc_submissions['results'][sample_token] = sample_results

        # Write results to JSON
        if jsonfile_prefix is None:
            jsonfile_prefix = 'submission'
        result_file = f'{jsonfile_prefix}_results.json'

        print(f"Writing results to {result_file}...")
        with open(result_file, 'w') as f:
            json.dump(nusc_submissions, f)

        # Run NuScenes evaluation
        print("Running NuScenes evaluation...")
        try:
            eval_set = 'mini_val' if 'mini' in nu_version else 'val'
            cfg = config_factory('detection_cvpr_2019')

            nusc_eval = NuScenesEval(
                renusc,
                config=cfg,
                result_path=result_file,
                eval_set=eval_set,
                output_dir='./eval_output',
                verbose=True
            )
            metrics = nusc_eval.main(render_curves=False)

            # Extract key metrics
            result_dict = {
                'mAP': float(metrics['mean_ap']),
                'NDS': float(metrics['nd_score']),
                'mATE': float(metrics['tp_errors']['trans_err']),
                'mASE': float(metrics['tp_errors']['scale_err']),
                'mAOE': float(metrics['tp_errors']['orient_err']),
                'mAVE': float(metrics['tp_errors']['vel_err']),
                'mAAE': float(metrics['tp_errors']['attr_err']),
            }

            print("\n" + "="*50)
            print("NuScenes Detection Metrics:")
            print("="*50)
            for k, v in result_dict.items():
                print(f"  {k}: {v:.4f}")
            print("="*50 + "\n")

            return result_dict

        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}



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

@DATASETS.register_module()
class CustomNuScenesDataset_radar(CustomNuScenesDataset):

    def get_data_info(self, index):
        """Override get_data_info to add scene_condition and use data_infos."""
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix

        scene_condition = self._get_scene_condition(info['token'])

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            pts_filename=info['lidar_path'],
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
            scene_condition=scene_condition,
        )

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []
            cam_intrinsics = []
            if self.camera_types is None:
                cams_to_use = info['cams'].items()
            else:
                cams_to_use = [(k, v) for k, v in info['cams'].items() if k in self.camera_types]

            for _, cam_info in cams_to_use:
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

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
                intrinsics=cam_intrinsics,
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def _get_scene_condition(self, sample_token):
        """
        Extract scene condition from nuScenes scene description.
        
        ORACLE TEST: This function parses the scene description to classify
        the condition as 'night', 'rain', or 'day' (default).
        
        nuScenes scene descriptions contain keywords like:
        - "night" for night scenes
        - "rain", "rainy" for rainy conditions
        - Most other scenes are daytime/clear
        
        Args:
            sample_token: The sample token to get condition for
            
        Returns:
            str: 'night', 'rain', or 'day'
        """
        try:
            sample = renusc.get('sample', sample_token)
            scene = renusc.get('scene', sample['scene_token'])
            description = scene.get('description', '').lower()
            
            # Check for night condition
            if 'night' in description:
                return 'night'
            # Check for rain condition
            elif 'rain' in description or 'rainy' in description:
                return 'rain'
            # Default to day/clear
            else:
                return 'day'
        except Exception:
            # If we can't determine, default to 'day'
            return 'day'


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