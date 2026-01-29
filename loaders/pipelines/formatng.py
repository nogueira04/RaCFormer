import numpy as np
from mmengine.registry import TRANSFORMS as PIPELINES
from mmdet3d.structures.points import BasePoints
from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes

# Compatibility wrapper for to_tensor
def to_tensor(data):
    import torch
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, (list, tuple)):
        return torch.tensor(data)
    return data

# Simple DataContainer replacement
class DC:
    def __init__(self, data, stack=True, cpu_only=False):
        self.data = data
        self.stack = stack
        self.cpu_only = cpu_only

@PIPELINES.register_module(force=True)
class RaCFormatBundle3D:
    """Custom formatting bundle for RaCFormer (standalone, not inheriting from Pack3DDetInputs)."""

    def __init__(self, class_names, with_gt=True, with_label=True):
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        import torch
        # Convert images to tensor
        # Model expects 5D: (batch, num_frames, C, H, W)
        if 'img' in results:
            if isinstance(results['img'], list):
                # List of images - stack and convert
                imgs = np.stack(results['img'], axis=0)  # (N, H, W, C)
                # Transpose to (N, C, H, W) for PyTorch
                imgs = imgs.transpose(0, 3, 1, 2)
                # Add batch dimension: (1, N, C, H, W)
                results['img'] = torch.from_numpy(imgs.copy()).float().unsqueeze(0)
            elif isinstance(results['img'], np.ndarray):
                img = results['img']
                if img.ndim == 3:  # (H, W, C)
                    img = img.transpose(2, 0, 1)  # (C, H, W)
                    img = img[np.newaxis, np.newaxis, ...]  # (1, 1, C, H, W)
                elif img.ndim == 4:  # (N, H, W, C)
                    img = img.transpose(0, 3, 1, 2)  # (N, C, H, W)
                    img = img[np.newaxis, ...]  # (1, N, C, H, W)
                results['img'] = torch.from_numpy(img.copy()).float()

        # Format 3D data - convert to raw tensors (no DC wrapper since we use pseudo_collate)
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = results['points'].tensor

        if 'radar_points' in results:
            if isinstance(results['radar_points'], list):
                radar_list = []
                for i in range(len(results['radar_points'])):
                    radar_points = results['radar_points'][i]
                    assert isinstance(radar_points, BasePoints)
                    radar_list.append(radar_points.tensor)
                results['radar_points'] = radar_list
            else:
                assert isinstance(results['radar_points'], BasePoints)
                results['radar_points'] = results['radar_points'].tensor

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = to_tensor(results[key])

        if self.with_gt:
            # Clean GT bboxes in the final
            if 'gt_bboxes_3d_mask' in results:
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][
                    gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][
                        gt_bboxes_3d_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][
                        gt_bboxes_3d_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                 dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                                                    dtype=np.int64)
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n)
                        for n in results['gt_names_3d']
                    ], dtype=np.int64)
        # Process results (standalone, no DC wrapper)
        for key in [
                'gt_labels_static3d', 'gt_labels_dynamic3d'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])
        if 'gt_bboxes_static3d' in results:
            if isinstance(results['gt_bboxes_static3d'], BaseInstance3DBoxes):
                results['gt_bboxes_static3d'] = results['gt_bboxes_static3d']
            else:
                results['gt_bboxes_static3d'] = to_tensor(results['gt_bboxes_static3d'])
        if 'gt_bboxes_dynamic3d' in results:
            if isinstance(results['gt_bboxes_dynamic3d'], BaseInstance3DBoxes):
                results['gt_bboxes_dynamic3d'] = results['gt_bboxes_dynamic3d']
            else:
                results['gt_bboxes_dynamic3d'] = to_tensor(results['gt_bboxes_dynamic3d'])
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str
