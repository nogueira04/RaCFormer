from __future__ import annotations

import copy
import json
from pathlib import Path
import yaml
from mmcv import Config

root = Path('/srv/nfs/shared/gnmp/RaCFormer')
bev_root = root / 'research/night_gen_phase1/teachers/bevfusion_src'
config_chain = [
    bev_root / 'configs/default.yaml',
    bev_root / 'configs/nuscenes/default.yaml',
    bev_root / 'configs/nuscenes/det/default.yaml',
    bev_root / 'configs/nuscenes/det/transfusion/default.yaml',
    bev_root / 'configs/nuscenes/det/transfusion/secfpn/default.yaml',
    bev_root / 'configs/nuscenes/det/transfusion/secfpn/camera+lidar/default.yaml',
    bev_root / 'configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/default.yaml',
    bev_root / 'configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml',
]

class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

def merge(a, b):
    a = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = merge(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a

def to_attr(obj):
    if isinstance(obj, dict):
        return AttrDict({k: to_attr(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_attr(v) for v in obj]
    return obj

def recursive_eval(obj, globals_=None):
    if globals_ is None:
        globals_ = to_attr(copy.deepcopy(obj))
    if isinstance(obj, dict):
        return {k: recursive_eval(v, globals_) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_eval(v, globals_) for v in obj]
    if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
        return recursive_eval(eval(obj[2:-1], globals_), globals_)
    return obj

def grid_hw(bound):
    return int(round((bound[1] - bound[0]) / bound[2]))

teacher = {}
for p in config_chain:
    with p.open() as f:
        teacher = merge(teacher, yaml.safe_load(f))
teacher = recursive_eval(teacher)
student = Config.fromfile(str(root / 'configs/racformer_r50_nuimg_704x256_f8.py'))

teacher_classes = list(teacher['object_classes'])
student_classes = list(student.class_names)
student_grid = student.grid_config
teacher_vt = teacher['model']['encoders']['camera']['vtransform']
teacher_fuser = teacher['model']['fuser']
student_shape = {
    'camera_lss_bev': [None, 'T', student.img_lss_view_transformer['out_channels'], grid_hw(student_grid['y']), grid_hw(student_grid['x'])],
    'radar_bev': [None, 'T', student.model.pts_bbox_head.transformer.embed_dims, 128, 128],
    'query_fusion_output': [None, student.model.pts_bbox_head.num_query, student.model.pts_bbox_head.transformer.embed_dims],
}
teacher_shape = {
    'camera_vtransform': [None, teacher_vt['out_channels'], grid_hw(teacher_vt['ybound']), grid_hw(teacher_vt['xbound'])],
    'fuser_output': [None, teacher_fuser['out_channels'], grid_hw(teacher_vt['ybound']), grid_hw(teacher_vt['xbound'])],
    'decoder_neck_output': [None, sum(teacher['model']['decoder']['neck']['out_channels']), grid_hw(teacher_vt['ybound']), grid_hw(teacher_vt['xbound'])],
}
result = {
    'teacher_config_chain': [str(p.relative_to(bev_root)) for p in config_chain],
    'teacher_classes': teacher_classes,
    'student_classes': student_classes,
    'class_order_match': teacher_classes == student_classes,
    'class_mismatches': [
        {'index': i, 'teacher': t, 'student': s}
        for i, (t, s) in enumerate(zip(teacher_classes, student_classes)) if t != s
    ],
    'teacher_bev_shapes_static': teacher_shape,
    'student_bev_shapes_static': student_shape,
    'student_dense_fused_bev_hook': {
        'exists': False,
        'reason': 'RaCFormer extract_feat returns separate camera/LSS all_bev_feats and radar_bev_feats; dense camera-radar fusion happens later as per-query Linear(embed_dims*3, embed_dims), not as a single dense fused BEV map.'
    },
    'adapter_single_1x1_possible_strict': False,
    'adapter_reason': 'Teacher BEV grid is 360x360 over [-54,54] at 0.3m; RaCFormer BEV grids are 128x128 over [-51.2,51.2] at 0.8m. A strict single learned 1x1 conv cannot crop/resample spatial grids, and decoder-neck teacher channels are 512 while student channels are 256.',
}
out = root / 'research/night_gen_phase1/teachers/diagnostics/D3_D4_D5_static.json'
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
