from pathlib import Path
import copy
import sys
import traceback
import yaml
import torch
from mmcv import Config

ROOT = Path('/srv/nfs/shared/gnmp/RaCFormer/research/night_gen_phase1/teachers/bevfusion_src')
CKPT = Path('/srv/nfs/shared/gnmp/RaCFormer/research/night_gen_phase1/teachers/bevfusion-det.pth')
CONFIGS = [
    ROOT / 'configs/default.yaml',
    ROOT / 'configs/nuscenes/default.yaml',
    ROOT / 'configs/nuscenes/det/default.yaml',
    ROOT / 'configs/nuscenes/det/transfusion/default.yaml',
    ROOT / 'configs/nuscenes/det/transfusion/secfpn/default.yaml',
    ROOT / 'configs/nuscenes/det/transfusion/secfpn/camera+lidar/default.yaml',
    ROOT / 'configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/default.yaml',
    ROOT / 'configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml',
]

def merge(a, b):
    a = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = merge(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a

class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

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
        out = eval(obj[2:-1], globals_)
        return recursive_eval(out, globals_)
    return obj

cfg = {}
for p in CONFIGS:
    with p.open() as f:
        cfg = merge(cfg, yaml.safe_load(f))
cfg = recursive_eval(cfg)
print('merged_config_files')
for p in CONFIGS:
    print(' -', p.relative_to(ROOT))
print('model_type', cfg['model']['type'])
print('object_classes', cfg['object_classes'])
print('model_keys', sorted(cfg['model'].keys()))

# Ensure local source wins over installed mmdet3d.
sys.path.insert(0, str(ROOT))



try:
    from mmcv.utils.registry import Registry
    _orig_register_module = Registry._register_module
    def _stage2_register_module(self, module, module_name=None, force=False):
        try:
            return _orig_register_module(self, module, module_name=module_name, force=force)
        except KeyError as exc:
            if 'already registered' not in str(exc):
                raise
            names = module_name if isinstance(module_name, list) else [module_name or module.__name__]
            for name in names:
                self._module_dict[name] = module
            return None
    Registry._register_module = _stage2_register_module
except Exception:
    pass

try:
    from mmcv.cnn import CONV_LAYERS
    for _k in list(CONV_LAYERS.module_dict.keys()):
        if _k.startswith('SparseConv') or _k.startswith('SparseInverseConv') or _k.startswith('SubMConv'):
            CONV_LAYERS._module_dict.pop(_k, None)
except Exception:
    pass

# D1-only import shim: state-dict loading does not execute custom CUDA ops.
import types
def _dummy_ext_module(name):
    mod = types.ModuleType(name)
    mod.__file__ = name.replace('.', '/') + '.so'
    def _missing(*args, **kwargs):
        raise RuntimeError(f'Dummy extension function called from {name}; Stage 2 D1 shim is import/load-only')
    def __getattr__(attr):
        return _missing
    mod.__getattr__ = __getattr__
    return mod

# Minimal flash_attn shim for unused radar encoder imports during D1 model construction.
import torch.nn as _nn
_flash_pkg = types.ModuleType('flash_attn')
_flash_mod = types.ModuleType('flash_attn.flash_attention')
class _FlashMHA(_nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, *args, **kwargs):
        raise RuntimeError('FlashMHA shim called; radar encoder is unused in BEVFusion C+L D1')
_flash_mod.FlashMHA = _FlashMHA
sys.modules.setdefault('flash_attn', _flash_pkg)
sys.modules.setdefault('flash_attn.flash_attention', _flash_mod)

for name in [
    'mmdet3d.ops.ball_query.ball_query_ext',
    'mmdet3d.ops.knn.knn_ext',
    'mmdet3d.ops.paconv.assign_score_withk_ext',
    'mmdet3d.ops.group_points.group_points_ext',
    'mmdet3d.ops.interpolate.interpolate_ext',
    'mmdet3d.ops.furthest_point_sample.furthest_point_sample_ext',
    'mmdet3d.ops.gather_points.gather_points_ext',
    'mmdet3d.ops.iou3d.iou3d_cuda',
    'mmdet3d.ops.voxel.voxel_layer',
    'mmdet3d.ops.bev_pool.bev_pool_ext',
    'mmdet3d.ops.feature_decorator.feature_decorator_ext',
    'mmdet3d.ops.roiaware_pool3d.roiaware_pool3d_ext',
    'mmdet3d.ops.spconv.sparse_conv_ext',
]:
    sys.modules.setdefault(name, _dummy_ext_module(name))

try:
    import mmdet3d  # noqa
    from mmdet3d.models import build_model
    model_cfg = Config(cfg).model
    model_cfg.train_cfg = None
    model = build_model(model_cfg, test_cfg=Config(cfg).get('test_cfg'))
    ckpt = torch.load(str(CKPT), map_location='cpu')
    state_dict = ckpt['state_dict']
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    print('build_model=OK')
    print('missing_count', len(missing))
    print('unexpected_count', len(unexpected))
    if missing:
        print('missing_first20', missing[:20])
    if unexpected:
        print('unexpected_first20', unexpected[:20])
except Exception as exc:
    print('build_or_load_exception', type(exc).__name__, str(exc))
    traceback.print_exc()
    raise
