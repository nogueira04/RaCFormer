# (c) radar Doppler/RCS noise -- sigma=3 on raw rcs/vx_comp/vy_comp, seed 0.
#
# Full 6019-sample val, canonical checkpoint, P2 pipeline; only the two radar loader
# entries differ from the base config. Importing the corruption module is what registers
# the corrupting pipeline classes -- the frozen eval driver resolves no `custom_imports`,
# so registration has to happen while this config is being parsed.
#
# Both radar loader entries are swapped (frame-t via Loadnuradarpoints and the sweep stack
# via LoadradarpointsFromMultiSweeps); build_corrupted_val_pipeline asserts it found
# exactly one of each, so a partially corrupted pipeline cannot run silently.

_base_ = ["/srv/nfs/shared/gnmp/RaCFormer/configs/racformer_eval_fullval_research.py"]

corruption = dict(family="radar_doppler_rcs_noise", sigma=3.0, corrupt_seed=0)


def _build_pipeline():
    import importlib.util
    import sys

    name = "robust_study_radar_noise"
    if name in sys.modules:
        mod = sys.modules[name]
    else:
        spec = importlib.util.spec_from_file_location(name, "/srv/nfs/shared/gnmp/RaCFormer/research/robust_study/corruptions/radar_noise.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod.build_corrupted_val_pipeline(_base_[0], corruption)


data = dict(val=dict(pipeline=_build_pipeline()))

del _build_pipeline
