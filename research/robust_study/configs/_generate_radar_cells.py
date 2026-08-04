"""Emit one config fragment per (severity, seed) cell for radar families (b) and (c).

Run once from the repo root:
    python -u research/robust_study/configs/_generate_radar_cells.py

Regenerating is idempotent -- the emitted text is a pure function of the ladders below,
which are the ones registered in fault-families.md (p in {25,50,75}%, sigma in {1,3,5},
seeds {0,1,2}).
"""

import os

REPO = "/srv/nfs/shared/gnmp/RaCFormer"
BASE = REPO + "/configs/racformer_eval_fullval_research.py"
MODULE = REPO + "/research/robust_study/corruptions/radar_noise.py"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE = '''# {title}
#
# Full 6019-sample val, canonical checkpoint, P2 pipeline; only the two radar loader
# entries differ from the base config. Importing the corruption module is what registers
# the corrupting pipeline classes -- the frozen eval driver resolves no `custom_imports`,
# so registration has to happen while this config is being parsed.
#
# Both radar loader entries are swapped (frame-t via Loadnuradarpoints and the sweep stack
# via LoadradarpointsFromMultiSweeps); build_corrupted_val_pipeline asserts it found
# exactly one of each, so a partially corrupted pipeline cannot run silently.

_base_ = ["{base}"]

corruption = dict(family="{family}", {param}={value!r}, corrupt_seed={seed})


def _build_pipeline():
    import importlib.util
    import sys

    name = "robust_study_radar_noise"
    if name in sys.modules:
        mod = sys.modules[name]
    else:
        spec = importlib.util.spec_from_file_location(name, "{module}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod.build_corrupted_val_pipeline(_base_[0], corruption)


data = dict(val=dict(pipeline=_build_pipeline()))

del _build_pipeline
'''

CELLS = []
for _p, _tag in ((0.25, "p25"), (0.50, "p50"), (0.75, "p75")):
    for _seed in (0, 1, 2):
        CELLS.append(dict(
            name="radar_dropout_{}_s{}".format(_tag, _seed),
            title="(b) radar point dropout -- {:.0f}% of points removed per sweep, seed {}.".format(
                _p * 100, _seed),
            family="radar_dropout", param="drop_p", value=_p, seed=_seed,
        ))
for _sigma, _tag in ((1.0, "sig1"), (3.0, "sig3"), (5.0, "sig5")):
    for _seed in (0, 1, 2):
        CELLS.append(dict(
            name="radar_noise_{}_s{}".format(_tag, _seed),
            title="(c) radar Doppler/RCS noise -- sigma={:g} on raw rcs/vx_comp/vy_comp, seed {}.".format(
                _sigma, _seed),
            family="radar_doppler_rcs_noise", param="sigma", value=_sigma, seed=_seed,
        ))


def main():
    for cell in CELLS:
        text = TEMPLATE.format(base=BASE, module=MODULE, **cell)
        path = os.path.join(OUT_DIR, cell["name"] + ".py")
        with open(path, "w") as fh:
            fh.write(text)
        print("wrote", path)
    print("total", len(CELLS))


if __name__ == "__main__":
    main()
