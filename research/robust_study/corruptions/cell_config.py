"""Shared helper for the per-cell config fragments of the robustness study.

A fragment says only which two pipeline entries it swaps and with what arguments; the rest
of the test pipeline is taken from the frozen eval config so a cell can never drift from
the clean path by transcription error.
"""

import os
import sys

from mmcv import Config

REPO_ROOT = "/srv/nfs/shared/gnmp/RaCFormer"
BASE_CONFIG = os.path.join(REPO_ROOT, "configs/racformer_eval_fullval_research.py")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Importing the module is what registers the corruption classes with PIPELINES. It is done
# here, at config-parse time, rather than through `custom_imports`, because the frozen
# driver (research/night_gen_phase1/eval_by_condition.py) never calls
# mmcv.utils.import_modules_from_strings.
from research.robust_study.corruptions import misalign  # noqa: E402,F401

RADAR_FRAME_T = "Loadnuradarpoints"
RADAR_SWEEPS = "LoadradarpointsFromMultiSweeps"


def build_pipeline(frame_t_type, sweeps_type, extra):
    """Copy the frozen test pipeline, swapping only the two radar loader entries.

    Both entries must be swapped together: `Loadnuradarpoints` supplies element 0 of
    `radar_points` and `LoadradarpointsFromMultiSweeps` supplies the rest, so injecting into
    one alone leaves the fault partial (the trap called out in fault-families (d1) and (d2)
    spec item 1).
    """
    pipeline = [dict(step) for step in Config.fromfile(BASE_CONFIG).test_pipeline]
    swapped = 0
    for step in pipeline:
        if step["type"] == RADAR_FRAME_T:
            step["type"] = frame_t_type
            step.update(extra)
            swapped += 1
        elif step["type"] == RADAR_SWEEPS:
            step["type"] = sweeps_type
            step.update(extra)
            swapped += 1
    if swapped != 2:
        raise RuntimeError(
            "expected to swap exactly 2 radar loader entries, swapped {}".format(swapped))
    return pipeline
