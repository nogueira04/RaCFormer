"""Fault-injection pipeline wrappers for the robustness study.

Importing this package registers the misalign ((d1)/(d2)) pipeline classes with the
mmdet3d PIPELINES registry. The radar_noise ((b)/(c)) classes are deliberately NOT
imported here: the 18 (b)/(c) cell configs side-load radar_noise.py under a private
module name (configs/_generate_radar_cells.py), and importing it here as well would
double-register the four Radar* classes and crash those cells with an mmcv duplicate-
registration error. No file under loaders/, configs/ or checkpoints/ is modified by
anything in here.
"""

from . import misalign  # noqa: F401
