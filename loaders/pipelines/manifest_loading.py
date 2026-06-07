"""
T4 — Generated-image manifest loader for fixed-partition (Interpretation B) night augmentation.

Wraps `LoadMultiViewImageFromFiles` (mmdet3d) so that, for each (sample_token, camera) pair
present in the manifest, the keyframe image is loaded from the generated PNG instead of the
original day JPG. Misses fall back to standard behavior. Original `cam_info['data_path']`
fields in `data_infos` are NEVER mutated; the swap only happens at decode time.

Manifest schema (research/night_gen_phase1/manifests/phase1_genaug_*.json):
    {
        "version": 1,
        "entries": [
            {"sample_token": "...", "camera": "CAM_FRONT",
             "original_pkl_data_path": "data/nuscenes/samples/CAM_FRONT/...jpg",
             "generated_path": "research/night_gen_phase1/outputs/.../CAM_FRONT/...png",
             "status": "ok"}
        ]
    }

Usage in a config train pipeline (replaces the standard LoadMultiViewImageFromFiles):
    dict(type='LoadMultiViewImageFromManifest',
         manifest_path='research/night_gen_phase1/manifests/phase1_genaug_seed20260425_manifest.json',
         to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1),
    ...
"""

import json
import logging
import os

import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES


_CAM_NAMES = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
)


def _camera_from_path(path: str):
    for cam in _CAM_NAMES:
        if f"/{cam}/" in path or f"\\{cam}\\" in path:
            return cam
    return None


@PIPELINES.register_module()
class LoadMultiViewImageFromManifest(object):
    """Drop-in replacement for `LoadMultiViewImageFromFiles` with manifest-based path swap.

    Args:
        manifest_path (str): JSON manifest written by Phase 1 generation. May be relative
            to cwd at construction time (resolved once).
        to_float32 (bool): Forwarded to mmdet3d behavior.
        color_type (str): mmcv color flag (default 'unchanged' to match upstream).
        sample_idx_key (str): Key in `results` carrying the sample token.
            Default: 'sample_idx' (matches CustomNuScenesDataset_radar.get_data_info).
        generated_sample_weight (float): If set below 1.0, samples with any generated
            keyframe camera get this loss weight via img_metas.
        log_summary (bool): If True, log a single hit/miss summary per worker the first time
            the loader runs.
    """

    def __init__(
        self,
        manifest_path,
        to_float32=False,
        color_type="unchanged",
        sample_idx_key="sample_idx",
        generated_sample_weight=1.0,
        log_summary=True,
    ):
        self.manifest_path = manifest_path
        self.to_float32 = to_float32
        self.color_type = color_type
        self.sample_idx_key = sample_idx_key
        self.generated_sample_weight = float(generated_sample_weight)
        self.log_summary = log_summary

        self._manifest = self._load_manifest(manifest_path)
        self._n_hits = 0
        self._n_misses = 0
        self._summary_logged = False

    @staticmethod
    def _load_manifest(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"manifest not found: {path}")
        with open(path) as fh:
            data = json.load(fh)
        entries = data.get("entries", data)
        index = {}
        for e in entries:
            if e.get("status") and e["status"] != "ok":
                continue
            tok = e.get("sample_token")
            cam = e.get("camera")
            gen = e.get("generated_path")
            if tok and cam and gen:
                index[(tok, cam)] = gen
        if not index:
            raise ValueError(f"manifest {path} contained no usable entries")
        return index

    def __call__(self, results):
        sample_token = results.get(self.sample_idx_key)
        filenames = list(results["img_filename"])
        new_filenames = list(filenames)
        hits_this_sample = 0
        for i, name in enumerate(filenames):
            cam = _camera_from_path(name)
            if cam is None or sample_token is None:
                continue
            swap = self._manifest.get((sample_token, cam))
            if swap is not None and os.path.exists(swap):
                new_filenames[i] = swap
                hits_this_sample += 1

        self._n_hits += hits_this_sample
        self._n_misses += len(filenames) - hits_this_sample

        if (
            self.log_summary
            and not self._summary_logged
            and (self._n_hits + self._n_misses) >= 6
        ):
            logging.info(
                "[LoadMultiViewImageFromManifest] first %d image-loads: %d hits / %d misses (manifest=%s, n_entries=%d)",
                self._n_hits + self._n_misses,
                self._n_hits,
                self._n_misses,
                self.manifest_path,
                len(self._manifest),
            )
            self._summary_logged = True

        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in new_filenames], axis=-1
        )
        if self.to_float32:
            img = img.astype(np.float32)
        results["filename"] = new_filenames
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        results["img_norm_cfg"] = dict(
            mean=np.zeros(img.shape[2], dtype=np.float32),
            std=np.ones(img.shape[2], dtype=np.float32),
            to_rgb=False,
        )
        # Auxiliary: count of generated-image hits for this sample (debug only).
        results["manifest_hits"] = hits_this_sample
        results["generated_sample_weight"] = (
            self.generated_sample_weight if hits_this_sample > 0 else 1.0
        )
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"manifest_path='{self.manifest_path}', "
            f"to_float32={self.to_float32}, color_type='{self.color_type}', "
            f"generated_sample_weight={self.generated_sample_weight}, "
            f"n_manifest_entries={len(self._manifest)})"
        )
