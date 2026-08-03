"""Write the per-run `provenance.json` for a robustness-study evaluation cell.

Runs inside the evaluation env (so the versions it records are the ones that actually ran) and is
called twice-safe: the job's normal path calls it after the end-state assertions, and the EXIT
trap calls it with a failure validity if the body died first. Every field is best-effort — a
missing artifact is recorded as null rather than crashing, because provenance for a failed run is
exactly when it matters most.

`validity` is decided by the caller, not here: VALID only when the eval succeeded AND the
start/end SHA + clean-tree assertions all held; anything else is INVALID with a reason string.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import OrderedDict


def _hash_file(path, algo):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions():
    """Record the versions that actually ran.

    The nuscenes devkit exposes no `__version__`, so every entry falls back to the installed
    distribution's metadata; recording "unknown" for the devkit would defeat the point of pinning
    it in the first place.
    """
    import importlib.metadata as md  # noqa: WPS433

    out = OrderedDict()
    out["python"] = sys.version.split()[0]
    devkit_path = None
    for name, module, dist in (("torch", "torch", "torch"), ("mmcv", "mmcv", "mmcv-full"),
                               ("mmdet", "mmdet", "mmdet"), ("mmdet3d", "mmdet3d", "mmdet3d"),
                               ("numpy", "numpy", "numpy"),
                               ("nuscenes_devkit", "nuscenes", "nuscenes-devkit")):
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", None)
            if version is None:
                version = md.version(dist)
            out[name] = version
            if module == "nuscenes":
                devkit_path = getattr(mod, "__file__", None)
        except Exception as exc:  # noqa: BLE001
            out[name] = "IMPORT_FAILED: %s" % exc
    try:
        import torch  # noqa: WPS433

        out["cuda"] = torch.version.cuda
        out["cudnn"] = str(torch.backends.cudnn.version())
        out["torch_cuda_available"] = torch.cuda.is_available()
        out["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:  # noqa: BLE001
        out["cuda"] = "UNAVAILABLE: %s" % exc
    return out, devkit_path


def _read_json(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _git(repo, *args):
    try:
        return subprocess.check_output(["git", "-C", repo] + list(args), stderr=subprocess.DEVNULL) \
            .decode().strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--validity", required=True, help='"VALID" or "INVALID: <reason>"')
    ap.add_argument("--start-sha", default=None)
    ap.add_argument("--end-sha", default=None)
    ap.add_argument("--dirty-start", default="unknown", choices=["true", "false", "unknown"])
    ap.add_argument("--dirty-end", default="unknown", choices=["true", "false", "unknown"])
    ap.add_argument("--git-tag", default=None)
    ap.add_argument("--exact-command", required=True)
    ap.add_argument("--slurm-job-id", default=None)
    ap.add_argument("--slurm-log-path", default=None)
    ap.add_argument("--node", default=None)
    ap.add_argument("--date-utc", default=None)
    ap.add_argument("--env-name", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--driver-script", default=None)
    ap.add_argument("--submission", default=None)
    ap.add_argument("--token-set-json", default=None)
    ap.add_argument("--seeds-json", default=None)
    ap.add_argument("--completion", default=None)
    ap.add_argument("--cell", default=None)
    ap.add_argument("--phase", default=None)
    args = ap.parse_args()

    versions, devkit_path = _versions()
    token_set = _read_json(args.token_set_json) or {}
    seeds = _read_json(args.seeds_json) or {}

    dirty_start = args.dirty_start == "true"
    dirty_end = args.dirty_end == "true"

    prov = OrderedDict()
    prov["phase"] = args.phase
    prov["cell"] = args.cell
    prov["start_sha"] = args.start_sha
    prov["end_sha"] = args.end_sha
    prov["validity"] = args.validity
    prov["git_tag"] = args.git_tag or _git(args.repo, "describe", "--tags", "--exact-match")
    # Single dirty verdict as specified; the two observations it is derived from are kept beside it
    # because "which end was dirty" is the first thing anyone debugging an INVALID run asks.
    prov["dirty"] = bool(dirty_start or dirty_end)
    prov["dirty_start"] = args.dirty_start
    prov["dirty_end"] = args.dirty_end
    prov["exact_command"] = args.exact_command
    prov["slurm_job_id"] = args.slurm_job_id
    prov["slurm_log_path"] = args.slurm_log_path
    prov["node"] = args.node
    prov["date_utc"] = args.date_utc
    prov["env_name"] = args.env_name
    prov["versions"] = versions
    prov["devkit_module_path"] = devkit_path
    prov["config_path"] = args.config
    prov["config_md5"] = _hash_file(args.config, "md5")
    prov["weights_path"] = args.weights
    prov["weights_md5"] = _hash_file(args.weights, "md5")
    prov["driver_script_path"] = args.driver_script
    prov["driver_script_sha256"] = _hash_file(args.driver_script, "sha256")
    prov["submission_path"] = args.submission
    prov["submission_json_sha256"] = _hash_file(args.submission, "sha256")
    prov["n_samples"] = token_set.get("n_submission_tokens")
    prov["token_set_sha256"] = token_set.get("token_set_sha256")
    prov["token_set_check"] = token_set or None
    prov["seeds"] = seeds
    prov["completion"] = args.completion

    with open(args.out, "w") as f:
        json.dump(prov, f, indent=2)
    print("[write_provenance] wrote %s (validity=%s)" % (args.out, args.validity))


if __name__ == "__main__":
    main()
