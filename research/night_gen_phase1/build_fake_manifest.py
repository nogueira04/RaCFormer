"""
Smoke helper — write a single-sample fake manifest that points generated_path back at
the original day JPGs. Used only by smoke_build_datasets.py to exercise the
LoadMultiViewImageFromManifest swap path without uploading any real generated PNGs.

The output file is NOT wired into racformer_train2k_genaug_research.py — the smoke
script overrides the manifest path in-memory only.

Usage (cluster, racformerfix env):
    conda run -n racformerfix python research/night_gen_phase1/build_fake_manifest.py \
        --src nuscenes_infos_train_2k_day.pkl \
        --out research/night_gen_phase1/manifests/phase1_genaug_FAKE_smoke.json
"""

import argparse
import json
import os
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src", required=True, help="day pkl whose infos[0] becomes the fake sample"
    )
    ap.add_argument("--out", required=True, help="output manifest JSON path")
    ap.add_argument(
        "--validate-with-dataroot",
        default=None,
        help="optional dataroot to check that the 6 cam JPGs exist; relative paths "
        "in the pkl are joined to this. Skip if running where data/nuscenes/ does not resolve.",
    )
    args = ap.parse_args()

    with open(args.src, "rb") as f:
        d = pickle.load(f)
    infos = d["infos"] if isinstance(d, dict) and "infos" in d else d
    if not infos:
        raise RuntimeError(f"{args.src} has no infos")

    info = infos[0]
    token = info["token"]
    cams = info["cams"]
    if len(cams) != 6:
        raise RuntimeError(
            f"expected 6 cams in info, got {len(cams)}: {sorted(cams.keys())}"
        )

    entries = []
    for cam_name, cam_info in cams.items():
        data_path = cam_info["data_path"]
        if args.validate_with_dataroot:
            # data_path looks like "data/nuscenes/samples/CAM_FRONT/foo.jpg".
            # Strip the leading "data/nuscenes/" if it's present and join to dataroot.
            rel = data_path
            if rel.startswith("data/nuscenes/"):
                rel = rel[len("data/nuscenes/") :]
            full = os.path.join(args.validate_with_dataroot, rel)
            if not os.path.exists(full):
                raise RuntimeError(
                    f"original cam JPG not found on disk: {full} (cam={cam_name}, data_path={data_path})"
                )
        entries.append(
            {
                "sample_token": token,
                "camera": cam_name,
                "original_pkl_data_path": data_path,
                "generated_path": data_path,
                "status": "ok",
            }
        )

    manifest = {
        "version": 1,
        "purpose": "smoke-only: re-routes 1 sample's 6 cameras back to the original JPGs",
        "source_pkl": args.src,
        "entries": entries,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[fake_manifest] wrote {args.out}: token={token}, n_cam_entries={len(entries)}"
    )


if __name__ == "__main__":
    main()
