"""Mini-screen probe for radar corruption families (b) dropout and (c) Doppler/RCS noise.

NO model is built and no inference runs -- this exercises only the two radar loader
pipeline stages on a handful of val samples and reports what they produced.

Emits, per family:
  * surviving point counts per severity and element (frame-t + each sweep-stack element)
  * raw-field mean/std of rcs / vx_comp / vy_comp, clean vs each sigma
  * CRN checks: (b) keep-set nesting p25 >= p50 >= p75 for a fixed seed
                (c) delta at sigma=5 equals 5x delta at sigma=1, same seed
  * determinism: rebuilding the pipeline reproduces bit-identical tensors
  * invariants: (c) leaves x/y/z/dt bit-identical; (b) returns a sub-multiset of clean rows
"""

import argparse
import collections
import datetime
import hashlib
import importlib.util
import json
import os
import sys

REPO = "/srv/nfs/shared/gnmp/RaCFormer"
MODULE_PATH = REPO + "/research/robust_study/corruptions/radar_noise.py"
BASE_CFG = REPO + "/configs/racformer_eval_fullval_research.py"

# renusc is constructed with a RELATIVE dataroot (loaders/nuscenes_dataset.py:21).
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmdet.datasets.builder import PIPELINES  # noqa: E402

import loaders  # noqa: F401,E402  (registers the frozen pipelines + dataset)
from mmdet3d.datasets import build_dataset  # noqa: E402

_spec = importlib.util.spec_from_file_location("robust_study_radar_noise", MODULE_PATH)
rn = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rn
_spec.loader.exec_module(rn)

RADAR_TYPES = {
    "Loadnuradarpoints",
    "LoadradarpointsFromMultiSweeps",
    "RadarDropoutLoadnuradarpoints",
    "RadarDropoutLoadradarpointsFromMultiSweeps",
    "RadarNoiseLoadnuradarpoints",
    "RadarNoiseLoadradarpointsFromMultiSweeps",
}

# Column layout of the 7-dim tensor the loader emits (loading.py:812).
COL_X, COL_Y, COL_Z, COL_RCS, COL_VX, COL_VY, COL_DT = range(7)
NOISE_COLS = {"rcs": COL_RCS, "vx_comp": COL_VX, "vy_comp": COL_VY}
UNTOUCHED_COLS = [COL_X, COL_Y, COL_Z, COL_DT]


def radar_entries(pipeline):
    return [dict(e) for e in pipeline if e["type"] in RADAR_TYPES]


def run(entries, info):
    results = dict(info)
    for entry in entries:
        transform = PIPELINES.build(dict(entry))
        results = transform(results)
    return [p.tensor.clone() for p in results["radar_points"]]


def digest_one(t):
    arr = np.ascontiguousarray(t.numpy())
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(digest_one(t).encode())
    return h.hexdigest()


def dup_groups(elem_digests):
    """Indices of stack elements that carry bit-identical CLEAN point sets."""
    by = collections.defaultdict(list)
    for i, d in enumerate(elem_digests):
        by[d].append(i)
    return [idx for idx in by.values() if len(idx) > 1]


def row_counter(t):
    """Multiset of position+time rows, used for the (b) subset test."""
    key = np.ascontiguousarray(t[:, UNTOUCHED_COLS].numpy())
    return collections.Counter(map(tuple, key.tolist()))


def is_sub_multiset(small, big):
    return all(big[k] >= v for k, v in small.items())


def stats(tensors, col):
    vals = np.concatenate([t[:, col].numpy() for t in tensors]) if tensors else np.zeros(0)
    if vals.size == 0:
        return dict(n=0, mean=None, std=None)
    return dict(n=int(vals.size), mean=float(vals.mean()), std=float(vals.std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample-indices", type=int, nargs="+", default=[0, 3000, 6000])
    args = ap.parse_args()

    out_dir = args.out_dir
    if os.path.exists(out_dir):
        raise SystemExit("refusing to reuse existing out-dir: " + out_dir)
    os.makedirs(out_dir)

    report = dict(
        utc=datetime.datetime.utcnow().isoformat() + "Z",
        repo=REPO,
        module_sha256=hashlib.sha256(open(MODULE_PATH, "rb").read()).hexdigest(),
        base_config=BASE_CFG,
        sample_indices=args.sample_indices,
    )

    cfg = Config.fromfile(BASE_CFG)
    dataset = build_dataset(cfg.data.val)
    clean_entries = radar_entries(cfg.data.val.pipeline)
    print("[probe] clean radar entries:", [e["type"] for e in clean_entries], flush=True)
    assert len(clean_entries) == 2, clean_entries

    infos, tokens = [], []
    for idx in args.sample_indices:
        info = dataset.get_data_info(idx)
        infos.append(info)
        tokens.append(info["sample_idx"])
    report["sample_tokens"] = tokens
    print("[probe] tokens:", tokens, flush=True)

    clean = [run(clean_entries, info) for info in infos]
    report["clean"] = [
        dict(token=tok,
             counts=[int(t.shape[0]) for t in tens],
             total=int(sum(int(t.shape[0]) for t in tens)),
             digest=digest(tens),
             elem_digests=[digest_one(t) for t in tens],
             dup_groups=dup_groups([digest_one(t) for t in tens]),
             fields={name: stats(tens, col) for name, col in NOISE_COLS.items()})
        for tok, tens in zip(tokens, clean)
    ]
    clean_dups = {tok: dup_groups([digest_one(t) for t in tens])
                  for tok, tens in zip(tokens, clean)}
    print("[probe] clean duplicate element groups:", clean_dups, flush=True)

    # ---------- family (b): dropout ----------
    drop_out = {}
    keep_rows = {}
    for p in rn.DROPOUT_P_LEVELS:
        for seed in rn.SEEDS:
            corruption = dict(family="radar_dropout", drop_p=p, corrupt_seed=seed)
            entries = rn.corrupt_pipeline(clean_entries, corruption)
            for tok, info, ctens in zip(tokens, infos, clean):
                tens = run(entries, info)
                key = "p{:.0f}_s{}_{}".format(p * 100, seed, tok)
                counts = [int(t.shape[0]) for t in tens]
                base = [int(t.shape[0]) for t in ctens]
                drop_out[key] = dict(
                    p=p, seed=seed, token=tok,
                    counts=counts, clean_counts=base,
                    total=int(sum(counts)), clean_total=int(sum(base)),
                    survival=float(sum(counts)) / float(sum(base)),
                    digest=digest(tens),
                    elem_digests=[digest_one(t) for t in tens],
                    sub_multiset=[bool(is_sub_multiset(row_counter(t), row_counter(c)))
                                  for t, c in zip(tens, ctens)],
                )
                keep_rows[(p, seed, tok)] = [row_counter(t) for t in tens]
            print("[probe] (b) p={} seed={} done".format(p, seed), flush=True)
    report["dropout"] = drop_out

    crn_b = []
    for seed in rn.SEEDS:
        for tok in tokens:
            for lo, hi in ((0.25, 0.50), (0.50, 0.75)):
                nested = all(is_sub_multiset(s, b) for s, b in
                             zip(keep_rows[(hi, seed, tok)], keep_rows[(lo, seed, tok)]))
                crn_b.append(dict(seed=seed, token=tok, superset_p=lo, subset_p=hi,
                                  nested=bool(nested)))
    report["crn_dropout_nesting"] = crn_b

    # ---------- family (c): Doppler/RCS noise ----------
    noise_out = {}
    deltas = {}
    for sigma in rn.NOISE_SIGMA_LEVELS:
        for seed in rn.SEEDS:
            corruption = dict(family="radar_doppler_rcs_noise", sigma=sigma, corrupt_seed=seed)
            entries = rn.corrupt_pipeline(clean_entries, corruption)
            for tok, info, ctens in zip(tokens, infos, clean):
                tens = run(entries, info)
                key = "sig{:g}_s{}_{}".format(sigma, seed, tok)
                shapes_match = all(t.shape == c.shape for t, c in zip(tens, ctens))
                pos_identical = shapes_match and all(
                    bool(torch.equal(t[:, UNTOUCHED_COLS], c[:, UNTOUCHED_COLS]))
                    for t, c in zip(tens, ctens))
                noise_out[key] = dict(
                    sigma=sigma, seed=seed, token=tok,
                    counts=[int(t.shape[0]) for t in tens],
                    shapes_match=bool(shapes_match),
                    positions_and_dt_bit_identical=bool(pos_identical),
                    digest=digest(tens),
                    elem_digests=[digest_one(t) for t in tens],
                    fields={name: stats(tens, col) for name, col in NOISE_COLS.items()},
                )
                deltas[(sigma, seed, tok)] = [
                    (t[:, [COL_RCS, COL_VX, COL_VY]] - c[:, [COL_RCS, COL_VX, COL_VY]]).numpy()
                    for t, c in zip(tens, ctens)]
            print("[probe] (c) sigma={} seed={} done".format(sigma, seed), flush=True)
    report["noise"] = noise_out

    crn_c = []
    for seed in rn.SEEDS:
        for tok in tokens:
            d1 = np.concatenate(deltas[(1.0, seed, tok)])
            d5 = np.concatenate(deltas[(5.0, seed, tok)])
            resid = np.abs(d5 - 5.0 * d1)
            scale = np.maximum(np.abs(d5), 1e-6)
            crn_c.append(dict(
                seed=seed, token=tok, n=int(d1.size),
                max_abs_resid=float(resid.max()),
                max_rel_resid=float((resid / scale).max()),
                median_ratio=float(np.median(d5[np.abs(d1) > 1e-3] / d1[np.abs(d1) > 1e-3])),
                proportional=bool(np.allclose(d5, 5.0 * d1, rtol=1e-4, atol=1e-4)),
            ))
    report["crn_noise_proportionality"] = crn_c

    # ---------- duplicated stack elements must share one corruption ----------
    dup_share = []
    for label, cells in (("dropout", drop_out), ("noise", noise_out)):
        for key, cell in cells.items():
            groups = clean_dups[cell["token"]]
            if not groups:
                continue
            shared = all(len({cell["elem_digests"][i] for i in grp}) == 1 for grp in groups)
            dup_share.append(dict(family=label, cell=key, token=cell["token"],
                                  groups=groups, all_shared=bool(shared)))
    report["duplicate_element_sharing"] = dup_share
    report["clean_dup_groups"] = clean_dups

    # ---------- determinism: rebuild everything from scratch ----------
    det = []
    for corruption, key_fmt in (
        (dict(family="radar_dropout", drop_p=0.50, corrupt_seed=1), "p50_s1_{}"),
        (dict(family="radar_doppler_rcs_noise", sigma=3.0, corrupt_seed=1), "sig3_s1_{}"),
    ):
        entries = rn.corrupt_pipeline(clean_entries, corruption)
        for tok, info in zip(tokens, infos):
            again = digest(run(entries, info))
            first = (drop_out if "drop_p" in corruption else noise_out)[key_fmt.format(tok)]["digest"]
            det.append(dict(corruption=corruption, token=tok, first=first, again=again,
                            bit_identical=bool(first == again)))
    report["determinism"] = det

    with open(os.path.join(out_dir, "probe_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    # ---------- histograms ----------
    np.savez_compressed(
        os.path.join(out_dir, "rcs_hist.npz"),
        clean=np.concatenate([t[:, COL_RCS].numpy() for t in clean[0]]),
        sig1=np.concatenate([d[:, 0] for d in deltas[(1.0, 0, tokens[0])]]),
        sig5=np.concatenate([d[:, 0] for d in deltas[(5.0, 0, tokens[0])]]),
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(np.concatenate([t[:, COL_RCS].numpy() for t in clean[0]]), bins=60)
        axes[0].set_title("clean rcs (dBsm), sample 0")
        for sigma in rn.NOISE_SIGMA_LEVELS:
            axes[1].hist(np.concatenate([d[:, 0] for d in deltas[(sigma, 0, tokens[0])]]),
                         bins=60, histtype="step", label="sigma={:g}".format(sigma))
        axes[1].set_title("applied rcs delta")
        axes[1].legend()
        surv = {p: [drop_out["p{:.0f}_s{}_{}".format(p * 100, s, t)]["survival"]
                    for s in rn.SEEDS for t in tokens] for p in rn.DROPOUT_P_LEVELS}
        axes[2].boxplot([surv[p] for p in rn.DROPOUT_P_LEVELS],
                        labels=["p={:.0f}%".format(p * 100) for p in rn.DROPOUT_P_LEVELS])
        axes[2].set_title("surviving point fraction")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "probe_hists.png"), dpi=110)
        report["histograms"] = "probe_hists.png"
    except Exception as exc:  # pragma: no cover
        print("[probe] matplotlib unavailable:", exc, flush=True)

    print("[probe] wrote", out_dir, flush=True)


if __name__ == "__main__":
    main()
