"""Phase 1 S5 — 2K mixed-condition training subset with night oversampling (T6-A).

Per T6.0 inspection (loaders/builder.py uses DistributedGroupSampler/GroupSampler with
dataset.flag group buckets), wrapping with a per-sample weighted sampler is more invasive
than a single-file change. Defaulting to T6-A: physical-duplication oversampled pkl that
duplicates each night info 3× more (night fraction climbs from ~10% to ~30%). Standard
sampler. Zero loader changes.

In-training val uses `max_samples=300` for fast feedback. Final reported metrics must be
re-evaluated with `configs/racformer_eval_fullval_research.py`.
"""

_base_ = ["./racformer_r50_nuimg_704x256_f8.py"]

data = dict(
    train=dict(
        ann_file="/srv/nfs/shared/gnmp/RaCFormer/nuscenes_infos_train_2k_mixed_oversampled.pkl",
        # Don't cap max_samples on train — duplicated pkl already encodes the desired count.
    ),
    val=dict(max_samples=300),
)

total_epochs = 12
eval_config = dict(interval=total_epochs)
