"""Phase 1 S0 baseline — 2K day-only training subset.

Train on `nuscenes_infos_train_2k_day.pkl` (2 000 day samples) for 12 epochs.
In-training val uses `max_samples=300` for fast feedback. Final reported metrics
must be re-evaluated with `configs/racformer_eval_fullval_research.py`.
"""

_base_ = ["./racformer_r50_nuimg_704x256_f8.py"]

data = dict(
    train=dict(
        ann_file="/srv/nfs/shared/gnmp/RaCFormer/nuscenes_infos_train_2k_day.pkl",
        max_samples=2000,
    ),
    val=dict(max_samples=300),
)

# Shorter training schedule for screening: 12 epochs
total_epochs = 12
eval_config = dict(interval=total_epochs)  # only at end
