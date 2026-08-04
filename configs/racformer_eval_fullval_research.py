"""Phase 1 final-eval config — full 6 019-sample val.

Used by `research/night_gen_phase1/eval_by_condition.py` (T5) AFTER training to compute
reported metrics. NEVER use any train2k_*_research config for reported metrics — those
have `data.val.max_samples=300` for fast in-training feedback only.

This config inherits the base, restores the full val pkl, removes any val cap, and is
otherwise unchanged. The `data.train` block is left as-is from the base; we don't train
with this config.
"""

_base_ = ["./racformer_r50_nuimg_704x256_f8.py"]

data = dict(
    val=dict(
        ann_file="/srv/nfs/shared/gnmp/RaCFormer/nuscenes_infos_val_sweep.pkl",
        max_samples=None,
    ),
)
