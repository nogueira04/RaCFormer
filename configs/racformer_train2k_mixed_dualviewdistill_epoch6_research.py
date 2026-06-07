"""Branch G Stage 3B epoch-6 halt config.

Used for the required full-val warning gate and user-review halt. It inherits the
same DualViewDistill mechanism as the 12-epoch Stage 3B config and changes only
the stop epoch.
"""

_base_ = ["./racformer_train2k_mixed_dualviewdistill_research.py"]

total_epochs = 6
eval_config = dict(interval=0)
checkpoint_config = dict(interval=1, max_keep_ckpts=4)
