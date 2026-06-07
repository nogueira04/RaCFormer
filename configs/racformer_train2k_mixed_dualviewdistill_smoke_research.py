"""Branch G Stage 3A DualViewDistill smoke config.

This is for one-batch smoke verification only, not for a 12-epoch training run.
"""

_base_ = ["./racformer_train2k_mixed_research.py"]

model = dict(
    dualview_distill=dict(
        teacher_dir="research/night_gen_phase1/teachers/dinov2_vitl14",
        student_channels=256,
        dino_channels=1024,
        loss_weight=0.05,
        cosine_weight=1.0,
        mse_weight=1.0,
        teacher_half=True,
    )
)

batch_size = 1
data = dict(
    workers_per_gpu=0,
    val=dict(max_samples=1),
)

total_epochs = 1
eval_config = dict(interval=0)
checkpoint_config = dict(interval=0, max_keep_ckpts=1)
