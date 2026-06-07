"""Branch G Stage 3B DualViewDistill train2k config.

This promotes the smoke-validated DualViewDistill mechanism to the normal train2k
schedule. The model block is intentionally identical to the Stage 3A smoke config;
only smoke-only runtime caps are omitted.
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

total_epochs = 12
eval_config = dict(interval=0)
checkpoint_config = dict(interval=1, max_keep_ckpts=4)
