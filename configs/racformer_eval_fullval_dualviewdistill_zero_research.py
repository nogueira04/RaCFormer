"""Full-val eval config for Branch G DualViewDistill checkpoints.

The aux module is present only so strict checkpoint loading accepts the trained
adapter weights. loss_weight=0 prevents DINOv2 teacher loading, and forward_test
does not call the aux loss path.
"""

_base_ = ["./racformer_eval_fullval_research.py"]

model = dict(
    dualview_distill=dict(
        teacher_dir="research/night_gen_phase1/teachers/dinov2_vitl14",
        student_channels=256,
        dino_channels=1024,
        loss_weight=0.0,
        cosine_weight=1.0,
        mse_weight=1.0,
        teacher_half=True,
    )
)
