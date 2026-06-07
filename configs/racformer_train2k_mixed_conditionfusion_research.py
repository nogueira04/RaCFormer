"""S5 plus explicit day/night/rain condition-aware fusion gate.

This tests a ContextualFusion-style hypothesis without new synthetic imagery:
can a trainable context gate keep S5's real-night benefit while avoiding the
large day/rain/overall collapse from mixed-condition oversampling?
"""

_base_ = ["./racformer_train2k_mixed_research.py"]

model = dict(
    pts_bbox_head=dict(
        transformer=dict(condition_fusion_gate=True),
    ),
)
