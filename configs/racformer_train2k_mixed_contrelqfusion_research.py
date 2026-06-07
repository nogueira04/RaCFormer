"""Branch A: mixed-condition train2k with continuous query reliability fusion.

This tests whether local per-query reliability cues can keep the mixed-condition
night benefit without repeating the rejected global day/night/rain condition gate.
The gate is opt-in and zero-initialized to multiplicative identity.
"""

_base_ = ["./racformer_train2k_mixed_research.py"]

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            continuous_reliability_query_fusion=True,
            reliability_hidden_dims=128,
            reliability_use_pairwise_cosine=True,
            reliability_use_query_geometry=True,
        ),
    ),
)
