"""Seed replication config for the S6 radar-query topk90 ablation.

Matches ``racformer_train2k_day_radarquery_topk90_research.py`` and only changes
the training seed. Use only if the seed-0 topk90 ablation passes or lands near
the gate.
"""

_base_ = ["./racformer_train2k_day_radarquery_topk90_research.py"]

random_seed = 20260502
