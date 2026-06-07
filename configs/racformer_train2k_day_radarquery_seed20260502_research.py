"""Seed replication config for S6 radar-guided query initialization.

Matches ``racformer_train2k_day_radarquery_research.py`` and only changes the
training seed. Use only if the seed-0 S6 screen passes or lands near the gate.
"""

_base_ = ["./racformer_train2k_day_radarquery_research.py"]

random_seed = 20260502
