"""Seed-1 confirmation for S0 occupancy + velocity + time BEV residual.

This repeats the passing S0_occveltimebev branch with a different training
seed and isolated output/stage names. It uses a copied seeded trainer rather
than mutating the shared train.py while other jobs are active.
"""

_base_ = ["./racformer_train2k_day_research.py"]

random_seed = 1

model = dict(
    radar_rcs_bev_residual=dict(
        output_shape=(128, 128),
        rcs_index=3,
        rcs_scale=1000000.0,
        extra_indices=(4, 5, 6),
        extra_scales=(20.0, 20.0, 1.0),
    ),
)
