# (a) camera removal, cell A1 — CAM_FRONT only (frontal-camera drop).
#
# Cell fragment for the robustness study. NEW file; nothing tracked is touched.
#
# This fragment changes NO pipeline entry: the (a) family's fault is installed at model level,
# on RaCFormer.extract_feat, exactly as the GATE-B G4 cell installs it. The `cam_removal` block
# below is the cell's camera scope, read by the runner.
#
# IMPORTANT: running the frozen driver directly on this config produces a CLEAN eval, not an
# (a) cell. Only the runner installs the fault:
#   conda run -n racformerfix python -u research/robust_study/tools/a_removal_subset.py \
#     --repo /srv/nfs/shared/gnmp/RaCFormer \
#     --config research/robust_study/configs/a_removal_front.py \
#     --weights checkpoints/racformer_r50_f8.pth \
#     --expect-samples 6019 --out-dir <run dir outside the checkout>
# The runner refuses a config with no `cam_removal` block, so the reverse mistake is caught;
# this direction is caught by the run's missing intervention_attestation.json.

from research.robust_study.corruptions.cell_config import BASE_CONFIG

_base_ = [BASE_CONFIG]

# Ladder position 1 of 3: frontal-camera drop < worst-sector drop (a_removal_back.py)
# < all-6 drop (the EXISTING GATE-B G4 + g4_repeat pair — never re-run).
# Deterministic, no seeds (fault-families.md "(a) Modality removal", Severity).
cam_removal = dict(
    cell="A1",
    cameras=["CAM_FRONT"],
    rationale="frontal camera, the preceding ladder level to the worst-sector drop",
)
