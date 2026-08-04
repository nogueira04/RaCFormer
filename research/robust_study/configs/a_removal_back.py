# (a) camera removal, cell A2 — CAM_BACK only (worst-sector single-camera drop).
#
# Cell fragment for the robustness study. NEW file; nothing tracked is touched.
#
# CAM_BACK is the worst-sector camera under the rule registered BEFORE any model run
# (fault-families.md "Worst-sector camera"): among the 5 non-frontal cameras, the one with the
# most GT box centers inside its view frustum over val, from annotations + calibrations alone.
# Computed 2026-08-03 on livenode03: CAM_BACK 59,570 vs CAM_FRONT_LEFT 27,613 (next), no
# tie-break needed. ** PENDING Aug-7 SIGN-OFF ** — if sign-off selects a different channel,
# change `cameras` here and nothing else.
#
# This fragment changes NO pipeline entry: the (a) family's fault is installed at model level,
# on RaCFormer.extract_feat, exactly as the GATE-B G4 cell installs it.
#
# IMPORTANT: running the frozen driver directly on this config produces a CLEAN eval, not an
# (a) cell. Only the runner installs the fault:
#   conda run -n racformerfix python -u research/robust_study/tools/a_removal_subset.py \
#     --repo /srv/nfs/shared/gnmp/RaCFormer \
#     --config research/robust_study/configs/a_removal_back.py \
#     --weights checkpoints/racformer_r50_f8.pth \
#     --expect-samples 6019 --out-dir <run dir outside the checkout>

from research.robust_study.corruptions.cell_config import BASE_CONFIG

_base_ = [BASE_CONFIG]

# Ladder position 2 of 3: frontal drop (a_removal_front.py) < worst-sector drop
# < all-6 drop (the EXISTING GATE-B G4 + g4_repeat pair — never re-run).
# Deterministic, no seeds (fault-families.md "(a) Modality removal", Severity).
cam_removal = dict(
    cell="A2",
    cameras=["CAM_BACK"],
    rationale="worst-sector camera by registered box-center-in-frustum rule; "
              "59,570 centers over val; pending Aug-7 sign-off",
)
