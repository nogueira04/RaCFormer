# Body of one GATE-B camera-removal cell. Sourced by the eight g*.sbatch templates, which differ
# ONLY in $CELL, $REMOVAL and the SLURM job name.
#
# Sharing the body is the point: GATE-B asks whether three implementations of "the camera is off"
# disagree, and that answer is only about the removal mechanism if nothing else can differ between
# the cells. Everything here is identical across all eight runs.
#
# Expects $CELL and $REMOVAL to be set by the caller.

: "${CELL:?set CELL before sourcing this body}"
: "${REMOVAL:?set REMOVAL to one of none|phase0|input|table9}"

case "$REMOVAL" in
  none|phase0|input|table9) ;;
  *) echo "[job] FATAL: unknown REMOVAL=$REMOVAL" >&2; exit 1 ;;
esac

rs_init "gate_b" "$CELL"
rs_write_thresholds_source

RS_CONFIG="$REPO/configs/racformer_eval_fullval_research.py"
RS_WEIGHTS="$REPO/checkpoints/racformer_r50_f8.pth"
RS_DRIVER="$TOOLS/gate_b_removal.py"
RS_SUBMISSION="$OUT_DIR/submission_overall/pts_bbox/results_nusc.json"

EXPECT_SAMPLES="${EXPECT_SAMPLES:-6019}"

[ -f "$RS_CONFIG" ] || rs_die "config missing: $RS_CONFIG"
[ -f "$RS_WEIGHTS" ] || rs_die "weights missing: $RS_WEIGHTS"
[ -f "$RS_DRIVER" ] || rs_die "runner missing: $RS_DRIVER"
[ -f "$REPO/research/night_gen_phase1/eval_by_condition.py" ] || rs_die "frozen eval driver missing"

echo "removal=$REMOVAL" > "$OUT_DIR/removal_mode.txt"

RS_EXACT_COMMAND="conda run -n racformerfix $CONDA_RUN_FLAGS python -u research/robust_study/tools/gate_b_removal.py --removal $REMOVAL --repo $REPO --config configs/racformer_eval_fullval_research.py --weights checkpoints/racformer_r50_f8.pth --out-dir $OUT_DIR --expect-samples $EXPECT_SAMPLES"
echo "[job] exact_command: $RS_EXACT_COMMAND"

cd "$REPO"
conda run -n "$ENV_NAME" $CONDA_RUN_FLAGS python -u research/robust_study/tools/gate_b_removal.py \
  --removal "$REMOVAL" \
  --repo "$REPO" \
  --config configs/racformer_eval_fullval_research.py \
  --weights checkpoints/racformer_r50_f8.pth \
  --out-dir "$OUT_DIR" \
  --expect-samples "$EXPECT_SAMPLES"

echo "=== post-run assertions ==="
[ -f "$RS_SUBMISSION" ] || rs_die "submission JSON not written: $RS_SUBMISSION"
[ -f "$OUT_DIR/eval_by_condition.json" ] || rs_die "eval_by_condition.json not written"

# The runner already exits non-zero on a failed attestation, which would have aborted this script
# and left the run without _COMPLETE. Re-reading the verdict here means the job never depends on
# that exit code alone -- a PASS has to be present in the artifact the orchestrator will read.
ATTEST="$OUT_DIR/intervention_attestation.json"
[ -f "$ATTEST" ] || rs_die "intervention attestation not written: $ATTEST"
ATTEST_VERDICT="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['verdict'])" "$ATTEST")"
if [ "$ATTEST_VERDICT" != "PASS" ]; then
  RS_VALIDITY="INVALID: intervention attestation verdict=$ATTEST_VERDICT for removal=$REMOVAL"
  rs_die "$RS_VALIDITY"
fi
echo "[job] intervention attestation: PASS"
python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
o, e = r['observed'], r['expected']
print('[job]   branch_hits=%s/%s  covered_view_frames=%s/%s  altered_view_frames=%s' % (
    o['branch_hits'], e['n_samples'], o['covered_view_frames'], e['view_frames'],
    o['altered_view_frames']))
" "$ATTEST"

conda run -n "$ENV_NAME" $CONDA_RUN_FLAGS python -u "$TOOLS/assert_val_token_set.py" \
  --submission "$RS_SUBMISSION" \
  --dataroot "$REPO/data/nuscenes" \
  --version v1.0-trainval \
  --out "$OUT_DIR/token_set.json"

rs_finalize "gate_b cell removal=$REMOVAL: full-val eval + intervention attestation + token-set assertion passed"
echo "[job] submission: $RS_SUBMISSION"
echo "[job] next: RUN_DIR=$OUT_DIR CELL=$CELL sbatch research/robust_study/jobs/g_crosscheck.sbatch"
