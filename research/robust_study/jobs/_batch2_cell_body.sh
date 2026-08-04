# Body of one Batch-2 corruption cell. Sourced by batch2_cell.sbatch after _job_common.sh and
# _gate_b_common.sh, never executed.
#
# One body for all five families, for the same reason GATE-B shares _gate_b_cell_body.sh:
# a family comparison is only about the corruption if nothing else differs between cells.
# The only per-cell facts are the cell name (= the config fragment's basename) and which
# runner installs the fault:
#   a_removal_*                      -> tools/a_removal_subset.py   (config carries cam_removal)
#   radar_dropout_* / radar_noise_*
#   d2_extrinsic_* / d1_async_*      -> tools/radar_cell_runner.py  (config swaps the two radar
#                                       loader entries; runner reads family + params off them)
# Both runners share one CLI, write intervention_attestation.json in the same shape, and exit 3
# on attestation FAIL -- which aborts this script (set -e) and leaves the run _FINALIZED but
# not _COMPLETE, with validity recorded INVALID by the assertions below.
#
# Expects $CELL to be set by the caller. Do NOT launch before (i) the wiring commit is the
# recorded exec_commit in $RUN_ROOT/exec_commit.txt and (ii) the Aug-7 sign-offs that gate the
# cells (A2's camera identity) have landed; the start assertion enforces (i) mechanically.

: "${CELL:?set CELL to a Batch-2 cell config basename before sourcing this body}"

case "$CELL" in
  a_removal_*) B2_RUNNER="a_removal_subset.py" ;;
  radar_dropout_*|radar_noise_*|d2_extrinsic_*|d1_async_*) B2_RUNNER="radar_cell_runner.py" ;;
  *) echo "[job] FATAL: unknown Batch-2 cell $CELL" >&2; exit 1 ;;
esac

rs_init "batch2" "$CELL"
rs_write_thresholds_source

RS_CONFIG="$REPO/research/robust_study/configs/$CELL.py"
RS_WEIGHTS="$REPO/checkpoints/racformer_r50_f8.pth"
RS_DRIVER="$TOOLS/$B2_RUNNER"
RS_SUBMISSION="$OUT_DIR/submission_overall/pts_bbox/results_nusc.json"

EXPECT_SAMPLES="${EXPECT_SAMPLES:-6019}"

[ -f "$RS_CONFIG" ] || rs_die "config missing: $RS_CONFIG"
[ -f "$RS_WEIGHTS" ] || rs_die "weights missing: $RS_WEIGHTS"
[ -f "$RS_DRIVER" ] || rs_die "runner missing: $RS_DRIVER"
[ -f "$REPO/research/night_gen_phase1/eval_by_condition.py" ] || rs_die "frozen eval driver missing"

RS_EXACT_COMMAND="conda run -n racformerfix $CONDA_RUN_FLAGS python -u research/robust_study/tools/$B2_RUNNER --repo $REPO --config research/robust_study/configs/$CELL.py --weights checkpoints/racformer_r50_f8.pth --out-dir $OUT_DIR --expect-samples $EXPECT_SAMPLES"
echo "[job] exact_command: $RS_EXACT_COMMAND"

cd "$REPO"
conda run -n "$ENV_NAME" $CONDA_RUN_FLAGS python -u "research/robust_study/tools/$B2_RUNNER" \
  --repo "$REPO" \
  --config "research/robust_study/configs/$CELL.py" \
  --weights "checkpoints/racformer_r50_f8.pth" \
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
  RS_VALIDITY="INVALID: intervention attestation verdict=$ATTEST_VERDICT for cell=$CELL"
  rs_die "$RS_VALIDITY"
fi
echo "[job] intervention attestation: PASS"

conda run -n "$ENV_NAME" $CONDA_RUN_FLAGS python -u "$TOOLS/assert_val_token_set.py" \
  --submission "$RS_SUBMISSION" \
  --dataroot "$REPO/data/nuscenes" \
  --version v1.0-trainval \
  --out "$OUT_DIR/token_set.json"

rs_finalize "batch2 cell $CELL: full-val eval + intervention attestation + token-set assertion passed"
echo "[job] submission: $RS_SUBMISSION"
echo "[job] next: RUN_DIR=$OUT_DIR CELL=$CELL sbatch research/robust_study/jobs/g_crosscheck.sbatch"
