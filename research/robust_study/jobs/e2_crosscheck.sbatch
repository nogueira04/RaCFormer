#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --time=02:00:00
#SBATCH --job-name=rs_e2_crosscheck
#SBATCH --output=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.out
#SBATCH --error=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.err
#
# Independent devkit cross-check of one completed evaluation cell. CPU only — no model, no GPU.
#
#   RUN_DIR=/srv/nfs/shared/gnmp/robust_study_runs/eval_oracle/clean_<stamp> \
#     sbatch research/robust_study/jobs/e2_crosscheck.sbatch
#
# Set FREEZE_THRESHOLDS=1 on the clean reference run to emit the per-class max-F1 thresholds. The
# file lands in the run directory (never inside the asserted checkout); committing it to
# research/robust_study/frozen/ is a separate, deliberate step that must happen before any cell
# consumes it.

set -euo pipefail
source "$HOME/.bashrc" 2>/dev/null || true
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_job_common.sh

: "${RUN_DIR:?set RUN_DIR to the completed run directory to cross-check}"

rs_init "eval_oracle" "e2_crosscheck"

[ -f "$RUN_DIR/_COMPLETE" ] || rs_die "$RUN_DIR has no _COMPLETE marker; it is not a readable result"

SUBMISSION="$RUN_DIR/submission_overall/pts_bbox/results_nusc.json"
P2_METRICS="$RUN_DIR/submission_overall/pts_bbox/metrics_summary.json"
[ -f "$SUBMISSION" ] || rs_die "submission JSON missing: $SUBMISSION"
[ -f "$P2_METRICS" ] || rs_die "wrapper metrics_summary.json missing: $P2_METRICS"

EXPECT_SUB_SHA="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('submission_json_sha256') or '')" "$RUN_DIR/provenance.json")"
EXPECT_TOK_SHA="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('token_set_sha256') or '')" "$RUN_DIR/provenance.json")"
[ -n "$EXPECT_SUB_SHA" ] || rs_die "no submission_json_sha256 in $RUN_DIR/provenance.json"

RS_SUBMISSION="$SUBMISSION"
RS_DRIVER="$TOOLS/devkit_crosscheck.py"

FREEZE_ARGS=()
if [ "${FREEZE_THRESHOLDS:-0}" = "1" ]; then
  FREEZE_ARGS=(--freeze-thresholds "$OUT_DIR/maxf1_thresholds.json")
elif [ -n "${THRESHOLDS_JSON:-}" ]; then
  FREEZE_ARGS=(--thresholds "$THRESHOLDS_JSON")
fi

# --allow-unverifiable-rows is passed HERE and nowhere else. Exactly one row of the tolerance table
# cannot close against a frozen driver — the wrapper records no pre-format prediction count, so
# "wrapper prediction count vs raw-JSON box total" has no left-hand side. Left fail-closed, that one
# row would make the gate unpassable by construction. The row is still computed, still printed, and
# still written to crosscheck.json as UNVERIFIABLE with its reasoning; whether that is acceptable is
# the orchestrator's call, not this job's. Every other row remains fail-closed.
RS_EXACT_COMMAND="$TOOLS/run_devkit_crosscheck.sh --submission $SUBMISSION --dataroot $REPO/data/nuscenes --split val --out-dir $OUT_DIR --p2-metrics $P2_METRICS --expect-submission-sha256 $EXPECT_SUB_SHA --allow-unverifiable-rows ${FREEZE_ARGS[*]+${FREEZE_ARGS[*]}}"
echo "[job] exact_command: $RS_EXACT_COMMAND"

"$TOOLS/run_devkit_crosscheck.sh" \
  --submission "$SUBMISSION" \
  --dataroot "$REPO/data/nuscenes" \
  --split val \
  --out-dir "$OUT_DIR" \
  --p2-metrics "$P2_METRICS" \
  --expect-submission-sha256 "$EXPECT_SUB_SHA" \
  --allow-unverifiable-rows \
  ${EXPECT_TOK_SHA:+--expect-token-set-sha256 "$EXPECT_TOK_SHA"} \
  ${FREEZE_ARGS[@]+"${FREEZE_ARGS[@]}"}

echo "=== rows that did NOT pass (orchestrator must read these) ==="
python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
bad = [row for row in r['comparison'] if row['status'] != 'PASS']
print('verdict:', r['verdict'], '| failed:', r['n_failed_rows'], '| unverifiable:', r['n_unverifiable_rows'])
for row in bad:
    print(' ', row['status'], row['row'])
" "$OUT_DIR/crosscheck.json"

cp "$RUN_DIR/token_set.json" "$OUT_DIR/token_set.json" 2>/dev/null || true
echo "cross-checked $RUN_DIR" > "$OUT_DIR/source_run.txt"

rs_finalize "devkit cross-check completed with a passing verdict"
echo "[job] crosscheck.json: $OUT_DIR/crosscheck.json"
