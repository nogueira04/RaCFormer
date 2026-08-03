#!/bin/bash
#SBATCH --partition=livecluster
#SBATCH --nodelist=livenode03
#SBATCH --chdir=/srv/nfs/shared/gnmp/RaCFormer
#SBATCH --time=02:00:00
#SBATCH --job-name=rs_g_crosscheck
#SBATCH --output=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.out
#SBATCH --error=/srv/nfs/shared/gnmp/robust_study_runs/slurm_logs/%x_%j.err
#
# Per-cell metric emission for one completed GATE-B cell. CPU only -- no model, no GPU.
#
#   RUN_DIR=/srv/nfs/shared/gnmp/robust_study_runs/gate_b/g2_phase0_<stamp> CELL=g2_phase0 \
#     sbatch research/robust_study/jobs/g_crosscheck.sbatch
#
# One parameterised script for all eight cells, running the same committed devkit_crosscheck.py the
# Evaluator Oracle ran, against the same frozen per-class max-F1 thresholds. That is what makes the
# cells comparable: identical extraction code, identical operating points, no per-cell variation.
# It emits, per class at centre-distance 2.0 m (the devkit's own dist_th_tp), TP/FP/FN/GT counts at
# both pre-registered operating points -- (i) no score cut and (ii) the frozen max-F1 threshold --
# alongside mAP, NDS, per-class AP and the five TP errors.
#
# Unlike the clean cells, this is NOT a validity gate on the removal cells' metrics: a camera-off
# cell is expected to move far from the clean run. The tolerance rows still run and are recorded;
# the orchestrator reads them.

set -euo pipefail
source "$HOME/.bashrc" 2>/dev/null || true
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_job_common.sh
source /srv/nfs/shared/gnmp/RaCFormer/research/robust_study/jobs/_gate_b_common.sh

: "${RUN_DIR:?set RUN_DIR to the completed gate_b cell directory to score}"
: "${CELL:?set CELL to the cell name, e.g. g2_phase0}"

rs_init "gate_b" "${CELL}_crosscheck"
rs_write_thresholds_source

[ -f "$RUN_DIR/_COMPLETE" ] || rs_die "$RUN_DIR has no _COMPLETE marker; it is not a readable result"
[ -f "$RUN_DIR/intervention_attestation.json" ] || rs_die "$RUN_DIR has no intervention attestation"

ATTEST_VERDICT="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['verdict'])" "$RUN_DIR/intervention_attestation.json")"
[ "$ATTEST_VERDICT" = "PASS" ] || rs_die "$RUN_DIR attestation verdict=$ATTEST_VERDICT; refusing to score an unattested cell"

SUBMISSION="$RUN_DIR/submission_overall/pts_bbox/results_nusc.json"
P2_METRICS="$RUN_DIR/submission_overall/pts_bbox/metrics_summary.json"
[ -f "$SUBMISSION" ] || rs_die "submission JSON missing: $SUBMISSION"
[ -f "$P2_METRICS" ] || rs_die "wrapper metrics_summary.json missing: $P2_METRICS"

EXPECT_SUB_SHA="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('submission_json_sha256') or '')" "$RUN_DIR/provenance.json")"
EXPECT_TOK_SHA="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('token_set_sha256') or '')" "$RUN_DIR/provenance.json")"
[ -n "$EXPECT_SUB_SHA" ] || rs_die "no submission_json_sha256 in $RUN_DIR/provenance.json"

RS_SUBMISSION="$SUBMISSION"
RS_DRIVER="$TOOLS/devkit_crosscheck.py"

# --allow-unverifiable-rows carries the same meaning as in e2_crosscheck.sbatch: exactly one row of
# the tolerance table (wrapper pre-format prediction count) has no left-hand side against a frozen
# driver. It is still computed, printed and recorded as UNVERIFIABLE. Every other row stays
# fail-closed.
RS_EXACT_COMMAND="$TOOLS/run_devkit_crosscheck.sh --submission $SUBMISSION --dataroot $REPO/data/nuscenes --split val --out-dir $OUT_DIR --p2-metrics $P2_METRICS --expect-submission-sha256 $EXPECT_SUB_SHA --allow-unverifiable-rows --thresholds $FROZEN_THRESHOLDS"
echo "[job] exact_command: $RS_EXACT_COMMAND"

"$TOOLS/run_devkit_crosscheck.sh" \
  --submission "$SUBMISSION" \
  --dataroot "$REPO/data/nuscenes" \
  --split val \
  --out-dir "$OUT_DIR" \
  --p2-metrics "$P2_METRICS" \
  --expect-submission-sha256 "$EXPECT_SUB_SHA" \
  --allow-unverifiable-rows \
  --thresholds "$FROZEN_THRESHOLDS" \
  ${EXPECT_TOK_SHA:+--expect-token-set-sha256 "$EXPECT_TOK_SHA"}

echo "=== cell summary (orchestrator reads crosscheck.json for the full vector) ==="
python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
m = r['metrics']
print('mAP_pp  %.6f' % m['mAP_pp'])
print('NDS_pp  %.6f' % m['NDS_pp'])
for name, ap in sorted(m['per_class_AP_pp'].items()):
    print('  AP_pp %-22s %.6f' % (name, ap))
print('tp_errors:', json.dumps(m['tp_errors']))
c = r['counts_at_dist_th_tp']
print('thresholds_source:', c['thresholds_source'])
for op in ('operating_point_i_no_score_cut', 'operating_point_ii_max_f1'):
    print(op)
    for name in sorted(c[op]):
        v = c[op][name]
        print('   %-22s TP=%-7s FP=%-8s FN=%-7s GT=%-7s' % (name, v['TP'], v['FP'], v['FN'], v['GT']))
print('near_zero_recall_at_op_i:', json.dumps(c['near_zero_recall_at_op_i']))
bad = [row for row in r['comparison'] if row['status'] != 'PASS']
print('verdict:', r['verdict'], '| failed:', r['n_failed_rows'], '| unverifiable:', r['n_unverifiable_rows'])
for row in bad:
    print(' ', row['status'], row['row'])
" "$OUT_DIR/crosscheck.json"

cp "$RUN_DIR/token_set.json" "$OUT_DIR/token_set.json" 2>/dev/null || true
cp "$RUN_DIR/intervention_attestation.json" "$OUT_DIR/intervention_attestation.json" 2>/dev/null || true
echo "scored $RUN_DIR" > "$OUT_DIR/source_run.txt"

rs_finalize "gate_b cell scored against the frozen max-F1 thresholds"
echo "[job] crosscheck.json: $OUT_DIR/crosscheck.json"
