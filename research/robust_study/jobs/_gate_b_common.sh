# GATE-B additions on top of _job_common.sh. Sourced after it, never executed.
#
# _job_common.sh is frozen tooling from the execution commit and is not modified by GATE-B;
# everything GATE-B needs beyond it lives here.

FROZEN_THRESHOLDS="${FROZEN_THRESHOLDS:-$REPO/research/robust_study/frozen/maxf1_thresholds_clean.json}"

# The per-class max-F1 thresholds were frozen from the clean E1 run and every GATE-B cell must be
# scored against that one file, so each run directory records where it came from -- repo-relative
# path, absolute path, content hash, and the commit the file is pinned at. `provenance.json` has no
# field for this and write_provenance.py is frozen, so it is a sidecar rather than a schema change.
rs_write_thresholds_source() {
  [ -f "$FROZEN_THRESHOLDS" ] || rs_die "frozen thresholds missing: $FROZEN_THRESHOLDS"
  local rel sha
  rel="${FROZEN_THRESHOLDS#$REPO/}"
  sha="$(sha256sum "$FROZEN_THRESHOLDS" | cut -d' ' -f1)"
  cat > "$OUT_DIR/thresholds_source.json" <<JSON
{
  "schema": "gate_b_thresholds_source/1",
  "repo_path": "$rel",
  "abs_path": "$FROZEN_THRESHOLDS",
  "sha256": "$sha",
  "commit": "$EXEC_COMMIT",
  "git_tag": "$GIT_TAG",
  "note": "per-class max-F1 score thresholds computed once from the clean E1 run and frozen; identical for every GATE-B cell"
}
JSON
  echo "[job] thresholds_source recorded: $rel ($sha)"
}
