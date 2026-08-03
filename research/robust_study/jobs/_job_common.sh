# Shared discipline for every robustness-study sbatch cell. Sourced, never executed.
#
# Guarantees each cell inherits:
#   * the checkout is the pinned execution commit and is clean, asserted at START and at END
#     (an end-assertion failure does not abort the run's bookkeeping — it marks it INVALID);
#   * the run directory is new (refuse if it already exists) and lives OUTSIDE the checkout, so
#     writing results can never be what breaks the clean-tree assertion;
#   * `_FINALIZED` is written unconditionally by the EXIT trap, `_COMPLETE` only on validity=VALID
#     — a directory without `_COMPLETE` must never be read as a result;
#   * seeds are exported and recorded;
#   * `provenance.json` is written on every path, including aborts.
#
# The execution commit is injected rather than hardcoded: these scripts are themselves part of the
# commit whose SHA they assert, so the value has to come from outside the repo. Set EXEC_COMMIT in
# the environment or put it in $RUN_ROOT/exec_commit.txt.

REPO="${REPO:-/srv/nfs/shared/gnmp/RaCFormer}"
RUN_ROOT="${RUN_ROOT:-/srv/nfs/shared/gnmp/robust_study_runs}"
EXEC_COMMIT_FILE="${EXEC_COMMIT_FILE:-$RUN_ROOT/exec_commit.txt}"
ENV_NAME="${ENV_NAME:-racformerfix}"
GIT_TAG="${GIT_TAG:-robust-study-freeze-20260802}"
TOOLS="$REPO/research/robust_study/tools"

# conda run buffers everything until exit unless told otherwise; on a multi-hour eval that means no
# live progress and total log loss if the job is killed. Set to "" to get the literal plan command.
CONDA_RUN_FLAGS="${CONDA_RUN_FLAGS:---no-capture-output}"

RS_VALIDITY=""
RS_START_SHA=""
RS_END_SHA=""
RS_DIRTY_START="unknown"
RS_DIRTY_END="unknown"
RS_EXACT_COMMAND=""
RS_CONFIG=""
RS_WEIGHTS=""
RS_DRIVER=""
RS_SUBMISSION=""

rs_die() {
  echo "[job] FATAL: $*" >&2
  exit 1
}

rs_tree_state() {
  # echoes "<sha> <dirty:true|false>"
  local sha dirty
  sha="$(git -C "$REPO" rev-parse HEAD)"
  if [ -z "$(git -C "$REPO" status --porcelain)" ]; then dirty="false"; else dirty="true"; fi
  echo "$sha $dirty"
}

rs_resolve_exec_commit() {
  if [ -z "${EXEC_COMMIT:-}" ]; then
    [ -f "$EXEC_COMMIT_FILE" ] || rs_die "EXEC_COMMIT unset and $EXEC_COMMIT_FILE missing"
    EXEC_COMMIT="$(tr -d '[:space:]' < "$EXEC_COMMIT_FILE")"
  fi
  [ -n "$EXEC_COMMIT" ] || rs_die "EXEC_COMMIT resolved empty"
  echo "[job] exec_commit=$EXEC_COMMIT"
}

rs_on_exit() {
  local rc=$?
  set +e
  {
    echo "exit_status=$rc"
    echo "finalized_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "validity=${RS_VALIDITY:-INVALID: aborted before validity was decided}"
    echo "slurm_job_id=${SLURM_JOB_ID:-none}"
  } > "$OUT_DIR/_FINALIZED"
  if [ ! -f "$OUT_DIR/provenance.json" ]; then
    rs_write_provenance "${RS_VALIDITY:-INVALID: aborted (exit=$rc) before end-state assertions}" \
      "aborted"
  fi
  echo "[job] _FINALIZED written (exit_status=$rc)"
}

# rs_init <phase> <cell>
rs_init() {
  local phase="$1" cell="$2"
  local stamp
  stamp="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
  RS_PHASE="$phase"
  RS_CELL="$cell"
  OUT_DIR="$RUN_ROOT/$phase/${cell}_${stamp}"

  if [ -e "$OUT_DIR" ]; then
    rs_die "run directory already exists, refusing to reuse it: $OUT_DIR"
  fi
  case "$OUT_DIR" in
    "$REPO"/*) rs_die "run directory is inside the asserted checkout: $OUT_DIR" ;;
  esac
  mkdir -p "$OUT_DIR"
  echo "[job] out_dir=$OUT_DIR"

  trap rs_on_exit EXIT
  RS_SLURM_LOG="$RUN_ROOT/slurm_logs/${SLURM_JOB_NAME:-local}_${SLURM_JOB_ID:-nojob}.out"
  ln -sfn "$RS_SLURM_LOG" "$OUT_DIR/slurm_log.txt" || true

  rs_resolve_exec_commit
  rs_assert_tree start

  rs_write_seeds
  echo "[job] node=$(hostname) date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# rs_assert_tree start|end ; at "end" a failure records INVALID instead of aborting silently.
rs_assert_tree() {
  local when="$1" state sha dirty
  state="$(rs_tree_state)"
  sha="${state% *}"
  dirty="${state#* }"
  echo "[job] $when: HEAD=$sha dirty=$dirty"
  if [ "$when" = "start" ]; then
    RS_START_SHA="$sha"
    RS_DIRTY_START="$dirty"
    [ "$sha" = "$EXEC_COMMIT" ] || rs_die "start SHA $sha != exec_commit $EXEC_COMMIT"
    [ "$dirty" = "false" ] || rs_die "working tree dirty at start: $(git -C "$REPO" status --porcelain | head -5)"
  else
    RS_END_SHA="$sha"
    RS_DIRTY_END="$dirty"
    if [ "$sha" != "$EXEC_COMMIT" ]; then
      RS_VALIDITY="INVALID: end SHA $sha != exec_commit $EXEC_COMMIT (tree mutated mid-job)"
      return 1
    fi
    if [ "$dirty" != "false" ]; then
      RS_VALIDITY="INVALID: working tree dirty at end: $(git -C "$REPO" status --porcelain | head -3 | tr '\n' ';')"
      return 1
    fi
  fi
  return 0
}

rs_write_seeds() {
  export PYTHONHASHSEED=0
  cat > "$OUT_DIR/seeds.json" <<'JSON'
{
  "PYTHONHASHSEED": "0",
  "torch_manual_seed": 0,
  "numpy_seed": 0,
  "python_random_seed": 0,
  "dataloader_seed": 0,
  "source": "PYTHONHASHSEED exported by the job; torch/numpy/random seeds are set inside the frozen driver by mmdet.apis.set_random_seed(0, deterministic=True) at research/night_gen_phase1/eval_by_condition.py:259, dataloader seed=0 at :270",
  "caveat": "the driver re-enables cudnn.benchmark at research/night_gen_phase1/eval_by_condition.py:260, which set_random_seed(deterministic=True) had just disabled; bit-identical repeats are therefore expected but not guaranteed, which is what the anchor repeats measure"
}
JSON
  echo "[job] seeds recorded in $OUT_DIR/seeds.json (PYTHONHASHSEED=$PYTHONHASHSEED)"
}

# rs_write_provenance <validity> <completion>
rs_write_provenance() {
  local validity="$1" completion="$2"
  conda run -n "$ENV_NAME" $CONDA_RUN_FLAGS python -u "$TOOLS/write_provenance.py" \
    --out "$OUT_DIR/provenance.json" \
    --repo "$REPO" \
    --phase "${RS_PHASE:-unknown}" \
    --cell "${RS_CELL:-unknown}" \
    --validity "$validity" \
    --start-sha "${RS_START_SHA:-unknown}" \
    --end-sha "${RS_END_SHA:-unknown}" \
    --dirty-start "$RS_DIRTY_START" \
    --dirty-end "$RS_DIRTY_END" \
    --git-tag "$GIT_TAG" \
    --exact-command "${RS_EXACT_COMMAND:-none}" \
    --slurm-job-id "${SLURM_JOB_ID:-none}" \
    --slurm-log-path "${RS_SLURM_LOG:-unknown}" \
    --node "$(hostname)" \
    --date-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --env-name "$ENV_NAME" \
    ${RS_CONFIG:+--config "$RS_CONFIG"} \
    ${RS_WEIGHTS:+--weights "$RS_WEIGHTS"} \
    ${RS_DRIVER:+--driver-script "$RS_DRIVER"} \
    ${RS_SUBMISSION:+--submission "$RS_SUBMISSION"} \
    --token-set-json "$OUT_DIR/token_set.json" \
    --seeds-json "$OUT_DIR/seeds.json" \
    --completion "$completion"
}

# Called on the success path only: end assertions -> validity -> provenance -> _COMPLETE.
rs_finalize() {
  local completion="${1:-eval completed}"
  if rs_assert_tree end; then
    RS_VALIDITY="VALID"
  else
    echo "[job] END-STATE ASSERTION FAILED: $RS_VALIDITY" >&2
  fi
  rs_write_provenance "$RS_VALIDITY" "$completion"
  if [ "$RS_VALIDITY" = "VALID" ]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT_DIR/_COMPLETE"
    echo "[job] _COMPLETE written; validity=VALID"
  else
    echo "[job] refusing to write _COMPLETE; validity=$RS_VALIDITY" >&2
    return 1
  fi
}
