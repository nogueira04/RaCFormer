# Body of a clean full-val evaluation cell. Sourced by e1_clean_fullval.sbatch and by the two
# anchor-repeat clones, which differ ONLY in $CELL and the SLURM job name.
#
# Sharing the body is deliberate: the three clean runs exist to measure run-to-run spread, and that
# number is only meaningful if the runs cannot drift apart in anything but their run directory.
#
# --full-val is passed EXPLICITLY rather than relying on the config baking it in.

rs_init "eval_oracle" "$CELL"

RS_CONFIG="$REPO/configs/racformer_eval_fullval_research.py"
RS_WEIGHTS="$REPO/checkpoints/racformer_r50_f8.pth"
RS_DRIVER="$REPO/research/night_gen_phase1/eval_by_condition.py"
RS_SUBMISSION="$OUT_DIR/submission_overall/pts_bbox/results_nusc.json"

[ -f "$RS_CONFIG" ] || rs_die "config missing: $RS_CONFIG"
[ -f "$RS_WEIGHTS" ] || rs_die "weights missing: $RS_WEIGHTS"
[ -f "$RS_DRIVER" ] || rs_die "driver missing: $RS_DRIVER"

RS_EXACT_COMMAND="conda run -n racformerfix $CONDA_RUN_FLAGS python -u research/night_gen_phase1/eval_by_condition.py --config configs/racformer_eval_fullval_research.py --weights checkpoints/racformer_r50_f8.pth --full-val --out-dir $OUT_DIR"
echo "[job] exact_command: $RS_EXACT_COMMAND"

cd "$REPO"
conda run -n "$ENV_NAME" $CONDA_RUN_FLAGS python -u research/night_gen_phase1/eval_by_condition.py \
  --config configs/racformer_eval_fullval_research.py \
  --weights checkpoints/racformer_r50_f8.pth \
  --full-val \
  --out-dir "$OUT_DIR"

echo "=== post-run assertions ==="
[ -f "$RS_SUBMISSION" ] || rs_die "submission JSON not written: $RS_SUBMISSION"
[ -f "$OUT_DIR/eval_by_condition.json" ] || rs_die "eval_by_condition.json not written"

conda run -n "$ENV_NAME" $CONDA_RUN_FLAGS python -u "$TOOLS/assert_val_token_set.py" \
  --submission "$RS_SUBMISSION" \
  --dataroot "$REPO/data/nuscenes" \
  --version v1.0-trainval \
  --out "$OUT_DIR/token_set.json"

rs_finalize "full-val eval + token-set assertion passed"
echo "[job] submission: $RS_SUBMISSION"
echo "[job] next: research/robust_study/jobs/e2_crosscheck.sbatch against this run directory."
