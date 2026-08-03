# frozen/ — values pinned by committing them

Anything in this directory is a number that must not move once downstream cells start consuming
it. Freezing means exactly one thing here: the file is written by a run, reviewed, and then
committed. Nothing regenerates it in place.

## `maxf1_thresholds_clean.json` (not yet present)

Per-class score thresholds maximising F1 over the `dist_th_tp = 2.0 m` match set. They define the
second of the two pre-registered operating points at which per-class TP/FP/FN are reported; the
first is the full post-cap prediction set with no score cut.

How they come to exist:

1. the clean reference cell runs and completes (`_COMPLETE` present);
2. `jobs/e2_crosscheck.sbatch` runs against it with `FREEZE_THRESHOLDS=1`, which computes the
   thresholds from **that** run and writes `maxf1_thresholds.json` into the cross-check run
   directory — outside the checkout, because a job that wrote into the repo would fail its own
   end-of-run clean-tree assertion;
3. that file is copied here and committed as its own commit;
4. every later cell passes `THRESHOLDS_JSON=<this file>`, which makes `devkit_crosscheck.py` read
   the thresholds instead of recomputing them. `crosscheck.json` records which of the two happened
   in `counts_at_dist_th_tp.thresholds_source`, so a run that silently recomputed is detectable.

Rules baked into the computation (`tools/devkit_crosscheck.py::max_f1_threshold`):

- candidate thresholds are the class's unique prediction scores;
- a prediction is kept when `score >= threshold`;
- equal-score predictions are matched in the devkit's own sort order;
- ties among F1 maxima are broken toward the **lowest** score, which keeps more predictions;
- an empty candidate set gives `+inf`, serialised as `null`, at which every count is 0.
