# Branch S2 Healthy Recovery Result

Status: terminal healthy recovery complete.

This report covers the isolated Branch S2 healthy day-specialist recovery, not the
original Branch S2 checkpoint chain. The original chain remains scientifically invalid
because the random-init `lr=4e-4` setup reproduced optimizer divergence in the day
sanity probe. This recovery kept the same day-specialist premise but used random init
with `AdamW lr=4e-5`, which was stable in the 2k sanity probe.

## Trajectory

| Epoch | Day mAP | Day NDS | Overall mAP | Overall NDS | Overall ECE | Night mAP | Rain mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.1222 | 0.2663 | 0.1117 | 0.2593 | 0.1239 | 0.0418 | 0.0754 |
| 6 | 0.1718 | 0.3186 | 0.1563 | 0.3073 | 0.1112 | 0.0337 | 0.1007 |
| 9 | 0.1935 | 0.3408 | 0.1759 | 0.3278 | 0.1060 | 0.0386 | 0.1136 |
| 12 | 0.2014 | 0.3487 | 0.1832 | 0.3347 | 0.1026 | 0.0405 | 0.1192 |

## Decision

Healthy training recovered a stable upward trajectory, so the evaluator was not the
root cause of the earlier nonsensical Branch S2 result. The recovery did not become
paper-competitive: final day mAP is `0.2014`, far below the tracked baseline-B day mAP
of `0.5387`, and the epoch 9 to 12 gain is small (`+0.0079` day mAP, `+0.0073`
overall mAP).

Terminate this healthy random-init S2 day-specialist branch as a negative diagnostic
result. Do not extend beyond epoch 12 without a new hypothesis, such as pretrained
initialization, a different optimizer/schedule, or a revised specialist setup. For
paper accounting, cite the healthy recovery as evidence that the original S2 failure
was optimizer divergence/invalid setup rather than an evaluation-counting bug.

## Artifacts

- State: `research/paper_goal_20260520/branch_s2/BRANCH_S2_HEALTHY_STATE.json`
- Epoch 3 summary: `research/paper_goal_20260520/branch_s2/DAY_SPECIALIST_HEALTHY_LR4E5_STEP2_SUMMARY.json`
- Epoch 6 summary: `research/paper_goal_20260520/branch_s2/DAY_SPECIALIST_HEALTHY_LR4E5_EPOCH6_SUMMARY.json`
- Epoch 9 summary: `research/paper_goal_20260520/branch_s2/DAY_SPECIALIST_HEALTHY_LR4E5_EPOCH9_SUMMARY.json`
- Epoch 12 summary: `research/paper_goal_20260520/branch_s2/DAY_SPECIALIST_HEALTHY_LR4E5_EPOCH12_SUMMARY.json`
- Final checkpoint: `research/paper_goal_20260520/branch_s2/checkpoints/day_specialist_s2_seed0_healthy_lr4e5_epoch12.pth`
- Final eval dir: `research/paper_goal_20260520/branch_s2/evals/day_specialist_s2_seed0_healthy_lr4e5_epoch12`
