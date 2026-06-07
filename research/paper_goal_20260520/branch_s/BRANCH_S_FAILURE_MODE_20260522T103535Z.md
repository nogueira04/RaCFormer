# Branch S Failure Mode

UTC: `20260522T103535Z`

Reason: `formal_gate_miss`

## Step 4 Gate

| Gate | Value | Threshold | Pass | Gap |
|---|---:|---:|---|---:|
| night mAP >= Baseline B night + 2.0 pp | 0.311753 | 0.342835 | FAIL | -0.031082 |
| day mAP >= Baseline B day - 0.5 pp | 0.520326 | 0.533689 | FAIL | -0.013363 |
| overall mAP >= Baseline B overall - 1.0 pp | 0.525075 | 0.519630 | PASS | 0.005445 |
| router accuracy >= 95% | 0.999802 | 0.950000 | PASS | 0.049802 |

## Integrated Summary

```json
{
  "overall_mAP": 0.5250754448257907,
  "overall_NDS": 0.5951216672277075,
  "overall_ECE": 0.08497930148800104,
  "split_counts": {
    "day": 4449,
    "night": 602,
    "rain": 968
  },
  "day_mAP": 0.5203263563556297,
  "day_NDS": 0.592896186282285,
  "day_ECE": 0.0850332992207291,
  "night_mAP": 0.31175257645986826,
  "night_NDS": 0.3808296083387042,
  "night_ECE": 0.07163087007386347,
  "rain_mAP": 0.5630137967761891,
  "rain_NDS": 0.6407035616920821,
  "rain_ECE": 0.08820659492470297,
  "route_counts": {
    "day": 5416,
    "night": 603
  },
  "runtime_seconds": 500.3716034889221,
  "step_gpu_hours": 0.0
}
```
