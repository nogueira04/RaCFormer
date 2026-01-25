#!/bin/bash
# Depth Oracle Experiment Runner
#
# This script runs the depth oracle experiment to determine if depth estimation
# is the bottleneck for night-time performance in RaCFormer.
#
# Usage:
#   bash tools/run_depth_oracle_experiment.sh [GPU_ID]

set -e

GPU_ID=${1:-0}
CONFIG="configs/racformer_r50_nuimg_704x256_f8.py"
CHECKPOINT="checkpoints/racformer_r50_f8.pth"
OUTPUT_DIR="results/depth_oracle"

echo "============================================================"
echo "DEPTH ORACLE EXPERIMENT"
echo "============================================================"
echo "GPU: ${GPU_ID}"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Step 1: Debug/Validation
echo ""
echo "Step 1: Running debug validation..."
echo "------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/debug_depth_oracle.py \
    ${CONFIG} \
    ${CHECKPOINT} \
    --n-samples 5 \
    --save-vis

# Check if debug passed
if [ $? -ne 0 ]; then
    echo "ERROR: Debug validation failed!"
    exit 1
fi

echo ""
echo "Debug validation passed!"

# Step 2: Full Night Evaluation
echo ""
echo "Step 2: Running full night evaluation..."
echo "------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/eval_depth_oracle.py \
    ${CONFIG} \
    ${CHECKPOINT} \
    --condition night \
    --brightness-threshold 60.0 \
    --out ${OUTPUT_DIR}/depth_oracle_night.json

# Step 3: Day Evaluation (Control)
echo ""
echo "Step 3: Running day evaluation (control)..."
echo "------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/eval_depth_oracle.py \
    ${CONFIG} \
    ${CHECKPOINT} \
    --condition day \
    --brightness-threshold 60.0 \
    --max-samples 500 \
    --out ${OUTPUT_DIR}/depth_oracle_day.json

# Step 4: Generate Report
echo ""
echo "Step 4: Generating report..."
echo "------------------------------------------------------------"
python -c "
import json
import os

output_dir = '${OUTPUT_DIR}'

# Load results
with open(f'{output_dir}/depth_oracle_night.json', 'r') as f:
    night_results = json.load(f)

with open(f'{output_dir}/depth_oracle_day.json', 'r') as f:
    day_results = json.load(f)

# Generate markdown report
report = '''# Depth Oracle Experiment Results

**Date**: $(date +%Y-%m-%d)
**Config**: ${CONFIG}
**Checkpoint**: ${CHECKPOINT}

---

## Executive Summary

'''

# Add interpretation
if 'interpretation' in night_results:
    interp = night_results['interpretation']
    report += f'''
### Key Finding

| Metric | Night Δ mAP |
|--------|-------------|
| With GT Depth | {interp.get('mAP_delta', 0)*100:+.1f}% |

**Conclusion**: {interp.get('conclusion', 'N/A')}

**Recommendation**: {interp.get('recommendation', 'N/A')}

'''

# Add detailed metrics
report += '''
---

## Detailed Results

### Night Condition

| Metric | Predicted Depth | GT Depth (Oracle) | Delta |
|--------|-----------------|-------------------|-------|
'''

if 'predicted_depth_metrics' in night_results and 'gt_depth_metrics' in night_results:
    pred = night_results['predicted_depth_metrics']
    gt = night_results['gt_depth_metrics']
    for metric in ['mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']:
        if metric in pred and metric in gt:
            delta = gt[metric] - pred[metric]
            sign = '+' if delta > 0 else ''
            report += f'| {metric} | {pred[metric]:.4f} | {gt[metric]:.4f} | {sign}{delta:.4f} |\\n'

report += '''
### Day Condition (Control)

| Metric | Predicted Depth | GT Depth (Oracle) | Delta |
|--------|-----------------|-------------------|-------|
'''

if 'predicted_depth_metrics' in day_results and 'gt_depth_metrics' in day_results:
    pred = day_results['predicted_depth_metrics']
    gt = day_results['gt_depth_metrics']
    for metric in ['mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']:
        if metric in pred and metric in gt:
            delta = gt[metric] - pred[metric]
            sign = '+' if delta > 0 else ''
            report += f'| {metric} | {pred[metric]:.4f} | {gt[metric]:.4f} | {sign}{delta:.4f} |\\n'

report += '''
---

## Configuration

'''
report += f'''
- GT Depth Coverage: {night_results.get('config', {}).get('mean_gt_coverage', 0)*100:.1f}%
- Night samples: {night_results.get('config', {}).get('n_night', 0)}
- Day samples: {day_results.get('config', {}).get('n_day', 0)}
- Brightness threshold: {night_results.get('config', {}).get('brightness_threshold', 60.0)}

'''

# Write report
report_path = 'research-notes/racformer/DEPTH_ORACLE_EXPERIMENT.md'
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, 'w') as f:
    f.write(report)

print(f'Report saved to {report_path}')
"

echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
echo "Results saved to: ${OUTPUT_DIR}/"
echo "Report saved to: research-notes/racformer/DEPTH_ORACLE_EXPERIMENT.md"
