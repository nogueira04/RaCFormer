# RaCFormer Paper Goal v5 Final Report

Verdict: `PASS-PAPER-NR-V5-STRENGTHENED`

UTC: `20260521T011657Z`

Endpoint: EXHAUSTION. v5 screened 8 autonomous candidates across robustness, efficiency, test-time, architecture-bounded, and data-axis-style scenarios. Seven candidates failed Phase B auto-promotion criteria. One candidate auto-promoted to Phase D and failed its pre-registered Stage 3B formal gate. No candidate reached `E_replicated`.

Pointer to v4 final report: `research/paper_goal_20260518/FINAL_REPORT_v4.md`

## Summary Metrics

- Cumulative GPU-h: `7.712979`
- Cumulative wall-clock hours in state: `0.0` (state wall-clock field was not actively accumulated)
- Candidates screened: `8`
- Auto-promotion rate: `1/8 = 12.5%`
- Positive candidates: `0`
- Terminal reason: all promoted candidates failed Phase D; all 8 candidates are in `B_blocked` or `D_fail`

## Candidates

| id | title | mechanism | scenario | Phase B verdict | Phase D verdict | Phase E verdict | GPU-h |
|---|---|---|---|---|---|---|---:|
| `multicorrupt-radar-drop-l2sp` | Radar-Drop Sensor-Corruption Fine-Tune with L2-SP | sensor-dropout-anti-forgetting-finetune | robust-radar-drop | `B_blocked` | not run | not run | 0.124078 |
| `cvfusion-radar-refine-camera-drop` | Radar Instance Refinement for Camera Drop | radar-instance-refinement-under-camera-mask | robust-camera-drop | `B_screened`, auto-promoted | `D_fail`: formal_gate_miss | not run | 5.344835 |
| `dyss-query-prune-efficiency` | Dynamic Query Pruning for Throughput | dynamic-query-efficiency | efficiency | `B_blocked` | not run | not run | 0.083203 |
| `monotta-night-layernorm` | Reliability-Driven Night Test-Time LayerNorm Adaptation | test-time-reliability-adaptation | temporal-night | `B_blocked` | not run | not run | 0.084173 |
| `unida-qddm-rain-alignment` | Query-Guided Rain Domain Alignment | query-domain-alignment | subset-rain | `B_blocked` | not run | not run | 0.129979 |
| `raydn-far-hard-negatives` | Ray Denoising Hard Negatives for Far Objects | depth-aware-denoising-training | subset-far | `B_blocked` | not run | not run | 0.119183 |
| `bevnext-crf-motorcycle-depth` | Object-Level CRF Depth Consistency for Motorcycle AP | object-depth-consistency-auxiliary-loss | class-motorcycle | `B_blocked` | not run | not run | 0.128742 |
| `ocrf-foreground-opacity-cv` | Foreground Opacity Query Attention for Construction Vehicles | foreground-opacity-query-attention | class-cv | `B_blocked` | not run | not run | 0.122877 |

## Phase D Evidence

Only `cvfusion-radar-refine-camera-drop` met the strict Phase B auto-promotion rule:

- Identity max diff: `0 <= 1e-6`
- Proxy camera-drop delta: `0.006751`, exceeding the `0.005` threshold by at least 20%

Stage 3B trained for 12 epochs / 24,000 iterations without hard training pathology:

- Nonfinite iterations: `0`
- Loss first / last: `44.17768096923828` / `16.64331817626953`
- Peak allocated memory: `5081.13623046875` MB
- Stage 3B GPU-h: `5.223925186859237`

Formal gate:

| metric | value | gate | verdict |
|---|---:|---:|---|
| camera-drop full-val mAP | 0.003828206588831159 | >= 0.0250012864 | FAIL |
| clean full-val mAP | 0.29807969129976597 | >= 0.2989905911 | FAIL |
| camera-drop full-val NDS | 0.08129685924405008 | informational | n/a |
| clean full-val NDS | 0.36355506574065616 | informational | n/a |

The candidate therefore failed with `formal_gate_miss`. Evidence:

- JSON: `research/paper_goal_20260520/cvfusion_stage3_train_eval_evidence.json`
- Failure markdown: `research/paper_goal_20260520/candidates/cvfusion-radar-refine-camera-drop_FAILURE_MODE_20260521T011546Z.md`
- Checkpoint retained remotely: `research/paper_goal_20260520/cvfusion_stage3/cvfusion_stage3b_epoch12_seed0.pth`

## Exhaustion Result

v5 added eight bounded candidate mechanisms on top of the v3/v4 closed families:

- Sensor-dropout anti-forgetting fine-tuning
- Radar-instance refinement under camera masking
- Dynamic-query efficiency pruning
- Test-time reliability adaptation
- Query/domain alignment for rain
- Depth-aware denoising for far objects
- Object-depth consistency auxiliary loss
- Foreground-opacity query attention

Together with the 5 v3 substrate-family branches and 6 v4 cross-literature candidates, the cumulative audit now covers 19 mechanism families or bounded representatives. v5 did not produce a surviving positive method, and the promoted candidate failed on full validation despite passing the stricter Phase B proxy and identity gates. This strengthens the negative-results framing rather than changing the endpoint.

