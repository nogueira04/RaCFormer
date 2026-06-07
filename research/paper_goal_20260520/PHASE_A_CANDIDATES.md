# v5 Phase A Candidates

UTC: 20260520T180056Z

Phase A generated exactly 8 candidates after applying the v5 closed-family checks.

Amendment: candidate 1 was corrected before Phase B to avoid the v3 exhausted weather-adaptive fusion-gate substrate.

## Baselines

- Clean S0 mAP: 0.3039905911
- Radar-drop S0 mAP: 0.2145426354
- Camera-drop S0 mAP: 0.0000012864
- S0 FPS median: 4.228253

## Candidate Table

| id | scenario | mechanism family | source |
|---|---|---|---|
| multicorrupt-radar-drop-l2sp | robust-radar-drop | sensor-dropout-anti-forgetting-finetune | arXiv:2502.13071, arXiv:2402.11677 |
| cvfusion-radar-refine-camera-drop | robust-camera-drop | radar-instance-refinement-under-camera-mask | arXiv:2507.04587, arXiv:2502.13071 |
| dyss-query-prune-efficiency | efficiency | dynamic-query-efficiency | CVPR2025 DySS, arXiv:2411.10715 |
| monotta-night-layernorm | temporal-night | test-time-reliability-adaptation | arXiv:2405.19682, arXiv:2406.13891 |
| unida-qddm-rain-alignment | subset-rain | query-domain-alignment | arXiv:2603.27995, arXiv:2410.22461 |
| raydn-far-hard-negatives | subset-far | depth-aware-denoising-training | arXiv:2402.03634 |
| bevnext-crf-motorcycle-depth | class-motorcycle | object-depth-consistency-auxiliary-loss | arXiv:2312.01696, CVPR2024 BEVNeXt |
| ocrf-foreground-opacity-cv | class-cv | foreground-opacity-query-attention | arXiv:2506.23565 |

## multicorrupt-radar-drop-l2sp

Title: Radar-Drop Sensor-Corruption Fine-Tune with L2-SP

Scenario: `robust-radar-drop`

Gate: radar-drop full-val mAP >= 0.2395426354 (S0_drop 0.2145426354 + 0.025) AND clean overall mAP >= 0.2989905911 (S0 0.3039905911 - 0.005).

Mechanism: MultiCorrupt and RobuRCDet motivate explicit evaluation and training under sensor/weather corruptions. Adapt this to RaCFormer without adding a new fusion gate: fine-tune a bounded set of existing fusion projection and decoder-normalization parameters with radar-depth/radar-BEV masking on random train2k batches, while an L2-SP penalty anchors all trainable parameters to the S0 checkpoint. The disable knob is the original S0 checkpoint; there is no query replacement, post-hoc calibration, or BEV distillation target.

Diagnostic: Verify the S0 checkpoint vs the diagnostic config with zero optimizer steps is identity on one full validation-format batch with max diff <=1e-6. Fine-tune only existing fusion projection and decoder LayerNorm affine parameters for <=300 train2k steps with radar feature masks p=0.5 and L2-SP anchoring to S0. Evaluate a 128-sample radar-drop mini split and 128 clean samples. Proxy gate: radar-drop mini mAP improves by >=0.005 and clean mini mAP drops by <=0.010; expected GPU use <1 GPU-h.

Kill condition: BLOCK if identity max diff >1e-6, if radar-drop proxy improvement <0.005, if clean proxy mAP drop >0.010, if NaN/Inf appears, or if fewer than 90% of trainable parameters remain within the registered L2-SP norm budget.

Sources: arXiv:2502.13071, https://arxiv.org/abs/2502.13071, arXiv:2402.11677, https://arxiv.org/abs/2402.11677

## cvfusion-radar-refine-camera-drop

Title: Radar Instance Refinement for Camera Drop

Scenario: `robust-camera-drop`

Gate: camera-drop full-val mAP >= 0.0250012864 (S0_drop 0.0000012864 + 0.025) AND clean overall mAP >= 0.2989905911 (S0 0.3039905911 - 0.005).

Mechanism: CVFusion uses radar-guided proposals and instance-level aggregation across points, image, and BEV views. Adapt only the instance-refinement idea: attach a lightweight radar-BEV/query residual that refines RaCFormer decoder features when image and camera-BEV features are masked. The baseline decoder queries remain intact; the residual is zero-initialized and can be disabled exactly.

Diagnostic: Verify residual_weight=0 identity on one batch with max diff <= 1e-6. Train the radar-instance residual and final fusion projection for <=300 train2k steps with camera feature masks p=0.5. Evaluate a 128-sample camera-drop mini split and 128 clean samples. Proxy gate: camera-drop mini mAP improves by >=0.005 and clean mini mAP drops by <=0.010; expected GPU use <1 GPU-h.

Kill condition: BLOCK if identity max diff > 1e-6, if camera-drop proxy improvement <0.005, if clean proxy mAP drop >0.010, or if radar-only residual has fewer than 20% nonzero query activations.

Sources: arXiv:2507.04587, https://arxiv.org/abs/2507.04587, arXiv:2502.13071, https://arxiv.org/abs/2502.13071

## dyss-query-prune-efficiency

Title: Dynamic Query Pruning for Throughput

Scenario: `efficiency`

Gate: FPS >= 6.342379 (1.5 x S0_FPS 4.228253) AND clean overall mAP >= 0.2989905911 (S0 0.3039905911 - 0.005).

Mechanism: DySS identifies the cost of carrying many sparse video queries and introduces dynamic queries for efficient multi-camera 3D detection. Adapt this as an inference-time query pruning schedule after decoder layer 3: stable low-objectness queries are masked from later cross-attention while high confidence and rare-class queries remain active. It changes computation only, not training labels.

Diagnostic: Verify prune_threshold=0 keeps logits and boxes identity with max diff <= 1e-6. Time 200 clean val samples and evaluate a 128-sample clean mini split with pruning enabled. Proxy gate: FPS >= 5.073903 (1.2 x S0_FPS) and clean mini mAP drops by <=0.010; expected GPU use <1 GPU-h.

Kill condition: BLOCK if identity max diff >1e-6, if FPS proxy <1.2 x S0_FPS, if clean mini mAP drop >0.010, or if pruning changes tensor shapes in a way that breaks the decoder cache.

Sources: CVPR2025 DySS, https://cvpr.thecvf.com/virtual/2025/35825, arXiv:2411.10715, https://arxiv.org/abs/2411.10715

## monotta-night-layernorm

Title: Reliability-Driven Night Test-Time LayerNorm Adaptation

Scenario: `temporal-night`

Gate: night condition mAP >= 0.1687749875 (S0 0.1487749875 + 0.020) AND day condition mAP >= 0.3102649818 (S0 0.3152649818 - 0.005).

Mechanism: MonoTTA proposes fully test-time adaptation for 3D detectors using reliable high-score objects and negative regularization for noisy low-score predictions; DPO further supports perturbation-stable 3D TTA. Adapt this conservatively to RaCFormer by updating only LayerNorm affine parameters and the fusion bias on unlabeled night batches, with high-score pseudo positives and low-score negative regularization. The source checkpoint is restored between clips.

Diagnostic: Run no-update identity on one batch with max diff <=1e-6. Then adapt LayerNorm/fusion-bias only for <=64 night samples and evaluate 64 held-out night plus 64 day samples. Proxy gate: night mini mAP improves by >=0.005 and day mini mAP drops by <=0.005; expected GPU use <1 GPU-h.

Kill condition: BLOCK if no-update max diff >1e-6, if night proxy improvement <0.005, if day proxy drop >0.005, if more than 10% of pseudo positives are non-finite, or if adaptation cannot restore checkpoint state.

Sources: arXiv:2405.19682, https://arxiv.org/abs/2405.19682, arXiv:2406.13891, https://arxiv.org/abs/2406.13891

## unida-qddm-rain-alignment

Title: Query-Guided Rain Domain Alignment

Scenario: `subset-rain`

Gate: rain condition mAP >= 0.2893174671 (S0 0.2743174671 + 0.015) AND clean overall mAP >= 0.2939905911 (S0 0.3039905911 - 0.010).

Mechanism: UniDA3D frames night, rain, and fog as target domains and uses query-guided domain discrepancy mitigation. Adapt the lightweight part only: add a small domain classifier and contrastive alignment loss on final decoder query features between day and rain train2k samples, leaving boxes, queries, and BEV tensors unchanged. The alignment head is train-only and disabled for identity checks.

Diagnostic: Verify align_weight=0 identity on one batch with max diff <=1e-6. Train the query-domain head and last fusion projection for <=300 train2k steps using day/rain metadata. Evaluate 128 rain and 128 clean samples. Proxy gate: rain mini mAP improves by >=0.005 and clean mini mAP drops by <=0.010; expected GPU use <1 GPU-h.

Kill condition: BLOCK if identity max diff >1e-6, if rain proxy improvement <0.005, if clean proxy mAP drop >0.010, or if the domain classifier accuracy remains within [45%,55%] after training.

Sources: arXiv:2603.27995, https://arxiv.org/abs/2603.27995, arXiv:2410.22461, https://arxiv.org/abs/2410.22461

## raydn-far-hard-negatives

Title: Ray Denoising Hard Negatives for Far Objects

Scenario: `subset-far`

Gate: object_far_ge30m mAP >= 0.1026307261 (S0 0.0876307261 + 0.015) AND clean overall mAP >= 0.2939905911 (S0 0.3039905911 - 0.010).

Mechanism: Ray Denoising improves DETR-style multi-view 3D detectors by adding depth-aware hard negative examples sampled along camera rays, without changing inference speed. Adapt this to RaCFormer's existing denoising branch: for far matched boxes, add train-only negative query targets displaced along the same camera ray/range bin. The decoder and polar query set at inference remain unchanged.

Diagnostic: Verify denoise_weight=0 identity on one batch with max diff <=1e-6. Train <=300 steps on far-object enriched train2k frames with ray hard negatives in the denoising branch only. Evaluate 128 far and 128 clean samples. Proxy gate: far mini mAP improves by >=0.005 and clean mini mAP drops by <=0.010; expected GPU use <1 GPU-h.

Kill condition: BLOCK if identity max diff >1e-6, if far proxy improvement <0.005, if clean proxy mAP drop >0.010, or if fewer than 30 far GT boxes are present in the diagnostic split.

Sources: arXiv:2402.03634, https://arxiv.org/abs/2402.03634

## bevnext-crf-motorcycle-depth

Title: Object-Level CRF Depth Consistency for Motorcycle AP

Scenario: `class-motorcycle`

Gate: motorcycle mean_dist_AP >= 0.3696619031 (S0 0.3396619031 + 0.030) AND clean overall mAP >= 0.2939905911 (S0 0.3039905911 - 0.010).

Mechanism: BEVNeXt revives dense BEV by using CRF-modulated depth estimation to enforce object-level depth consistency. Adapt this as a small auxiliary loss over RaCFormer depth logits for motorcycle-heavy samples: the loss smooths depth only within projected object regions and does not supervise or distil camera-BEV features. At aux_weight=0 the forward path is unchanged.

Diagnostic: Verify aux_weight=0 identity on one batch with max diff <=1e-6. Train <=300 steps on train2k frames ranked by motorcycle count with object-depth CRF loss plus original detection/depth losses. Evaluate a 128-sample motorcycle-heavy split and 128 clean samples. Proxy gate: motorcycle AP improves by >=0.010 and clean mini mAP drops by <=0.010; expected GPU use <1 GPU-h.

Kill condition: BLOCK if identity max diff >1e-6, if motorcycle AP proxy improvement <0.010, if clean proxy mAP drop >0.010, or if projected-object depth masks cover fewer than 25 motorcycle GT instances.

Sources: arXiv:2312.01696, https://arxiv.org/abs/2312.01696, CVPR2024 BEVNeXt, https://openaccess.thecvf.com/content/CVPR2024/papers/Li_BEVNeXt_Reviving_Dense_BEV_Frameworks_for_3D_Object_Detection_CVPR_2024_paper.pdf

## ocrf-foreground-opacity-cv

Title: Foreground Opacity Query Attention for Construction Vehicles

Scenario: `class-cv`

Gate: construction_vehicle mean_dist_AP >= 0.1121073377 (S0 0.0821073377 + 0.030) AND clean overall mAP >= 0.2939905911 (S0 0.3039905911 - 0.010).

Mechanism: OcRFDet finds that whole-scene radiance rendering can hurt detection, but object-centric foreground opacity can enhance 2D foreground BEV features. Adapt a bounded version: estimate a lightweight foreground-opacity scalar from projected construction-vehicle query regions and use it as a zero-initialized query attention residual. No rendering network, no whole-scene radiance field, and no BEV distillation target are introduced.

Diagnostic: Verify opacity_weight=0 identity on one batch with max diff <=1e-6. Train <=300 steps on construction-vehicle-enriched train2k frames with the query opacity residual and original losses. Evaluate a 128-sample construction-vehicle split and 128 clean samples. Proxy gate: construction vehicle AP improves by >=0.010 and clean mini mAP drops by <=0.010; expected GPU use <1 GPU-h.

Kill condition: BLOCK if identity max diff >1e-6, if construction_vehicle AP proxy improvement <0.010, if clean proxy mAP drop >0.010, or if the diagnostic split has fewer than 25 positive construction_vehicle GT boxes.

Sources: arXiv:2506.23565, https://arxiv.org/abs/2506.23565
