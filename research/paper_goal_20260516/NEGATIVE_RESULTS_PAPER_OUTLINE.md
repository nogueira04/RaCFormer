# Night Up, Day Down: An Empirical Audit of Radar-Camera 3D Detection Robustness

Updated: 20260517T211223Z

## Abstract

We audit more than 25 radar-camera 3D detection interventions for nuScenes robustness and find that most apparent night gains are coupled to day degradation, seed instability, or geometry-label drift. The paper argues that negative interventions are useful when they are organized into falsifiable mechanism classes: Prior Corruption, Gate Collapse, Geometry-Label Drift, and Stochastic Gate Flipping.

## 1. Introduction

The central claim is not that RaCFormer is uniquely brittle; it is that radar-camera robustness work needs failure accounting at the same standard as positive methods. We use the existing intervention history as a controlled audit and show that several plausible mechanisms fail for different, repeatable reasons.

Claims:
- Query replacement can actively corrupt the learned polar-anchor prior.
- Decoder-side condition gates at the tested fusion site trade night gains for day losses rather than separating domains.
- Generated-night augmentation is unsafe without geometry-preservation QC.
- Motion/temporal gates need seed-stability evidence before their gains can be claimed.

## 2. Benchmark and Audit Protocol

Dataset: nuScenes full validation split with day, night, and rain condition partitions. Baseline: S0 train2k day model. Metrics: official nuScenes mAP/NDS, per-class AP, nuScenes TP errors, exact CPU-only nuScenes subset re-eval rows for every audited variant, and CPU-only prediction-level TP/FN proxies for agreement and calibration.

Figures:
- Fig. 1: audit map of variants by family and verdict.
- Fig. 2: condition mAP/NDS deltas vs S0 with 95% class-bootstrap CIs.
- Fig. 3: per-class AP breakdown across conditions.

## 3. Failure Taxonomy

### 3.1 Prior Corruption (D Family)

Branch D replaces learned polar-anchor queries with radar-derived object tokens. The dose-response pattern is the key evidence: the larger N=180 replacement is worse than top-k90, so the failure is not simply underpowered use of radar. The mechanism likely overwrites a load-bearing query prior.

Figure: Branch D query-replacement monotonicity, with all-sample and radar-rich subset deltas vs S0.

### 3.2 Gate Collapse (A Family)

The S5 family tests multiple decoder-side gating substrates at the same fusion site. Their shared signature is night improvement pressure paired with day degradation, and WiSE-FT shows the night-biased solution is not linearly connected to S0 without catastrophic overall loss.

Figure: S5/S5_conditionfusion/S5_contrelqfusion day-night Pareto and WiSE-FT curve.

### 3.3 Geometry-Label Drift (NB2 and DriveGEN)

Night synthesis is only useful if the 3D labels remain valid. The audit positions NB2/DriveGEN failures as geometry-label drift: apparent photometric robustness can be offset by broken depth, scale, and multi-view consistency.

Figure: generated-night QC examples and condition metrics.

### 3.4 Stochastic Gate Flipping (C Family)

The C-family seed matrix shows why single-seed night wins are not sufficient. Seed0/seed1/seed2 and V10/V40 are compared by per-class AP, subset metrics, and TP/FN kappa agreement.

Figure: C-family seed-stability kappa matrix and per-class variance.

## 4. Cross-Variant Evidence

This section uses the Stage 1B artifacts: cross-variant agreement matrix, failure-attribution heatmap, calibration diagrams, condition-shift histograms, exact all-variant subset re-eval rows. These provide the paper's evidence that the taxonomy is not just a narrative grouping.

Figures:
- Fig. 4: Cohen's kappa agreement matrix.
- Fig. 5: per-sample failure attribution heatmap.
- Fig. 6: reliability diagrams by condition.
- Fig. 7: score-distribution KL divergence heatmap.

## 5. Negative Results as Design Constraints

The audit converts failures into constraints for future branches: do not write radar into query parameters, do not repeat decoder-side gates at the exhausted site, do not train on generated images without geometry QC, and do not claim temporal/motion improvements without multi-seed replication.

## 6. Related Work

Position against radar-camera 3D detection, robustness under night/rain, multimodal distillation, and negative-results methodology. The paper should emphasize that it complements positive methods by identifying intervention families that are experimentally exhausted under this setup.

## 7. Limitations

The audit is centered on RaCFormer and nuScenes. Prediction-level agreement/calibration analyses use a documented same-class 2m matching proxy, while official claims remain anchored to nuScenes mAP/NDS. Some intervention families have more variants than others.

## 8. Conclusion

A publishable negative-results paper is viable if it presents the intervention history as a mechanism audit rather than a list of failed runs. The Stage 1A and Stage 1B artifacts now cover the required figure and claim inventory for that framing.

## 9. v5 Search-Effort Appendix

The v5 unattended continuation (20260520-20260521) screened 8 additional bounded candidates across radar-drop robustness, camera-drop robustness, efficiency, test-time adaptation, rain/far subsets, and motorcycle/construction-vehicle class axes. Seven candidates failed the strict Phase B auto-promotion rule. One candidate, `cvfusion-radar-refine-camera-drop`, passed identity-at-zero and proxy promotion but failed full Stage 3B validation: camera-drop mAP `0.003828` versus gate `0.025001`, clean mAP `0.298080` versus gate `0.298991`. No v5 candidate reached replication. Combined with the 5 v3 substrate-family branches and 6 v4 cross-literature candidates, the cumulative audit now covers 19 mechanism families or bounded representatives without a surviving positive endpoint.
