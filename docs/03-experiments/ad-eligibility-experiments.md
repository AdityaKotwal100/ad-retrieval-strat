# Ad Eligibility Experiments

## Experiment 1: Baseline bipartite k-means appropriateness

- Hypothesis
- A simple centroid-purity model over crisis vs commercial phrases can produce usable eligibility scores.

- Change made
- Implemented in `experiments/k-means-for-appropriateness/experiment.py`.
- Corpus: 371 sensitivity phrases plus 110 commercial exemplars.

- Evaluation method
- Measured MAE, Spearman rank correlation, hard-block accuracy, and per-query errors.
- Report in `experiments/k-means-for-appropriateness/findings.md`.

- Results
- Best MAE at `k=2`: 0.195.
- Spearman: +0.779.
- Hard-block: 100%.
- Within ±0.20: 11/20.

- Decision and rationale
- Not sufficient for production alone.
- Kept concept of separate safety/commercial signals, rejected single-pass calibration as final approach.

## Experiment 2: 3-tier corpus extension

- Hypothesis
- Adding medium-tier phrases should improve calibration for distress but non-crisis queries.

- Change made
- Added financial hardship, ambiguous stress, and informational phrase tiers.
- Implemented in `experiment_3tier.py` and reported in `findings_3tier.md`.

- Evaluation method
- Same metrics as baseline across k sweep.

- Results
- Best MAE: 0.182 (`k=16`).
- Spearman: +0.794.
- Within ±0.20 improved from 11/20 to 13/20.
- Regressions observed on some pure commercial queries.

- Decision and rationale
- Partially kept as evidence that medium-tier examples matter.
- Rejected as standalone scoring model due commercial-score dilution.

## Experiment 3: Anchored commercial centroid and two-pass variants

- Hypothesis
- Anchoring commercial representation or separating safety and affinity heads will improve calibration.

- Change made
- Anchored centroid and two-pass prototypes in `experiment_two_pass.py`.

- Evaluation method
- Compared MAE, Spearman, hard-block, and within-0.20 outcomes against prior baselines.

- Results
- Anchored single-pass best: MAE 0.178, hard-block 100%.
- Two-pass orthogonal best MAE: 0.146, but hard-block dropped to 78%.

- Decision and rationale
- Confirmed two-head direction is structurally better.
- Rejected global-threshold two-pass because hard-block failures are unacceptable.

## Experiment 4: Option E production-mirror two-pass

- Hypothesis
- Production-like design with hard cluster gating plus continuous penalty and commercial floor can recover both safety and calibration.

- Change made
- Implemented in `experiment_option_e.py` and reported in `findings_option_e.md`.

- Evaluation method
- Same benchmark suite and multi-k affinity sweep.

- Results
- Best: MAE 0.087, Spearman +0.905, hard-block 100%, 19/20 within ±0.20.
- One persistent miss: "why do runners get blisters" (0.55 vs expected 0.80).

- Decision and rationale
- Kept core layered architecture and caution-first gating philosophy.
- This experiment series directly informed the current `EligibilityScorer` structure.

## Experiment 5: Run against production test-suite outputs

- Hypothesis
- Option E should track current runtime scores on saved test-suite outputs.

- Change made
- Evaluated in `run_on_test_suite.py`, results in `findings_test_suite.md`.

- Evaluation method
- Compared against `data/test_results/*/output.json` derived ground truth.

- Results
- Comparable 36-case subset: MAE 0.134, Spearman +0.700, hard-block 100%, within ±0.20 = 27/36.
- Errors concentrated on high-eligibility commercial queries with incidental sensitivity overlap.

- Decision and rationale
- Kept caution bias and hard-gate behavior.
- Identified need for better handling of weak incidental sensitivity in commercial queries.

## Experiment 6: Toxicity model iteration sweeps

- Hypothesis
- Iterative model architecture and threshold tuning can improve insensitive-query detection with low inference cost.

- Change made
- Multi-version training/testing in `experiments/toxicity` and `generate-ads-or-not/models/toxicity`.

- Evaluation method
- Tracked ROC AUC, PR AUC, micro metrics, thresholds, model size, and latency.

- Results
- `experiments/toxicity` best objective among shown test reports: v10 objective 1.135, micro ROC AUC 0.9716.
- Runtime model path currently points to v5 artifact (`app/config.py`), with threshold loaded from model artifact.
- `generate-ads-or-not/models/toxicity/final_summary.json` shows very high offline AUC and sub-millisecond p95 inference for iterations.

- Decision and rationale
- Kept lightweight model-based sensitive gating with fail-closed behavior.
- Inference: runtime kept v5 likely for stability and known behavior compatibility; explicit model-version selection rationale is not documented.

## Evidence gaps

- Full commit-by-commit rationale for each threshold change is not reconstructable from visible git history (only two commits available).
- Not found in repo: a formal confusion matrix report for the exact current production checkpoint and exact held-out query set used for gating policy sign-off.

## Proposed replacement experiments (if rerunning)

1. Build a fixed, versioned evaluation set for sensitive and borderline benign queries.
2. Run grid search over model threshold and `ELIGIBILITY_GATE` with leakage constraints.
3. Track false-negative crisis leakage as hard constraint, then optimize commercial false-positive rate.
4. Save signed JSON reports under `experiments/eligibility/<date>/` for reproducibility.
