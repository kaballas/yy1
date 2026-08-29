# Market-Blind Winner Ranker Audit v1

Git commit: `3d64889feade7da3ab6f530cbba4d7cb839cdd20`

Development used training and validation only. Historical test/test2 were not loaded or scored.

Test-3 races available: **63**. Test-3 opened: **no**.

## Baseline seed variance

Mean Top-1: 30.78%; standard deviation: 0.30%; range: 0.80%.

## Feature audit

Features: 361; verified: 216; unknown: 145; suspect: 0.

## Group ablations

| Removed group | Features | Mean Top-1 | Delta | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| recent_form | 58 | 30.56% | -0.22% | [-1.20%, +0.78%] |
| sectionals_speed | 33 | 30.96% | +0.18% | [-0.88%, +1.24%] |
| career_profile | 16 | 30.02% | -0.76% | [-1.88%, +0.38%] |
| distance | 31 | 30.54% | -0.24% | [-0.96%, +0.46%] |
| track_condition | 32 | 30.06% | -0.72% | [-1.82%, +0.38%] |
| class | 12 | 30.28% | -0.50% | [-1.12%, +0.10%] |
| weight_barrier | 59 | 30.64% | -0.14% | [-1.26%, +0.98%] |
| connections | 21 | 30.00% | -0.78% | [-2.32%, +0.76%] |
| freshness_preparation | 16 | 30.74% | -0.04% | [-1.02%, +0.94%] |
| race_context_relative | 77 | 30.22% | -0.56% | [-1.52%, +0.42%] |
| prize_money_profile | 5 | 30.26% | -0.52% | [-1.44%, +0.38%] |
| other_profile | 1 | 30.74% | -0.04% | [-0.70%, +0.62%] |

## Architecture comparison

| Model | Features | Parameters | Mean Top-1 | Std | Delta | Bootstrap 95% CI | Seeds +/=/- |
|---|---:|---:|---:|---:|---:|---:|---:|
| reduced_current_mlp | 302 | 54849 | 30.64% | 0.29% | -0.14% | [-1.26%, +0.98%] | 2/1/2 |
| wider_mlp | 302 | 104897 | 30.44% | 0.38% | -0.34% | [-1.48%, +0.80%] | 1/0/4 |
| residual_mlp | 302 | 109057 | 30.46% | 0.39% | -0.32% | [-1.52%, +0.88%] | 1/0/4 |
| xgboost_ranker | 302 | None | 29.64% | 0.30% | -1.14% | [-2.88%, +0.60%] | 0/0/5 |

## Final selection

Selected: **full_current_mlp**.

No challenger meets the predeclared validation-only improvement rule.

Configuration SHA256: `a4771768b9b09d81a8351a5a56e0dca9a70ca4b9216a554c48e7cfb30ff5c0e9`

