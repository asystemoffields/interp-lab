# Changelog

All notable changes to interp-lab are documented here. This project adheres to
[semantic versioning](https://semver.org/).

## 2.1.0 — rigor & correctness pass

This release tightens the toolkit's core promise — honest correlational-vs-causal
separation and calibrated uncertainty — by fixing a cluster of statistical and
ranking bugs and adding a shared statistics module. All fixes were located by an
adversarial multi-agent review and each is covered by a regression test
(`tests/test_stats.py`, `tests/test_rigor_fixes.py`).

### Added

- **`interp_lab.stats`** — a dependency-free statistics module:
  - `mean_confidence_interval` uses a **Student-t** critical value (not a fixed
    `z = 1.96`) and the sample standard deviation, so an interval from 3 points is
    wider than one from 300, and a single observation reports **no** interval
    (`method = "insufficient_n"`) instead of a misleading zero-width one.
  - `bootstrap_mean_interval` — seeded, reproducible percentile bootstrap.
  - `permutation_test` — a null distribution for a correlation statistic, so an
    association can be reported with a p-value.
- **`FeatureFingerprint.causal_provenance`** (`"intervention"` / `"association"` /
  `"none"`) — records whether a fingerprint's causal vector came from measured
  interventions or a correlational proxy. Matching refuses to compare causal
  vectors of different provenance.
- Intervention CIs now report `criterion_ci_method` and `criterion_ci_n` alongside
  the bounds; the activation-records provider tags its proxy effects with
  `metadata.causal_evidence = "association_proxy"`.

### Fixed

- **Importance double-counting** (`scoring.py`): an absent `strong_causal_score` no
  longer defaults to `causal_effect` (which counted the same causal number at
  0.50), and an absent `specificity` no longer borrows `association`.
- **Silent length-mismatch comparisons** (`math_utils.cosine`/`pearson`): vectors of
  different lengths (e.g. two models with different hidden sizes) are no longer
  trimmed to a shared prefix and compared; they return `0.0` ("no evidence").
- **Free 0.5 "half match" for absent causal evidence** (`matching.py`): a missing
  causal vector is now excluded and the remaining weights renormalized, instead of
  contributing a neutral 0.5.
- **Mismatched causal-vector axes** (`matching.py` + `fingerprints.py`): a measured
  causal effect is no longer cosine-compared against a correlational proxy on the
  same axis (they used different key sets).
- **Ranking vs. validation disagreement on signed effects** (`matching.py`): the
  opposite-direction cap now uses the same `min_abs_signed_effect` (0.02) threshold
  the validation layer uses, instead of an orphan 0.05.
- **Zero-width / mis-calibrated confidence intervals** (`adapters/interventions.py`,
  `graph_validation.py`): both CI helpers now route through `interp_lab.stats`
  (Student-t, sample variance, honest small-n handling).
- **`KeyError` on high-precision strength sweeps** (`feature_interventions.py`):
  intervention-strength selection now uses the raw float key rather than its
  8-decimal display rounding.
- **Wrong-sign scoring of unknown interventions** (`adapters/interventions.py`): an
  unrecognized intervention name is now rejected at parse time (its effect
  direction is undefined) rather than silently scored as "rise-is-good".
- **Suggestive sign gate cleared by a coin-flip** (`graph_validation.py`): a path
  whose effect sign is split 50/50 (`sign_consistency == 0.5`) is now classified
  `weak`, not `suggestive`; the gate requires a strict majority.
- **L1 sparsity that weakens with width** (`sae_training.py`): the SAE L1 penalty
  now sums over the latent axis (mean over the batch) so sparsity pressure is
  stable as `latent_dim` grows.

## 2.0.0

- Renamed the package to `interp_lab`, added semantic text embeddings, and
  archived a golden DistilGPT-2 real-model demo. See git history for detail.
