# Changelog

All notable changes to interp-lab are documented here. This project adheres to
[semantic versioning](https://semver.org/).

## 2.2.0 — usability, robustness & new tools

This release makes interp-lab easier to pick up and harder to crash, demonstrates
its core thesis in the front-door demo, and adds genuinely useful new features —
without any new required dependencies. Every change is covered by tests
(`tests/test_cli_errors.py`, `tests/test_api_ergonomics.py`, `tests/test_run_diff.py`,
`tests/test_matching_tuning.py`, plus additions to existing suites).

### Added

- **`compare-runs`** (CLI) / **`compare_runs`** (API) — diff two inspection reports
  to surface rank drift, per-score deltas, and added/dropped features between two
  seeds, checkpoints, or versions. Writes JSON + a readable Markdown summary
  (`interp-lab.run_diff.v1`).
- **`quickstart`** (alias `tutorial`) — a short guided walkthrough of the workflow
  and what each metric means.
- **`inspect --csv-out`** / `inspect(csv_out=...)` — export the ranked features as a
  spreadsheet/paper-friendly CSV (pure stdlib).
- **`interp-lab --version`**, and a grouped, "start here" **`--help`** epilog instead
  of a flat 36-command wall.
- **Notebook-friendly Python API**: `Written*` results returned by `inspect`/`compare`
  now chain directly into `compare`/`validate_matches`; compact `__repr__`s that no
  longer dump float vectors; `InspectionReport.cards_table()` /
  `MatchReport.matches_table()` for `pandas.DataFrame(...)`-ready rows;
  top-level `load_inspection_report` / `load_match_report`; a PEP 561 `py.typed`
  marker; documented `FeatureCard` score fields.
- **`match --min-score` / `--weights`** (and `compare(min_score=, weights=)`) — drop
  weak candidate matches and override fingerprint component weights for sensitivity
  analysis.
- **Report polish**: an inline-SVG *importance-by-layer* profile (color-coded by
  measured-causal vs correlational), strength-colored metric meters, and a
  reproducibility stamp (`metadata.tool`: version + platform) in every `report.json`.
- The **demo now demonstrates a causal claim** end-to-end: the toy intervention
  runner gained an opt-in `measured` mode (strong causal scores, signed effects,
  CIs, controls), so `interp-lab demo` shows *validated* cross-model equivalents and
  measured effects instead of "causal claims untested". The demo also writes a
  clickable `index.html` hub and prints a narrative summary with what to open next.

### Fixed

- **Raw tracebacks on ordinary user errors** are now clean `interp-lab: error: …`
  (exit 2) messages: the CLI boundary catches `OSError`/`ImportError`, an unknown
  `--text-embedder` no longer crashes, and a JSONL feature row missing `feature_id`
  reports the file:line and field instead of a `KeyError`.
- **Real-world files no longer crash**: inputs are read with `utf-8-sig` so a
  Windows UTF-8 BOM is tolerated; a config value containing literal `{`/`}` (e.g. a
  criterion describing JSON or set notation) is substituted safely; `match --out foo`
  without a `.json` suffix writes `foo.md` beside it instead of `FileExistsError`.
- **Free 0.5 "half match" for absent/length-mismatched activation & decoder
  signatures** (`matching.py`): these components are now gated and renormalized like
  text and causal (previously only text/causal were), and a provenance-`"none"`
  causal vector is excluded rather than scored.
- **Garbage-in protection**: non-finite numbers (`NaN`/`Infinity`, which `json` accepts)
  are rejected at ingestion with a file:line message; an empty/whitespace `--criterion`
  is rejected instead of emitting a placeholder report; a JSONL `--model` typo warns
  instead of silently producing an empty report.
- **Honest report labels**: the HTML report's effect column/meter now reads
  "Causal effect" only when interventions were measured (otherwise "Criterion
  score"); the metric legend defines every rendered field (including Importance and
  Stability); the match-validation "Causal" column is relabeled "Causal-vec sim" (a
  structural similarity, not a measured effect).
- **CLI consistency**: `init-run`'s overwrite message references `--force` (not the
  `force=True` kwarg); `validate-assay <file>` accepts a positional path;
  `prepare-sae-prompts` accepts `--out` as an alias for `--out-dir`; `--json` output
  for `release-check`/`demo-sweep` keeps stdout pure JSON (the "Wrote …" line goes to
  stderr); `doctor` now points to the next step.

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
