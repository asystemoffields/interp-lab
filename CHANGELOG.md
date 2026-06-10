# Changelog

All notable changes to interp-lab are documented here. This project adheres to
[semantic versioning](https://semver.org/).

## 3.1.0 — the criterion compiler release

Natural-language criteria are the front door to interp-lab, and until now the
text→dataset step rested on lexical templating and hash-bucket cosine — the
weakest link in an otherwise provenance-obsessed pipeline. 3.1.0 rebuilds it
around a clean division of labor: **generation is delegated to big models**
(the agent driving the toolkit, or a local GGUF), **verification runs on a
tiny purpose-built NLI cross-encoder** (~70 M, optional extra, nothing
bundled), and **everything is gated** through NLI margins plus the existing
assay validation. Generator and verifier have different failure modes, which
is the point. 539 tests (up from 500).

### Added

- **`compile-criterion`** / `interp_lab.compile_criterion()` — turn a
  natural-language criterion into a verified prompt dataset + criterion-lab
  preset (`interp-lab.criterion_compile.v1`). Three generators:
  - `agent` (recommended): two-phase flow — phase 1 writes a
    `generation-request.json` (`interp-lab.criterion_generation_request.v1`)
    with exact candidate format and counts plus canonical `agent_next_actions`;
    the driving agent writes `candidates.jsonl` and finishes with
    `compile-criterion --candidates`. The model already in the loop does the
    generation; no bundled generator.
  - `llamacpp`: generate candidates with a local GGUF via llama-cpp-python.
  - `heuristic`: the previous templating engine, kept as a dependency-free
    floor.
  Candidates pass a margin gate (positives ≥ 0.7, negatives ≤ 0.3,
  ≥ 8 survivors per side, balance-trimmed lowest-margin-first, every exclusion
  recorded) and the resulting preset is validated through the real
  criterion-assay machinery before anything is written. Gate failure writes
  the report and names the failed gate.
- **`score-prompts`** / `interp_lab.score_prompts()` — score any prompt
  dataset against a criterion hypothesis. Default scorer is a tiny zero-shot
  NLI cross-encoder (`MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1`),
  installed via the new **`interp-lab[criteria]`** extra; outputs are exact
  `ScoredPrompt` JSONL re-validated through the real record loader, with
  per-row `criterion_score_source` (`nli:<model>` / `hash_cosine`) and
  `--binarize` preserving `criterion_score_raw`.
- **Honest degradation** — without `[criteria]`, the hash-cosine scorer still
  runs but is labeled weak everywhere it appears: the margin gate downgrades
  to advisory (no candidates excluded on score), reports carry explicit
  warnings, and `criterion_score_source` says so per row.
- **2 new MCP tools** (21 total): `compile_criterion` (the agent generator
  returns the generation request directly — the natural two-phase fit for
  MCP) and `score_prompts`.
- Docs: `AGENTS.md` now opens investigations with "operationalize the
  criterion first"; COMMANDS.md and README cover the compile flow.

### Noted for future work

ONNX scorer backend (torch-free `[criteria]`), a `doctor` check for the
`[criteria]` extra, real-model NLI smoke test in CI, and a fine-tuned
purpose-built scorer once usage data accumulates.

## 3.0.0 — the self-driving investigation release

interp-lab 2.x ranked features and graded claims; 3.0.0 closes the loop. The
toolkit now plans its own next experiments, accumulates evidence across runs,
audits its own grading against planted ground truth, and turns validated
findings into reusable artifacts — all drivable end-to-end over MCP. The suite
grew from 363 to 500 tests, including a generative (Hypothesis) invariant
suite over the evidence rules.

### Added — the investigation loop

- **`plan-evidence`** / `interp_lab.plan_evidence()` — "what should I run
  next?" Diagnoses each card's evidence gaps against the real grading
  semantics (`no_causal_evidence`, `no_signed_effect`, `insufficient_power`,
  `no_controls`, `sign_inconsistency`), solves the Student-t sample size that
  would close each gap, and ranks runnable `intervene` commands by expected
  claim-grade movement per prompt (`interp-lab.evidence_plan.v1`).
  Association-derived effect priors are labeled `association_prior`, never as
  measured.
- **Criterion dossiers** — `dossier-update` / `dossier-show` /
  `interp_lab.update_dossier()` accumulate evidence for one (model, criterion)
  across runs: per-feature grade history and transitions, score trajectories,
  same-provenance sign-flip contradictions (provenance changes are never
  flagged as contradictions), attached match/graph-validation artifacts with
  hashes (`interp-lab.dossier.v1`, atomic writes).
- **`calibrate`** / `interp_lab.calibrate()` — the trust anchor. Plants
  synthetic worlds with known causal features, equally-correlated decoys, and
  noise; runs the real records+interventions pipeline blind; reports discovery
  precision/recall, P(truly causal | claim grade) with Wilson CIs, decoy
  resistance, and effect-size rank correlation
  (`interp-lab.calibration_report.v1`). Current machinery scores
  precision@k = 1.0, decoy resistance = 1.0,
  P(truly causal | validated) = 1.0 on default worlds.
- **`quant-diff`** / `interp_lab.quant_diff()` — which validated features did
  quantization break? Matches and validates two same-criterion reports (e.g.
  FP16 vs Q4 of one model) and verdicts every feature: preserved / degraded /
  lost / emerged, with `degraded_validated` as the headline and all thresholds
  echoed in the report (`interp-lab.quant_diff.v1`). Ships with a `run` preset
  (`examples/presets/quant-diff-run.json`), a workflow builder, and
  docs/QUANT_DIFF.md.
- **Steering artifacts** — `export-steering` packages an
  intervention-validated feature as a reusable steering vector
  (`interp-lab.steering_vector.v1`; refuses unvalidated cards unless
  `--allow-unvalidated`, which stamps `provenance: "unvalidated"`);
  `apply-steering` generates baseline-vs-steered continuations through the
  existing hidden-steering hooks (`interp-lab.steering_generation.v1`).
- **`migrate-report`** / `interp_lab.migrate_inspection_report()` — re-scores
  pre-2.3 reports under current scoring semantics with per-field deltas
  recorded under `metadata.migration`, so old/new diffs reflect the model,
  not the scorer.
- **GGUF bridge** (`docs/GGUF_BRIDGE.md`) — `export-gguf-records` pulls
  final-layer activation records from GGUF models via llama-cpp-python
  (`pip install "interp-lab[gguf]"`; honestly labeled `layers_available:
  final_only`), and `convert-hidden-dump` converts a simple documented
  hidden-state dump format from ANY runtime into full-fidelity multi-layer
  activation records — CPU-only labs need no torch.

### Added — agents & assurance

- **MCP server now covers the whole loop**: 19 tools (was 10) — adding
  `plan_evidence`, `dossier_update`, `dossier_show`, `quant_diff`,
  `calibrate`, `migrate_report`, `export_steering`, `intervene` (dry-run by
  default), and `train_sae`. `apply-steering` is deliberately not served
  (text generation is a host-agent decision).
- **Generative evidence invariants** (`tests/test_evidence_invariants.py`,
  Hypothesis, dev extra): association-only inputs can never produce
  causal-labeled outputs anywhere — scoring, matching, validation, graphs,
  rendered reports, explanation reports, end-to-end serialized artifacts.
  The 2.3.0 point regressions are now properties of the system.
- A **Claude Code skill** (`.claude/skills/interp-lab-investigate/`) that
  drives the full investigation loop, and an updated AGENTS.md.

### Fixed

- **`contradicted` / `contradicted_effect` now require intervention
  provenance** (found by the new invariant suite): two opposite-sign
  correlations previously earned a causal-sounding contradiction verdict with
  zero interventions. Association-only opposite pairs now grade
  `needs_causal_evidence` / `needs_more_evidence` with reason code
  `opposite_associations_lack_intervention_provenance`, mirroring the
  `validated` gate. Run-level: such reports grade `causal_evidence_needed`
  instead of `contradicted_matches_present`.

### Breaking

- `agent_next_actions` entries emit only the canonical shape
  `{id, title, command?+argv?, instruction?, requires?}`. The 2.3.0
  compatibility aliases are removed from emitted payloads: `next_action` on
  release-check actions, `description` on explanation-report prose actions,
  and the flat `agent_next_action` / `agent_next_action_argv` /
  `agent_next_action_requires` keys on feature-search results and text-pivot
  matches. Renderers still tolerate legacy keys when reading pre-3.0 artifacts
  from disk.
- Match-validation grade flow: association-only opposite-sign pairs no longer
  grade `contradicted` (see Fixed above) — consumers keying on that grade for
  correlational data must read the new reason code instead.

## 2.3.0 — evidence-integrity hardening & agent ergonomics

A full re-audit of the codebase (four independent review passes) found that
correlational `signed_association` values could still leak onto causal-labeled
axes through several doors the 2.1.0 rigor pass missed, plus a set of
agent-facing surfaces that had drifted from the executable truth. This release
closes all of them, then builds out the agent-facing surface: a single
discovery endpoint, a Model Context Protocol server, one canonical
next-action schema across every report type, and an `AGENTS.md` operating
manual. The suite grew from 272 to 362 tests.

### Added — for AI agents

- **`interp-lab capabilities [--json] [--out FILE]`** / **`interp_lab.capabilities()`**
  — one discovery payload (`interp-lab.capabilities.v1`): the full structured
  CLI surface (37 commands with options), the public Python API contract,
  optional-module availability, and the tool's conventions (JSON-first
  outputs, error shape, next-action schema, placeholder convention).
- **`interp-lab mcp`** — a pure-stdlib Model Context Protocol server over
  stdio exposing ten tools (`capabilities`, `doctor`, `inspect`, `compare`,
  `validate_matches`, `search_features`, `compare_runs`,
  `check_explanation_consistency`, `attribution_graph`,
  `validate_attribution_graph`) plus README/COMMANDS/AGENTS docs as
  resources. Artifact-producing tools write to disk and return compact
  summaries with paths. Client config: `{"command": "interp-lab", "args": ["mcp"]}`.
- **One canonical `agent_next_actions` shape everywhere**:
  `{id, title, command?+argv?, instruction?, requires?}` — runnable actions
  carry both a shlex-quoted `command` and its `argv`; prose guidance carries
  `instruction`. Previously four different shapes existed across inspection,
  run-diff, env-profile, release-check, demo-sweep, and explanation reports.
  Legacy keys (`next_action`, `description`, flat `agent_next_action*`) are
  still emitted for this release; demo-sweep's plain strings are now objects.
  A meta-test generates every report type and asserts shape conformance and
  that every embedded argv parses against the real CLI parser.
- **`AGENTS.md`** — the operating manual for agents driving interp-lab:
  evidence philosophy (provenance semantics, claim grades, what not to
  claim), discovery, MCP setup, conventions, and the core inspect →
  intervene → re-inspect → match → validate loop as runnable commands.
  Linked from a new "For AI agents" README section.

### Fixed — evidence integrity

- **Purely correlational matches can no longer grade `validated`/`validated_equivalent`.**
  Match components now carry signed-effect provenance markers
  (`signed_effect_provenance_intervention` / `_association`); `validate-matches`
  requires intervention provenance for `validated` and caps association-only
  (and legacy pre-provenance) pairs at `needs_causal_evidence` with reason code
  `signed_effects_lack_intervention_provenance`.
- **Mixed-provenance signed effects are never compared.** A measured
  `signed_causal_effect` on one side is no longer blended with a correlational
  `signed_association` on the other — no opposite-direction cap, no
  `contradicted_effect` from a correlation. Shared provenance-aware accessors
  (`signed_effect_with_provenance`, `has_intervention_provenance`) now back
  scoring, matching, graphs, and match validation.
- **`score_feature` no longer double-counts association-proxy evidence as
  causal.** The `criterion` key feeds the causal axis (and
  `FeatureCard.causal_effect`) only with intervention provenance; records-backend
  cards without interventions now report `causal_effect = 0.0`.
- **The text-cosine association fallback lost its free ~0.5 baseline** — a
  no-evidence card no longer outranks a measured-but-weak one.
- **Graph edges labeled `measured_intervention` never carry a correlational
  signed effect** (no `signed_association` fallback in the measured branch), and
  supernode `aggregate_causal_effect` edges average intervention-backed members
  only (with `measured_member_count` + `evidence` fields).
- **Goodfire and Neuronpedia adapters no longer fabricate `specificity`.**
  Goodfire's constant `0.5` is gone; Neuronpedia's autointerp explanation score
  moved to `metadata["autointerp_score"]` where it belongs.
- **`validate-attribution-graph --allow-missing-controls` is honest about it**:
  robust paths with zero control records get reason code
  `passed_effect_and_sign_thresholds_no_controls` and matching interpretation
  text instead of claiming they "beat controls".
- **`t_critical` is no longer overconfident at small df for non-0.95
  confidence**: exact two-sided tables for 0.90 and 0.99 (df 1–30), conservative
  round-up for uncovered confidences below df 10 (0.95 behavior unchanged).

### Fixed — intervention & export correctness

- **Multi-piece target tokens are scored on the first BPE piece, not the last**
  (`" centimeters"` now measures P(" cent"), not P("imeters")), with the
  resolved id→token mapping recorded in row metadata
  (`resolved_target_token_ids`).
- **Fallback-dictionary SAE artifacts encode exactly what their metrics
  describe** (`encoder_bias = -l1_coefficient`; metrics derived through
  `encode_with_artifact`).
- **`train-sae --method auto` no longer silently downgrades to the fallback
  dictionary** on a broken torch install; torch availability is detected up
  front, training errors propagate, and the auto fallback prints an advisory.
- **Layer-0 (embedding) features are rejected up front** in intervention
  exports, before model load — no more truncated JSONL after a late crash.
- **Cross-backend layer numbering is unified**: nnsight/TransformerLens
  resid-post exports now use the HF `hidden_states` convention
  (block *i* → layer *i*+1) and all three backends stamp
  `layer_convention: hidden_state_index` in feature metadata.

### Fixed — agent surfaces & artifacts

- **`search-features` next actions are runnable**: the embedded `intervene`
  argv now includes the required `--model`/`--criterion`/`--dataset`/`--out`
  arguments (placeholder convention matching inspection-report actions), plus a
  new `agent_next_action_requires` field. A new meta-test asserts every emitted
  next-action argv parses against the real CLI parser.
- **`public_api_contract()` lists every required parameter** (added
  `validate_hf_sae_paths`'s `path_records_out`), enforced by a new
  introspection meta-test.
- **`interp-lab run` writes its manifest into the rendered output directory**
  when `out` contains `--var` templates (no more literal `reports/${name}/`).
- **The demo-sweep internal-command allowlist is derived from the CLI parser**
  (eight commands had drifted out of it), and verify-only `demo-sweep` no longer
  clobbers the archived release-evidence report at the default `--out`.
- **`doctor` reports the actually-configured text embedder** (flag and
  programmatic configuration included, not just the env var), and
  `compare-runs --markdown-out` works without `--out`.
- Newline-containing tokens render escaped (`\n`) in report and graph token
  displays; toy backend tolerates criteria without examples; Neuronpedia adapter
  skips non-numeric API values instead of crashing; NLA JSONL errors include
  `file:line` context.

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
