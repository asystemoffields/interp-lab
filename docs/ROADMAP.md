# Roadmap

The north star:

> Give interp-lab a model and a criterion in natural language. It returns the internal features most responsible for that criterion, explains them, tests them causally, and, when desired, searches for equivalent features across other models.

## Shipped In 1.0

- CLI for `inspect`, `match`, `validate-matches`, and `demo`.
- Typed feature evidence, feature cards, reports, and match reports.
- JSONL feature import.
- Activation records backend for per-prompt feature activations.
- Deterministic toy backend.
- Fingerprint similarity across models.
- Match validation with claim grades and agent next actions.
- Self-contained HTML feature-card reports.
- Self-contained HTML match-validation viewer.
- Self-contained Studio frontend for CLI command and run-config generation.
- Markdown and JSON reports.
- `SAELens` provider for public sparse autoencoders.
- Neuronpedia feature import.
- Goodfire feature import.
- Gemma Scope and Qwen-Scope wrappers.
- TransformerLens activation export.
- NNsight activation export.
- Crosscoder latent import via the model-agnostic activation-records path (a dedicated `CrosscoderFeatureProvider` is a planned provider — see `ARCHITECTURE.md`).
- Intervention records import.
- Hidden-dimension ablation export.
- Contrast-direction steering export.
- SAE-latent steering export.
- SAE-latent path patching across layers.
- Side-effect suite with unrelated criteria.
- Control rows and confidence intervals.
- Attribution graph export with candidate feature groups, coactivation paths, and validation next steps.
- Config files for repeatable runs.
- Dataset manifests and anchor prompt sets.
- Streaming activation-record ranking.
- Scale planning for large model runs.
- Hugging Face artifact publishing.
- HTML feature cards.

## Available On Main After 1.0

- Natural Language Autoencoder verbalizer adapter.
- Explanation consistency checks across paraphrases.
- Explanation-to-feature search.
- Text-pivot cross-model matching.
- Model-family comparison reports.

## Shipped In 2.2 — usability, robustness & new tools

- `compare-runs` — diff two inspection reports for rank drift and regressions.
- `quickstart`/`tutorial` guided walkthrough; grouped `--help`; `--version`.
- `inspect --csv-out` and an inline importance-by-layer chart in the HTML report.
- Notebook-friendly, chainable Python API (`Written*` results chain into `compare`/`validate_matches`; `cards_table()`/`matches_table()`; top-level loaders; `py.typed`).
- `match --min-score`/`--weights` matcher tuning.
- Friendly CLI errors and real-world-file hardening (UTF-8 BOM, literal braces, non-finite rejection), plus a demo that demonstrates a measured causal claim end-to-end.

## Next Research And Engineering Milestones

- Match calibration against held-out intervention transfer.
- Distributed SAE training manifests.
- Remote causal validation workers.
- Public example gallery with archived real-model reports.
