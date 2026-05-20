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
- Crosscoder feature import.
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
- Model-family comparison reports.

## Next Research And Engineering Milestones

- Text-pivot cross-model matching.
- Match calibration against held-out intervention transfer.
- Distributed SAE training manifests.
- Remote causal validation workers.
- Public example gallery with archived real-model reports.
