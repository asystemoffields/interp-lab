# Roadmap

The north star:

> Give interp-lab a model and a criterion in natural language. It returns the internal features most responsible for that criterion, explains them, tests them causally, and, when desired, searches for equivalent features across other models.

## Milestone 1: Useful Local Skeleton

- CLI for `inspect`, `match`, `validate-matches`, and `demo`.
- Typed feature evidence, feature cards, reports, and match reports.
- JSONL feature import.
- Activation records backend for per-prompt feature activations.
- Deterministic toy backend.
- Fingerprint similarity across models.
- Match validation with claim grades and agent next actions.
- Markdown and JSON reports.

## Milestone 2: Real Feature Sources

- `SAELens` provider for public sparse autoencoders.
- Neuronpedia feature import.
- Goodfire feature import.
- Gemma Scope and Qwen-Scope wrappers.
- TransformerLens activation export.
- NNsight activation export.
- Crosscoder feature import.

## Milestone 3: Causal Testing

- Intervention records import.
- Hidden-dimension ablation export.
- Contrast-direction steering export.
- SAE-latent steering export.
- SAE-latent path patching across layers.
- Side-effect suite with unrelated criteria.
- Control rows and confidence intervals.
- Attribution graph export with candidate feature groups, coactivation paths, and validation next steps.

## Milestone 4: Natural-Language Oracles

- NLA verbalizer adapter.
- Explanation consistency checks across paraphrases.
- Explanation-to-feature search.
- Text-pivot cross-model matching.

## Milestone 5: Research-Grade Robustness

- Config files for repeatable runs.
- Dataset manifests and anchor prompt sets.
- Streaming activation-record ranking.
- Scale planning for large model runs.
- Hugging Face artifact publishing.
- HTML feature cards.
- Match calibration against held-out intervention transfer.
- Model-family comparison reports.
- Distributed SAE training manifests.
- Remote causal validation workers.
