# Roadmap

The north star:

> Give Oracle SAE a model and a criterion in natural language. It returns the internal features most responsible for that criterion, explains them, tests them causally, and, when desired, searches for equivalent features across other models.

## Milestone 1: Useful Local Skeleton

- CLI for `inspect`, `match`, and `demo`.
- Typed feature evidence, feature cards, reports, and match reports.
- JSONL feature import.
- Activation records backend for per-prompt feature activations.
- Deterministic toy backend.
- Fingerprint similarity across models.
- Markdown and JSON reports.

## Milestone 2: Real Feature Sources

- `SAELens` provider for public sparse autoencoders.
- Neuronpedia feature import.
- Crosscoder feature import.
- Activation cache import from `TransformerLens` or `nnsight`.
- Conversion scripts from activation caches to activation records.

## Milestone 3: Causal Testing

- Intervention records import.
- Ablation runner.
- Amplification runner.
- Activation patching runner.
- Clamp-to-feature-value runner.
- Side-effect suite with unrelated criteria.

## Milestone 4: Natural-Language Oracles

- NLA verbalizer adapter.
- Explanation consistency checks across paraphrases.
- Explanation-to-feature search.
- Text-pivot cross-model matching.

## Milestone 5: Research-Grade Robustness

- Config files for repeatable runs.
- Dataset manifests and anchor prompt sets.
- HTML feature cards.
- Match calibration against held-out intervention transfer.
- Model-family comparison reports.
