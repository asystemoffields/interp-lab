# Contributing

interp-lab is early, and the best contributions make one part of the inspection loop more real.

## High-Value Contributions

- Add a feature provider for `SAELens`, Neuronpedia, crosscoders, or a lab-specific feature dump.
- Add an intervention runner for ablation, amplification, clamping, activation patching, or steering.
- Add a verbalizer backed by a Natural Language Autoencoder or another activation-to-text method.
- Improve criterion compilation from natural-language goals into datasets and counterfactuals.
- Add report views that help researchers inspect evidence quickly.

## Development Setup

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Adapter Guidelines

Adapters should return stable `FeatureEvidence` objects and put backend-specific details in `metadata`.

Keep these fields meaningful:

- `feature_id`
- `model`
- `layer`
- `label`
- `examples`
- `activation_signature`
- `decoder_signature`
- `causal_effects`
- `source`

## Tests

Add focused tests for scoring, matching, file formats, and CLI behavior. When adding a backend that depends on large models, keep a small fixture or mock path so CI stays fast.
