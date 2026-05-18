# Goodfire Brief

Oracle SAE / Interp Lab is an open-source criterion-driven interpretability tool.

The target workflow:

> Give it a model and a criterion in natural language. It returns the internal features most responsible for that criterion, explains them, tests them causally, and searches for equivalent features in other models.

## What Works Now

- Feature ranking from JSONL feature dumps and activation records.
- Neuronpedia and SAE Lens feature import.
- Hugging Face hidden-state activation export.
- On-demand SAE training for models with no public SAE.
- Minimal and production SAE presets.
- Token-level activation collection.
- Top-k and JumpReLU-style sparse codes.
- Held-out reconstruction metrics and dead-latent reporting.
- SAE-latent causal validation through decoder-direction steering.
- Hidden-dimension ablations and contrast-direction steering.
- Specificity-aware strength sweeps using negative prompts as side-effect checks.
- Cross-model feature matching through text, activation, decoder, and causal fingerprints.
- Config-driven runs with manifests for reproducibility.

## Validated Smoke Path

The current real-model smoke path uses DistilGPT-2 and GPT-2 on a physical-unit prediction criterion.

The production SAE run trains on token-level rows, exports learned latent activations, runs decoder-direction steering, and feeds the resulting intervention records back into the normal inspection report.

Recent local result:

- model: `distilgpt2`
- layer: 6
- SAE width: 32 in the audit-fix smoke, 64 in the longer production smoke
- sparse code: top-k
- causal feature example: `SAE:L6:F30`
- directed effect: `0.077`
- specificity: `0.077`
- measured side effect: near zero on the tiny negative set

## Why It May Be Useful

The tool aims to be an orchestration layer for interpretability workflows:

- researchers can plug in better feature sources, SAEs, crosscoders, NLAs, and intervention runners;
- applied users can run a natural-language criterion against a model and get an evidence-ranked report;
- teams can compare candidate features across models and preserve provenance through manifests.

## Near-Term Gaps

- richer activation adapters for TransformerLens, nnsight, and local model wrappers;
- criterion dataset generation from natural language;
- causal validation suite with confidence intervals and controls;
- HTML reports for feature cards and cross-model matches;
- transfer tests for cross-model feature equivalence.

## Ask

Feedback on the design would be valuable, especially:

- whether the adapter contract matches real interpretability workflows;
- how to represent causal feature strength cleanly;
- what evidence would make cross-model feature equivalence persuasive;
- where this could complement existing feature-analysis and model-control tooling.
