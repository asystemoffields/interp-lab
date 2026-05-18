# Architecture

interp-lab is organized around one idea: every candidate feature should produce a comparable fingerprint and a falsifiable causal claim.

## Pipeline

```text
criterion text
  -> CriterionCompiler
  -> FeatureProvider
  -> Verbalizer
  -> InterventionRunner
  -> FeatureFingerprint
  -> FeatureCard
```

The same feature card can come from a sparse autoencoder, crosscoder, natural-language autoencoder, manual feature dump, or a future activation oracle.

## Feature Providers

A feature provider returns `FeatureEvidence`.

Current providers:

- `ToyFeatureProvider`: deterministic demo evidence.
- `JsonlFeatureProvider`: feature evidence imported from JSONL.
- `ActivationRecordFeatureProvider`: per-prompt feature activations aggregated into feature evidence.
- `NeuronpediaFeatureProvider`: public Neuronpedia feature dashboards imported as evidence.
- `SAELensFeatureProvider`: selected pretrained SAE Lens features imported as evidence.
- `GoodfireFeatureProvider`: semantically searched Goodfire features imported as evidence.
- `ScopeFeatureProvider`: named wrappers for Gemma Scope and Qwen-Scope metadata.

Planned providers:

- `CrosscoderFeatureProvider`
- `NLAFeatureProvider`

## Activation Records

Activation records make the tool useful across model families before a library-specific adapter exists. Each row contains a prompt or token position, a criterion score, and a set of feature activations.

The provider estimates:

- signed activation/criterion correlation,
- mean positive-vs-negative separation,
- top activating examples,
- activation signatures over anchor prompts.

Those estimates become `causal_effects` and fingerprint components. Direct interventions should still be run when a model adapter is available.

The provider streams JSONL and keeps per-feature sufficient statistics, compact signatures, and top examples. This keeps the ranking path usable for sharded activation corpora produced by large remote harvesters.

Current activation exporters:

- Hugging Face hidden states.
- TransformerLens hook caches.
- NNsight trace paths.
- HF contrast directions.
- On-demand SAE latent activations.

## Verbalizers

A verbalizer turns feature evidence into an explanation. In the toy backend this is a short heuristic sentence. A real NLA adapter can generate explanations directly from residual-stream activations or from candidate feature activations.

## Intervention Runners

An intervention runner estimates the behavioral effect of changing a feature.

The score schema is intentionally simple:

- `criterion`: how much the feature moves the requested criterion.
- `specificity`: how targeted the effect appears.
- `side_effect`: how much unrelated behavior moves.
- `strong_causal_score`: criterion effect adjusted by side effects and control interventions.
- `criterion_ci_low` / `criterion_ci_high`: a normal-approximation confidence interval over intervention rows.

Runners can implement ablation, amplification, clamping, activation patching, or text-level estimates.

Current runners:

- `ToyInterventionRunner`: passes through demo or imported causal scores.
- `InterventionRecordRunner`: aggregates external ablation, amplification, clamp, patch, and steering records.

Intervention records are model-agnostic. A model-specific backend can produce them today, and interp-lab will fold them into feature ranking and reports.

The Hugging Face exporters currently generate two useful intervention families:

- hidden-dimension ablations, including grouped top-k ablations;
- contrast-direction steering, with optional strength sweeps that select the setting with the best criterion effect minus side-effect movement.

Prompt datasets with `criterion_score` support specificity checks: positive-scored prompts estimate the requested criterion effect, and negative-scored prompts estimate side effects.

Intervention records can also carry controls through `metadata.control_type`, with values such as `random_feature`, `matched_frequency`, or `placebo`. Reports preserve target effects, control effects, and confidence intervals.

## On-Demand SAE Training

When a model has no public SAE, `train-sae` can create one from:

- existing activation-record JSONL;
- Hugging Face hidden states collected directly from a model and prompt set.

The trainer writes two artifacts:

- an SAE artifact JSON with mean vector, encoder weights, decoder weights, source feature ids, config, and metrics;
- optional activation records for learned SAE latents, ready for the normal `inspect --backend records` pipeline.

Training methods:

- `auto`: use PyTorch when available, then fall back.
- `torch`: optimize a ReLU SAE with MSE plus L1 activation penalty.
- `fallback`: build a deterministic sparse dictionary from high-variance source dimensions plus seeded random directions.

Training presets:

- `minimal`: quick iteration, one row per prompt, small default expansion, ReLU plus L1 sparsity.
- `production`: token-level activation rows, top-k sparse codes, larger default expansion, held-out metrics, dead-latent reporting, and optional SAE-latent steering validation.
- `custom`: preserve the lower-level knobs for targeted experiments.

The PyTorch trainer supports `relu-l1`, `topk`, and `jumprelu` sparse code rules. Artifacts store train and validation reconstruction MSE, average L0, per-latent firing rates, dead-latent indices, and active-latent fraction. JSONL inputs can be bounded with deterministic reservoir sampling through `--max-records`.

When `--causal-out` is supplied for an HF-trained SAE, the trainer ranks learned latents by criterion association, steers along their decoder directions, sweeps strengths, rejects high-side-effect settings, and writes intervention records for the normal causal report path.

This keeps the user path continuous: no public SAE -> collect activations -> train SAE -> rank learned latents -> run causal tests -> compare features across models.

## Matching

Cross-model matching compares fingerprints:

```text
score =
  0.35 * text similarity
+ 0.25 * activation signature similarity
+ 0.20 * decoder signature similarity
+ 0.20 * causal vector similarity
```

The weights are defaults. Real projects should tune them against held-out transfer tests.

Match reports preserve labels and signed effects. Candidate equivalents with opposite signed causal effects are downgraded by the causal fingerprint component and surfaced in the markdown report.

## Attribution Graphs

`export-attribution-graph` converts a report into a graph JSON with:

- a criterion node;
- feature nodes;
- feature-to-criterion causal edges;
- optional feature-to-feature fingerprint-similarity edges.

The graph schema keeps effect sizes, signed effects, specificity, side effects, strong causal scores, confidence intervals, and intervention record counts.

## Scaling Model

For very large models, interp-lab treats model execution as an adapter concern. A 1T+ model can harvest activations through a colocated runtime, Goodfire-style API, NNsight remote execution, or a custom cluster job. The stable interchange layer is sharded activation records, SAE artifacts, intervention records, and manifests.

`plan-scale` estimates activation storage, per-shard size, SAE parameter storage, and recommended execution shape before a harvesting run begins.

## Adapter Contract

Adapters should keep raw, backend-specific details in `metadata` and populate these stable fields:

- `feature_id`
- `model`
- `layer`
- `label`
- `examples`
- `activation_signature`
- `decoder_signature`
- `causal_effects`
- `source`

That gives the rest of the tool a stable surface while the research layer keeps moving.
