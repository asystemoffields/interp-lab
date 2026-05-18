# Real Model Smoke Test

This smoke test uses real Hugging Face models and a small physical-measurement criterion.

## Dataset

Two prompt sets are included:

- `examples/hf_prompts_measurements.jsonl`: full measurement sentences vs ordinary sentences.
- `examples/hf_prompts_unit_prediction.jsonl`: prefixes ending after a number, scored as prompts where the next token should likely be a measurement unit.

## Commands

Export hidden-state activation records:

```bash
oracle-sae export-hf-records \
  --model distilgpt2 \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --out reports/real-small/distilgpt2-unit/records.jsonl \
  --features-per-layer 12 \
  --pool last \
  --max-length 64
```

Inspect criterion-associated features:

```bash
oracle-sae inspect \
  --model distilgpt2 \
  --criterion "the next token should be a physical measurement unit" \
  --backend records \
  --records reports/real-small/distilgpt2-unit/records.jsonl \
  --out reports/real-small/distilgpt2-unit/inspect
```

Export real ablation records:

```bash
oracle-sae export-hf-interventions \
  --model distilgpt2 \
  --report reports/real-small/distilgpt2-unit/inspect/report.json \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --criterion "the next token should be a physical measurement unit" \
  --top-k 6 \
  --out reports/real-small/distilgpt2-unit/interventions.jsonl \
  --max-length 64
```

Inspect only features with intervention evidence:

```bash
oracle-sae inspect \
  --model distilgpt2 \
  --criterion "the next token should be a physical measurement unit" \
  --backend records \
  --records reports/real-small/distilgpt2-unit/records.jsonl \
  --interventions reports/real-small/distilgpt2-unit/interventions.jsonl \
  --require-interventions \
  --out reports/real-small/distilgpt2-unit/inspect-causal-required
```

Export a contrast-direction feature with a specificity-aware steering sweep:

```bash
oracle-sae export-hf-contrast \
  --model distilgpt2 \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --criterion "the next token should be a physical measurement unit" \
  --records-out reports/real-small/distilgpt2-unit/contrast-records-specific-sweep.jsonl \
  --interventions-out reports/real-small/distilgpt2-unit/contrast-interventions-specific-sweep.jsonl \
  --strength-sweep "3,10,30,100" \
  --max-length 64
```

Then inspect it:

```bash
oracle-sae inspect \
  --model distilgpt2 \
  --criterion "the next token should be a physical measurement unit" \
  --backend records \
  --records reports/real-small/distilgpt2-unit/contrast-records-specific-sweep.jsonl \
  --interventions reports/real-small/distilgpt2-unit/contrast-interventions-specific-sweep.jsonl \
  --require-interventions \
  --out reports/real-small/distilgpt2-unit/inspect-contrast-specific-sweep
```

Train an SAE directly from model activations:

```bash
oracle-sae train-sae \
  --preset minimal \
  --hf-model distilgpt2 \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --layer 6 \
  --latent-dim 64 \
  --epochs 50 \
  --batch-size 6 \
  --lr 0.003 \
  --l1 0.001 \
  --out reports/real-small/distilgpt2-unit/trained-sae/sae.json \
  --records-out reports/real-small/distilgpt2-unit/trained-sae/records.jsonl \
  --max-length 64
```

Run the production-oriented path with token rows, top-k sparse codes, and causal validation:

```bash
oracle-sae train-sae \
  --preset production \
  --hf-model distilgpt2 \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --layer 6 \
  --latent-dim 64 \
  --epochs 30 \
  --batch-size 32 \
  --max-length 64 \
  --causal-top-k 4 \
  --out reports/real-small/distilgpt2-unit/trained-sae-production/sae.json \
  --records-out reports/real-small/distilgpt2-unit/trained-sae-production/records.jsonl \
  --causal-out reports/real-small/distilgpt2-unit/trained-sae-production/interventions.jsonl \
  --criterion "the next token should be a physical measurement unit" \
  --causal-strength-sweep "3,10,30"
```

Inspect the trained SAE latents:

```bash
oracle-sae inspect \
  --model distilgpt2 \
  --criterion "the next token should be a physical measurement unit" \
  --backend records \
  --records reports/real-small/distilgpt2-unit/trained-sae/records.jsonl \
  --out reports/real-small/distilgpt2-unit/trained-sae/inspect
```

Compare DistilGPT-2 and GPT-2 reports:

```bash
oracle-sae match \
  --left reports/real-small/distilgpt2-unit/inspect/report.json \
  --right reports/real-small/gpt2-unit/inspect/report.json \
  --top-k 12 \
  --out reports/real-small/distilgpt2-vs-gpt2-unit/matches.json
```

## Observations

The association pass found strong criterion-correlated hidden dimensions in real models. GPT-2 and DistilGPT-2 surfaced overlapping final-layer dimensions for the unit-prediction criterion, including dimensions `314`, `643`, `260`, `644`, and `652`.

The causal ablation pass was more conservative. Ablating single raw hidden dimensions barely changed the probability mass on physical-unit tokens. Grouped top-k ablations moved the behavior more than individual dimensions, but remained weaker than a learned contrast direction.

The contrast-direction pass produced stronger and more specific causal evidence:

- DistilGPT-2 selected steering strength `10`, with mean positive-prompt directed effect `0.078` and mean side effect `0.001`.
- GPT-2 selected steering strength `30`, with mean positive-prompt directed effect `0.070` and mean side effect `0.001`.
- Strength `100` moved unrelated prompts too much, so the specificity-aware sweep rejected it for these runs.

The on-demand SAE training pass also worked on the same tiny dataset:

- DistilGPT-2 layer 6 trained a 64-latent SAE from 768-dimensional hidden states.
- The PyTorch SAE reached reconstruction MSE `0.0020` and average L0 `24.5`.
- The exported SAE activation records wrote all 64 latents per prompt and inspected cleanly with the standard records backend.

The production-oriented SAE smoke run trained on token-level rows with top-k sparse codes:

- token mode `all`, sparsity `topk`, top-k `32`;
- reconstruction MSE `0.9318`, validation MSE `0.8468`, average L0 `27.64`, dead latents `0`;
- SAE latent `SAE:L6:F58` produced mean directed effect `0.080` and mean side effect `0.001` under decoder steering.

The cross-model matcher handled both cases:

- Association reports produced high-scoring candidate equivalents across the two GPT-style models.
- Causal-required reports lowered the causal component when intervention effects were near zero.
- Contrast-direction reports matched the DistilGPT-2 and GPT-2 criterion directions with aligned signed effects.

## Interpretation

This validates the tool path on real small models:

1. Export real hidden activations.
2. Rank features by criterion association.
3. Export intervention records.
4. Re-rank with causal evidence.
5. Match candidate equivalents across models.

It also shows the useful split between two modes:

- Raw hidden dimensions are convenient smoke-test units and often reveal associated structure.
- Learned directions, SAE features, or crosscoder features are stronger candidates for causal steering and cross-model equivalence.
