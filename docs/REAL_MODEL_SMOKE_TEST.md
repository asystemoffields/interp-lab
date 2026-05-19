# Real Model Smoke Test

This smoke test uses real Hugging Face models and a small physical-measurement criterion.

## Dataset

Two prompt sets are included:

- `examples/hf_prompts_measurements.jsonl`: full measurement sentences vs ordinary sentences.
- `examples/hf_prompts_unit_prediction.jsonl`: prefixes ending after a number, scored as prompts where the next token should likely be a measurement unit.

## Commands

Export hidden-state activation records:

```bash
interp-lab export-hf-records \
  --model distilgpt2 \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --out reports/real-small/distilgpt2-unit/records.jsonl \
  --features-per-layer 12 \
  --pool last \
  --max-length 64
```

Inspect criterion-associated features:

```bash
interp-lab inspect \
  --model distilgpt2 \
  --criterion "the next token should be a physical measurement unit" \
  --backend records \
  --records reports/real-small/distilgpt2-unit/records.jsonl \
  --out reports/real-small/distilgpt2-unit/inspect
```

Export real ablation records:

```bash
interp-lab export-hf-interventions \
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
interp-lab inspect \
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
interp-lab export-hf-contrast \
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
interp-lab inspect \
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
interp-lab train-sae \
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
interp-lab train-sae \
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
interp-lab inspect \
  --model distilgpt2 \
  --criterion "the next token should be a physical measurement unit" \
  --backend records \
  --records reports/real-small/distilgpt2-unit/trained-sae/records.jsonl \
  --out reports/real-small/distilgpt2-unit/trained-sae/inspect
```

Run an SAE path-patching smoke test with a tiny local/open model:

```bash
interp-lab train-sae \
  --preset minimal \
  --hf-model sshleifer/tiny-gpt2 \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --layer 1 \
  --latent-dim 4 \
  --epochs 5 \
  --batch-size 4 \
  --method torch \
  --out reports/real-small/tiny-gpt2-unit/sae-layer1/sae.json \
  --records-out reports/real-small/tiny-gpt2-unit/sae-layer1/records.jsonl \
  --max-length 32

interp-lab train-sae \
  --preset minimal \
  --hf-model sshleifer/tiny-gpt2 \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --layer 2 \
  --latent-dim 4 \
  --epochs 5 \
  --batch-size 4 \
  --method torch \
  --out reports/real-small/tiny-gpt2-unit/sae-layer2/sae.json \
  --records-out reports/real-small/tiny-gpt2-unit/sae-layer2/records.jsonl \
  --max-length 32
```

Inspect both SAE layers:

```bash
interp-lab inspect \
  --model sshleifer/tiny-gpt2 \
  --criterion "the next token should be a physical measurement unit" \
  --backend records \
  --records reports/real-small/tiny-gpt2-unit/sae-layer1/records.jsonl \
  --top-k 4 \
  --out reports/real-small/tiny-gpt2-unit/sae-layer1/report

interp-lab inspect \
  --model sshleifer/tiny-gpt2 \
  --criterion "the next token should be a physical measurement unit" \
  --backend records \
  --records reports/real-small/tiny-gpt2-unit/sae-layer2/records.jsonl \
  --top-k 4 \
  --out reports/real-small/tiny-gpt2-unit/sae-layer2/report
```

Patch source SAE latents from layer 1 and measure downstream layer-2 SAE latents:

```bash
interp-lab export-hf-sae-paths \
  --model sshleifer/tiny-gpt2 \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --criterion "the next token should be a physical measurement unit" \
  --source-sae reports/real-small/tiny-gpt2-unit/sae-layer1/sae.json \
  --target-sae reports/real-small/tiny-gpt2-unit/sae-layer2/sae.json \
  --source-report reports/real-small/tiny-gpt2-unit/sae-layer1/report/report.json \
  --target-report reports/real-small/tiny-gpt2-unit/sae-layer2/report/report.json \
  --source-top-k 2 \
  --target-top-k 2 \
  --skip-behavior-score \
  --strength-sweep=-2,2 \
  --random-source-controls 1 \
  --max-length 32 \
  --out reports/real-small/tiny-gpt2-unit/paths/layer1-to-layer2.jsonl
```

Optionally include output behavior scoring:

```bash
interp-lab export-hf-sae-paths \
  --model sshleifer/tiny-gpt2 \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --criterion "the next token should be a physical measurement unit" \
  --source-sae reports/real-small/tiny-gpt2-unit/sae-layer1/sae.json \
  --target-sae reports/real-small/tiny-gpt2-unit/sae-layer2/sae.json \
  --source-report reports/real-small/tiny-gpt2-unit/sae-layer1/report/report.json \
  --target-report reports/real-small/tiny-gpt2-unit/sae-layer2/report/report.json \
  --source-top-k 1 \
  --target-top-k 1 \
  --target-token auto \
  --strength-sweep=2 \
  --max-length 32 \
  --out reports/real-small/tiny-gpt2-unit/paths/layer1-to-layer2-behavior.jsonl
```

Fuse the layer reports and measured path records into an attribution graph:

```bash
interp-lab export-attribution-graph \
  --report reports/real-small/tiny-gpt2-unit/sae-layer1/report/report.json \
  --report reports/real-small/tiny-gpt2-unit/sae-layer2/report/report.json \
  --path-records reports/real-small/tiny-gpt2-unit/paths/layer1-to-layer2.jsonl \
  --path-records reports/real-small/tiny-gpt2-unit/paths/layer1-to-layer2-behavior.jsonl \
  --include-similarity-edges \
  --out reports/real-small/tiny-gpt2-unit/paths/graph.json
```

Compare DistilGPT-2 and GPT-2 reports:

```bash
interp-lab match \
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

The tiny GPT-2 path-patching smoke run validated the local/open attribution loop:

- layer-1 and layer-2 four-latent SAEs trained successfully on CPU;
- `export-hf-sae-paths` wrote `96` source-to-target effect rows for two source latents, two target latents, two steering strengths, and twelve prompts;
- with one random-source control per selected source latent, it also wrote `96` matched control rows;
- the strongest measured internal path in the fused graph had mean absolute target-latent delta about `0.62`, while the random-source controls moved the target latents by a similar amount on this tiny sanity setup;
- the fused graph reports path specificity beside raw target-latent movement;
- behavior-scored path rows also wrote successfully, with tiny output score movement on `sshleifer/tiny-gpt2`, as expected for a very small sanity model.

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
6. Patch source SAE latents and measure downstream SAE latent paths.
7. Fuse reports and path records into an attribution graph.

It also shows the useful split between two modes:

- Raw hidden dimensions are convenient smoke-test units and often reveal associated structure.
- Learned directions, SAE features, or crosscoder features are stronger candidates for causal steering and cross-model equivalence.
- Path-patching records are the bridge from ranked feature cards to circuit-style claims: they measure whether one feature intervention changes another feature downstream.
