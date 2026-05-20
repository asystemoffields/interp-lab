# Golden Real-Model Demo

This is the release-gate demo for the core loop on a small open model:

1. Prepare scored prompts.
2. Train a small SAE from real Hugging Face activations.
3. Inspect learned latents for a natural-language criterion.
4. Suppress top features with an agent-reviewable intervention plan.
5. Re-inspect with causal evidence.
6. Export an attribution graph and compact summary.

The example uses DistilGPT-2 and a physical-unit next-token criterion so it can run on CPU.

## Commands

```bash
mkdir -p reports/golden-real/distilgpt2-unit

interp-lab prepare-sae-prompts \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --out-dir reports/golden-real/distilgpt2-unit/prompt-pack \
  --latent-dim 32 \
  --max-length 64

interp-lab train-sae \
  --preset minimal \
  --hf-model distilgpt2 \
  --dataset reports/golden-real/distilgpt2-unit/prompt-pack/train.jsonl \
  --layer 6 \
  --latent-dim 32 \
  --epochs 20 \
  --batch-size 6 \
  --lr 0.003 \
  --l1 0.001 \
  --out reports/golden-real/distilgpt2-unit/sae/sae.json \
  --records-out reports/golden-real/distilgpt2-unit/sae/records.jsonl \
  --max-length 64

interp-lab inspect \
  --model distilgpt2 \
  --criterion "the next token should be a physical measurement unit" \
  --backend records \
  --records reports/golden-real/distilgpt2-unit/sae/records.jsonl \
  --out reports/golden-real/distilgpt2-unit/inspect \
  --html-out reports/golden-real/distilgpt2-unit/inspect/report.html

interp-lab intervene \
  --model distilgpt2 \
  --dataset reports/golden-real/distilgpt2-unit/prompt-pack/causal.jsonl \
  --criterion "the next token should be a physical measurement unit" \
  --report reports/golden-real/distilgpt2-unit/inspect/report.json \
  --records reports/golden-real/distilgpt2-unit/sae/records.jsonl \
  --top-k 3 \
  --sae reports/golden-real/distilgpt2-unit/sae/sae.json \
  --mode suppress \
  --strength-sweep "1,3,10" \
  --target-token auto \
  --max-length 64 \
  --out reports/golden-real/distilgpt2-unit/interventions.jsonl \
  --plan-out reports/golden-real/distilgpt2-unit/intervention-plan.json \
  --dry-run \
  --json

interp-lab intervene \
  --model distilgpt2 \
  --dataset reports/golden-real/distilgpt2-unit/prompt-pack/causal.jsonl \
  --criterion "the next token should be a physical measurement unit" \
  --report reports/golden-real/distilgpt2-unit/inspect/report.json \
  --records reports/golden-real/distilgpt2-unit/sae/records.jsonl \
  --top-k 3 \
  --sae reports/golden-real/distilgpt2-unit/sae/sae.json \
  --mode suppress \
  --strength-sweep "1,3,10" \
  --target-token auto \
  --max-length 64 \
  --out reports/golden-real/distilgpt2-unit/interventions.jsonl \
  --plan-out reports/golden-real/distilgpt2-unit/intervention-plan.json

interp-lab inspect \
  --model distilgpt2 \
  --criterion "the next token should be a physical measurement unit" \
  --backend records \
  --records reports/golden-real/distilgpt2-unit/sae/records.jsonl \
  --interventions reports/golden-real/distilgpt2-unit/interventions.jsonl \
  --require-interventions \
  --out reports/golden-real/distilgpt2-unit/inspect-causal \
  --html-out reports/golden-real/distilgpt2-unit/inspect-causal/report.html

interp-lab export-attribution-graph \
  --report reports/golden-real/distilgpt2-unit/inspect-causal/report.json \
  --out reports/golden-real/distilgpt2-unit/graph.json \
  --markdown-out reports/golden-real/distilgpt2-unit/graph.md \
  --html-out reports/golden-real/distilgpt2-unit/graph.html

interp-lab summarize-attribution-graph \
  --graph reports/golden-real/distilgpt2-unit/graph.json \
  --out reports/golden-real/distilgpt2-unit/graph-summary.json
```

## What To Check

The intervention dry run should write a JSON plan with:

- selected `SAE:L...:F...` features from the report;
- `estimated_forward_passes`;
- advisories about small prompt packs when applicable;
- `agent_actions[0].argv` containing `inspect --backend records --records ... --interventions ...`.

The final artifacts to open are:

- `reports/golden-real/distilgpt2-unit/inspect-causal/report.html`
- `reports/golden-real/distilgpt2-unit/graph.html`
- `reports/golden-real/distilgpt2-unit/graph-summary.json`

For real research claims, increase prompt count, SAE width, layers, training rows, and held-out validation. This demo is a compact end-to-end check that the workflow, reports, causal evidence loop, graph export, and agent-facing plans are wired correctly.
