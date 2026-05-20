# Golden real-model demo — DistilGPT-2 (archived run)

These are **real artifacts** from an end-to-end interp-lab run on `distilgpt2`, not
toy data. They are the trust anchor for the core loop described in
[`docs/GOLDEN_REAL_MODEL_DEMO.md`](../../../docs/GOLDEN_REAL_MODEL_DEMO.md): train an
SAE from real Hugging Face activations, rank its latents against a criterion,
suppress the top features, re-measure with causal interventions, and export an
attribution graph.

**Criterion:** *"the next token should be a physical measurement unit"*

## How it was produced

Run on Modal (CPU) so it is cheap and reproducible, with semantic text embeddings
enabled:

```bash
modal run --detach examples/modal_golden_demo.py
```

- Model: `distilgpt2`, residual stream at layer 6.
- SAE: trained here (`minimal` preset, latent-dim 32, 20 epochs) from 12 scored prompts.
- Text embedder: **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim) — every
  fingerprint in the reports records `"text_embedder": "st-all-MiniLM-L6-v2"`, so this
  run also exercises the semantic matching path on a real model.

## What the numbers say

The top-ranked latent is a genuine criterion promoter with measured causal effect:

| feature   | role               | association | causal effect | strong-causal |
|-----------|--------------------|------------:|--------------:|--------------:|
| SAE:L6:F10 | criterion_promoter |       0.765 |         0.226 |         0.224 |
| SAE:L6:F30 | —                  |       0.962 |         0.016 |             — |
| SAE:L6:F13 | —                  |       0.924 |         0.023 |             — |

`interventions.jsonl` shows an authentic dose-response: suppressing F10/F13/F30 lowers
the probability mass on auto-derived unit tokens (` feet`, ` meters`, ` miles`, ` km`, …)
monotonically with strength (mean directed effect ≈ −0.02 → −0.05 → −0.09 at strengths
1 / 3 / 10), while side effects stay near zero. Note how F30 has the highest *association*
but almost no *causal* effect — exactly the correlational-vs-causal distinction the tool
is built to surface.

## Files to open

- `inspect-causal/report.html` — feature cards with causal evidence (start here).
- `graph.html` — attribution graph viewer.
- `graph-summary.json` — compact, agent/script-readable summary.
- `interventions.jsonl` — the raw suppression measurements.

## Caveat

This is a **compact** end-to-end check (12 prompts, latent-dim 32, single layer), meant
to prove the workflow, causal loop, and graph export are wired correctly on a real model.
For research claims, scale the prompt count, SAE width, layers, training rows, and add
held-out path validation.
