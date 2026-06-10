# Quant-Diff: Which Validated Features Does Quantization Break?

Mixed-precision quantization work (e.g. PMRA-style tensor-level GGUF re-quants)
needs a sharper question than "did perplexity move?": **which of the features
you previously validated does the quantized variant no longer support?**

`quant-diff` answers it by running the same model at two precisions through the
records path, inspecting both against the **same criterion**, matching and
validating the feature pairs with interp-lab's existing cross-model machinery,
and reporting a per-feature verdict plus one headline list:
`degraded_validated` — the intervention-validated baseline features whose
causal claim the variant broke.

## 1. Export records at two precisions

The typical source is a GGUF model under llama.cpp, exported once per precision
(see `docs/GGUF_BRIDGE.md`):

```bash
# Baseline: FP16 GGUF (final-layer records via llama.cpp embeddings)
interp-lab export-gguf-records \
  --model models/llama-3-8b-f16.gguf \
  --dataset prompts.jsonl \
  --out records-f16.jsonl

# Variant: the quantized GGUF you are evaluating
interp-lab export-gguf-records \
  --model models/llama-3-8b-q4_k_m.gguf \
  --dataset prompts.jsonl \
  --out records-q4.jsonl
```

If you already have per-layer hidden-state dumps (from your own runtime, any
precision), convert them instead with `interp-lab convert-hidden-dump --dump
hidden.jsonl --out records.jsonl`. And for Hugging Face checkpoints the same
comparison works today at two dtypes:

```bash
interp-lab export-hf-records --model meta-llama/Llama-3-8B --torch-dtype float16 \
  --dataset prompts.jsonl --out records-f16.jsonl
interp-lab export-hf-records --model your/llama-3-8b-quantized --torch-dtype auto \
  --dataset prompts.jsonl --out records-q4.jsonl
```

Use the **same prompt dataset and the same criterion** for both exports —
quant-diff refuses cross-criterion comparisons because every feature would
trivially read as "broken".

## 2. Run the preset

`examples/presets/quant-diff-run.json` is a ready-made `interp-lab run` config.
JSON has no comments, so its variables are documented here:

| Variable | Meaning |
| --- | --- |
| `${left_records}` | Activation-record JSONL for the **baseline** (higher-precision) model. |
| `${right_records}` | Activation-record JSONL for the **quantized variant**. |
| `${criterion}` | The natural-language criterion, identical for both sides. |
| `${name}` | Run name; outputs land in `reports/quant-diff/${name}`. |

```bash
interp-lab run examples/presets/quant-diff-run.json \
  --var left_records=records-f16.jsonl \
  --var right_records=records-q4.jsonl \
  --var criterion="the model is aware it is being evaluated" \
  --var name=llama3-q4km
```

The preset's side labels default to `baseline`/`variant`; the config is an
editable template (the `init-run` house pattern), so set the `model` /
`*_label` args to real ids like `llama-3-8b-f16` / `llama-3-8b-q4_k_m` if you
want them stamped on the report. From Python,
`interp_lab.workflows.quant_diff_workflow(left_records, right_records,
criterion, out_dir, model_left=..., model_right=..., top_k=...)` builds the
same config with your labels filled in.

The run executes five steps: `inspect` (baseline, records backend) →
`inspect` (variant) → `match` → `validate-matches` → `quant-diff`, writing
`quant-diff.json` and `quant-diff.md` into the run directory. You can also call
`quant-diff` directly on two existing reports:

```bash
interp-lab quant-diff \
  --left-report reports/f16/report.json \
  --right-report reports/q4/report.json \
  --left-label f16 --right-label q4_k_m \
  --out reports/quant-diff.json
```

(Omit `--matches` / `--match-validation` and quant-diff computes them
in-process with the same matching and validation functions the CLI uses.)

## 3. Read the verdicts

The markdown opens with the only table that usually matters — **Features
broken by quantization** — followed by the full verdict table. Verdicts:

| Verdict | Meaning |
| --- | --- |
| `preserved` | Baseline feature was intervention-validated; the variant keeps comparable, same-direction causal evidence within thresholds. |
| `degraded` | Baseline feature was intervention-validated; the variant flipped its signed effect, lost the intervention evidence, washed the effect out, or dropped sharply in importance. **These populate `summary.degraded_validated`.** |
| `preserved_correlational` | The pair matches, but the baseline side only ever had correlational evidence — there was no validated causal claim to break. Never counted in `degraded_validated`. |
| `changed_correlational` | Correlational-only pair whose association moved (sign flip or large drop). A lead to investigate, not a broken validated feature. |
| `lost` | No acceptable match for the baseline feature in the variant report. If the baseline feature was intervention-validated it also appears in the broken-features table (`summary.validated_lost`). |
| `emerged` | Variant-only feature — quantization sometimes creates new behavior worth inspecting. |

### What "degraded" means evidence-wise

A feature is only ever `degraded` when the **baseline side carried real
intervention evidence** (`signed_causal_effect` / intervention records — the
same provenance accessors matching uses). The degradation reasons are:

- `signed_effect_direction_flipped` — same-provenance signed effects with
  opposite directions (both above the 0.02 noise floor). The variant's feature
  now pushes the criterion the other way.
- `right_lost_intervention_evidence` — the variant card has no intervention
  measurements; its signed association is a correlational proxy, which is
  **never compared numerically** against the baseline's measured effect
  (the pair is labeled `mixed_provenance_not_compared` instead).
- `signed_effect_magnitude_dropped` / `signed_effect_washed_out` — both sides
  are intervention-backed, but the variant's effect lost more than 0.15 of
  magnitude, or fell below the 0.02 floor entirely.
- `importance_dropped` — the variant's overall rank score fell by more than
  0.15 even though the signed effect still agrees.

### Verdict thresholds

Reproduced in every report under `summary.verdict_thresholds`:

| Threshold | Default | Why |
| --- | ---: | --- |
| `min_match_score` | 0.40 | Minimum match score to count as the "same" feature. Deliberately **below** matching's 0.49 opposite-direction cap, so a sign-flipped feature classifies as `degraded` instead of silently falling out of the pairing and reading as `lost`. |
| `min_structural_component` | 0.65 | At least `min_structural_components` structural axes (text/activation/decoder) must clear this bar. Two unrelated features with orthogonal fingerprints pair at a free 0.5 cosine floor otherwise, and a genuinely lost feature would masquerade as a survivor. |
| `min_structural_components` | 1 | See above. |
| `max_importance_drop` | 0.15 | Importance fall (baseline − variant) beyond which a validated feature is degraded. |
| `max_signed_effect_drop` | 0.15 | Tolerated same-provenance signed-effect magnitude loss; matches `validate-matches`' signed-effect delta threshold. |
| `min_abs_signed_effect` | 0.02 | Noise floor shared with matching/validation; a baseline effect above it that lands below it in the variant is "washed out". |

All are keyword-overridable in `interp_lab.quant_diff.build_quant_diff`.

## 4. Honest caveats

- **Final-layer-only records limit layer attribution.** The
  `export-gguf-records` path reads llama.cpp's embeddings output, which exposes
  only the final hidden state. A "degraded" verdict there tells you *the model*
  lost the feature, not *which layer's* quantization caused it. For per-layer
  attribution use `convert-hidden-dump` on a full hidden-state dump, or
  `export-hf-records --layers` on a Transformers-compatible checkpoint, then
  diff layer by layer.
- **Same criterion required, same prompts strongly recommended.** The verdicts
  compare evidence *about one behavior*; different prompt sets confound
  quantization damage with prompt-distribution shift.
- **Correlational verdicts are leads, not claims.** Records without attached
  interventions only ever support `*_correlational` verdicts. To turn a
  `changed_correlational` lead into a real claim, run matched interventions on
  both precisions (`plan-evidence` on the variant report tells you the cheapest
  path) and re-run the diff.
- **`degraded` is threshold-relative.** A 0.14 effect drop reads `preserved`;
  0.16 reads `degraded`. The thresholds are printed in every report so the
  cut is auditable — tighten them for release gates, loosen them for triage.
- **Matching is structural, not identity.** Feature pairing relies on
  fingerprint similarity (labels, activation/decoder signatures, causal
  vectors). With short label-only records the text axis dominates; enrich the
  records (decoder signatures, more examples) if `lost`/`emerged` counts look
  implausibly high.
