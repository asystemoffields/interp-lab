# GGUF Bridge: llama.cpp models in the records pipeline

interp-lab's activation-records backend (`inspect --backend records`) is
model-agnostic: any runtime that can dump hidden states can feed it. This
bridge connects CPU GGUF labs — small models served by llama.cpp with no
torch/transformers stack — to the full inspection pipeline. The output is the
existing activation-records JSONL format; no new schema is introduced.

There are two paths with an explicit fidelity tradeoff:

| Path | Command | Layers | Needs |
| --- | --- | --- | --- |
| A. Direct llama.cpp export | `export-gguf-records` | **Final layer only** | `interp-lab[gguf]` (llama-cpp-python) |
| B. Hidden-state dump converter | `convert-hidden-dump` | **Any layers the runtime dumps** | nothing beyond stdlib |

Use Path A for a zero-effort smoke pass on a GGUF model. Use Path B whenever
you care about intermediate layers — it is the full-fidelity route.

## Install

```bash
python -m pip install "interp-lab[gguf]"   # Path A only; Path B needs no extras
```

The `gguf` extra installs `llama-cpp-python>=0.3` (compiles llama.cpp; on
CPU-only machines plain `pip install` works, no CUDA toolchain needed).

## Path A — direct export via llama-cpp-python

llama.cpp's embedding API, constructed with `embedding=True` and pooling
disabled (`pooling_type=LLAMA_POOLING_TYPE_NONE`), returns one vector per
token: the **final-layer hidden state, after the final norm**. That gives
genuine last-layer activation records.

**Honest limitation:** llama.cpp does not expose intermediate layers through
the embedding API. Path A can never tell you *where* in the network a feature
lives — only what the final residual stream encodes. Every record is stamped
`"source": "llama_cpp_embeddings"` and `"layers_available": "final_only"` so
reports cannot be mistaken for multi-layer evidence. For intermediate layers,
use Path B.

Feature ids follow the Hugging Face hidden-state convention used by
`export-hf-records`: a model with `n_layers` transformer blocks has
`hidden_states[0..n_layers]`, and the final-norm output is index `n_layers`.
So a 30-block GGUF model produces `L30:D<dim>` features with
`"layer_convention": "hidden_state_index"`, directly comparable to
`export-hf-records` output for the same architecture.

End-to-end:

```bash
# 1. Scored prompts (the same JSONL format used everywhere else):
#    {"prompt_id": "...", "text": "...", "criterion_score": 1.0}
interp-lab build-prompts \
  --positive-prompt "This looks like a constructed benchmark scenario." \
  --negative-prompt "Can you help me reschedule a meeting?" \
  --out prompts/eval-awareness.jsonl

# 2. Export final-layer records from the GGUF model.
interp-lab export-gguf-records \
  --model models/smollm2-135m.Q8_0.gguf \
  --dataset prompts/eval-awareness.jsonl \
  --out reports/gguf/smollm2-135m/records.jsonl \
  --top-features 64

# 3. Inspect with the standard records backend.
interp-lab inspect \
  --model smollm2-135m.Q8_0.gguf \
  --criterion "the model is aware it is being evaluated" \
  --backend records \
  --records reports/gguf/smollm2-135m/records.jsonl \
  --out reports/gguf/smollm2-135m/inspect
```

Notes:

- `--model-name` overrides the model string stamped on records (default: the
  GGUF file name). The `inspect --model` value must match it.
- Dimensions are ranked by **activation variance across prompts** and the top
  `--top-features` kept. Variance is used (rather than the criterion
  correlation the HF exporter uses at export time) because it stays well
  defined for single-sided prompt packs; the records backend recomputes
  criterion associations from the full record set on load either way. Each
  feature's `selection_variance` is recorded in its metadata.
- The layer count is read from `n_layer()` when llama-cpp-python exposes it,
  otherwise from GGUF metadata (`<architecture>.block_count`). If neither is
  available, pass `--n-layers` explicitly — the export refuses to guess.

### API-version caveats (llama-cpp-python)

- The bridge constructs `Llama(model_path=..., embedding=True, n_ctx=...,
  pooling_type=llama_cpp.LLAMA_POOLING_TYPE_NONE, verbose=False)` and calls
  `llama.embed(text)`. With pooling disabled, `embed` returns a list of
  per-token vectors; interp-lab pools them (`--pool last` by default,
  matching the HF exporter, or `--pool mean`). If a build returns a single
  already-pooled vector (older versions, or embedding-tuned models with baked
  pooling), it is accepted as-is.
- `pooling_type` was added to the `Llama` constructor around 0.2.57; the
  `gguf` extra pins `>=0.3`. If the constant is missing the bridge omits the
  argument, and you should verify per-token output before trusting `--pool`.
- Some llama-cpp-python versions L2-normalize embeddings (notably
  `create_embedding`, and `embed(..., normalize=...)` defaults have changed
  across releases). The bridge uses `embed()`'s default. Per-prompt
  normalization rescales vectors but the records backend's rankings are
  correlation-based, so criterion associations are largely preserved;
  absolute activation magnitudes should not be compared across
  llama-cpp-python versions.
- Programmatic use can bypass llama-cpp-python entirely: pass
  `llama_factory=` any callable returning an object with
  `embed(text) -> list[list[float]]` and either `n_layer() -> int` or a
  `metadata` dict containing `<arch>.block_count` (this is also the seam the
  tests stub).

## Path B — universal hidden-state dump converter

Any runtime that can write hidden states to JSONL can feed interp-lab at
**every layer**. The converter validates the dump, pools per-token vectors,
selects high-variance dimensions per layer, attaches criterion scores from a
scored-prompts file, writes activation records, and re-validates the output
through the real records loader before returning.

### Dump format (`interp-lab.hidden_state_dump.v1`)

One JSON object per line:

```json
{"prompt_index": 0, "layer": 5, "tokens": ["The", " sky"], "hidden": [[0.1, -0.2], [0.3, 0.4]]}
```

- `prompt_index` (required): 0-based index linking lines for the same prompt
  across layers, and matching the line order of the `--dataset` prompts file.
- `layer` (required): non-negative integer. Use the HF hidden-state index
  convention when you can (embeddings = 0, block `i` output = `i + 1`); if
  your runtime counts differently, say so with `--layer-convention` and the
  stamp travels with every feature.
- `hidden` (required): one row of floats per token, or a single flat vector
  if the runtime already pooled.
- `tokens` (optional): token strings; if present, the count must match the
  `hidden` rows.
- Optional per-line: `prompt_id`, `text`, `criterion_score`, `model`. Inline
  values win over the prompts file. Scores must come from the line or the
  prompts file — records without criterion scores are rejected.

Every prompt must cover every layer that appears in the dump (rectangular
coverage), and duplicate `(prompt_index, layer)` lines are rejected.
Malformed lines fail with `path:line:` diagnostics.

### Producing a dump

From any Python runtime in ~10 lines (here: a llama.cpp build patched with a
`--dump-hidden` style callback, or any harness that yields per-layer states —
the same shape works for a custom GGML loop, a JAX model, or an ONNX session):

```python
import json

with open("dump.jsonl", "w") as out:
    for prompt_index, prompt in enumerate(prompts):
        # run_model is your runtime: text -> {layer: [[float per dim] per token]}
        hidden_by_layer, tokens = run_model(prompt["text"])
        for layer, hidden in hidden_by_layer.items():
            out.write(json.dumps({
                "prompt_index": prompt_index,
                "layer": layer,
                "tokens": tokens,
                "hidden": hidden,
            }) + "\n")
```

For a patched llama.cpp, dump `inpL`/`cur` after each block in
`build_graph` (or use a `ggml_backend_sched_eval_callback` to capture `l_out-*`
tensors), write one line per `(prompt, layer)`, and label layers as
`block_index + 1` to match the HF convention.

### End-to-end

```bash
interp-lab convert-hidden-dump \
  --dump reports/gguf/parcae-140m/dump.jsonl \
  --dataset prompts/eval-awareness.jsonl \
  --out reports/gguf/parcae-140m/records.jsonl \
  --features-per-layer 64

interp-lab inspect \
  --model parcae-140m \
  --criterion "the model is aware it is being evaluated" \
  --backend records \
  --records reports/gguf/parcae-140m/records.jsonl \
  --out reports/gguf/parcae-140m/inspect
```

Pass `--model-name` (converter) so the record model string matches
`inspect --model`; otherwise the converter uses the dump lines' `model` field
or `hidden-state-dump`. `--features-per-layer 0` keeps every dimension.

## Limitations, honestly

- **Path A is final-layer only.** llama.cpp's embedding API cannot see
  intermediate layers; the records say so (`layers_available: final_only`).
- **Quantization changes activations.** A Q4 GGUF's hidden states are not the
  fp32 model's hidden states. Records carry the GGUF file name so reports
  stay attributable to the exact artifact.
- **Dimension selection is correlational bookkeeping, not causal evidence.**
  Variance/association ranking finds candidates; the records backend already
  labels its effects `causal_evidence: association_proxy`. Run real
  interventions for causal claims.
- **Embedding normalization differs across llama-cpp-python versions** (see
  caveats above); compare activation magnitudes only within one export.
- **Path B trusts the producing runtime's layer labels.** The
  `layer_convention` stamp records the claim; it cannot verify it.
