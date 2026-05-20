# Gemma 4 Walkthrough

This walkthrough runs interp-lab on a Gemma 4 text workflow using either the official Hugging Face checkpoint or a local quantized checkpoint saved in a Transformers-compatible directory.

Official model:

```text
google/gemma-4-E2B-it
```

References: [Google Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) and [Transformers Gemma 4 docs](https://huggingface.co/docs/transformers/v5.5.0/model_doc/gemma4).

Local model:

```text
C:/models/your-gemma4-quant
```

The local path should contain the tokenizer/config files and weights loadable by Transformers. If another runtime exports activation records, point `interp-lab inspect --backend records` at those records directly.

## Install Extras

```bash
python -m pip install -U "interp-lab[hf,train]"
```

For quantized local checkpoints, install the quantization runtime required by that checkpoint, such as bitsandbytes, accelerate, torchao, or the package that produced the local artifact.

For Modal GPU runs:

```bash
python -m pip install -U "interp-lab[modal]"
modal setup
```

## Profile The Environment

```bash
interp-lab profile-env \
  --path reports/gemma4-code \
  --out reports/gemma4-code/env-profile.json \
  --json
```

Then plan the run:

```bash
interp-lab plan-scale \
  --model-params 5B \
  --model-weight-size 9.543GB \
  --tokens 12 \
  --d-model 1536 \
  --selected-layers 1 \
  --latent-dim 256 \
  --from-env \
  --out reports/gemma4-code/scale-plan.json
```

The planner will suggest a starting route and leave every route selectable with `--profile`.
For `google/gemma-4-E2B-it`, the current text stack reports hidden size `1536`, `35` text layers, and about `9.543GB` of bfloat16 safetensors.

## Run On Modal

The Modal example runs Gemma 4 on a remote GPU, caches Hugging Face weights in a Modal volume, and writes compact records and reports back to the local `reports/` directory.

Start with the contrast workflow:

```bash
modal run examples/modal_gemma4.py \
  --workflow contrast \
  --dataset examples/gemma4_code_prompts.jsonl \
  --out-dir reports/gemma4-modal/contrast
```

Run hidden-dimension discovery with causal validation:

```bash
modal run examples/modal_gemma4.py \
  --workflow hidden \
  --dataset examples/gemma4_code_prompts.jsonl \
  --layers 20,28,35 \
  --features-per-layer 32 \
  --top-k 16 \
  --group-top-k 8 \
  --out-dir reports/gemma4-modal/hidden
```

Run a tool-call behavior assay:

```bash
interp-lab validate-assay \
  --preset-file examples/presets/successful-tool-calls.json \
  --out reports/gemma4-tool-calls/assay-validation.json

modal run examples/modal_gemma4.py \
  --workflow hidden \
  --dataset examples/gemma4_tool_call_prompts.jsonl \
  --criterion "the assistant should produce a valid schema-following tool call that successfully executes the user's requested operation" \
  --layers 20,28,35 \
  --features-per-layer 32 \
  --top-k 16 \
  --group-top-k 8 \
  --target-token auto \
  --max-length 96 \
  --out-dir reports/gemma4-tool-calls/modal-hidden
```

After the run, export the causal report as an attribution graph:

```bash
interp-lab export-attribution-graph \
  --report reports/gemma4-tool-calls/modal-hidden/hidden-causal/report.json \
  --out reports/gemma4-tool-calls/modal-hidden/graph.json \
  --markdown-out reports/gemma4-tool-calls/modal-hidden/graph.md \
  --html-out reports/gemma4-tool-calls/modal-hidden/graph.html

interp-lab summarize-attribution-graph \
  --graph reports/gemma4-tool-calls/modal-hidden/graph.json \
  --out reports/gemma4-tool-calls/modal-hidden/graph-summary.json
```

Set `INTERP_LAB_MODAL_GPU=L40S` or another Modal GPU name before `modal run` when you want a larger accelerator. The default is `A10G`.

The Modal workflows default to `--target-token auto`. Read the causal report's behavior-score line after the run: if the baseline score is saturated, rerun with a narrower explicit target-token set for the behavior you care about. If the score is near zero even with auto targets, inspect the target-token sample in the report and pass explicit `raw:` or `space:` tokens for the behavior.

On Windows PowerShell, set UTF-8 output before running Modal in a redirected or background process:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## Choose The Model

Use the official checkpoint as `<GEMMA4_MODEL>`:

```bash
google/gemma-4-E2B-it
```

Use a local quantized checkpoint as `<GEMMA4_MODEL>`:

```bash
C:/models/your-gemma4-quant
```

Gemma 4 uses a conditional-generation class in current Transformers builds:

```text
--model-class gemma4-conditional
```

For a local checkpoint with custom code or local-only files, add:

```text
--trust-remote-code --local-files-only
```

For a sharded or quantized GPU load, add the checkpoint-specific loading flags:

```text
--device cuda --device-map auto --torch-dtype auto
```

Extra loader kwargs can be passed as JSON:

```text
--model-kwargs-json "{\"attn_implementation\":\"eager\"}"
```

To find the hidden size for a local or official checkpoint:

```bash
python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('<GEMMA4_MODEL>', trust_remote_code=True); t=getattr(c, 'text_config', c); print(getattr(t, 'hidden_size', 'unknown'))"
```

## Export Hidden-State Records

The example dataset asks for code or structured output completions and contrasts them with ordinary prose prompts. Use the official Transformers checkpoint, or a local safetensors/Transformers checkpoint, for this step:

```bash
interp-lab export-hf-records \
  --model <GEMMA4_MODEL> \
  --model-class gemma4-conditional \
  --dataset examples/gemma4_code_prompts.jsonl \
  --out reports/gemma4-code/records.jsonl \
  --layers 12 \
  --features-per-layer 24 \
  --pool last \
  --max-length 96
```

## Inspect Candidate Features

```bash
interp-lab inspect \
  --model <GEMMA4_MODEL> \
  --criterion "the next token should begin code, a command, or structured data" \
  --backend records \
  --records reports/gemma4-code/records.jsonl \
  --out reports/gemma4-code/inspect
```

## Causal Ablation

Run ablations against code-like target tokens:

```bash
interp-lab export-hf-interventions \
  --model <GEMMA4_MODEL> \
  --model-class gemma4-conditional \
  --report reports/gemma4-code/inspect/report.json \
  --dataset examples/gemma4_code_prompts.jsonl \
  --criterion "the next token should begin code, a command, or structured data" \
  --top-k 8 \
  --group-top-k 6 \
  --records reports/gemma4-code/records.jsonl \
  --append-group-records reports/gemma4-code/records-with-group.jsonl \
  --target-token auto \
  --out reports/gemma4-code/interventions.jsonl \
  --max-length 96
```

Then re-inspect with causal evidence:

```bash
interp-lab inspect \
  --model <GEMMA4_MODEL> \
  --criterion "the next token should begin code, a command, or structured data" \
  --backend records \
  --records reports/gemma4-code/records-with-group.jsonl \
  --interventions reports/gemma4-code/interventions.jsonl \
  --require-interventions \
  --out reports/gemma4-code/inspect-causal
```

## Train A Tiny SAE

Use this when the local custom Gemma 4 checkpoint has no public SAE.

```bash
interp-lab train-sae \
  --preset minimal \
  --hf-model <GEMMA4_MODEL> \
  --model-class gemma4-conditional \
  --dataset examples/gemma4_code_prompts.jsonl \
  --layer 12 \
  --latent-dim 256 \
  --epochs 40 \
  --batch-size 6 \
  --max-length 96 \
  --out reports/gemma4-code/trained-sae/sae.json \
  --records-out reports/gemma4-code/trained-sae/records.jsonl
```

Inspect the learned SAE latents:

```bash
interp-lab inspect \
  --model <GEMMA4_MODEL> \
  --criterion "the next token should begin code, a command, or structured data" \
  --backend records \
  --records reports/gemma4-code/trained-sae/records.jsonl \
  --out reports/gemma4-code/trained-sae/inspect
```

## Agent Loop

For an AI agent, the intended loop is:

```text
profile-env -> plan-scale -> export-hf-records or train-sae -> inspect -> export-hf-interventions -> inspect --require-interventions
```

The files to read are:

- `reports/gemma4-code/env-profile.json`
- `reports/gemma4-code/scale-plan.json`
- `reports/gemma4-code/inspect/report.json`
- `reports/gemma4-code/inspect-causal/report.json`
- `reports/gemma4-code/trained-sae/sae.json`

These are all JSON and can be passed directly into follow-up tooling.

Use `--target-token auto` to derive behavior-scoring tokens from positive prompts. Target tokens otherwise default to GPT-style leading-space forms. Prefix a token with `raw:` to score exact token text as well, which is useful for Gemma-style tokenizers.
The causal report includes behavior-score diagnostics so agents and researchers can spot saturated or near-zero scoring setups before over-reading a weak causal effect.
