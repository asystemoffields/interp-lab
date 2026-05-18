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

The local path should contain the tokenizer/config files and weights loadable by Transformers. GGUF or llama.cpp-only files can still be used by exporting activation records from that runtime and then using the `records` backend.

## Install Extras

```bash
python -m pip install -U "interp-lab[hf,train]"
```

For quantized local checkpoints, install the quantization runtime required by that checkpoint, such as bitsandbytes, accelerate, torchao, or the package that produced the local artifact.

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
  --model-params 2B \
  --tokens 12 \
  --d-model <hidden-size> \
  --selected-layers 1 \
  --latent-dim 256 \
  --from-env \
  --out reports/gemma4-code/scale-plan.json
```

The planner will suggest a starting route and leave every route selectable with `--profile`.

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

The example dataset asks for code or structured output completions and contrasts them with ordinary prose prompts.

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
  --target-token "def,import,return,class,function,const,{,[" \
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
