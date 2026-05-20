# interp-lab

interp-lab is an open-source starter kit for criterion-driven mechanistic interpretability.

Give it a model, a criterion, and feature evidence. It ranks internal features, explains them, tests causal impact, and searches for equivalent features in other models.

Quick start:

```bash
interp-lab inspect \
  --model google/gemma-2-2b \
  --criterion "the model is aware it is being evaluated" \
  --backend toy \
  --out reports/eval-awareness
```

Python API:

```python
from interp_lab import compare, inspect, validate_matches

left = inspect(
    "toy/model-a",
    "the model is aware it is being evaluated",
    backend="toy",
    out="reports/model-a",
)
right = inspect(
    "toy/model-b",
    "the model is aware it is being evaluated",
    backend="toy",
    out="reports/model-b",
)
matches = compare(left.report, right.report, out="reports/matches.json")
validation = validate_matches(matches.report, out="reports/match-validation.json")
```

The package includes toy, JSONL, activation-record, Neuronpedia, SAE Lens, Goodfire, Gemma Scope/Qwen-Scope, Hugging Face, TransformerLens, NNsight, contrast-direction, and on-demand SAE training paths. It is shaped around adapter interfaces for real activation hooks, SAEs, crosscoders, and natural-language autoencoders.

## Why This Exists

The goal is to get close to an "oracular SAE" workflow:

1. Compile a natural-language criterion into examples and scores.
2. Collect candidate features from SAEs, crosscoders, NLA explanations, or feature dumps.
3. Rank features by criterion association, specificity, causal evidence, and stability.
4. Build a feature fingerprint that can be compared across models.
5. Validate cross-model equivalents with interventions.

## Commands

Check your local environment:

```bash
interp-lab doctor
```

Check stable-release readiness:

```bash
interp-lab release-check --strict --out reports/release-check.json
```

Profile the current machine and route options:

```bash
interp-lab profile-env --out reports/env-profile.json --json
```

Run a criterion inspection:

```bash
interp-lab inspect \
  --model toy/a \
  --criterion "Python security bug" \
  --backend toy \
  --html-out reports/inspection/report.html
```

This writes JSON and Markdown by default. `--html-out` adds a self-contained searchable feature-card report with layer/source/evidence filters and copyable next-action commands. Reports include `agent_next_actions` with exact follow-up command templates for intervention planning, causal re-inspection, and graph export.

Build a prompt dataset from prompts you wrote:

```bash
interp-lab build-prompts \
  --positive prompts/code-positive.txt \
  --negative prompts/code-controls.txt \
  --split paragraphs \
  --out prompts/code-criterion.jsonl
```

Prompt files can use one prompt per paragraph or one prompt per line with `--split lines`. Use `--positive-prompt` and `--negative-prompt` for inline prompts, and `--delimiter` for multi-line chat-style prompts separated by a literal marker. The output JSONL works anywhere interp-lab accepts `--dataset`.

The same custom-prompt path is available from Python with `interp_lab.build_prompts(...)`.

Compare two reports:

```bash
interp-lab match \
  --left reports/a/report.json \
  --right reports/b/report.json \
  --out reports/matches.json
```

This writes both `matches.json` and a readable markdown report with labels, component scores, and signed effects when present.

Validate the match claims:

```bash
interp-lab validate-matches \
  --matches reports/matches.json \
  --out reports/match-validation.json \
  --html-out reports/match-validation.html
```

This grades each pair as `validated`, `needs_causal_evidence`, `plausible`, `contradicted`, or `weak`, with reason codes and next actions for agents or researchers. The HTML output is self-contained and includes search, status filters, score components, and evidence details.

Create a demo run:

```bash
interp-lab demo --out reports/demo
```

The demo writes a complete tour: feature reports for two toy models, cross-model matches, match validation, an attribution graph, HTML viewers, a compact graph summary, and a local Studio page.

For a compact real-model release check, follow `docs/GOLDEN_REAL_MODEL_DEMO.md`. It trains a small DistilGPT-2 SAE, suppresses selected SAE latents, re-inspects with intervention evidence, and exports an HTML attribution graph. The broader stable-demo suite is cataloged in `docs/REAL_MODEL_DEMOS.md` and `examples/real_model_demos/`.

Verify or run the cataloged real-model demos with:

```bash
interp-lab demo-sweep --out reports/real-model-demo-sweep.json
```

Use `--run` to execute the manifest commands and `--demo <id>` to focus on one walkthrough.

Open the browser Studio command builder:

```bash
interp-lab studio --out reports/interp-lab-studio.html
```

Studio is a self-contained local HTML app generated from the CLI parser. It covers the interp-lab command surface, builds shell commands, and emits single-step run-config JSON that can be pasted into a larger workflow or handed to an agent.

Serve Studio locally when you want to run jobs and browse reports from the browser:

```bash
interp-lab studio --serve --reports-dir reports
```

The served app keeps persistent job history under `reports/.studio/jobs.json`, launches known interp-lab commands, imports pasted run-config JSON, and exposes generated HTML, JSON, Markdown, and graph artifacts under the current workspace. The static file remains useful for sharing commands and configs; served mode adds the local runner.

For agent integrations and downstream wrappers, `interp_lab.public_api_contract()` returns the current stable exports, schema ids, and core callable parameters as JSON-serializable data.

Start Criterion Lab from a prompt assay. The default path is discovery-first: it writes prompt pairs, exports activation records across every hidden-state layer, ranks the features that actually track the criterion, and builds a graph/report you can use to choose SAE and causal follow-up runs.

```bash
interp-lab criterion-lab \
  --model distilgpt2 \
  --preset overconfidence \
  --run-dir reports/overconfidence-lab \
  --out reports/overconfidence-lab/run.json
```

Presets are JSON files containing the criterion and contrast prompts. The bundled `overconfidence` preset is just one data file; you can point at your own preset file or directory:

```bash
interp-lab validate-assay \
  --preset-file examples/presets/math-reasoning.json \
  --out reports/math-reasoning-lab/assay-validation.json

interp-lab criterion-lab \
  --model distilgpt2 \
  --preset-file examples/presets/math-reasoning.json \
  --run-dir reports/math-reasoning-lab \
  --out reports/math-reasoning-lab/run.json
```

For an ad hoc agent-driven run, skip presets and provide the criterion plus prompt pairs directly:

```bash
interp-lab criterion-lab \
  --model distilgpt2 \
  --criterion "the model is doing multi-step mathematical reasoning" \
  --positive-prompt "Solve step by step: If 3 notebooks cost $7.50, how much do 11 cost?" \
  --negative-prompt "Write a friendly greeting to a new teammate." \
  --out reports/math-custom/run.json
```

Use `--list-presets` to see discoverable presets, `--preset-dir presets` to add a local registry, and `--workflow sae --layer <N>` after discovery identifies promising layers. If you enable SAE causal scoring without explicit target tokens, Criterion Lab uses model-derived `auto` targets; preset target hints are only used when `--use-preset-target-hints` is supplied.

Run a reproducible workflow from config:

```bash
interp-lab run examples/run_records.json
```

This writes a run manifest with the tool version, platform, input hashes, executed steps, per-step output artifacts, and an aggregate output inventory. Run configs can be JSON, TOML, or YAML.

Generate an editable run config for a common workflow:

```bash
interp-lab init-run \
  --workflow sae \
  --model distilgpt2 \
  --criterion "the next token should be a physical measurement unit" \
  --positive-prompt "The answer is measured in meters." \
  --negative-prompt "The answer is a person's name." \
  --include-causal \
  --target-token auto \
  --latent-dim 1024 \
  --run-dir reports/distilgpt2-sae-run \
  --out runs/distilgpt2-sae.json
```

Then run it with `interp-lab run runs/distilgpt2-sae.json`. The generated JSON is meant to be edited before larger runs. SAE workflows add a `prepare-sae-prompts` step by default, then train on `train.jsonl`, score interventions on `causal.jsonl`, and keep `validation.jsonl` available for held-out checks. With `--include-causal`, the generated SAE inspection focuses on features that received causal intervention rows. Pass `--skip-prompt-pack` when your dataset is already split and you want to use it directly.

For a two-layer path-patching workflow, use `--workflow sae-paths` with `--source-layer` and `--target-layer`. This scaffolds source and target SAE training, causal feature reports, measured SAE-latent paths, graph exports, compact graph summaries, and optional held-out path validation. Add `--validation-dataset` when you have a separate held-out prompt set:

```bash
interp-lab init-run \
  --workflow sae-paths \
  --model distilgpt2 \
  --criterion "the next token should be a physical measurement unit" \
  --positive-prompt "The answer is measured in meters." \
  --negative-prompt "The answer is a person's name." \
  --source-layer 2 \
  --target-layer 4 \
  --include-causal \
  --target-token auto \
  --validate-paths \
  --latent-dim 1024 \
  --run-dir reports/distilgpt2-sae-paths \
  --out runs/distilgpt2-sae-paths.json
```

HF-backed scaffolds also accept model-loading flags such as `--model-class`, `--trust-remote-code`, `--torch-dtype`, `--device-map`, and `--local-files-only`, and pass them through to every generated HF step.

Export activation records from a real Hugging Face model:

```bash
interp-lab export-hf-records \
  --model distilgpt2 \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --out reports/real-small/distilgpt2-unit/records.jsonl
```

For a current Gemma 4 walkthrough, including local quantized Transformers-compatible checkpoints, see `docs/GEMMA4_WALKTHROUGH.md`.

Export activation records from TransformerLens hooks:

```bash
python -m pip install "interp-lab[transformerlens]"

interp-lab export-transformerlens-records \
  --model gpt2-small \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --layers 6 \
  --out reports/tl/gpt2-small-layer6-records.jsonl
```

Export activation records from NNsight traces:

```bash
python -m pip install "interp-lab[nnsight]"

interp-lab export-nnsight-records \
  --model openai-community/gpt2 \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --activation-path transformer.h[6].output[0] \
  --out reports/nnsight/gpt2-layer6-records.jsonl
```

Export ablation records for top hidden-dimension features:

```bash
interp-lab export-hf-interventions \
  --model distilgpt2 \
  --report reports/real-small/distilgpt2-unit/inspect/report.json \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --criterion "the next token should be a physical measurement unit" \
  --out reports/real-small/distilgpt2-unit/interventions.jsonl
```

Amplify, suppress, or ablate specific discovered features with one agent-friendly command:

```bash
interp-lab intervene \
  --model distilgpt2 \
  --dataset prompts/unit-sae-pack/causal.jsonl \
  --criterion "the next token should be a physical measurement unit" \
  --feature SAE:L6:F30 \
  --records reports/production-sae/records.jsonl \
  --sae reports/production-sae/sae.json \
  --mode suppress \
  --strength-sweep "1,3,10" \
  --target-token auto \
  --out reports/production-sae/feature-interventions.jsonl \
  --plan-out reports/production-sae/intervention-plan.json
```

Use `--dry-run --json` first when an agent should inspect the plan before spending model time. The plan includes selected features, expected forward passes, exact next-action commands, and advisories. Pass `--records` when the intervention comes from an existing activation-record inspection so the plan can emit a complete `inspect --backend records --records ... --interventions ...` follow-up command. The output JSONL can be passed back into `interp-lab inspect --interventions ...` so causal evidence updates the feature report.

Export a contrast-direction feature and calibrate a causal steering strength:

```bash
interp-lab export-hf-contrast \
  --model distilgpt2 \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --criterion "the next token should be a physical measurement unit" \
  --records-out reports/real-small/distilgpt2-unit/contrast-records.jsonl \
  --interventions-out reports/real-small/distilgpt2-unit/contrast-interventions.jsonl \
  --strength-sweep "3,10,30,100"
```

`export-hf-contrast` learns a positive-minus-negative hidden-state direction from scored prompts. When `--strength-sweep` is set, it tests each steering strength on positive prompts, uses negative prompts as side-effect checks, and writes intervention rows for the most specific setting.

Prepare train, causal, and held-out prompt splits for behavior SAE runs:

```bash
interp-lab prepare-sae-prompts \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --out-dir prompts/unit-sae-pack \
  --latent-dim 1024 \
  --max-length 128
```

This writes `train.jsonl`, `causal.jsonl`, `validation.jsonl`, and `manifest.json`. The split is deterministic, stratified by criterion score, keeps duplicate prompt text in one split, and adds advisories when the pack looks too small for the requested SAE width.

Train an SAE when no public SAE exists:

```bash
interp-lab train-sae \
  --preset minimal \
  --hf-model distilgpt2 \
  --dataset prompts/unit-sae-pack/train.jsonl \
  --layer 6 \
  --latent-dim 64 \
  --epochs 50 \
  --out reports/real-small/distilgpt2-unit/trained-sae/sae.json \
  --records-out reports/real-small/distilgpt2-unit/trained-sae/records.jsonl
```

Use `--preset minimal` for quick local exploration. It trains on one activation row per prompt and keeps the compute footprint small.

Use `--preset production` when you want a stronger artifact:

```bash
interp-lab train-sae \
  --preset production \
  --hf-model distilgpt2 \
  --dataset prompts/unit-sae-pack/train.jsonl \
  --causal-dataset prompts/unit-sae-pack/causal.jsonl \
  --layer 6 \
  --latent-dim 1024 \
  --out reports/production-sae/sae.json \
  --records-out reports/production-sae/records.jsonl \
  --causal-out reports/production-sae/interventions.jsonl \
  --criterion "the next token should be a physical measurement unit"
```

Production mode uses token-level activation rows, top-k sparse codes, held-out reconstruction metrics, dead-latent reporting, and optional SAE-latent steering interventions when `--causal-out` is provided. The exported SAE records include training diagnostics such as `rows_per_latent`, train/validation reconstruction MSE, validation/train MSE ratio, active-latent fraction, dead-latent count, and advisories for sparse data or validation drift. You can override any preset choice, such as `--epochs`, `--batch-size`, `--top-k`, or `--max-records`.

Add `--target-token auto` when you want causal scoring tokens derived from the positive prompts. Prefix tokens with `raw:` for exact tokenizer text, which is often useful outside GPT-style leading-space tokenizers.

Then inspect the learned SAE latents with the normal records backend:

```bash
interp-lab inspect \
  --model distilgpt2 \
  --criterion "the next token should be a physical measurement unit" \
  --backend records \
  --records reports/real-small/distilgpt2-unit/trained-sae/records.jsonl \
  --out reports/real-small/distilgpt2-unit/trained-sae/inspect
```

Measure real source-to-target SAE paths between two trained layers:

```bash
interp-lab export-hf-sae-paths \
  --model distilgpt2 \
  --dataset examples/hf_prompts_unit_prediction.jsonl \
  --criterion "the next token should be a physical measurement unit" \
  --source-sae reports/sae-layer6/sae.json \
  --target-sae reports/sae-layer10/sae.json \
  --source-report reports/sae-layer6/report/report.json \
  --target-report reports/sae-layer10/report/report.json \
  --out reports/sae-paths/layer6-to-layer10.jsonl \
  --strength-sweep=-4,-2,2,4 \
  --random-source-controls 2
```

This steers selected source SAE decoder directions, re-encodes the downstream hidden state with the target SAE, and writes measured target-latent deltas plus optional behavior-score deltas. `--random-source-controls` adds matched rows from random source SAE latents so the attribution graph can report path specificity next to raw effect size. Feed the result into the attribution graph:

```bash
interp-lab export-attribution-graph \
  --report reports/sae-layer6/report/report.json \
  --report reports/sae-layer10/report/report.json \
  --path-records reports/sae-paths/layer6-to-layer10.jsonl \
  --out reports/sae-paths/graph.json \
  --markdown-out reports/sae-paths/graph.md \
  --html-out reports/sae-paths/graph.html
```

The Markdown graph digest summarizes strong causal features, measured candidate paths, validation status counts when present, feature groups, and the next validation checks. The HTML graph viewer is a self-contained local file with an evidence summary, role/status filters, searchable feature rows, candidate paths, copyable agent actions, and an SVG graph.

For automation, write a compact graph summary JSON:

```bash
interp-lab summarize-attribution-graph \
  --graph reports/sae-paths/graph.json \
  --out reports/sae-paths/graph-summary.json
```

Validate candidate graph paths with repeated or held-out path records:

```bash
interp-lab validate-attribution-graph \
  --graph reports/sae-paths/graph.json \
  --path-records reports/sae-paths/heldout-layer6-to-layer10.jsonl \
  --out reports/sae-paths/validation.json \
  --graph-out reports/sae-paths/validated-graph.json
```

This writes JSON and Markdown summaries with effect sizes, control comparisons, sign consistency, confidence intervals, path status, claim grade, validation reason codes, run-level `agent_next_actions`, and a next action for each path. `--graph-out` writes a copy of the attribution graph with validation attached to matching path edges and candidate paths, plus `validated-graph.md` and `validated-graph.html` digests by default.

Rerun the graph's top SAE paths on held-out prompts and validate them in one step:

```bash
interp-lab validate-hf-sae-paths \
  --graph reports/sae-paths/graph.json \
  --model distilgpt2 \
  --dataset prompts/heldout-code-criterion.jsonl \
  --source-sae reports/sae-layer6/sae.json \
  --target-sae reports/sae-layer10/sae.json \
  --path-records-out reports/sae-paths/heldout-paths.jsonl \
  --out reports/sae-paths/heldout-validation.json \
  --graph-out reports/sae-paths/heldout-validated-graph.json \
  --graph-markdown-out reports/sae-paths/heldout-validated-graph.md \
  --graph-html-out reports/sae-paths/heldout-validated-graph.html \
  --random-source-controls 2
```

The validation command selects exact graph path pairs, reruns only those source-target pairs, adds random-source controls, and writes the validation report. To measure specific pairs manually, pass `--path-pair SOURCE=TARGET` to `export-hf-sae-paths`.

`train-sae` can also train from an existing activation-record JSONL:

```bash
interp-lab train-sae \
  --records reports/real-small/distilgpt2-unit/records.jsonl \
  --model distilgpt2 \
  --latent-dim 256 \
  --method auto \
  --out reports/sae/sae.json \
  --records-out reports/sae/records.jsonl
```

Training uses PyTorch when available. `--method fallback` uses a deterministic sparse dictionary trainer, which is useful for small runs, constrained environments, and smoke tests. Set `--latent-dim` directly for any SAE width, or use `--expansion-factor` to scale from the input dimension. By default, the exported activation records write every learned latent; `--top-k-features` can compress large runs. `--max-records` bounds training on large JSONL streams with deterministic reservoir sampling.

Rank features from per-prompt activation records:

```bash
interp-lab inspect \
  --model my/model \
  --criterion "the model is aware it is being evaluated" \
  --backend records \
  --records examples/activation_records.jsonl \
  --out reports/eval-awareness
```

Add causal intervention evidence:

```bash
interp-lab inspect \
  --model my/model \
  --criterion "the model is aware it is being evaluated" \
  --backend records \
  --records examples/activation_records.jsonl \
  --interventions examples/interventions.jsonl \
  --out reports/eval-awareness-causal
```

Import selected features from Neuronpedia:

```bash
interp-lab inspect \
  --model gpt2-small \
  --criterion "mentions of measurements in meters or feet" \
  --backend neuronpedia \
  --neuronpedia-feature gpt2-small@6-res_scefr-ajt:650 \
  --out reports/neuronpedia-measurements
```

Import selected features from a pretrained SAE Lens SAE:

```bash
python -m pip install "interp-lab[saelens]"

interp-lab inspect \
  --model gpt2-small \
  --criterion "numeric measurements" \
  --backend saelens \
  --saelens-release gpt2-small-res-jb \
  --saelens-sae-id blocks.6.hook_resid_pre \
  --saelens-feature-indexes 650 \
  --out reports/saelens-feature
```

Import Goodfire features:

```bash
python -m pip install "interp-lab[goodfire]"

interp-lab inspect \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --criterion "formal writing style" \
  --backend goodfire \
  --goodfire-top-k 20 \
  --out reports/goodfire-formal-style
```

Import selected features from named SAE suites:

```bash
interp-lab inspect \
  --model google/gemma-2-2b \
  --criterion "numeric measurements" \
  --backend scope \
  --scope-source gemma-scope \
  --scope-release <saelens-release-or-hf-repo> \
  --scope-sae-id blocks.6.hook_resid_post \
  --scope-feature-indexes 650 \
  --out reports/gemma-scope-feature
```

Publish reports or artifact folders to Hugging Face Hub:

```bash
python -m pip install "interp-lab[publish]"

interp-lab publish-hf-artifact \
  --repo-id your-user/interp-lab-demo \
  --repo-type dataset \
  --path reports/real-small/distilgpt2-unit \
  --tag sae \
  --tag activation-records
```

Export a report as an attribution graph:

```bash
interp-lab export-attribution-graph \
  --report reports/eval-awareness/report.json \
  --out reports/eval-awareness/graph.json \
  --markdown-out reports/eval-awareness/graph.md \
  --html-out reports/eval-awareness/graph.html \
  --include-similarity-edges
```

Repeat `--report` to fuse reports for the same criterion. Graph exports include namespaced feature nodes, criterion-association edges, measured causal-effect edges when intervention evidence exists, candidate feature-group supernodes, coactivation edges from aligned activation signatures, a readable Markdown digest, an offline HTML viewer, and a `mechanism_summary` with candidate paths plus validation next steps. `summarize-attribution-graph` writes a compact JSON view with counts, top path claims, validation assessment, and agent actions.

Plan a large run before harvesting activations:

```bash
interp-lab plan-scale \
  --model-params 1T \
  --tokens 1B \
  --d-model 16384 \
  --selected-layers 8 \
  --latent-dim 1M \
  --from-env \
  --target-shard-size 64GB \
  --out reports/scale-plan.json
```

## JSONL Feature Dumps

You can inspect a model from a JSONL feature dump:

```bash
interp-lab inspect \
  --model my/model \
  --criterion "refusal behavior" \
  --features examples/features.jsonl \
  --out reports/refusal
```

Each row should look like this:

```json
{
  "feature_id": "L18:F104921",
  "model": "my/model",
  "layer": 18,
  "label": "constructed benchmark or test scenario",
  "examples": ["This looks like a test case...", "The prompt appears artificial..."],
  "activation_signature": [0.9, 0.2, 0.1],
  "decoder_signature": [0.1, -0.4, 0.3],
  "causal_effects": {"criterion": 0.34, "refusal": 0.12},
  "source": "sae"
}
```

## Activation Records

Activation records are the most flexible import path. Use them when you have per-prompt or per-token feature activations from an SAE, crosscoder, NLA probe, Neuronpedia script, remote activation harvester, or custom hook.

Each row is one prompt or token position:

```json
{
  "model": "my/model",
  "prompt_id": "eval-1",
  "text": "This looks like a benchmark task...",
  "criterion_score": 1.0,
  "features": [
    {
      "feature_id": "L18:F104921",
      "activation": 0.92,
      "label": "constructed benchmark or test scenario",
      "layer": 18,
      "decoder_signature": [0.1, -0.4, 0.3, 0.2]
    }
  ]
}
```

interp-lab streams records by feature, estimates criterion association from sufficient statistics, preserves top activating examples, and creates a feature fingerprint for matching. Add intervention records when you want causal evidence in the report.

## Intervention Records

Intervention records let the report distinguish correlational evidence from causal evidence. Each row is one ablation, amplification, clamp, patch, or steering run:

```json
{
  "model": "my/model",
  "feature_id": "L18:F104921",
  "criterion": "the model is aware it is being evaluated",
  "intervention": "ablate",
  "prompt_id": "eval-1",
  "baseline_score": 0.92,
  "intervention_score": 0.31,
  "side_effect_score": 0.04
}
```

For `ablate`, `zero`, `remove`, `knockout`, `suppress`, and `clamp_down`, a score drop is treated as evidence the feature promotes the criterion. For `amplify`, `steer`, `patch`, `patch_in`, `clamp`, and `clamp_up`, a score rise is treated as evidence the feature promotes the criterion.

Hugging Face exporters use positive-scored prompts for criterion effects and negative-scored prompts for side-effect estimates. That makes a report prefer features that move the requested behavior while leaving nearby unrelated prompts stable.

When an intervention row includes `metadata.behavior_score`, reports summarize the baseline behavior score, target-token strategy, target-token count, and a small target-token sample. If the score is saturated or near zero, the report adds a note suggesting narrower tokens, harder prompts, `auto` targets when explicit tokens were used, or exact `raw:`/`space:` tokenizer forms when auto targets are already near zero.

Rows with a `criterion` field are matched to the CLI criterion by normalized exact text. Omit `criterion`, or pass `--allow-intervention-criterion-mismatch`, when you want to reuse intervention files across paraphrased criteria.

Control rows can be included in the same intervention JSONL by setting `metadata.control_type` to values such as `random_feature`, `matched_frequency`, or `placebo`. Reports include confidence intervals, control-effect summaries, and a `strong_causal_score`.

## Neuronpedia

The Neuronpedia backend reads the public feature JSON endpoint documented by Neuronpedia. It accepts refs like:

```text
gpt2-small@6-res_scefr-ajt:650
https://www.neuronpedia.org/gpt2-small/6-res_scefr-ajt/650
https://www.neuronpedia.org/api/feature/gpt2-small/6-res_scefr-ajt/650
```

Neuronpedia features include dashboard evidence, autointerp explanations, top activating examples, logits, sparsity, and related metadata. interp-lab converts those into feature evidence and fingerprints.

## SAE Lens

The SAE Lens backend is optional because it can pull in heavier model tooling. It uses `SAE.from_pretrained_with_cfg_and_sparsity()` when available, extracts selected decoder rows, and wraps them as interp-lab feature evidence. For criterion ranking over real prompts, export SAE activations into activation records and run the `records` backend.

## Ecosystem Bridges

- Goodfire: semantic feature search through the Goodfire SDK.
- Neuronpedia: public feature endpoint import.
- SAE Lens: pretrained SAE decoder-row import.
- Gemma Scope and Qwen-Scope: named wrappers around SAE-suite metadata.
- TransformerLens: hook-cache activation export.
- NNsight: trace-based activation export for local or remote model execution.
- Modal: remote GPU activation runs that return compact records and reports.
- Hugging Face Hub: artifact publishing for reports, records, interventions, and trained SAE metadata.

Each bridge is optional. The base package keeps the portable JSONL evidence formats stable, while heavier model tooling lives behind extras.

## Scaling

For large models, use interp-lab as the orchestration and evidence layer:

1. Harvest activations through the environment that can run the model.
2. Write sharded activation records or SAE feature records.
3. Train or import SAEs against those shards.
4. Stream records into inspection reports.
5. Run causal validation in resumable batches.
6. Publish reports, graphs, and artifacts with manifests.

`interp-lab profile-env` inspects CPU cores, RAM, disk space, local accelerators, optional packages, and sanitized environment flags such as whether Goodfire or NNsight credentials are present. It returns advisory route options, including local CPU, single GPU, cluster, remote API, and frontier-lab style harvesting.

`interp-lab plan-scale` accepts human-friendly sizes such as `70B`, `1T`, `1B`, and `64GB`. It estimates model-weight load, dense activation storage, sparse feature-record storage, SAE parameter storage, causal validation forward passes, shard counts, risk flags, and agent next actions. Add `--model-weight-size` when the checkpoint size is known, `--from-env` to profile the current machine while planning, or `--env-profile other-machine.json` to plan against a saved profile from another environment. Every route suggestion can be overridden with `--profile`. Use `--json` or `--out scale-plan.json` when an AI agent or workflow should consume the plan directly. See `docs/SCALING.md` for the 1T+ path.

Modal users can run the Gemma 4 remote workflow directly:

```bash
modal run examples/modal_gemma4.py --workflow contrast --out-dir reports/gemma4-modal/contrast
modal run examples/modal_gemma4.py --workflow hidden --out-dir reports/gemma4-modal/hidden
modal run examples/modal_gemma4.py --workflow hidden --dataset examples/gemma4_tool_call_prompts.jsonl --out-dir reports/gemma4-tool-calls/modal-hidden
```

## Architecture

The core object is a `FeatureFingerprint`:

```text
activation signature
+ text explanation embedding
+ decoder signature
+ causal effect vector
+ examples
```

Cross-model equivalence is scored by fingerprint similarity. `validate-matches` turns those candidates into explicit evidence grades using score, text/activation/decoder components, causal fingerprint similarity, signed-effect direction, and signed-effect calibration.

Adapters are intentionally small:

- `FeatureProvider`: returns candidate features.
- `Verbalizer`: adds NLA-style text explanations.
- `InterventionRunner`: ablates, amplifies, patches, or estimates causal effects.
- `CriterionCompiler`: turns natural-language criteria into examples and scoring hints.

## Roadmap

- Natural Language Autoencoder adapter.
- Crosscoder training and import.
- Rich HTML feature cards.
- Studio workflows that prepare local or remote runs from the browser.
- Distributed SAE training manifests.
- Remote causal validation workers.
- Feature transfer tests across model families.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```
