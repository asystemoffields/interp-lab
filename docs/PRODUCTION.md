# Production Guide

interp-lab is meant to work for quick local experiments and repeatable research runs. The production path is:

```bash
interp-lab doctor
interp-lab profile-env --out reports/env-profile.json --json
interp-lab run examples/run_records.json
```

For Python users, the same path is available through `interp_lab`:

```python
from interp_lab import doctor, run

print(doctor()["ok"])
run("examples/run_records.json")
```

For browser-driven command setup, generate the local Studio page:

```bash
interp-lab studio --out reports/interp-lab-studio.html
```

Studio reads the CLI parser, exposes every command as a form, and writes both shell commands and single-step run-config JSON. The generated file is static HTML, so it can live next to reports, demos, and shared artifacts.

For less technical users, run the local Studio server:

```bash
interp-lab studio --serve --reports-dir reports
```

Served Studio adds a local runner, persistent job history in `reports/.studio/jobs.json`, run-config import, artifact browsing, HTML report preview, and graph JSON summaries. It uses the same CLI command specs as the static page, only launches known interp-lab subcommands, and keeps artifacts inside the current workspace.

For behavior-led work, start with Criterion Lab:

```bash
interp-lab criterion-lab --model distilgpt2 --preset overconfidence --out reports/overconfidence-lab/run.json
```

Criterion Lab presets are prompt assays. Validate authored assays with `interp-lab validate-assay --preset-file path/to/preset.json` before launch. The generated config first scans activation records across all hidden-state layers, ranks the features and layers that actually track the criterion, and exports reports plus graph artifacts for review. Researchers can then train SAEs or run path validation on discovered layers. Use `--preset-file path/to/preset.json` or `--preset-dir presets` for project-specific assays, and `--workflow sae --layer <N>` when discovery has identified a layer worth testing causally.

Before training a behavior SAE, split scored prompts into a prompt pack:

```bash
interp-lab prepare-sae-prompts --dataset prompts/criterion.jsonl --out-dir prompts/sae-pack --latent-dim 4096
```

Use `train.jsonl` for activation collection, `causal.jsonl` for intervention scoring, and `validation.jsonl` for held-out path validation. The manifest records split counts, duplicate handling, row estimates, and next actions for agents.

## Supported Platforms

CI runs on:

- Ubuntu latest
- macOS latest
- Windows latest
- Python 3.10, 3.11, and 3.12

The package avoids shell-specific workflows in the core CLI. Commands in docs use forward slashes because Python accepts them on Windows, macOS, and Linux.

## Reproducible Runs

Use `interp-lab run` when you want a saved, replayable workflow.

Use `interp-lab init-run` when you want a starting config for `records`, `hf-records`, `sae`, or `sae-paths` workflows:

```bash
interp-lab init-run --workflow records --model toy-records/model --criterion "benchmark awareness" --records examples/activation_records.jsonl --out runs/records.json
```

`sae-paths` is the paper-inspired path workflow: it trains source and target layer SAEs, writes causal feature reports, measures SAE-latent path patches, exports graph JSON, Markdown, and HTML files, writes compact graph summaries for agents, and can add held-out validation with `--validate-paths`. Use `--validation-dataset` to point validation at a separate prompt set.

Generated `sae` and `sae-paths` configs add `prepare-sae-prompts` by default. Training uses `train.jsonl`, causal scoring and first-pass path patching use `causal.jsonl`, and held-out path validation uses `validation.jsonl` unless `--validation-dataset` is supplied. Pass `--skip-prompt-pack` when you want the generated run to use `--dataset` directly.

HF-backed scaffolds accept the same loading flags as the generated commands: `--model-class`, `--trust-remote-code`, `--local-files-only`, `--torch-dtype`, `--device-map`, `--model-kwargs-json`, and `--tokenizer-kwargs-json`.

Minimal config:

```json
{
  "out": "reports/example-records-run",
  "model": "toy-records/model",
  "criterion": "benchmark awareness",
  "backend": "records",
  "records": "examples/activation_records.jsonl",
  "top_k": 5
}
```

Multi-step config:

```json
{
  "out": "reports/full-run",
  "steps": [
    {
      "name": "inspect",
      "command": "inspect",
      "args": {
        "model": "toy-records/model",
        "criterion": "benchmark awareness",
        "backend": "records",
        "records": "examples/activation_records.jsonl",
        "out": "{run_dir}/inspect"
      }
    },
    {
      "name": "match-demo",
      "command": "demo",
      "args": {
        "out": "{run_dir}/demo"
      }
    }
  ]
}
```

Every run writes `manifest.json` with:

- interp-lab version
- Python and platform info
- rendered config
- input file hashes for small/medium inputs
- ordered step records
- per-step output artifacts and aggregate output inventory
- exit codes and timestamps

Use `--dry-run` to print commands:

```bash
interp-lab run examples/run_records.json --dry-run
```

Use `--var KEY=VALUE` for templates:

```bash
interp-lab run run.json --var MODEL=distilgpt2
```

Config strings can reference `{MODEL}`, `{run_dir}`, `{config_dir}`, or `${MODEL}`.

## Environment Diagnostics

`interp-lab doctor` reports core runtime status and optional adapter dependencies:

- `PyYAML` for YAML configs
- `torch` and `transformers` for Hugging Face activation collection and SAE training
- `sae-lens` for public SAE Lens adapters
- `transformer-lens` for hook-cache activation export
- `nnsight` for trace-path activation export
- `goodfire` for Goodfire feature search
- `huggingface-hub` for artifact publishing

Use JSON output in scripts:

```bash
interp-lab doctor --json
```

`interp-lab profile-env` reports machine capacity and route options:

```bash
interp-lab profile-env --path reports --json
```

The JSON includes CPU cores, RAM, free disk at the inspected path, visible accelerators, optional packages, sanitized credential presence checks, route options, alerts, and agent next actions.

## Release Checklist

Before publishing a release:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m compileall src tests
python -m build
python -m twine check dist/*
interp-lab doctor
interp-lab profile-env --out reports/env-profile.json --json
interp-lab run examples/run_records.json
interp-lab release-check --strict --out reports/release-check.json
```

`release-check --strict` is the stable-release gate. It should fail while the project is still classified as alpha or has unresolved stable-release blockers. The release bar is documented in `docs/STABLE_RELEASE.md`.

For model-backed release examples:

```bash
python -m pip install -e ".[hf,train,transformerlens,nnsight,goodfire,publish]"
interp-lab train-sae --help
interp-lab export-hf-records --help
interp-lab export-transformerlens-records --help
interp-lab export-nnsight-records --help
interp-lab publish-hf-artifact --help
```

The reproducible real-model demo suite is cataloged in `docs/REAL_MODEL_DEMOS.md`. Each demo has a JSON manifest under `examples/real_model_demos/` with ordered commands, expected artifacts, and interpretation notes for reviewers and agents.

PyPI release details live in `docs/RELEASE.md`.

## Large Runs

Use `interp-lab plan-scale` before harvesting large activation corpora. It accepts sizes like `70B`, `1T`, `1B`, and `64GB`, writes JSON with `--out`, and includes agent next actions for automated workflows. Add `--from-env` for local advisory routing or pass `--env-profile` from a target machine. For 1T+ models, keep the model runtime colocated with the infrastructure that can serve it, write sharded activation records or SAE records, then run interp-lab ranking and report generation over the evidence layer.

Details live in `docs/SCALING.md`.

## Operator Notes

- Keep large generated artifacts outside git. The default `.gitignore` excludes `reports/`.
- Commit small examples and smoke-test configs.
- Prefer config runs for shareable results.
- Include the `manifest.json` when sending reports to collaborators.
