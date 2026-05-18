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

## Supported Platforms

CI runs on:

- Ubuntu latest
- macOS latest
- Windows latest
- Python 3.10, 3.11, and 3.12

The package avoids shell-specific workflows in the core CLI. Commands in docs use forward slashes because Python accepts them on Windows, macOS, and Linux.

## Reproducible Runs

Use `interp-lab run` when you want a saved, replayable workflow.

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
```

For model-backed release examples:

```bash
python -m pip install -e ".[hf,train,transformerlens,nnsight,goodfire,publish]"
interp-lab train-sae --help
interp-lab export-hf-records --help
interp-lab export-transformerlens-records --help
interp-lab export-nnsight-records --help
interp-lab publish-hf-artifact --help
```

PyPI release details live in `docs/RELEASE.md`.

## Large Runs

Use `interp-lab plan-scale` before harvesting large activation corpora. It accepts sizes like `70B`, `1T`, `1B`, and `64GB`, writes JSON with `--out`, and includes agent next actions for automated workflows. Add `--from-env` for local advisory routing or pass `--env-profile` from a target machine. For 1T+ models, keep the model runtime colocated with the infrastructure that can serve it, write sharded activation records or SAE records, then run interp-lab ranking and report generation over the evidence layer.

Details live in `docs/SCALING.md`.

## Operator Notes

- Keep large generated artifacts outside git. The default `.gitignore` excludes `reports/`.
- Commit small examples and smoke-test configs.
- Prefer config runs for shareable results.
- Include the `manifest.json` when sending reports to collaborators.
