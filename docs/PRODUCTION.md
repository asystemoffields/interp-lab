# Production Guide

Oracle SAE is meant to work for quick local experiments and repeatable research runs. The production path is:

```bash
oracle-sae doctor
oracle-sae run examples/run_records.json
```

## Supported Platforms

CI runs on:

- Ubuntu latest
- macOS latest
- Windows latest
- Python 3.10, 3.11, and 3.12

The package avoids shell-specific workflows in the core CLI. Commands in docs use forward slashes because Python accepts them on Windows, macOS, and Linux.

## Reproducible Runs

Use `oracle-sae run` when you want a saved, replayable workflow.

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

- Oracle SAE version
- Python and platform info
- rendered config
- input file hashes for small/medium inputs
- ordered step records
- exit codes and timestamps

Use `--dry-run` to print commands:

```bash
oracle-sae run examples/run_records.json --dry-run
```

Use `--var KEY=VALUE` for templates:

```bash
oracle-sae run run.json --var MODEL=distilgpt2
```

Config strings can reference `{MODEL}`, `{run_dir}`, `{config_dir}`, or `${MODEL}`.

## Environment Diagnostics

`oracle-sae doctor` reports core runtime status and optional adapter dependencies:

- `PyYAML` for YAML configs
- `torch` and `transformers` for Hugging Face activation collection and SAE training
- `sae-lens` for public SAE Lens adapters

Use JSON output in scripts:

```bash
oracle-sae doctor --json
```

## Release Checklist

Before publishing a release:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m compileall src tests
python -m build
oracle-sae doctor
oracle-sae run examples/run_records.json
```

For model-backed release examples:

```bash
python -m pip install -e ".[hf,train]"
oracle-sae train-sae --help
oracle-sae export-hf-records --help
```

## Operator Notes

- Keep large generated artifacts outside git. The default `.gitignore` excludes `reports/`.
- Commit small examples and smoke-test configs.
- Prefer config runs for shareable results.
- Include the `manifest.json` when sending reports to collaborators.
