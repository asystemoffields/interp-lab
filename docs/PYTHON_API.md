# Python API

The public package is `interp_lab`.

## Inspect

```python
from interp_lab import inspect

report = inspect(
    "toy/model-a",
    "the model is aware it is being evaluated",
    backend="toy",
)
```

Write JSON and Markdown reports by passing `out`:

```python
result = inspect(
    "toy/model-a",
    "the model is aware it is being evaluated",
    backend="toy",
    out="reports/model-a",
)

print(result.json_path)
print(result.markdown_path)
```

## Compare

```python
from interp_lab import compare, inspect

left = inspect("toy/model-a", "benchmark awareness", backend="toy")
right = inspect("toy/model-b", "benchmark awareness", backend="toy")

matches = compare(left, right, out="reports/matches.json")
```

`compare` accepts in-memory reports or `report.json` paths.

## Train An SAE

```python
from interp_lab import train_sae

result = train_sae(
    records="examples/activation_records.jsonl",
    model="toy-records/model",
    out="reports/sae/sae.json",
    records_out="reports/sae/records.jsonl",
    method="fallback",
    latent_dim=64,
)
```

For Hugging Face models:

```python
result = train_sae(
    hf_model="distilgpt2",
    dataset="examples/hf_prompts_unit_prediction.jsonl",
    criterion="the next token should be a physical measurement unit",
    layer=6,
    preset="production",
    latent_dim=256,
    out="reports/distilgpt2-sae/sae.json",
    records_out="reports/distilgpt2-sae/records.jsonl",
    causal_out="reports/distilgpt2-sae/interventions.jsonl",
)
```

## Run Configs

```python
from interp_lab import run

run("examples/run_records.json")
```

## Diagnostics

```python
from interp_lab import doctor

diagnostics = doctor()
print(diagnostics["ok"])
```

## Extension Points

Advanced users can import provider and runner classes from `interp_lab.providers`:

```python
from interp_lab.providers import FeatureProvider, InterventionRunner
```

Stable report and schema helpers live in `interp_lab.artifacts`:

```python
from interp_lab.artifacts import InspectionReport, load_inspection_report
```

The API facade is intentionally thin. New model adapters, feature sources, causal evaluators, and report formats can be added behind it without changing the main happy path.
