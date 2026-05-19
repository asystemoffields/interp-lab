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

## Build Prompts

```python
from interp_lab import build_prompts

dataset = build_prompts(
    positive="prompts/code-positive.txt",
    negative_prompt="A neutral control prompt.",
    split="paragraphs",
    id_prefix="code",
    out="prompts/code-criterion.jsonl",
)

print(dataset.record_count)
```

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
from interp_lab import run, scaffold_run

scaffold = scaffold_run(
    out="runs/distilgpt2-sae.json",
    workflow="sae",
    model="distilgpt2",
    criterion="the next token should be a physical measurement unit",
    positive_prompt="The answer is measured in meters.",
    negative_prompt="The answer is a person's name.",
    include_causal=True,
    target_token="auto",
    run_dir="reports/distilgpt2-sae-run",
)

run(scaffold.path)
```

## Diagnostics

```python
from interp_lab import doctor

diagnostics = doctor()
print(diagnostics["ok"])
```

## Environment Profiles

```python
from interp_lab import profile_environment, scale_plan

env = profile_environment("reports")
print(env["routing"]["suggested_profile"])

plan = scale_plan(
    model_params=70_000_000_000,
    tokens=10_000_000,
    d_model=8192,
    selected_layers=4,
    env_profile=env,
)
```

Use `env_profile="reports/env-profile.json"` to plan against a saved profile from another machine.

## Graphs, Publishing, And Scale Plans

```python
from interp_lab import (
    attribution_graph,
    attribution_graph_summary,
    publish_hf_artifact,
    scale_plan,
    validate_attribution_graph,
)

graph = attribution_graph("reports/model-a/report.json")

written_graph = attribution_graph(
    "reports/model-a/report.json",
    out="reports/model-a/graph.json",
    markdown_out="reports/model-a/graph.md",
)

validation = validate_attribution_graph(
    "reports/model-a/graph.json",
    path_records="reports/model-a/heldout-paths.jsonl",
    out="reports/model-a/validation.json",
    graph_out="reports/model-a/validated-graph.json",
)

summary = attribution_graph_summary(
    "reports/model-a/validated-graph.json",
    out="reports/model-a/graph-summary.json",
)

publish_hf_artifact(
    repo_id="your-user/interp-lab-demo",
    paths=["reports/model-a"],
    repo_type="dataset",
    dry_run=True,
)

plan = scale_plan(
    model_params=1e12,
    tokens=1_000_000_000,
    d_model=16384,
    selected_layers=8,
    latent_dim=1_048_576,
    profile="frontier-lab",
    target_shard_size_bytes=64 * 1024**3,
    top_k_active=64,
)
```

## Extension Points

Advanced users can import provider and runner classes from `interp_lab.providers`:

```python
from interp_lab.providers import FeatureProvider, GoodfireFeatureProvider, InterventionRunner
```

Stable report and schema helpers live in `interp_lab.artifacts`:

```python
from interp_lab.artifacts import InspectionReport, load_inspection_report
```

The API facade is intentionally thin. New model adapters, feature sources, causal evaluators, and report formats can be added behind it without changing the main happy path.
