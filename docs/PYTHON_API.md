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

Write JSON and Markdown reports by passing `out`. Pass `html_out` for a self-contained searchable feature-card report, and `csv_out` for a spreadsheet/paper-friendly CSV of the ranked features:

```python
result = inspect(
    "toy/model-a",
    "the model is aware it is being evaluated",
    backend="toy",
    out="reports/model-a",
    html_out="reports/model-a/report.html",
    csv_out="reports/model-a/features.csv",
)

print(result.json_path, result.markdown_path, result.html_path, result.csv_path)
```

### In a notebook

The result objects and dataclasses are notebook-friendly: `repr()` is compact (no
dumped float vectors), and you can get a table without depending on pandas:

```python
report = inspect("toy/model-a", "benchmark awareness", backend="toy")
report                      # InspectionReport(model='toy/model-a', criterion='…', cards=8)
report.cards_table()        # list[dict]: rank, feature_id, importance, causal_provenance, …
# import pandas as pd; pd.DataFrame(report.cards_table())
```

Load a written report back with the top-level loaders:

```python
from interp_lab import load_inspection_report, load_match_report
report = load_inspection_report("reports/model-a/report.json")
```

## Compare

```python
from interp_lab import compare, inspect

left = inspect("toy/model-a", "benchmark awareness", backend="toy")
right = inspect("toy/model-b", "benchmark awareness", backend="toy")

matches = compare(left, right, out="reports/matches.json")
```

`compare` accepts in-memory reports, `report.json` paths, or the `WrittenInspection`
objects returned by `inspect(out=...)` — so the whole workflow chains:

```python
a = inspect("toy/model-a", "benchmark awareness", backend="toy", out="reports/a")
b = inspect("toy/model-b", "benchmark awareness", backend="toy", out="reports/b")
matches = compare(a, b, out="reports/matches.json")          # accepts WrittenInspection
validation = validate_matches(matches, out="reports/validation.json")  # accepts WrittenMatch
```

Pass `min_score=` to drop weak pairs or `weights={"text": 0.4, "causal": 0.3, ...}` to
tune the fingerprint components. `MatchReport.matches_table()` gives a notebook table.

Validate candidate equivalents:

```python
from interp_lab import validate_matches

validation = validate_matches(
    matches.report,
    out="reports/match-validation.json",
    html_out="reports/match-validation.html",
)
print(validation.report["summary"]["overall_claim_grade"])
```

`validate_matches` accepts an in-memory `MatchReport`, a `matches.json` path, or the `WrittenMatch` from `compare(out=...)`, and writes JSON plus Markdown when `out` is supplied. Pass `html_out` for a self-contained searchable report.

## Compare runs

Diff two inspection reports to catch rank drift or regressions across seeds, checkpoints, or tool versions:

```python
from interp_lab import compare_runs

diff = compare_runs("reports/baseline/report.json", "reports/candidate/report.json")
print(diff["summary"]["rank_stability"], diff["interpretation"])
for mover in diff["changed_features"][:5]:
    print(mover["feature_id"], mover["importance_delta"], mover["rank_delta"])
```

Pass `out=` to write the diff as JSON plus a readable Markdown summary. `left` is the baseline, `right` the candidate.

## Explanation Workflows

Use external Natural Language Autoencoder or autointerp records when inspecting features:

```python
from interp_lab import inspect

report = inspect(
    "toy/model-a",
    "successful tool calls",
    backend="jsonl",
    features="reports/features.jsonl",
    verbalizer="nla",
    nla_explanations="reports/nla-explanations.jsonl",
)
```

Check paraphrase consistency, search explanations, and compare model families:

```python
from interp_lab import (
    check_explanation_consistency,
    compare_model_families,
    match_text_pivot,
    search_features,
)

consistency = check_explanation_consistency(
    ["reports/tool-calls/report.json", "reports/successful-tool-use/report.json"],
    out="reports/explanation-consistency.json",
    html_out="reports/explanation-consistency.html",
)

hits = search_features(
    "features that represent valid tool-call arguments",
    "reports/tool-calls/report.json",
    out="reports/feature-search.json",
    html_out="reports/feature-search.html",
)

text_matches = match_text_pivot(
    "reports/gemma/report.json",
    "reports/qwen/report.json",
    out="reports/text-pivot-matches.json",
    html_out="reports/text-pivot-matches.html",
)

families = compare_model_families(
    [
        {"family": "gemma", "report": "reports/gemma/report.json"},
        {"family": "qwen", "report": "reports/qwen/report.json"},
    ],
    out="reports/model-family-comparison.json",
    html_out="reports/model-family-comparison.html",
)
```

Each function returns machine-readable JSON when `out` is omitted, or a `WrittenAnalysis` with JSON, Markdown, and optional HTML paths when `out` is supplied.

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

Prepare train, causal, and held-out splits for SAE training:

```python
from interp_lab import prepare_sae_prompts

pack = prepare_sae_prompts(
    dataset="prompts/code-criterion.jsonl",
    out_dir="prompts/code-sae-pack",
    latent_dim=1024,
    max_length=128,
)

print(pack.manifest_path)
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

## Intervene On Features

```python
from interp_lab import intervene

plan = intervene(
    model="distilgpt2",
    dataset="prompts/unit-sae-pack/causal.jsonl",
    criterion="the next token should be a physical measurement unit",
    features=["SAE:L6:F30"],
    records="reports/distilgpt2-sae/records.jsonl",
    sae="reports/distilgpt2-sae/sae.json",
    mode="suppress",
    strength_sweep=[1.0, 3.0, 10.0],
    target_tokens=["auto"],
    out="reports/distilgpt2-sae/interventions.jsonl",
    plan_out="reports/distilgpt2-sae/intervention-plan.json",
    dry_run=True,
)

print(plan.plan["estimated_forward_passes"])
```

Set `dry_run=False` to write intervention records that can be passed into `inspect(..., interventions=...)`.

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
    latent_dim=1024,
    run_dir="reports/distilgpt2-sae-run",
)

run(scaffold.path)
```

SAE scaffolds prepare prompt packs by default. Set `prepare_sae_prompts=False` when `dataset` already points to the exact training split you want.

For SAE-latent path experiments, scaffold a two-layer workflow:

```python
scaffold = scaffold_run(
    out="runs/distilgpt2-sae-paths.json",
    workflow="sae-paths",
    model="distilgpt2",
    criterion="the next token should be a physical measurement unit",
    dataset="examples/hf_prompts_unit_prediction.jsonl",
    validation_dataset="examples/hf_prompts_unit_prediction.jsonl",
    source_layer=2,
    target_layer=4,
    include_causal=True,
    target_token="auto",
    validate_paths=True,
    latent_dim=1024,
    torch_dtype="auto",
    device_map="auto",
    run_dir="reports/distilgpt2-sae-paths",
)
```

## Criterion Labs

```python
from interp_lab import criterion_lab, run

lab = criterion_lab(
    model="distilgpt2",
    preset="overconfidence",
    out="reports/overconfidence-lab/run.json",
    run_dir="reports/overconfidence-lab",
)

run(lab.path)
```

Criterion Lab presets are JSON prompt assays. The default config runs all-layer discovery with activation records, feature inspection, and graph export. Use a project preset file when an agent or researcher defines a new criterion:

```python
from interp_lab import validate_criterion_assay

validation = validate_criterion_assay(
    preset_file="examples/presets/math-reasoning.json",
)

lab = criterion_lab(
    model="distilgpt2",
    preset_file="examples/presets/math-reasoning.json",
    out="reports/math-reasoning-lab/run.json",
)
```

Use `criterion_lab_presets()` to list bundled and project presets. For a one-off run, pass `criterion`, `positive_prompt`, and `negative_prompt` directly. After discovery surfaces promising layers, call `criterion_lab(..., workflow="sae", layer=layer_index)` to train and causally test an SAE on that layer.

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

## Release Readiness

```python
from interp_lab import demo_sweep, public_api_contract, release_check

report = release_check(".")
print(report["ready_for_stable_release"])
print(report["agent_next_actions"])

sweep = demo_sweep(out="reports/real-model-demo-sweep.json")
print(sweep["status"])

contract = public_api_contract()
print(contract["schema_version"])
print(contract["exports"])
print(contract["schemas"])
```

`public_api_contract()` returns a JSON-serializable contract for agents, integration tests, and downstream wrappers. It lists the stable exports, schema ids, and core callable parameters that should be changed intentionally.

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
    html_out="reports/model-a/graph.html",
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
