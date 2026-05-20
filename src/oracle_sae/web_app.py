from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


COMMAND_SPECS: list[dict[str, Any]] = [
    {
        "id": "inspect",
        "group": "Discovery",
        "label": "Inspect Features",
        "description": "Rank and explain internal features for a natural-language criterion.",
        "fields": [
            {"key": "model", "flag": "--model", "label": "Model", "required": True, "placeholder": "google/gemma-2-2b"},
            {"key": "criterion", "flag": "--criterion", "label": "Criterion", "required": True, "placeholder": "the model is aware it is being evaluated"},
            {"key": "backend", "flag": "--backend", "label": "Backend", "type": "select", "default": "toy", "options": ["toy", "jsonl", "records", "neuronpedia", "saelens", "goodfire", "scope"]},
            {"key": "records", "flag": "--records", "label": "Activation Records", "placeholder": "records.jsonl"},
            {"key": "features", "flag": "--features", "label": "Feature JSONL", "placeholder": "features.jsonl"},
            {"key": "interventions", "flag": "--interventions", "label": "Intervention Records", "placeholder": "interventions.jsonl"},
            {"key": "top_k", "flag": "--top-k", "label": "Top K", "type": "number", "default": "8"},
            {"key": "out", "flag": "--out", "label": "Output Directory", "default": "reports/inspection"},
            {"key": "html_out", "flag": "--html-out", "label": "HTML Report", "default": "reports/inspection/report.html"},
            {"key": "require_interventions", "flag": "--require-interventions", "label": "Require intervention evidence", "type": "boolean"},
        ],
    },
    {
        "id": "build-prompts",
        "group": "Data",
        "label": "Build Prompts",
        "description": "Create a scored prompt JSONL dataset from files or inline prompts.",
        "fields": [
            {"key": "positive", "flag": "--positive", "label": "Positive prompt files", "type": "repeat", "placeholder": "prompts/positive.txt"},
            {"key": "negative", "flag": "--negative", "label": "Negative prompt files", "type": "repeat", "placeholder": "prompts/negative.txt"},
            {"key": "positive_prompt", "flag": "--positive-prompt", "label": "Inline positive prompts", "type": "repeat"},
            {"key": "negative_prompt", "flag": "--negative-prompt", "label": "Inline negative prompts", "type": "repeat"},
            {"key": "split", "flag": "--split", "label": "Split", "type": "select", "default": "paragraphs", "options": ["paragraphs", "lines"]},
            {"key": "out", "flag": "--out", "label": "Output JSONL", "default": "prompts/criterion.jsonl"},
        ],
    },
    {
        "id": "criterion-lab",
        "group": "Guided Labs",
        "label": "Criterion Lab",
        "description": "Scaffold a discovery-first behavior workflow from a criterion or preset JSON file.",
        "fields": [
            {"key": "model", "flag": "--model", "label": "Model", "required": True, "placeholder": "distilgpt2"},
            {"key": "preset", "flag": "--preset", "label": "Preset name or JSON path", "placeholder": "overconfidence or presets/math-reasoning.json"},
            {"key": "preset_file", "flag": "--preset-file", "label": "Preset JSON file", "placeholder": "examples/presets/math-reasoning.json"},
            {"key": "preset_dir", "flag": "--preset-dir", "label": "Preset directories", "type": "repeat", "placeholder": "presets"},
            {"key": "list_presets", "flag": "--list-presets", "label": "List discoverable presets", "type": "boolean"},
            {"key": "criterion", "flag": "--criterion", "label": "Criterion override"},
            {"key": "workflow", "flag": "--workflow", "label": "Workflow", "type": "select", "default": "discovery", "options": ["discovery", "hf-records", "sae", "sae-paths"]},
            {"key": "training_preset", "flag": "--training-preset", "label": "SAE preset", "type": "select", "default": "minimal", "options": ["minimal", "production", "custom"]},
            {"key": "layers", "flag": "--layers", "label": "Discovery layers", "default": "all", "placeholder": "all, 0,4,8, or 2-6"},
            {"key": "run_dir", "flag": "--run-dir", "label": "Run directory", "default": "reports/criterion-lab"},
            {"key": "out", "flag": "--out", "label": "Run config path", "default": "reports/criterion-lab/run.json"},
            {"key": "positive_prompt", "flag": "--positive-prompt", "label": "Extra positive prompts", "type": "repeat"},
            {"key": "negative_prompt", "flag": "--negative-prompt", "label": "Extra negative prompts", "type": "repeat"},
            {"key": "no_preset_prompts", "flag": "--no-preset-prompts", "label": "Use only my prompts", "type": "boolean"},
            {"key": "use_preset_target_hints", "flag": "--use-preset-target-hints", "label": "Use preset target hints", "type": "boolean"},
            {"key": "skip_causal", "flag": "--skip-causal", "label": "Skip first-pass causal scoring", "type": "boolean"},
            {"key": "target_token", "flag": "--target-token", "label": "Target tokens", "type": "repeat"},
            {"key": "layer", "flag": "--layer", "label": "SAE layer", "type": "number"},
            {"key": "source_layer", "flag": "--source-layer", "label": "Source layer", "type": "number"},
            {"key": "target_layer", "flag": "--target-layer", "label": "Target layer", "type": "number"},
            {"key": "device", "flag": "--device", "label": "Device", "default": "cpu"},
            {"key": "execute", "flag": "--execute", "label": "Execute after writing config", "type": "boolean"},
            {"key": "force", "flag": "--force", "label": "Overwrite config", "type": "boolean"},
        ],
    },
    {
        "id": "validate-assay",
        "group": "Guided Labs",
        "label": "Validate Assay",
        "description": "Check a user-authored Criterion Lab assay before launching discovery.",
        "fields": [
            {"key": "preset", "flag": "--preset", "label": "Preset name or JSON path", "placeholder": "overconfidence or presets/refusal.json"},
            {"key": "preset_file", "flag": "--preset-file", "label": "Assay JSON file", "placeholder": "examples/presets/math-reasoning.json"},
            {"key": "preset_dir", "flag": "--preset-dir", "label": "Preset directories", "type": "repeat", "placeholder": "presets"},
            {"key": "out", "flag": "--out", "label": "Validation JSON", "default": "reports/assay-validation.json"},
            {"key": "fail_on_warning", "flag": "--fail-on-warning", "label": "Fail on warnings", "type": "boolean"},
        ],
    },
    {
        "id": "export-hf-records",
        "group": "Data",
        "label": "Export HF Records",
        "description": "Export Hugging Face hidden states as activation records.",
        "fields": [
            {"key": "model", "flag": "--model", "label": "Model", "required": True},
            {"key": "dataset", "flag": "--dataset", "label": "Prompt Dataset", "required": True},
            {"key": "layer", "flag": "--layer", "label": "Layer", "type": "number"},
            {"key": "out", "flag": "--out", "label": "Output JSONL", "default": "records/hf-records.jsonl"},
            {"key": "features_per_layer", "flag": "--features-per-layer", "label": "Features per layer", "type": "number", "default": "16"},
            {"key": "pool", "flag": "--pool", "label": "Pool", "type": "select", "default": "last", "options": ["last", "mean"]},
            {"key": "device", "flag": "--device", "label": "Device", "default": "cpu"},
        ],
    },
    {
        "id": "train-sae",
        "group": "SAE",
        "label": "Train SAE",
        "description": "Train an on-demand SAE from records or HF activations.",
        "fields": [
            {"key": "out", "flag": "--out", "label": "SAE artifact", "required": True, "default": "reports/sae/sae.json"},
            {"key": "records", "flag": "--records", "label": "Activation records"},
            {"key": "model", "flag": "--model", "label": "Model name for records"},
            {"key": "hf_model", "flag": "--hf-model", "label": "HF model"},
            {"key": "dataset", "flag": "--dataset", "label": "Prompt dataset"},
            {"key": "preset", "flag": "--preset", "label": "Preset", "type": "select", "default": "minimal", "options": ["minimal", "production", "custom"]},
            {"key": "layer", "flag": "--layer", "label": "Layer", "type": "number"},
            {"key": "device", "flag": "--device", "label": "Device", "default": "cpu"},
            {"key": "records_out", "flag": "--records-out", "label": "Learned latent records"},
            {"key": "causal_out", "flag": "--causal-out", "label": "Causal intervention records"},
            {"key": "criterion", "flag": "--criterion", "label": "Criterion for causal eval"},
            {"key": "target_tokens", "flag": "--target-token", "label": "Target tokens", "type": "repeat"},
        ],
    },
    {
        "id": "export-hf-interventions",
        "group": "Causal",
        "label": "Export HF Interventions",
        "description": "Run hidden-dimension ablations or steering and write intervention records.",
        "fields": [
            {"key": "model", "flag": "--model", "label": "Model", "required": True},
            {"key": "dataset", "flag": "--dataset", "label": "Prompt Dataset", "required": True},
            {"key": "report", "flag": "--report", "label": "Inspection Report", "required": True},
            {"key": "out", "flag": "--out", "label": "Output JSONL", "default": "reports/interventions.jsonl"},
            {"key": "criterion", "flag": "--criterion", "label": "Criterion"},
            {"key": "feature", "flag": "--feature", "label": "Feature ids", "type": "repeat"},
            {"key": "target_tokens", "flag": "--target-token", "label": "Target tokens", "type": "repeat"},
            {"key": "device", "flag": "--device", "label": "Device", "default": "cpu"},
        ],
    },
    {
        "id": "export-hf-contrast",
        "group": "Causal",
        "label": "Export HF Contrast",
        "description": "Export a contrast-direction feature and optional steering interventions.",
        "fields": [
            {"key": "model", "flag": "--model", "label": "Model", "required": True},
            {"key": "dataset", "flag": "--dataset", "label": "Prompt Dataset", "required": True},
            {"key": "criterion", "flag": "--criterion", "label": "Criterion", "required": True},
            {"key": "out", "flag": "--out", "label": "Feature JSONL", "default": "reports/contrast-feature.jsonl"},
            {"key": "interventions_out", "flag": "--interventions-out", "label": "Interventions JSONL"},
            {"key": "layer", "flag": "--layer", "label": "Layer", "type": "number"},
            {"key": "target_tokens", "flag": "--target-token", "label": "Target tokens", "type": "repeat"},
        ],
    },
    {
        "id": "match",
        "group": "Cross-Model",
        "label": "Match Features",
        "description": "Search for candidate equivalent features across two inspection reports.",
        "fields": [
            {"key": "left", "flag": "--left", "label": "Left report", "required": True, "default": "reports/a/report.json"},
            {"key": "right", "flag": "--right", "label": "Right report", "required": True, "default": "reports/b/report.json"},
            {"key": "top_k", "flag": "--top-k", "label": "Top K", "type": "number", "default": "10"},
            {"key": "out", "flag": "--out", "label": "Output JSON", "default": "reports/matches.json"},
        ],
    },
    {
        "id": "validate-matches",
        "group": "Cross-Model",
        "label": "Validate Matches",
        "description": "Grade cross-model equivalence claims and write reason codes.",
        "fields": [
            {"key": "matches", "flag": "--matches", "label": "Matches JSON", "required": True, "default": "reports/matches.json"},
            {"key": "out", "flag": "--out", "label": "Output JSON", "default": "reports/match-validation.json"},
            {"key": "html_out", "flag": "--html-out", "label": "HTML Report", "default": "reports/match-validation.html"},
            {"key": "top_k", "flag": "--top-k", "label": "Top K", "type": "number"},
        ],
    },
    {
        "id": "export-attribution-graph",
        "group": "Graphs",
        "label": "Export Attribution Graph",
        "description": "Build a mechanism graph from one or more inspection reports.",
        "fields": [
            {"key": "report", "flag": "--report", "label": "Inspection reports", "type": "repeat", "required": True, "placeholder": "reports/model-a/report.json"},
            {"key": "out", "flag": "--out", "label": "Graph JSON", "default": "reports/graph.json"},
            {"key": "markdown_out", "flag": "--markdown-out", "label": "Markdown report", "default": "reports/graph.md"},
            {"key": "html_out", "flag": "--html-out", "label": "HTML viewer", "default": "reports/graph.html"},
            {"key": "include_similarity_edges", "flag": "--include-similarity-edges", "label": "Include similarity edges", "type": "boolean"},
            {"key": "similarity_threshold", "flag": "--similarity-threshold", "label": "Similarity threshold", "type": "number", "default": "0.9"},
        ],
    },
    {
        "id": "summarize-attribution-graph",
        "group": "Graphs",
        "label": "Summarize Graph",
        "description": "Write a compact graph summary for agents and scripts.",
        "fields": [
            {"key": "graph", "flag": "--graph", "label": "Graph JSON", "required": True, "default": "reports/graph.json"},
            {"key": "out", "flag": "--out", "label": "Summary JSON", "default": "reports/graph-summary.json"},
        ],
    },
    {
        "id": "validate-attribution-graph",
        "group": "Graphs",
        "label": "Validate Graph",
        "description": "Validate measured attribution graph paths with path-patching records.",
        "fields": [
            {"key": "graph", "flag": "--graph", "label": "Graph JSON", "required": True, "default": "reports/graph.json"},
            {"key": "path_records", "flag": "--path-records", "label": "Path records", "type": "repeat", "required": True},
            {"key": "out", "flag": "--out", "label": "Validation JSON", "default": "reports/graph-validation.json"},
            {"key": "graph_out", "flag": "--graph-out", "label": "Annotated graph JSON"},
            {"key": "graph_html_out", "flag": "--graph-html-out", "label": "Annotated graph HTML"},
            {"key": "allow_missing_controls", "flag": "--allow-missing-controls", "label": "Allow missing controls", "type": "boolean"},
        ],
    },
    {
        "id": "export-hf-sae-paths",
        "group": "SAE Paths",
        "label": "Export SAE Paths",
        "description": "Patch source SAE latents and measure target SAE latent paths.",
        "fields": [
            {"key": "model", "flag": "--model", "label": "Model", "required": True},
            {"key": "dataset", "flag": "--dataset", "label": "Prompt Dataset", "required": True},
            {"key": "source_sae", "flag": "--source-sae", "label": "Source SAE", "required": True},
            {"key": "target_sae", "flag": "--target-sae", "label": "Target SAE", "required": True},
            {"key": "out", "flag": "--out", "label": "Path records JSONL", "default": "reports/sae-paths.jsonl"},
            {"key": "criterion", "flag": "--criterion", "label": "Criterion"},
            {"key": "source_report", "flag": "--source-report", "label": "Source report"},
            {"key": "target_report", "flag": "--target-report", "label": "Target report"},
            {"key": "random_source_controls", "flag": "--random-source-controls", "label": "Random controls", "type": "number", "default": "2"},
        ],
    },
    {
        "id": "validate-hf-sae-paths",
        "group": "SAE Paths",
        "label": "Validate SAE Paths",
        "description": "Rerun graph candidate SAE paths on held-out prompts and validate them.",
        "fields": [
            {"key": "graph", "flag": "--graph", "label": "Graph JSON", "required": True},
            {"key": "model", "flag": "--model", "label": "Model", "required": True},
            {"key": "dataset", "flag": "--dataset", "label": "Held-out Dataset", "required": True},
            {"key": "source_sae", "flag": "--source-sae", "label": "Source SAE", "required": True},
            {"key": "target_sae", "flag": "--target-sae", "label": "Target SAE", "required": True},
            {"key": "path_records_out", "flag": "--path-records-out", "label": "Path records out", "default": "reports/heldout-paths.jsonl"},
            {"key": "out", "flag": "--out", "label": "Validation JSON", "default": "reports/path-validation.json"},
            {"key": "graph_out", "flag": "--graph-out", "label": "Annotated graph JSON"},
            {"key": "graph_html_out", "flag": "--graph-html-out", "label": "Annotated graph HTML"},
        ],
    },
    {
        "id": "profile-env",
        "group": "Planning",
        "label": "Profile Environment",
        "description": "Inspect local compute, storage, and route options.",
        "fields": [
            {"key": "out", "flag": "--out", "label": "Output JSON", "default": "reports/env-profile.json"},
            {"key": "path", "flag": "--path", "label": "Filesystem path", "default": "."},
            {"key": "json", "flag": "--json", "label": "Print JSON", "type": "boolean", "default_checked": True},
        ],
    },
    {
        "id": "plan-scale",
        "group": "Planning",
        "label": "Plan Scale",
        "description": "Estimate storage and execution shape for large model runs.",
        "fields": [
            {"key": "model_params", "flag": "--model-params", "label": "Model params", "required": True, "placeholder": "1T"},
            {"key": "tokens", "flag": "--tokens", "label": "Tokens", "required": True, "placeholder": "100M"},
            {"key": "d_model", "flag": "--d-model", "label": "d_model", "required": True, "placeholder": "8192"},
            {"key": "selected_layers", "flag": "--selected-layers", "label": "Selected layers", "type": "number", "default": "1"},
            {"key": "latent_dim", "flag": "--latent-dim", "label": "Latent dim", "default": "131072"},
            {"key": "profile", "flag": "--profile", "label": "Profile", "type": "select", "default": "auto", "options": ["auto", "local", "single-gpu", "cluster", "remote"]},
        ],
    },
    {
        "id": "init-run",
        "group": "Automation",
        "label": "Scaffold Run Config",
        "description": "Write an editable run config for common workflows.",
        "fields": [
            {"key": "out", "flag": "--out", "label": "Config path", "required": True, "default": "runs/interp-lab-run.json"},
            {"key": "workflow", "flag": "--workflow", "label": "Workflow", "type": "select", "default": "records", "options": ["records", "hf-records", "sae", "sae-paths"]},
            {"key": "model", "flag": "--model", "label": "Model", "required": True},
            {"key": "criterion", "flag": "--criterion", "label": "Criterion", "required": True},
            {"key": "run_dir", "flag": "--run-dir", "label": "Run directory", "default": "reports/interp-run"},
        ],
    },
    {
        "id": "run",
        "group": "Automation",
        "label": "Run Config",
        "description": "Run a reproducible JSON/TOML/YAML workflow config.",
        "positional": [{"key": "config", "label": "Config path", "required": True, "default": "runs/interp-lab-run.json"}],
        "fields": [{"key": "dry_run", "flag": "--dry-run", "label": "Dry run", "type": "boolean"}],
    },
    {
        "id": "publish-hf-artifact",
        "group": "Sharing",
        "label": "Publish HF Artifact",
        "description": "Package reports and records for Hugging Face Hub.",
        "fields": [
            {"key": "repo_id", "flag": "--repo-id", "label": "Repo id", "required": True, "placeholder": "user/interp-lab-demo"},
            {"key": "path", "flag": "--path", "label": "Artifact paths", "type": "repeat", "required": True, "placeholder": "reports/demo"},
            {"key": "repo_type", "flag": "--repo-type", "label": "Repo type", "type": "select", "default": "dataset", "options": ["dataset", "model", "space"]},
            {"key": "private", "flag": "--private", "label": "Private repo", "type": "boolean"},
            {"key": "dry_run", "flag": "--dry-run", "label": "Dry run", "type": "boolean", "default_checked": True},
        ],
    },
    {
        "id": "doctor",
        "group": "Utility",
        "label": "Doctor",
        "description": "Check the local interp-lab environment.",
        "fields": [{"key": "json", "flag": "--json", "label": "Print JSON", "type": "boolean"}],
    },
    {
        "id": "demo",
        "group": "Utility",
        "label": "Demo Tour",
        "description": "Write a complete toy demo with reports, HTML, matches, graph, and summary.",
        "fields": [{"key": "out", "flag": "--out", "label": "Output directory", "default": "reports/demo"}],
    },
]


def command_specs_from_parser(
    parser: argparse.ArgumentParser,
    curated_specs: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build Studio command specs from the real CLI parser.

    The curated specs provide friendlier labels, grouping, and examples. The
    parser remains the source of truth for available commands and options.
    """
    curated = list(COMMAND_SPECS if curated_specs is None else curated_specs)
    curated_by_id = {spec["id"]: spec for spec in curated}
    subparsers = _find_subparsers(parser)
    if subparsers is None:
        return curated

    help_by_name = {
        getattr(action, "dest", ""): getattr(action, "help", "")
        for action in getattr(subparsers, "_choices_actions", [])
    }
    specs: list[dict[str, Any]] = []
    seen_parser_ids: set[int] = set()
    for name, subparser in subparsers.choices.items():
        parser_id = id(subparser)
        if parser_id in seen_parser_ids:
            continue
        seen_parser_ids.add(parser_id)
        generated = _spec_from_subparser(name, subparser, help_by_name.get(name, ""))
        curated_spec = curated_by_id.get(name)
        if curated_spec is not None:
            generated["group"] = curated_spec.get("group", generated["group"])
            generated["label"] = curated_spec.get("label", generated["label"])
            generated["description"] = curated_spec.get("description", generated["description"])
            _merge_curated_fields(generated, curated_spec)
        specs.append(generated)
    return specs


def write_web_app(out_path: str | Path, command_specs: Sequence[dict[str, Any]] | None = None) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_web_app_html(command_specs=command_specs), encoding="utf-8")
    return path


def render_web_app_html(command_specs: Sequence[dict[str, Any]] | None = None) -> str:
    effective_specs = list(COMMAND_SPECS if command_specs is None else command_specs)
    specs = _json_payload(effective_specs)
    command_count = len(effective_specs)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interp Lab Studio</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --ink: #1c2528;
      --muted: #5c686d;
      --line: #d9dedb;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-soft: #e7f5f1;
      --blue: #285e9e;
      --warn: #946200;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      width: min(1220px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 30px 0 56px;
    }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ font-size: 32px; letter-spacing: 0; }}
    h2 {{ font-size: 18px; }}
    h3 {{ font-size: 15px; }}
    code, pre {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    }}
    header {{
      display: grid;
      gap: 16px;
      margin-bottom: 18px;
    }}
    .subhead {{ color: var(--muted); max-width: 920px; }}
    .shell {{
      display: grid;
      grid-template-columns: 300px 1fr;
      gap: 14px;
      align-items: start;
    }}
    .panel, .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .panel {{ padding: 16px; }}
    .sidebar {{
      position: sticky;
      top: 16px;
      display: grid;
      gap: 10px;
    }}
    .command-list {{
      display: grid;
      gap: 6px;
      max-height: calc(100vh - 210px);
      overflow: auto;
      padding-right: 2px;
    }}
    .command-button {{
      width: 100%;
      text-align: left;
      border: 1px solid transparent;
      border-radius: 7px;
      background: #f6f8f7;
      padding: 9px 10px;
      color: var(--ink);
      cursor: pointer;
      font: inherit;
    }}
    .command-button.active {{
      border-color: #89c9be;
      background: var(--accent-soft);
    }}
    .command-button span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
    }}
    .group-label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
      margin-top: 8px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    label {{
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    input, select, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-weight: 400;
    }}
    textarea {{ min-height: 76px; resize: vertical; }}
    .checkbox {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--ink);
      font-size: 14px;
      font-weight: 500;
    }}
    .checkbox input {{ width: auto; }}
    .required::after {{ content: " *"; color: #b42318; }}
    .workspace {{
      display: grid;
      gap: 14px;
    }}
    .hero-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    button.action {{
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--ink);
      padding: 8px 10px;
      cursor: pointer;
      font: inherit;
      font-weight: 650;
    }}
    button.action.primary {{
      border-color: #0f766e;
      background: #0f766e;
      color: #fff;
    }}
    .output-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    pre {{
      margin: 0;
      min-height: 122px;
      max-height: 320px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f3f5f4;
      padding: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 12px;
    }}
    .workflow {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .card {{
      padding: 12px;
      display: grid;
      gap: 8px;
    }}
    .pill {{
      display: inline-flex;
      width: fit-content;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      background: #fff;
    }}
    .note {{
      border-left: 3px solid var(--accent);
      padding-left: 10px;
      color: var(--muted);
    }}
    .status-line {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }}
    .history-list, .artifact-list {{
      display: grid;
      gap: 8px;
      max-height: 260px;
      overflow: auto;
    }}
    .history-item, .artifact-item {{
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #f8faf9;
      padding: 9px 10px;
      display: grid;
      gap: 4px;
      color: var(--ink);
      font: inherit;
      text-align: left;
    }}
    button.artifact-item {{
      cursor: pointer;
    }}
    .item-meta {{
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .artifact-layout {{
      display: grid;
      grid-template-columns: minmax(220px, 0.85fr) minmax(0, 1.15fr);
      gap: 12px;
    }}
    .artifact-frame {{
      width: 100%;
      min-height: 360px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      display: none;
    }}
    .graph-summary {{
      display: grid;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .graph-counts {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .server-disabled {{
      opacity: 0.58;
    }}
    .extra {{
      grid-column: 1 / -1;
    }}
    @media (max-width: 900px) {{
      main {{ width: min(100vw - 20px, 1220px); }}
      .shell, .output-grid, .grid, .workflow, .artifact-layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: static; }}
      .command-list {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Interp Lab Studio</h1>
        <p class="subhead">Build interp-lab commands and reproducible run configs from a browser. The generated output is plain text, so it works in terminals, scripts, and agent workflows.</p>
      </div>
      <div class="hero-actions">
        <span class="pill">{command_count} CLI surfaces</span>
        <span class="pill">static HTML</span>
        <span class="pill">run-config output</span>
      </div>
    </header>
    <section class="panel" style="margin-bottom:14px">
      <h2>Guided Starts</h2>
      <div class="workflow" id="workflow-cards"></div>
    </section>
    <div class="shell">
      <aside class="sidebar panel">
        <h2>Commands</h2>
        <input id="command-search" type="search" placeholder="Filter commands">
        <div id="command-list" class="command-list"></div>
      </aside>
      <section class="workspace">
        <div class="panel">
          <h2 id="command-title"></h2>
          <p id="command-description" class="subhead"></p>
        </div>
        <form id="command-form" class="panel">
          <div id="fields" class="grid"></div>
          <label class="extra">
            Extra flags
            <textarea id="extra-flags" placeholder="--trust-remote-code --torch-dtype auto"></textarea>
          </label>
        </form>
        <section class="output-grid">
          <div class="panel">
            <div class="hero-actions" style="justify-content:space-between;margin-bottom:10px">
              <h2>Command</h2>
              <button class="action primary" id="copy-command" type="button">Copy</button>
            </div>
            <pre id="generated-command"></pre>
          </div>
          <div class="panel">
            <div class="hero-actions" style="justify-content:space-between;margin-bottom:10px">
              <h2>Run Config Step</h2>
              <button class="action" id="copy-config" type="button">Copy</button>
            </div>
            <pre id="run-config-output"></pre>
          </div>
        </section>
        <section class="panel">
          <h2>Suggested Next Actions</h2>
          <div class="note" id="next-action"></div>
        </section>
        <section class="panel" id="server-panel">
          <div class="status-line">
            <div>
              <h2>Local Runner</h2>
              <p class="subhead" id="server-detail">Checking for the local Studio server.</p>
            </div>
            <span class="pill" id="server-status">checking</span>
          </div>
          <div class="hero-actions">
            <button class="action primary" id="start-job" type="button">Run Command</button>
            <button class="action" id="start-config-job" type="button">Run Config</button>
            <button class="action" id="refresh-jobs" type="button">Refresh Jobs</button>
          </div>
          <div class="history-list" id="job-list" style="margin-top:10px"></div>
        </section>
        <section class="panel" id="artifact-panel">
          <div class="status-line">
            <div>
              <h2>Reports And Graphs</h2>
              <p class="subhead">Browse generated artifacts when Studio is served locally.</p>
            </div>
            <button class="action" id="refresh-artifacts" type="button">Refresh Artifacts</button>
          </div>
          <div class="artifact-layout">
            <div class="artifact-list" id="artifact-list"></div>
            <div>
              <h3 id="artifact-title">No artifact selected</h3>
              <div class="graph-summary" id="graph-overview"></div>
              <iframe class="artifact-frame" id="artifact-frame" title="Artifact preview"></iframe>
              <pre id="artifact-preview"></pre>
            </div>
          </div>
        </section>
      </section>
    </div>
  </main>
  <script id="command-specs" type="application/json">{specs}</script>
  <script>
    const commandSpecs = JSON.parse(document.getElementById("command-specs").textContent);
    const workflows = [
      {{ title: "Start with a toy tour", command: "demo", values: {{ out: "reports/demo" }} }},
      {{ title: "Validate an assay", command: "validate-assay", values: {{ preset_file: "examples/presets/math-reasoning.json", out: "reports/assay-validation.json" }} }},
      {{ title: "Discovery-first Criterion Lab", command: "criterion-lab", values: {{ model: "distilgpt2", preset: "overconfidence", workflow: "discovery", layers: "all", training_preset: "minimal", run_dir: "reports/criterion-lab", out: "reports/criterion-lab/run.json", execute: false, force: true }} }},
      {{ title: "Inspect a local/open model", command: "inspect", values: {{ backend: "records", out: "reports/inspection", html_out: "reports/inspection/report.html" }} }},
      {{ title: "Train and inspect an SAE", command: "train-sae", values: {{ preset: "production", out: "reports/sae/sae.json", records_out: "reports/sae/records.jsonl" }} }},
      {{ title: "Compare two models", command: "match", values: {{ left: "reports/a/report.json", right: "reports/b/report.json", out: "reports/matches.json" }} }},
      {{ title: "Validate cross-model matches", command: "validate-matches", values: {{ matches: "reports/matches.json", out: "reports/match-validation.json", html_out: "reports/match-validation.html" }} }},
      {{ title: "Build graph review", command: "export-attribution-graph", values: {{ report: ["reports/model-a/report.json", "reports/model-b/report.json"], out: "reports/graph.json", html_out: "reports/graph.html" }} }},
    ];
    let selected = commandSpecs[0];
    const values = new Map();
    let serverAvailable = false;
    let jobs = [];
    let artifacts = [];

    const byId = (id) => document.getElementById(id);
    const commandList = byId("command-list");
    const fields = byId("fields");

    function renderWorkflows() {{
      byId("workflow-cards").innerHTML = workflows.map((item) => `
        <article class="card">
          <span class="pill">${{escapeHtml(item.command)}}</span>
          <h3>${{escapeHtml(item.title)}}</h3>
          <button class="action" type="button" data-workflow="${{escapeAttr(item.title)}}">Use this</button>
        </article>
      `).join("");
      document.querySelectorAll("[data-workflow]").forEach((button) => {{
        button.addEventListener("click", () => {{
          const item = workflows.find((candidate) => candidate.title === button.dataset.workflow);
          selectCommand(item.command, item.values);
        }});
      }});
    }}

    function renderCommandList() {{
      const query = byId("command-search").value.trim().toLowerCase();
      const groups = new Map();
      for (const spec of commandSpecs) {{
        const haystack = `${{spec.id}} ${{spec.label}} ${{spec.description}} ${{spec.group}}`.toLowerCase();
        if (query && !haystack.includes(query)) continue;
        if (!groups.has(spec.group)) groups.set(spec.group, []);
        groups.get(spec.group).push(spec);
      }}
      commandList.innerHTML = "";
      for (const [group, specs] of groups.entries()) {{
        const label = document.createElement("div");
        label.className = "group-label";
        label.textContent = group;
        commandList.appendChild(label);
        for (const spec of specs) {{
          const button = document.createElement("button");
          button.type = "button";
          button.className = "command-button" + (spec.id === selected.id ? " active" : "");
          button.innerHTML = `${{escapeHtml(spec.label)}}<span>${{escapeHtml(spec.id)}}</span>`;
          button.addEventListener("click", () => selectCommand(spec.id));
          commandList.appendChild(button);
        }}
      }}
    }}

    function selectCommand(id, patchValues = null) {{
      selected = commandSpecs.find((spec) => spec.id === id) || commandSpecs[0];
      if (patchValues) values.set(selected.id, {{ ...currentValues(), ...patchValues }});
      renderCommandList();
      renderForm();
      updateOutputs();
    }}

    function currentValues() {{
      return values.get(selected.id) || {{}};
    }}

    function renderForm() {{
      byId("command-title").textContent = selected.label;
      byId("command-description").textContent = selected.description;
      const saved = currentValues();
      const allFields = [...(selected.positional || []), ...(selected.fields || [])];
      fields.innerHTML = allFields.map((field) => renderField(field, saved)).join("");
      fields.querySelectorAll("input, select, textarea").forEach((input) => {{
        input.addEventListener("input", updateOutputs);
        input.addEventListener("change", updateOutputs);
      }});
      byId("extra-flags").value = saved.__extra || "";
    }}

    function renderField(field, saved) {{
      const value = saved[field.key] ?? field.default ?? "";
      const required = field.required ? " required" : "";
      if (field.type === "boolean") {{
        const checked = saved[field.key] ?? field.default_checked ?? false;
        return `<label class="checkbox"><input data-key="${{escapeAttr(field.key)}}" data-kind="boolean" type="checkbox" ${{checked ? "checked" : ""}}> ${{escapeHtml(field.label)}}</label>`;
      }}
      if (field.type === "select") {{
        const emptyOption = field.required ? "" : `<option value="" ${{value === "" ? "selected" : ""}}>Choose...</option>`;
        const options = emptyOption + (field.options || []).map((option) => `<option value="${{escapeAttr(option)}}" ${{String(value) === String(option) ? "selected" : ""}}>${{escapeHtml(option)}}</option>`).join("");
        return `<label class="${{required}}">${{escapeHtml(field.label)}}<select data-key="${{escapeAttr(field.key)}}" data-kind="value">${{options}}</select></label>`;
      }}
      if (field.type === "repeat") {{
        const text = Array.isArray(value) ? value.join("\\n") : value;
        return `<label class="${{required}}">${{escapeHtml(field.label)}}<textarea data-key="${{escapeAttr(field.key)}}" data-kind="repeat" placeholder="${{escapeAttr(field.placeholder || 'one value per line')}}">${{escapeHtml(text || "")}}</textarea></label>`;
      }}
      return `<label class="${{required}}">${{escapeHtml(field.label)}}<input data-key="${{escapeAttr(field.key)}}" data-kind="value" type="${{field.type || 'text'}}" value="${{escapeAttr(value)}}" placeholder="${{escapeAttr(field.placeholder || '')}}"></label>`;
    }}

    function readForm() {{
      const data = {{}};
      fields.querySelectorAll("[data-key]").forEach((input) => {{
        const key = input.dataset.key;
        if (input.dataset.kind === "boolean") {{
          if (input.checked) data[key] = true;
        }} else if (input.dataset.kind === "repeat") {{
          const parts = input.value.split(/\\r?\\n/).map((item) => item.trim()).filter(Boolean);
          if (parts.length) data[key] = parts;
        }} else if (input.value.trim()) {{
          data[key] = input.value.trim();
        }}
      }});
      const extra = byId("extra-flags").value.trim();
      if (extra) data.__extra = extra;
      values.set(selected.id, data);
      return data;
    }}

    function updateOutputs() {{
      const data = readForm();
      const command = buildCommand(selected, data);
      byId("generated-command").textContent = command;
      byId("run-config-output").textContent = JSON.stringify(buildRunConfig(selected, data), null, 2);
      byId("next-action").textContent = nextAction(selected.id);
    }}

    function buildCommand(spec, data) {{
      return buildArgv(spec, data, true).map(shellQuote).join(" ");
    }}

    function buildArgv(spec, data, includeProgram = false) {{
      const command = includeProgram ? ["interp-lab", spec.id] : [];
      if (!includeProgram) command.push(spec.id);
      for (const field of spec.positional || []) {{
        appendValue(command, null, data[field.key]);
      }}
      for (const field of spec.fields || []) {{
        if (field.type === "boolean") {{
          if (data[field.key]) command.push(field.flag);
        }} else {{
          appendValue(command, field.flag, data[field.key]);
        }}
      }}
      if (data.__extra) command.push(...splitExtraFlags(data.__extra));
      return command;
    }}

    function appendValue(command, flag, value) {{
      if (value === undefined || value === null || value === "") return;
      const values = Array.isArray(value) ? value : [value];
      for (const item of values) {{
        if (flag) command.push(flag);
        command.push(String(item));
      }}
    }}

    function buildRunConfig(spec, data) {{
      const usesListArgs = (spec.positional || []).length > 0 || Boolean(data.__extra);
      const args = {{}};
      for (const field of [...(spec.positional || []), ...(spec.fields || [])]) {{
        const value = data[field.key];
        if (value === undefined || value === null || value === "") continue;
        args[field.key] = value;
      }}
      const stepArgs = usesListArgs ? buildArgv(spec, data, false).slice(1) : args;
      return {{
        schema_version: "interp-lab.run.v1",
        out: "reports/studio-run",
        steps: [
          {{
            name: spec.id,
            command: spec.id,
            args: stepArgs,
          }},
        ],
      }};
    }}

    function nextAction(id) {{
      const actions = {{
        inspect: "Review report.html, then export an attribution graph or collect interventions for top features.",
        "train-sae": "Inspect the learned latent records, then run causal validation for the most important features.",
        match: "Run validate-matches on the match report before treating pairs as equivalents.",
        "validate-matches": "Use validated pairs in graph review; collect interventions for pairs that need evidence.",
        "export-attribution-graph": "Open graph.html, then validate candidate paths with held-out path records.",
        "validate-assay": "Fix any assay errors, then launch Criterion Lab discovery with the validated preset file.",
        "criterion-lab": "Run the generated config, then open the inspection report and graph artifacts from Reports And Graphs.",
        demo: "Open the generated HTML reports in the output folder for a quick product tour.",
      }};
      return actions[id] || "Copy the command or run-config step, then run it in the environment that has access to the model and data.";
    }}

    async function checkServer() {{
      try {{
        const response = await fetch("/api/health", {{ cache: "no-store" }});
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const health = await response.json();
        setServerState(true, `Workspace: ${{health.workspace}}`);
        await Promise.all([refreshJobs(), refreshArtifacts()]);
      }} catch (error) {{
        setServerState(false, "Static HTML mode. Run `interp-lab studio --serve` to launch commands and browse artifacts here.");
      }}
    }}

    function setServerState(available, detail) {{
      serverAvailable = available;
      byId("server-status").textContent = available ? "connected" : "static";
      byId("server-detail").textContent = detail;
      byId("server-panel").classList.toggle("server-disabled", !available);
      byId("artifact-panel").classList.toggle("server-disabled", !available);
      byId("start-job").disabled = !available;
      byId("start-config-job").disabled = !available;
      byId("refresh-jobs").disabled = !available;
      byId("refresh-artifacts").disabled = !available;
      if (!available) {{
        byId("job-list").innerHTML = `<div class="history-item"><span>Local runner unavailable</span><span class="item-meta">Serve Studio locally to run jobs from this page.</span></div>`;
        byId("artifact-list").innerHTML = `<div class="history-item"><span>Artifact browser unavailable</span><span class="item-meta">Serve Studio locally to browse generated reports.</span></div>`;
      }}
    }}

    async function startJob(useConfig) {{
      if (!serverAvailable) return;
      const data = readForm();
      const payload = useConfig
        ? {{ run_config: buildRunConfig(selected, data) }}
        : {{ argv: buildArgv(selected, data, false) }};
      const response = await fetch("/api/jobs", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload),
      }});
      if (!response.ok) {{
        const error = await response.json().catch(() => ({{ error: response.statusText }}));
        byId("job-list").innerHTML = `<div class="history-item"><strong>Job rejected</strong><span class="item-meta">${{escapeHtml(error.error || "Unknown error")}}</span></div>`;
        return;
      }}
      const payloadJson = await response.json();
      jobs = [payloadJson.job, ...jobs.filter((job) => job.id !== payloadJson.job.id)];
      renderJobs();
      pollJob(payloadJson.job.id);
    }}

    async function pollJob(jobId) {{
      for (let attempt = 0; attempt < 180; attempt += 1) {{
        await delay(1000);
        const response = await fetch(`/api/jobs/${{encodeURIComponent(jobId)}}`, {{ cache: "no-store" }});
        if (!response.ok) return;
        const payload = await response.json();
        jobs = [payload.job, ...jobs.filter((job) => job.id !== payload.job.id)];
        renderJobs();
        if (payload.job.status === "succeeded" || payload.job.status === "failed") {{
          await refreshArtifacts();
          return;
        }}
      }}
    }}

    async function refreshJobs() {{
      if (!serverAvailable) return;
      const response = await fetch("/api/jobs", {{ cache: "no-store" }});
      if (!response.ok) return;
      const payload = await response.json();
      jobs = payload.jobs || [];
      renderJobs();
    }}

    function renderJobs() {{
      const list = byId("job-list");
      if (!jobs.length) {{
        list.innerHTML = `<div class="history-item"><span>No jobs yet</span><span class="item-meta">Run a command or run-config from this page.</span></div>`;
        return;
      }}
      list.innerHTML = "";
      for (const job of jobs.slice(0, 12)) {{
        const item = document.createElement("div");
        item.className = "history-item";
        const output = [job.stdout, job.stderr].filter(Boolean).join("\\n").slice(-1800);
        item.innerHTML = `
          <strong>${{escapeHtml(job.command || job.id)}} <span class="pill">${{escapeHtml(job.status)}}</span></strong>
          <span class="item-meta">${{escapeHtml((job.argv || []).join(" "))}}</span>
          <span class="item-meta">${{job.exit_code === null || job.exit_code === undefined ? "" : `exit ${{job.exit_code}}`}} ${{escapeHtml(job.finished_at || job.started_at || job.created_at || "")}}</span>
          ${{output ? `<pre>${{escapeHtml(output)}}</pre>` : ""}}
        `;
        list.appendChild(item);
      }}
    }}

    async function refreshArtifacts() {{
      if (!serverAvailable) return;
      const response = await fetch("/api/artifacts", {{ cache: "no-store" }});
      if (!response.ok) return;
      const payload = await response.json();
      artifacts = payload.artifacts || [];
      renderArtifacts();
    }}

    function renderArtifacts() {{
      const list = byId("artifact-list");
      if (!artifacts.length) {{
        list.innerHTML = `<div class="history-item"><span>No artifacts found</span><span class="item-meta">Run a demo, inspection, or graph export to populate reports.</span></div>`;
        return;
      }}
      list.innerHTML = "";
      for (const artifact of artifacts.slice(0, 80)) {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "artifact-item";
        button.innerHTML = `
          <strong>${{escapeHtml(artifact.relative_path || artifact.name)}}</strong>
          <span class="item-meta">${{escapeHtml(artifact.kind)}} · ${{formatBytes(artifact.size_bytes)}} · ${{escapeHtml(artifact.modified_at)}}</span>
        `;
        button.addEventListener("click", () => loadArtifact(artifact));
        list.appendChild(button);
      }}
    }}

    async function loadArtifact(artifact) {{
      byId("artifact-title").textContent = artifact.relative_path || artifact.name;
      const response = await fetch(`/api/artifact?path=${{encodeURIComponent(artifact.path)}}`, {{ cache: "no-store" }});
      if (!response.ok) return;
      const payload = await response.json();
      const preview = byId("artifact-preview");
      const frame = byId("artifact-frame");
      renderGraphOverview(payload.text, artifact.kind);
      if (artifact.kind === "html") {{
        frame.src = `/api/raw?path=${{encodeURIComponent(artifact.path)}}`;
        frame.style.display = "block";
        preview.style.display = "none";
      }} else {{
        frame.style.display = "none";
        preview.style.display = "block";
        preview.textContent = payload.text.slice(0, 32000);
      }}
    }}

    function renderGraphOverview(text, kind) {{
      const target = byId("graph-overview");
      target.innerHTML = "";
      if (kind !== "graph") return;
      try {{
        const graph = JSON.parse(text);
        const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
        const edges = Array.isArray(graph.edges) ? graph.edges : [];
        const edgeTypes = {{}};
        for (const edge of edges) {{
          const type = edge.type || "edge";
          edgeTypes[type] = (edgeTypes[type] || 0) + 1;
        }}
        const typeText = Object.entries(edgeTypes)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 6)
          .map(([type, count]) => `${{type}}: ${{count}}`)
          .join(", ");
        target.innerHTML = `
          <div class="graph-counts">
            <span class="pill">${{nodes.length}} nodes</span>
            <span class="pill">${{edges.length}} edges</span>
            <span class="pill">${{escapeHtml(graph.schema_version || "graph")}}</span>
          </div>
          <div class="item-meta">${{escapeHtml(typeText || "No edge types found")}}</div>
        `;
      }} catch (error) {{
        target.innerHTML = `<div class="item-meta">Graph preview unavailable: ${{escapeHtml(error.message)}}</div>`;
      }}
    }}

    function delay(ms) {{
      return new Promise((resolve) => setTimeout(resolve, ms));
    }}

    function formatBytes(value) {{
      const bytes = Number(value || 0);
      if (bytes < 1024) return `${{bytes}} B`;
      if (bytes < 1024 * 1024) return `${{(bytes / 1024).toFixed(1)}} KB`;
      return `${{(bytes / (1024 * 1024)).toFixed(1)}} MB`;
    }}

    function shellQuote(value) {{
      if (/^[A-Za-z0-9_./:=,@+-]+$/.test(value)) return value;
      return "'" + String(value).replace(/'/g, "'\\\"'\\\"'") + "'";
    }}

    function splitExtraFlags(value) {{
      const tokens = [];
      let current = "";
      let quote = null;
      let escaped = false;
      for (const char of String(value)) {{
        if (escaped) {{
          current += char;
          escaped = false;
          continue;
        }}
        if (char === "\\\\") {{
          escaped = true;
          continue;
        }}
        if (quote) {{
          if (char === quote) quote = null;
          else current += char;
          continue;
        }}
        if (char === "'" || char === '"') {{
          quote = char;
          continue;
        }}
        if (/\\s/.test(char)) {{
          if (current) {{
            tokens.push(current);
            current = "";
          }}
          continue;
        }}
        current += char;
      }}
      if (escaped) current += "\\\\";
      if (current) tokens.push(current);
      return tokens;
    }}

    function escapeHtml(value) {{
      return String(value).replace(/[&<>]/g, (char) => ({{ "&": "&amp;", "<": "&lt;", ">": "&gt;" }}[char]));
    }}

    function escapeAttr(value) {{
      return escapeHtml(value).replace(/"/g, "&quot;");
    }}

    byId("command-search").addEventListener("input", renderCommandList);
    byId("extra-flags").addEventListener("input", updateOutputs);
    byId("copy-command").addEventListener("click", async () => navigator.clipboard?.writeText(byId("generated-command").textContent));
    byId("copy-config").addEventListener("click", async () => navigator.clipboard?.writeText(byId("run-config-output").textContent));
    byId("start-job").addEventListener("click", () => startJob(false));
    byId("start-config-job").addEventListener("click", () => startJob(true));
    byId("refresh-jobs").addEventListener("click", refreshJobs);
    byId("refresh-artifacts").addEventListener("click", refreshArtifacts);
    renderWorkflows();
    renderCommandList();
    renderForm();
    updateOutputs();
    checkServer();
  </script>
</body>
</html>
"""


def build_web_app_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write or serve the interp-lab Studio web app.")
    parser.add_argument("--out", default="reports/interp-lab-studio.html", help="Output HTML path.")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve Studio locally with job launching, run history, and artifact browsing.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for --serve.")
    parser.add_argument("--port", type=int, default=8765, help="Port for --serve. Use 0 for any free port.")
    parser.add_argument("--reports-dir", default="reports", help="Reports directory exposed by --serve.")
    parser.add_argument("--open", action="store_true", help="Open the served Studio page in a browser.")
    return parser


def run_web_app_from_args(
    args: argparse.Namespace,
    command_specs: Sequence[dict[str, Any]] | None = None,
) -> Path:
    return write_web_app(args.out, command_specs=command_specs)


def _find_subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _spec_from_subparser(name: str, parser: argparse.ArgumentParser, help_text: str) -> dict[str, Any]:
    positionals: list[dict[str, Any]] = []
    fields: list[dict[str, Any]] = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        field = _field_from_action(action)
        if field is None:
            continue
        if action.option_strings:
            fields.append(field)
        else:
            positionals.append(field)

    spec: dict[str, Any] = {
        "id": name,
        "group": _default_group(name),
        "label": _label_from_id(name),
        "description": help_text or parser.description or f"Configure interp-lab {name}.",
        "fields": fields,
    }
    if positionals:
        spec["positional"] = positionals
    return spec


def _field_from_action(action: argparse.Action) -> dict[str, Any] | None:
    if action.dest == argparse.SUPPRESS:
        return None
    field: dict[str, Any] = {
        "key": action.dest,
        "label": _label_from_id(action.dest),
    }
    if action.option_strings:
        field["flag"] = _preferred_flag(action.option_strings)
        if getattr(action, "required", False):
            field["required"] = True
    elif getattr(action, "nargs", None) not in ("?", "*"):
        field["required"] = True

    if action.metavar:
        field["placeholder"] = _metavar_text(action.metavar)

    if getattr(action, "choices", None):
        field["type"] = "select"
        field["options"] = [str(choice) for choice in action.choices]
    elif _is_boolean_action(action):
        field["type"] = "boolean"
    elif _is_repeat_action(action):
        field["type"] = "repeat"
    elif action.type in (int, float):
        field["type"] = "number"
    return field


def _merge_curated_fields(generated: dict[str, Any], curated: dict[str, Any]) -> None:
    curated_fields: dict[str, dict[str, Any]] = {}
    for field in curated.get("positional", []) + curated.get("fields", []):
        curated_fields[field["key"]] = field

    for section in ("positional", "fields"):
        merged: list[dict[str, Any]] = []
        for field in generated.get(section, []):
            override = curated_fields.get(field["key"])
            if override is not None:
                field = {**field}
                for key in ("label", "placeholder", "default", "default_checked"):
                    if key in override:
                        field[key] = override[key]
            merged.append(field)
        if merged:
            generated[section] = merged
        elif section in generated:
            generated.pop(section)


def _preferred_flag(option_strings: Sequence[str]) -> str:
    long_flags = [flag for flag in option_strings if flag.startswith("--")]
    return long_flags[0] if long_flags else option_strings[0]


def _is_boolean_action(action: argparse.Action) -> bool:
    return isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))


def _is_repeat_action(action: argparse.Action) -> bool:
    return isinstance(action, argparse._AppendAction) or getattr(action, "nargs", None) in ("+", "*")


def _metavar_text(metavar: Any) -> str:
    if isinstance(metavar, tuple):
        return " ".join(str(part) for part in metavar)
    return str(metavar)


def _label_from_id(value: str) -> str:
    words = value.replace("_", "-").split("-")
    return " ".join(word.upper() if word in {"hf", "sae", "json", "jsonl", "html"} else word.capitalize() for word in words)


def _default_group(command: str) -> str:
    if command in {"inspect"}:
        return "Discovery"
    if command in {
        "build-prompts",
        "export-hf-records",
        "export-transformerlens-records",
        "export-nnsight-records",
    }:
        return "Data"
    if command in {"train-sae"}:
        return "SAE"
    if command in {"export-hf-interventions", "export-hf-contrast"}:
        return "Causal"
    if command in {"match", "validate-matches"}:
        return "Cross-Model"
    if command in {
        "export-attribution-graph",
        "summarize-attribution-graph",
        "validate-attribution-graph",
    }:
        return "Graphs"
    if command in {"export-hf-sae-paths", "validate-hf-sae-paths"}:
        return "SAE Paths"
    if command in {"profile-env", "plan-scale"}:
        return "Planning"
    if command in {"init-run", "run"}:
        return "Automation"
    if command in {"publish-hf-artifact"}:
        return "Sharing"
    return "Utility"


def _json_payload(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True).replace("</", "<\\/")
