from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunTemplateWriteResult:
    path: Path
    config: dict[str, Any]


def build_run_template(
    *,
    workflow: str,
    model: str,
    criterion: str,
    run_dir: str | Path,
    records: str | Path | None = None,
    interventions: str | Path | None = None,
    dataset: str | Path | None = None,
    positive: list[str | Path] | None = None,
    negative: list[str | Path] | None = None,
    positive_prompt: list[str] | None = None,
    negative_prompt: list[str] | None = None,
    split: str = "paragraphs",
    delimiter: str | None = None,
    top_k: int = 8,
    features_per_layer: int = 16,
    layers: str | None = None,
    preset: str = "minimal",
    layer: int | None = None,
    device: str = "cpu",
    max_length: int | None = None,
    include_causal: bool = False,
    target_token: list[str] | None = None,
) -> dict[str, Any]:
    """Build an editable run config for common interp-lab workflows."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    workflow = workflow.lower()
    config: dict[str, Any] = {
        "out": str(run_dir),
        "steps": [],
    }
    dataset_ref = str(dataset) if dataset is not None else None
    prompt_args = _prompt_step_args(
        positive=positive,
        negative=negative,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        split=split,
        delimiter=delimiter,
    )
    if prompt_args:
        dataset_ref = "{run_dir}/prompts.jsonl"
        prompt_args["out"] = dataset_ref
        config["steps"].append(
            {
                "name": "build-prompts",
                "command": "build-prompts",
                "args": prompt_args,
            }
        )
    if workflow == "records":
        if records is None:
            raise ValueError("records is required for the records workflow")
        _add_inspect_and_graph_steps(
            config,
            model=model,
            criterion=criterion,
            records=str(records),
            interventions=str(interventions) if interventions is not None else None,
            top_k=top_k,
        )
        return config
    if workflow == "hf-records":
        if dataset_ref is None:
            raise ValueError("dataset or prompt inputs are required for the hf-records workflow")
        records_path = "{run_dir}/records.jsonl"
        export_args: dict[str, Any] = {
            "model": model,
            "dataset": dataset_ref,
            "out": records_path,
            "features_per_layer": features_per_layer,
            "device": device,
        }
        if layers is not None:
            export_args["layers"] = layers
        if max_length is not None:
            export_args["max_length"] = max_length
        config["steps"].append(
            {
                "name": "export-records",
                "command": "export-hf-records",
                "args": export_args,
            }
        )
        _add_inspect_and_graph_steps(
            config,
            model=model,
            criterion=criterion,
            records=records_path,
            interventions=None,
            top_k=top_k,
        )
        return config
    if workflow == "sae":
        if dataset_ref is None:
            raise ValueError("dataset or prompt inputs are required for the sae workflow")
        records_path = "{run_dir}/sae/records.jsonl"
        train_args: dict[str, Any] = {
            "hf_model": model,
            "dataset": dataset_ref,
            "criterion": criterion,
            "preset": preset,
            "out": "{run_dir}/sae/sae.json",
            "records_out": records_path,
            "device": device,
        }
        if layer is not None:
            train_args["layer"] = layer
        if max_length is not None:
            train_args["max_length"] = max_length
        causal_path = None
        if include_causal:
            causal_path = "{run_dir}/sae/interventions.jsonl"
            train_args["causal_out"] = causal_path
            if target_token:
                train_args["target_token"] = target_token
        config["steps"].append(
            {
                "name": "train-sae",
                "command": "train-sae",
                "args": train_args,
            }
        )
        _add_inspect_and_graph_steps(
            config,
            model=model,
            criterion=criterion,
            records=records_path,
            interventions=causal_path,
            top_k=top_k,
        )
        return config
    raise ValueError("workflow must be one of: records, hf-records, sae")


def write_run_template(
    *,
    out: str | Path,
    force: bool = False,
    **kwargs: Any,
) -> RunTemplateWriteResult:
    config = build_run_template(**kwargs)
    path = Path(out)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass force=True to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return RunTemplateWriteResult(path=path, config=config)


def build_init_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write an editable interp-lab run config.")
    parser.add_argument("--out", required=True, help="Output run config JSON path.")
    parser.add_argument("--workflow", choices=["records", "hf-records", "sae"], default="records")
    parser.add_argument("--run-dir", default="reports/interp-run", help="Directory the generated run will write to.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--criterion", required=True)
    parser.add_argument("--records", help="Activation-record JSONL for --workflow records.")
    parser.add_argument("--interventions", help="Optional intervention JSONL for --workflow records.")
    parser.add_argument("--dataset", help="Prompt JSONL for HF-backed workflows.")
    parser.add_argument("--positive", action="append", default=[], help="Positive prompt file. Repeatable.")
    parser.add_argument("--negative", action="append", default=[], help="Negative prompt file. Repeatable.")
    parser.add_argument("--positive-prompt", action="append", default=[], help="Inline positive prompt. Repeatable.")
    parser.add_argument("--negative-prompt", action="append", default=[], help="Inline negative prompt. Repeatable.")
    parser.add_argument("--split", choices=["lines", "paragraphs"], default="paragraphs")
    parser.add_argument("--delimiter", help="Literal delimiter between prompts in prompt files.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--features-per-layer", type=int, default=16)
    parser.add_argument("--layers", help="Hidden-state layers for --workflow hf-records, e.g. 6 or 2-6.")
    parser.add_argument("--preset", choices=["minimal", "production", "custom"], default="minimal")
    parser.add_argument("--layer", type=int, help="HF hidden-state layer for --workflow sae.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--include-causal", action="store_true", help="Add SAE causal validation output to SAE runs.")
    parser.add_argument("--target-token", action="append", default=[], help="Target token for SAE causal scoring.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing config path.")
    return parser


def run_init_run_from_args(args: argparse.Namespace) -> RunTemplateWriteResult:
    try:
        return write_run_template(
            out=args.out,
            force=args.force,
            workflow=args.workflow,
            model=args.model,
            criterion=args.criterion,
            run_dir=args.run_dir,
            records=args.records,
            interventions=args.interventions,
            dataset=args.dataset,
            positive=args.positive,
            negative=args.negative,
            positive_prompt=args.positive_prompt,
            negative_prompt=args.negative_prompt,
            split=args.split,
            delimiter=args.delimiter,
            top_k=args.top_k,
            features_per_layer=args.features_per_layer,
            layers=args.layers,
            preset=args.preset,
            layer=args.layer,
            device=args.device,
            max_length=args.max_length,
            include_causal=args.include_causal,
            target_token=args.target_token,
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _prompt_step_args(
    *,
    positive: list[str | Path] | None,
    negative: list[str | Path] | None,
    positive_prompt: list[str] | None,
    negative_prompt: list[str] | None,
    split: str,
    delimiter: str | None,
) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if positive:
        args["positive"] = [str(path) for path in positive]
    if negative:
        args["negative"] = [str(path) for path in negative]
    if positive_prompt:
        args["positive_prompt"] = list(positive_prompt)
    if negative_prompt:
        args["negative_prompt"] = list(negative_prompt)
    if not args:
        return {}
    args["split"] = split
    if delimiter is not None:
        args["delimiter"] = delimiter
    return args


def _add_inspect_and_graph_steps(
    config: dict[str, Any],
    *,
    model: str,
    criterion: str,
    records: str,
    interventions: str | None,
    top_k: int,
) -> None:
    inspect_args: dict[str, Any] = {
        "model": model,
        "criterion": criterion,
        "backend": "records",
        "records": records,
        "top_k": top_k,
        "out": "{run_dir}/inspect",
    }
    if interventions is not None:
        inspect_args["interventions"] = interventions
    config["steps"].append(
        {
            "name": "inspect",
            "command": "inspect",
            "args": inspect_args,
        }
    )
    config["steps"].append(
        {
            "name": "graph",
            "command": "export-attribution-graph",
            "args": {
                "report": "{run_dir}/inspect/report.json",
                "out": "{run_dir}/graph.json",
                "markdown_out": "{run_dir}/graph.md",
            },
        }
    )
