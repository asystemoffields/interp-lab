from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oracle_sae.hf_loading import MODEL_CLASS_CHOICES


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
    validation_dataset: str | Path | None = None,
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
    latent_dim: int | None = None,
    layer: int | None = None,
    source_layer: int | None = None,
    target_layer: int | None = None,
    path_top_k: int = 8,
    source_top_k: int = 4,
    target_top_k: int = 8,
    random_source_controls: int = 2,
    validate_paths: bool = False,
    pool: str = "last",
    device: str = "cpu",
    max_length: int | None = None,
    include_causal: bool = False,
    prepare_sae_prompts: bool = True,
    target_token: list[str] | None = None,
    model_class: str = "auto-causal-lm",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    model_kwargs_json: str | None = None,
    tokenizer_kwargs_json: str | None = None,
) -> dict[str, Any]:
    """Build an editable run config for common interp-lab workflows."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    workflow = workflow.lower()
    hf_loading_args = _hf_loading_step_args(
        model_class=model_class,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        device_map=device_map,
        model_kwargs_json=model_kwargs_json,
        tokenizer_kwargs_json=tokenizer_kwargs_json,
    )
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
        export_args.update(hf_loading_args)
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
        sae_dataset_ref = dataset_ref
        causal_dataset_ref = None
        if prepare_sae_prompts:
            prompt_pack = _add_prepare_sae_prompt_step(
                config,
                dataset=dataset_ref,
                out_dir="{run_dir}/sae-prompts",
                latent_dim=latent_dim,
                max_length=max_length,
            )
            sae_dataset_ref = prompt_pack["train"]
            causal_dataset_ref = prompt_pack["causal"]
        records_path = "{run_dir}/sae/records.jsonl"
        train_args: dict[str, Any] = {
            "hf_model": model,
            "dataset": sae_dataset_ref,
            "criterion": criterion,
            "preset": preset,
            "out": "{run_dir}/sae/sae.json",
            "records_out": records_path,
            "device": device,
        }
        train_args.update(hf_loading_args)
        if latent_dim is not None:
            train_args["latent_dim"] = latent_dim
        if layer is not None:
            train_args["layer"] = layer
        if max_length is not None:
            train_args["max_length"] = max_length
        causal_path = None
        if include_causal:
            causal_path = "{run_dir}/sae/interventions.jsonl"
            train_args["causal_out"] = causal_path
            if causal_dataset_ref is not None:
                train_args["causal_dataset"] = causal_dataset_ref
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
            require_interventions=causal_path is not None,
            top_k=top_k,
        )
        return config
    if workflow == "sae-paths":
        if path_top_k <= 0:
            raise ValueError("path_top_k must be positive")
        if source_top_k <= 0:
            raise ValueError("source_top_k must be positive")
        if target_top_k <= 0:
            raise ValueError("target_top_k must be positive")
        if random_source_controls < 0:
            raise ValueError("random_source_controls must be non-negative")
        if dataset_ref is None:
            raise ValueError("dataset or prompt inputs are required for the sae-paths workflow")
        if source_layer is None:
            raise ValueError("source_layer is required for the sae-paths workflow")
        if target_layer is None:
            raise ValueError("target_layer is required for the sae-paths workflow")
        if source_layer >= target_layer:
            raise ValueError("source_layer must be lower than target_layer for the sae-paths workflow")
        source_sae_path = "{run_dir}/source-sae/sae.json"
        target_sae_path = "{run_dir}/target-sae/sae.json"
        source_records_path = "{run_dir}/source-sae/records.jsonl"
        target_records_path = "{run_dir}/target-sae/records.jsonl"
        source_interventions_path = "{run_dir}/source-sae/interventions.jsonl" if include_causal else None
        target_interventions_path = "{run_dir}/target-sae/interventions.jsonl" if include_causal else None
        source_report_path = "{run_dir}/source-report/report.json"
        target_report_path = "{run_dir}/target-report/report.json"
        path_records_path = "{run_dir}/paths.jsonl"
        graph_path = "{run_dir}/graph.json"
        sae_dataset_ref = dataset_ref
        causal_dataset_ref = None
        validation_dataset_ref = str(validation_dataset) if validation_dataset is not None else dataset_ref
        if prepare_sae_prompts:
            prompt_pack = _add_prepare_sae_prompt_step(
                config,
                dataset=dataset_ref,
                out_dir="{run_dir}/sae-prompts",
                latent_dim=latent_dim,
                max_length=max_length,
            )
            sae_dataset_ref = prompt_pack["train"]
            causal_dataset_ref = prompt_pack["causal"]
            if validation_dataset is None:
                validation_dataset_ref = prompt_pack["validation"]
        config["steps"].append(
            {
                "name": "train-source-sae",
                "command": "train-sae",
                "args": _hf_sae_train_args(
                    model=model,
                    dataset=sae_dataset_ref,
                    criterion=criterion,
                    preset=preset,
                    latent_dim=latent_dim,
                    layer=source_layer,
                    out=source_sae_path,
                    records_out=source_records_path,
                    causal_out=source_interventions_path,
                    causal_dataset=causal_dataset_ref,
                    pool=pool,
                    device=device,
                    max_length=max_length,
                    target_token=target_token,
                    hf_loading_args=hf_loading_args,
                ),
            }
        )
        config["steps"].append(
            {
                "name": "train-target-sae",
                "command": "train-sae",
                "args": _hf_sae_train_args(
                    model=model,
                    dataset=sae_dataset_ref,
                    criterion=criterion,
                    preset=preset,
                    latent_dim=latent_dim,
                    layer=target_layer,
                    out=target_sae_path,
                    records_out=target_records_path,
                    causal_out=target_interventions_path,
                    causal_dataset=causal_dataset_ref,
                    pool=pool,
                    device=device,
                    max_length=max_length,
                    target_token=target_token,
                    hf_loading_args=hf_loading_args,
                ),
            }
        )
        _add_inspect_step(
            config,
            name="inspect-source",
            model=model,
            criterion=criterion,
            records=source_records_path,
            interventions=source_interventions_path,
            require_interventions=source_interventions_path is not None,
            top_k=top_k,
            out="{run_dir}/source-report",
        )
        _add_inspect_step(
            config,
            name="inspect-target",
            model=model,
            criterion=criterion,
            records=target_records_path,
            interventions=target_interventions_path,
            require_interventions=target_interventions_path is not None,
            top_k=top_k,
            out="{run_dir}/target-report",
        )
        path_args: dict[str, Any] = {
            "model": model,
            "dataset": causal_dataset_ref or dataset_ref,
            "criterion": criterion,
            "source_sae": source_sae_path,
            "target_sae": target_sae_path,
            "source_report": source_report_path,
            "target_report": target_report_path,
            "source_top_k": source_top_k,
            "target_top_k": target_top_k,
            "random_source_controls": random_source_controls,
            "pool": pool,
            "out": path_records_path,
            "device": device,
        }
        path_args.update(hf_loading_args)
        if max_length is not None:
            path_args["max_length"] = max_length
        if target_token:
            path_args["target_token"] = target_token
        config["steps"].append(
            {
                "name": "export-paths",
                "command": "export-hf-sae-paths",
                "args": path_args,
            }
        )
        _add_graph_step(
            config,
            report=[source_report_path, target_report_path],
            path_records=path_records_path,
            out=graph_path,
            markdown_out="{run_dir}/graph.md",
            html_out="{run_dir}/graph.html",
        )
        _add_graph_summary_step(
            config,
            name="summarize-graph",
            graph=graph_path,
            out="{run_dir}/graph-summary.json",
        )
        if validate_paths:
            validation_args: dict[str, Any] = {
                "graph": graph_path,
                "model": model,
                "dataset": validation_dataset_ref,
                "criterion": criterion,
                "source_sae": source_sae_path,
                "target_sae": target_sae_path,
                "source_report": source_report_path,
                "target_report": target_report_path,
                "path_records_out": "{run_dir}/validated-paths.jsonl",
                "out": "{run_dir}/validation.json",
                "markdown_out": "{run_dir}/validation.md",
                "graph_out": "{run_dir}/validated-graph.json",
                "graph_markdown_out": "{run_dir}/validated-graph.md",
                "graph_html_out": "{run_dir}/validated-graph.html",
                "top_k": path_top_k,
                "random_source_controls": random_source_controls,
                "pool": pool,
                "device": device,
            }
            validation_args.update(hf_loading_args)
            if max_length is not None:
                validation_args["max_length"] = max_length
            if target_token:
                validation_args["target_token"] = target_token
            config["steps"].append(
                {
                    "name": "validate-paths",
                    "command": "validate-hf-sae-paths",
                    "args": validation_args,
                }
            )
            _add_graph_summary_step(
                config,
                name="summarize-validated-graph",
                graph="{run_dir}/validated-graph.json",
                out="{run_dir}/validated-graph-summary.json",
            )
        return config
    raise ValueError("workflow must be one of: records, hf-records, sae, sae-paths")


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
    parser.add_argument("--workflow", choices=["records", "hf-records", "sae", "sae-paths"], default="records")
    parser.add_argument("--run-dir", default="reports/interp-run", help="Directory the generated run will write to.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--criterion", required=True)
    parser.add_argument("--records", help="Activation-record JSONL for --workflow records.")
    parser.add_argument("--interventions", help="Optional intervention JSONL for --workflow records.")
    parser.add_argument("--dataset", help="Prompt JSONL for HF-backed workflows.")
    parser.add_argument(
        "--validation-dataset",
        help="Held-out prompt JSONL for --workflow sae-paths --validate-paths. Defaults to --dataset.",
    )
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
    parser.add_argument("--latent-dim", type=int, help="SAE latent count for generated SAE training steps.")
    parser.add_argument("--layer", type=int, help="HF hidden-state layer for --workflow sae.")
    parser.add_argument("--source-layer", type=int, help="Upstream HF hidden-state layer for --workflow sae-paths.")
    parser.add_argument("--target-layer", type=int, help="Downstream HF hidden-state layer for --workflow sae-paths.")
    parser.add_argument("--path-top-k", type=int, default=8, help="Candidate graph paths to validate.")
    parser.add_argument("--source-top-k", type=int, default=4, help="Source SAE features to path-patch.")
    parser.add_argument("--target-top-k", type=int, default=8, help="Target SAE features to measure per source feature.")
    parser.add_argument(
        "--random-source-controls",
        type=int,
        default=2,
        help="Random source SAE latents to patch as controls in path workflows.",
    )
    parser.add_argument("--validate-paths", action="store_true", help="Add held-out SAE path validation to path runs.")
    parser.add_argument("--pool", choices=["last", "mean"], default="last")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--include-causal", action="store_true", help="Add SAE causal validation output to SAE runs.")
    parser.add_argument(
        "--skip-prompt-pack",
        action="store_true",
        help="Use --dataset directly in SAE workflows instead of preparing train/causal/validation prompt splits.",
    )
    parser.add_argument("--target-token", action="append", default=[], help="Target token for SAE causal scoring.")
    parser.add_argument("--model-class", choices=MODEL_CLASS_CHOICES, default="auto-causal-lm")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", help="Optional device_map passed to HF model loading, e.g. auto.")
    parser.add_argument("--model-kwargs-json", help="Extra JSON object passed to model from_pretrained.")
    parser.add_argument("--tokenizer-kwargs-json", help="Extra JSON object passed to tokenizer from_pretrained.")
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
            validation_dataset=args.validation_dataset,
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
            latent_dim=args.latent_dim,
            layer=args.layer,
            source_layer=args.source_layer,
            target_layer=args.target_layer,
            path_top_k=args.path_top_k,
            source_top_k=args.source_top_k,
            target_top_k=args.target_top_k,
            random_source_controls=args.random_source_controls,
            validate_paths=args.validate_paths,
            pool=args.pool,
            device=args.device,
            max_length=args.max_length,
            include_causal=args.include_causal,
            prepare_sae_prompts=not args.skip_prompt_pack,
            target_token=args.target_token,
            model_class=args.model_class,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            model_kwargs_json=args.model_kwargs_json,
            tokenizer_kwargs_json=args.tokenizer_kwargs_json,
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


def _hf_loading_step_args(
    *,
    model_class: str,
    trust_remote_code: bool,
    local_files_only: bool,
    torch_dtype: str | None,
    device_map: str | None,
    model_kwargs_json: str | None,
    tokenizer_kwargs_json: str | None,
) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if model_class != "auto-causal-lm":
        args["model_class"] = model_class
    if trust_remote_code:
        args["trust_remote_code"] = True
    if local_files_only:
        args["local_files_only"] = True
    if torch_dtype is not None:
        args["torch_dtype"] = torch_dtype
    if device_map is not None:
        args["device_map"] = device_map
    if model_kwargs_json is not None:
        args["model_kwargs_json"] = model_kwargs_json
    if tokenizer_kwargs_json is not None:
        args["tokenizer_kwargs_json"] = tokenizer_kwargs_json
    return args


def _add_prepare_sae_prompt_step(
    config: dict[str, Any],
    *,
    dataset: str,
    out_dir: str,
    latent_dim: int | None,
    max_length: int | None,
) -> dict[str, str]:
    args: dict[str, Any] = {
        "dataset": dataset,
        "out_dir": out_dir,
    }
    if latent_dim is not None:
        args["latent_dim"] = latent_dim
    if max_length is not None:
        args["max_length"] = max_length
    config["steps"].append(
        {
            "name": "prepare-sae-prompts",
            "command": "prepare-sae-prompts",
            "args": args,
        }
    )
    return {
        "train": f"{out_dir}/train.jsonl",
        "causal": f"{out_dir}/causal.jsonl",
        "validation": f"{out_dir}/validation.jsonl",
    }


def _hf_sae_train_args(
    *,
    model: str,
    dataset: str,
    criterion: str,
    preset: str,
    latent_dim: int | None,
    layer: int,
    out: str,
    records_out: str,
    causal_out: str | None,
    causal_dataset: str | None,
    pool: str,
    device: str,
    max_length: int | None,
    target_token: list[str] | None,
    hf_loading_args: dict[str, Any],
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "hf_model": model,
        "dataset": dataset,
        "criterion": criterion,
        "preset": preset,
        "layer": layer,
        "pool": pool,
        "out": out,
        "records_out": records_out,
        "device": device,
    }
    args.update(hf_loading_args)
    if latent_dim is not None:
        args["latent_dim"] = latent_dim
    if max_length is not None:
        args["max_length"] = max_length
    if causal_out is not None:
        args["causal_out"] = causal_out
        if causal_dataset is not None:
            args["causal_dataset"] = causal_dataset
        if target_token:
            args["target_token"] = target_token
    return args


def _add_inspect_and_graph_steps(
    config: dict[str, Any],
    *,
    model: str,
    criterion: str,
    records: str,
    interventions: str | None,
    require_interventions: bool = False,
    top_k: int,
) -> None:
    inspect_out = "{run_dir}/inspect"
    _add_inspect_step(
        config,
        name="inspect",
        model=model,
        criterion=criterion,
        records=records,
        interventions=interventions,
        require_interventions=require_interventions,
        top_k=top_k,
        out=inspect_out,
    )
    _add_graph_step(
        config,
        report=f"{inspect_out}/report.json",
        out="{run_dir}/graph.json",
        markdown_out="{run_dir}/graph.md",
        html_out="{run_dir}/graph.html",
    )


def _add_inspect_step(
    config: dict[str, Any],
    *,
    name: str,
    model: str,
    criterion: str,
    records: str,
    interventions: str | None,
    require_interventions: bool = False,
    top_k: int,
    out: str,
) -> None:
    inspect_args: dict[str, Any] = {
        "model": model,
        "criterion": criterion,
        "backend": "records",
        "records": records,
        "top_k": top_k,
        "out": out,
    }
    if interventions is not None:
        inspect_args["interventions"] = interventions
    if require_interventions:
        inspect_args["require_interventions"] = True
    config["steps"].append(
        {
            "name": name,
            "command": "inspect",
            "args": inspect_args,
        }
    )


def _add_graph_step(
    config: dict[str, Any],
    *,
    report: str | list[str],
    out: str,
    markdown_out: str,
    html_out: str | None = None,
    path_records: str | None = None,
) -> None:
    args: dict[str, Any] = {
        "report": report,
        "out": out,
        "markdown_out": markdown_out,
    }
    if html_out is not None:
        args["html_out"] = html_out
    if path_records is not None:
        args["path_records"] = path_records
    config["steps"].append(
        {
            "name": "graph",
            "command": "export-attribution-graph",
            "args": args,
        }
    )


def _add_graph_summary_step(
    config: dict[str, Any],
    *,
    name: str,
    graph: str,
    out: str,
) -> None:
    config["steps"].append(
        {
            "name": name,
            "command": "summarize-attribution-graph",
            "args": {
                "graph": graph,
                "out": out,
            },
        }
    )
