from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oracle_sae.graph_validation import GraphValidationWriteResult, export_graph_validation_report, select_graph_path_pairs
from oracle_sae.hf_contrast import parse_strength_sweep
from oracle_sae.hf_interventions import parse_target_tokens
from oracle_sae.hf_loading import add_hf_loading_args, hf_loading_options_from_args
from oracle_sae.hf_sae_paths import export_hf_sae_path_records


@dataclass(frozen=True)
class HfSaePathValidationResult:
    selected_path_pairs: list[tuple[str, str]]
    path_records_path: Path
    validation: GraphValidationWriteResult


def export_hf_sae_path_validation(
    *,
    graph_path: str | Path,
    model_name: str,
    dataset_path: str | Path,
    source_artifact_path: str | Path,
    target_artifact_path: str | Path,
    path_records_out_path: str | Path,
    validation_out_path: str | Path,
    criterion: str | None = None,
    markdown_out_path: str | Path | None = None,
    graph_out_path: str | Path | None = None,
    graph_markdown_out_path: str | Path | None = None,
    graph_html_out_path: str | Path | None = None,
    source_report_path: str | Path | None = None,
    target_report_path: str | Path | None = None,
    top_k: int = 8,
    pool: str = "last",
    strength_sweep: list[float] | None = None,
    random_source_controls: int = 2,
    control_seed: int = 0,
    score_behavior: bool = True,
    target_tokens: list[str] | None = None,
    device: str = "cpu",
    max_length: int = 128,
    min_effect: float = 0.05,
    min_specificity: float = 0.02,
    min_effect_control_ratio: float = 1.5,
    min_prompt_count: int = 3,
    min_sign_consistency: float = 0.75,
    require_controls: bool = True,
    model_class: str = "auto-causal-lm",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> HfSaePathValidationResult:
    graph_file = Path(graph_path)
    graph = json.loads(graph_file.read_text(encoding="utf-8"))
    selected_path_pairs = select_graph_path_pairs(graph, top_k=top_k)
    if not selected_path_pairs:
        raise ValueError(f"{graph_file}: no path_patch candidate paths found")
    criterion_text = criterion or _criterion_text(graph)
    path_records_path = export_hf_sae_path_records(
        model_name=model_name,
        dataset_path=dataset_path,
        source_artifact_path=source_artifact_path,
        target_artifact_path=target_artifact_path,
        out_path=path_records_out_path,
        criterion=criterion_text,
        path_pairs=selected_path_pairs,
        source_report_path=source_report_path,
        target_report_path=target_report_path,
        source_top_k=top_k,
        target_top_k=top_k,
        pool=pool,
        strength_sweep=strength_sweep,
        random_source_controls=random_source_controls,
        control_seed=control_seed,
        score_behavior=score_behavior,
        target_tokens=target_tokens,
        device=device,
        max_length=max_length,
        model_class=model_class,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        device_map=device_map,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    validation = export_graph_validation_report(
        graph_path=graph_file,
        path_records_path=path_records_path,
        out_path=validation_out_path,
        markdown_out_path=markdown_out_path,
        graph_out_path=graph_out_path,
        graph_markdown_out_path=graph_markdown_out_path,
        graph_html_out_path=graph_html_out_path,
        top_k=top_k,
        min_effect=min_effect,
        min_specificity=min_specificity,
        min_effect_control_ratio=min_effect_control_ratio,
        min_prompt_count=min_prompt_count,
        min_sign_consistency=min_sign_consistency,
        require_controls=require_controls,
    )
    return HfSaePathValidationResult(
        selected_path_pairs=selected_path_pairs,
        path_records_path=path_records_path,
        validation=validation,
    )


def build_hf_sae_validation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerun graph candidate SAE paths on held-out HF prompts and validate them.",
    )
    parser.add_argument("--graph", required=True, help="Attribution graph JSON with path_patch candidates.")
    parser.add_argument("--model", required=True, help="Hugging Face model name.")
    parser.add_argument("--dataset", required=True, help="Held-out prompt JSONL with text and criterion_score.")
    parser.add_argument("--criterion", help="Criterion text stored in validation path records.")
    parser.add_argument("--source-sae", required=True, help="Source SAE artifact JSON.")
    parser.add_argument("--target-sae", required=True, help="Target/downstream SAE artifact JSON.")
    parser.add_argument("--path-records-out", required=True, help="Output path-patch JSONL path.")
    parser.add_argument("--out", required=True, help="Output validation JSON path.")
    parser.add_argument("--markdown-out", help="Output validation Markdown path. Defaults to --out with .md.")
    parser.add_argument("--graph-out", help="Optional output graph JSON annotated with validation status.")
    parser.add_argument(
        "--graph-markdown-out",
        help="Output annotated graph Markdown path. Defaults to --graph-out with .md when --graph-out is set.",
    )
    parser.add_argument(
        "--graph-html-out",
        help="Output annotated graph HTML viewer path. Defaults to --graph-out with .html when --graph-out is set.",
    )
    parser.add_argument("--source-report", help="Optional source-layer report JSON for labels.")
    parser.add_argument("--target-report", help="Optional target-layer report JSON for labels.")
    parser.add_argument("--top-k", type=int, default=8, help="Candidate graph paths to validate.")
    parser.add_argument("--pool", choices=["last", "mean"], default="last")
    parser.add_argument("--strength-sweep", help="Comma-separated signed source steering strengths.")
    parser.add_argument("--random-source-controls", type=int, default=2)
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument("--target-token", action="append")
    parser.add_argument("--skip-behavior-score", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--min-effect", type=float, default=0.05)
    parser.add_argument("--min-specificity", type=float, default=0.02)
    parser.add_argument("--min-effect-control-ratio", type=float, default=1.5)
    parser.add_argument("--min-prompt-count", type=int, default=3)
    parser.add_argument("--min-sign-consistency", type=float, default=0.75)
    parser.add_argument("--allow-missing-controls", action="store_true")
    add_hf_loading_args(parser)
    return parser


def run_hf_sae_validation_from_args(args: argparse.Namespace) -> HfSaePathValidationResult:
    try:
        loading_options = hf_loading_options_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        return export_hf_sae_path_validation(
            graph_path=args.graph,
            model_name=args.model,
            dataset_path=args.dataset,
            source_artifact_path=args.source_sae,
            target_artifact_path=args.target_sae,
            path_records_out_path=args.path_records_out,
            validation_out_path=args.out,
            criterion=args.criterion,
            markdown_out_path=args.markdown_out,
            graph_out_path=args.graph_out,
            graph_markdown_out_path=args.graph_markdown_out,
            graph_html_out_path=args.graph_html_out,
            source_report_path=args.source_report,
            target_report_path=args.target_report,
            top_k=args.top_k,
            pool=args.pool,
            strength_sweep=parse_strength_sweep(args.strength_sweep),
            random_source_controls=args.random_source_controls,
            control_seed=args.control_seed,
            score_behavior=not args.skip_behavior_score,
            target_tokens=parse_target_tokens(args.target_token),
            device=args.device,
            max_length=args.max_length,
            min_effect=args.min_effect,
            min_specificity=args.min_specificity,
            min_effect_control_ratio=args.min_effect_control_ratio,
            min_prompt_count=args.min_prompt_count,
            min_sign_consistency=args.min_sign_consistency,
            require_controls=not args.allow_missing_controls,
            **loading_options,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _criterion_text(graph: dict[str, Any]) -> str:
    criterion = graph.get("criterion")
    if isinstance(criterion, dict):
        value = criterion.get("text")
        if value:
            return str(value)
    if criterion:
        return str(criterion)
    return ""
