from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oracle_sae.graphs import load_path_patch_records


DEFAULT_MIN_EFFECT = 0.05
DEFAULT_MIN_SPECIFICITY = 0.02
DEFAULT_MIN_EFFECT_CONTROL_RATIO = 1.5
DEFAULT_MIN_PROMPT_COUNT = 3
DEFAULT_MIN_SIGN_CONSISTENCY = 0.75


@dataclass(frozen=True)
class GraphValidationWriteResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path


def export_graph_validation_report(
    *,
    graph_path: str | Path,
    path_records_path: str | Path | list[str | Path],
    out_path: str | Path,
    markdown_out_path: str | Path | None = None,
    top_k: int = 8,
    min_effect: float = DEFAULT_MIN_EFFECT,
    min_specificity: float = DEFAULT_MIN_SPECIFICITY,
    min_effect_control_ratio: float = DEFAULT_MIN_EFFECT_CONTROL_RATIO,
    min_prompt_count: int = DEFAULT_MIN_PROMPT_COUNT,
    min_sign_consistency: float = DEFAULT_MIN_SIGN_CONSISTENCY,
    require_controls: bool = True,
) -> GraphValidationWriteResult:
    graph_file = Path(graph_path)
    graph = json.loads(graph_file.read_text(encoding="utf-8"))
    path_records = load_path_patch_records(path_records_path)
    report = build_graph_validation_report(
        graph,
        path_records=path_records,
        graph_path=str(graph_file),
        top_k=top_k,
        min_effect=min_effect,
        min_specificity=min_specificity,
        min_effect_control_ratio=min_effect_control_ratio,
        min_prompt_count=min_prompt_count,
        min_sign_consistency=min_sign_consistency,
        require_controls=require_controls,
    )
    json_path = Path(out_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = Path(markdown_out_path) if markdown_out_path is not None else json_path.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_graph_validation_markdown(report), encoding="utf-8")
    return GraphValidationWriteResult(report=report, json_path=json_path, markdown_path=markdown_path)


def build_graph_validation_report(
    graph: dict[str, Any],
    *,
    path_records: list[dict[str, Any]],
    graph_path: str | None = None,
    top_k: int = 8,
    min_effect: float = DEFAULT_MIN_EFFECT,
    min_specificity: float = DEFAULT_MIN_SPECIFICITY,
    min_effect_control_ratio: float = DEFAULT_MIN_EFFECT_CONTROL_RATIO,
    min_prompt_count: int = DEFAULT_MIN_PROMPT_COUNT,
    min_sign_consistency: float = DEFAULT_MIN_SIGN_CONSISTENCY,
    require_controls: bool = True,
) -> dict[str, Any]:
    thresholds = {
        "min_effect": float(min_effect),
        "min_specificity": float(min_specificity),
        "min_effect_control_ratio": float(min_effect_control_ratio),
        "min_prompt_count": int(min_prompt_count),
        "min_sign_consistency": float(min_sign_consistency),
        "require_controls": bool(require_controls),
    }
    candidates = _path_patch_candidates(graph)[:top_k]
    grouped_records = _group_path_records(path_records)
    validations = [
        _validate_path(
            candidate,
            grouped_records.get((candidate["source_feature_id"], candidate["target_feature_id"]), []),
            thresholds=thresholds,
        )
        for candidate in candidates
    ]
    status_counts: dict[str, int] = {}
    for validation in validations:
        status = str(validation["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema_version": "interp-lab.graph_validation.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "graph_path": graph_path,
        "model": graph.get("model"),
        "criterion": graph.get("criterion"),
        "thresholds": thresholds,
        "summary": {
            "candidate_count": len(candidates),
            "path_record_count": len(path_records),
            "validated_path_count": len(validations),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "path_validations": validations,
    }


def select_graph_path_pairs(graph: dict[str, Any], *, top_k: int = 8) -> list[tuple[str, str]]:
    pairs = []
    seen = set()
    for candidate in _path_patch_candidates(graph):
        pair = (candidate["source_feature_id"], candidate["target_feature_id"])
        if pair in seen:
            continue
        pairs.append(pair)
        seen.add(pair)
        if len(pairs) >= top_k:
            break
    return pairs


def render_graph_validation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Attribution Graph Validation",
        "",
        f"Model: `{report.get('model', '')}`",
        f"Path records: `{report['summary']['path_record_count']}`",
        "",
        "| Status | Path | Effect | Control | Specificity | Sign | Prompts |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report.get("path_validations", []):
        lines.append(
            "| "
            f"{item['status']} | "
            f"`{item['source_feature_id']} -> {item['target_feature_id']}` | "
            f"{_markdown_number(item['mean_abs_target_activation_delta'])} | "
            f"{_markdown_number(item['control_mean_abs_target_activation_delta'])} | "
            f"{_markdown_number(item['path_specificity_score'])} | "
            f"{_markdown_number(item['sign_consistency'])} | "
            f"{item['prompt_count']} |"
        )
    lines.extend(["", "## Notes", ""])
    for item in report.get("path_validations", []):
        lines.append(
            f"- `{item['source_feature_id']} -> {item['target_feature_id']}`: {item['interpretation']}"
        )
    lines.append("")
    return "\n".join(lines)


def build_graph_validation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate measured attribution-graph paths with path records.")
    parser.add_argument("--graph", required=True, help="Attribution graph JSON.")
    parser.add_argument(
        "--path-records",
        action="append",
        required=True,
        help="Path-patching JSONL from a validation run. Repeatable.",
    )
    parser.add_argument("--out", required=True, help="Output validation JSON path.")
    parser.add_argument("--markdown-out", help="Output validation Markdown path. Defaults to --out with .md.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-effect", type=float, default=DEFAULT_MIN_EFFECT)
    parser.add_argument("--min-specificity", type=float, default=DEFAULT_MIN_SPECIFICITY)
    parser.add_argument("--min-effect-control-ratio", type=float, default=DEFAULT_MIN_EFFECT_CONTROL_RATIO)
    parser.add_argument("--min-prompt-count", type=int, default=DEFAULT_MIN_PROMPT_COUNT)
    parser.add_argument("--min-sign-consistency", type=float, default=DEFAULT_MIN_SIGN_CONSISTENCY)
    parser.add_argument(
        "--allow-missing-controls",
        action="store_true",
        help="Classify paths without control rows using effect and sign evidence only.",
    )
    return parser


def run_graph_validation_from_args(args: argparse.Namespace) -> GraphValidationWriteResult:
    return export_graph_validation_report(
        graph_path=args.graph,
        path_records_path=args.path_records,
        out_path=args.out,
        markdown_out_path=args.markdown_out,
        top_k=args.top_k,
        min_effect=args.min_effect,
        min_specificity=args.min_specificity,
        min_effect_control_ratio=args.min_effect_control_ratio,
        min_prompt_count=args.min_prompt_count,
        min_sign_consistency=args.min_sign_consistency,
        require_controls=not args.allow_missing_controls,
    )


def _path_patch_candidates(graph: dict[str, Any]) -> list[dict[str, Any]]:
    summary_paths = graph.get("mechanism_summary", {}).get("candidate_paths", [])
    candidates = [
        {
            "source_feature_id": str(path["source_feature_id"]),
            "target_feature_id": str(path["target_feature_id"]),
            "source_label": path.get("source_label"),
            "target_label": path.get("target_label"),
        }
        for path in summary_paths
        if path.get("evidence") == "path_patch"
        and path.get("source_feature_id")
        and path.get("target_feature_id")
    ]
    if candidates:
        return candidates
    return [
        {
            "source_feature_id": str(edge["source"]),
            "target_feature_id": str(edge["target"]),
            "source_label": None,
            "target_label": None,
        }
        for edge in graph.get("edges", [])
        if edge.get("type") == "path_patch" and edge.get("source") and edge.get("target")
    ]


def _group_path_records(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        source = str(record.get("source_feature_id", ""))
        target = str(record.get("target_feature_id", ""))
        if source and target:
            grouped.setdefault((source, target), []).append(record)
    return grouped


def _validate_path(
    candidate: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    effect_rows = [record for record in records if not _is_control(record)]
    control_rows = [record for record in records if _is_control(record)]
    effect_deltas = [_float(record.get("target_activation_delta")) for record in effect_rows]
    control_deltas = [_float(record.get("target_activation_delta")) for record in control_rows]
    score_deltas = [
        _float(record.get("score_delta"))
        for record in effect_rows
        if record.get("score_delta") is not None
    ]
    control_score_deltas = [
        _float(record.get("score_delta")) for record in control_rows if record.get("score_delta") is not None
    ]
    mean_abs_effect = _mean([abs(value) for value in effect_deltas])
    mean_abs_control = _mean([abs(value) for value in control_deltas])
    specificity = max(0.0, mean_abs_effect - mean_abs_control)
    ratio = _ratio(mean_abs_effect, mean_abs_control)
    prompt_count = len(
        {str(row.get("prompt_id", "")) for row in effect_rows if row.get("prompt_id") is not None}
    )
    sign_consistency = _sign_consistency(effect_deltas)
    status = _validation_status(
        effect_count=len(effect_rows),
        control_count=len(control_rows),
        prompt_count=prompt_count,
        mean_abs_effect=mean_abs_effect,
        specificity=specificity,
        ratio=ratio,
        sign_consistency=sign_consistency,
        thresholds=thresholds,
    )
    return {
        "source_feature_id": candidate["source_feature_id"],
        "target_feature_id": candidate["target_feature_id"],
        "source_label": candidate.get("source_label"),
        "target_label": candidate.get("target_label"),
        "status": status,
        "interpretation": _interpret_status(status),
        "record_count": len(effect_rows),
        "control_record_count": len(control_rows),
        "prompt_count": prompt_count,
        "mean_target_activation_delta": round(_mean(effect_deltas), 6),
        "mean_abs_target_activation_delta": round(mean_abs_effect, 6),
        "target_activation_delta_ci": _mean_ci(effect_deltas),
        "control_mean_abs_target_activation_delta": round(mean_abs_control, 6) if control_rows else None,
        "path_specificity_score": round(specificity, 6) if control_rows else None,
        "effect_control_ratio": round(ratio, 6) if ratio is not None else None,
        "sign_consistency": round(sign_consistency, 6),
        "mean_score_delta": round(_mean(score_deltas), 6) if score_deltas else None,
        "control_mean_abs_score_delta": round(_mean([abs(value) for value in control_score_deltas]), 6)
        if control_score_deltas
        else None,
    }


def _validation_status(
    *,
    effect_count: int,
    control_count: int,
    prompt_count: int,
    mean_abs_effect: float,
    specificity: float,
    ratio: float | None,
    sign_consistency: float,
    thresholds: dict[str, Any],
) -> str:
    if effect_count == 0:
        return "weak"
    if prompt_count < int(thresholds["min_prompt_count"]):
        return "weak"
    if bool(thresholds["require_controls"]) and control_count == 0:
        return "failed_control"
    controls_pass = control_count == 0 or (
        specificity >= float(thresholds["min_specificity"])
        and (ratio is None or ratio >= float(thresholds["min_effect_control_ratio"]))
    )
    if not controls_pass:
        return "failed_control"
    if (
        mean_abs_effect >= float(thresholds["min_effect"])
        and sign_consistency >= float(thresholds["min_sign_consistency"])
    ):
        return "robust"
    if mean_abs_effect >= float(thresholds["min_effect"]) * 0.5 and sign_consistency >= 0.5:
        return "suggestive"
    return "weak"


def _interpret_status(status: str) -> str:
    if status == "robust":
        return "The path replicated with a target-latent effect that beat controls and kept a consistent sign."
    if status == "suggestive":
        return "The path has measurable evidence but should be repeated with more prompts or stronger controls."
    if status == "failed_control":
        return "The measured effect did not separate cleanly from control interventions."
    return "The available path records are weak for this candidate."


def _is_control(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata", {})
    return isinstance(metadata, dict) and bool(metadata.get("control_type"))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_ci(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    mean = _mean(values)
    if len(values) == 1:
        return {"low": round(mean, 6), "high": round(mean, 6)}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half_width = 1.96 * math.sqrt(variance) / math.sqrt(len(values))
    return {"low": round(mean - half_width, 6), "high": round(mean + half_width, 6)}


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _sign_consistency(values: list[float]) -> float:
    nonzero = [value for value in values if abs(value) > 1e-12]
    if not nonzero:
        return 0.0
    direction = 1.0 if _mean(nonzero) >= 0.0 else -1.0
    aligned = [value for value in nonzero if value * direction > 0.0]
    return len(aligned) / len(nonzero)


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _markdown_number(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"
