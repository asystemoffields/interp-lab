from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oracle_sae.graphs import load_path_patch_records, write_attribution_graph_html, write_attribution_graph_markdown


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
    annotated_graph_path: Path | None = None
    annotated_graph_markdown_path: Path | None = None
    annotated_graph_html_path: Path | None = None


def export_graph_validation_report(
    *,
    graph_path: str | Path,
    path_records_path: str | Path | list[str | Path],
    out_path: str | Path,
    markdown_out_path: str | Path | None = None,
    graph_out_path: str | Path | None = None,
    graph_markdown_out_path: str | Path | None = None,
    graph_html_out_path: str | Path | None = None,
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
    annotated_graph_path = None
    annotated_graph_markdown_path = None
    annotated_graph_html_path = None
    if graph_out_path is not None or graph_markdown_out_path is not None or graph_html_out_path is not None:
        annotated_graph = annotate_graph_with_validation(graph, report)
    if graph_out_path is not None:
        annotated_graph_path = Path(graph_out_path)
        annotated_graph_path.parent.mkdir(parents=True, exist_ok=True)
        annotated_graph_path.write_text(json.dumps(annotated_graph, indent=2, sort_keys=True), encoding="utf-8")
    if graph_markdown_out_path is not None or graph_out_path is not None:
        graph_markdown_path = (
            Path(graph_markdown_out_path)
            if graph_markdown_out_path is not None
            else Path(graph_out_path).with_suffix(".md")
        )
        annotated_graph_markdown_path = write_attribution_graph_markdown(annotated_graph, graph_markdown_path)
    if graph_html_out_path is not None or graph_out_path is not None:
        graph_html_path = (
            Path(graph_html_out_path)
            if graph_html_out_path is not None
            else Path(graph_out_path).with_suffix(".html")
        )
        annotated_graph_html_path = write_attribution_graph_html(annotated_graph, graph_html_path)
    return GraphValidationWriteResult(
        report=report,
        json_path=json_path,
        markdown_path=markdown_path,
        annotated_graph_path=annotated_graph_path,
        annotated_graph_markdown_path=annotated_graph_markdown_path,
        annotated_graph_html_path=annotated_graph_html_path,
    )


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
    claim_grade_counts: dict[str, int] = {}
    for validation in validations:
        grade = str(validation["claim_grade"])
        claim_grade_counts[grade] = claim_grade_counts.get(grade, 0) + 1
    run_assessment = _validation_run_assessment(validations)
    return {
        "schema_version": "interp-lab.graph_validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "graph_path": graph_path,
        "model": graph.get("model"),
        "criterion": graph.get("criterion"),
        "thresholds": thresholds,
        "summary": {
            "candidate_count": len(candidates),
            "path_record_count": len(path_records),
            "validated_path_count": len(validations),
            "claim_grade_counts": dict(sorted(claim_grade_counts.items())),
            "overall_claim_grade": run_assessment["overall_claim_grade"],
            "recommended_next_action": run_assessment["recommended_next_action"],
            "status_counts": dict(sorted(status_counts.items())),
        },
        "run_assessment": run_assessment,
        "agent_next_actions": _validation_agent_next_actions(run_assessment),
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


def annotate_graph_with_validation(graph: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    annotated = copy.deepcopy(graph)
    validations = {
        (item["source_feature_id"], item["target_feature_id"]): _validation_annotation(item)
        for item in report.get("path_validations", [])
    }
    for edge in annotated.get("edges", []):
        if edge.get("type") != "path_patch":
            continue
        validation = validations.get(
            (
                str(edge.get("source_feature_id", edge.get("source"))),
                str(edge.get("target_feature_id", edge.get("target"))),
            )
        )
        if validation is not None:
            edge["validation"] = validation
    mechanism_summary = annotated.setdefault("mechanism_summary", {})
    for path in mechanism_summary.get("candidate_paths", []):
        validation = validations.get((str(path.get("source_feature_id")), str(path.get("target_feature_id"))))
        if validation is not None:
            path["validation"] = validation
    mechanism_summary["path_validation_status_counts"] = report.get("summary", {}).get("status_counts", {})
    metadata = annotated.setdefault("metadata", {})
    metadata["graph_validation"] = {
        "schema_version": report.get("schema_version"),
        "created_at": report.get("created_at"),
        "summary": report.get("summary", {}),
        "run_assessment": report.get("run_assessment", {}),
        "agent_next_actions": report.get("agent_next_actions", []),
    }
    return annotated


def render_graph_validation_markdown(report: dict[str, Any]) -> str:
    assessment = report.get("run_assessment", {})
    lines = [
        "# Attribution Graph Validation",
        "",
        f"Model: `{report.get('model', '')}`",
        f"Path records: `{report['summary']['path_record_count']}`",
        f"Overall: `{assessment.get('overall_claim_grade', report['summary'].get('overall_claim_grade', ''))}`",
        f"Recommended next action: {assessment.get('recommended_next_action', report['summary'].get('recommended_next_action', ''))}",
        "",
        "| Status | Claim | Path | Effect | Control | Specificity | Sign | Prompts |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report.get("path_validations", []):
        lines.append(
            "| "
            f"{item['status']} | "
            f"{item['claim_grade']} | "
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
            f"- `{item['source_feature_id']} -> {item['target_feature_id']}`: "
            f"{item['interpretation']} Next: {item['next_action']}"
        )
    lines.append("")
    actions = report.get("agent_next_actions", [])
    if actions:
        lines.extend(["## Agent Next Actions", ""])
        for action in actions:
            lines.append(f"- `{action['id']}`: {action['title']}: `{action['command']}`")
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
    parser.add_argument("--graph-out", help="Optional output graph JSON annotated with validation status.")
    parser.add_argument(
        "--graph-markdown-out",
        help="Output annotated graph Markdown path. Defaults to --graph-out with .md when --graph-out is set.",
    )
    parser.add_argument(
        "--graph-html-out",
        help="Output annotated graph HTML viewer path. Defaults to --graph-out with .html when --graph-out is set.",
    )
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
        graph_out_path=args.graph_out,
        graph_markdown_out_path=args.graph_markdown_out,
        graph_html_out_path=args.graph_html_out,
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
    edge_candidates = [
        {
            "source_feature_id": str(edge.get("source_feature_id", edge.get("source"))),
            "target_feature_id": str(edge.get("target_feature_id", edge.get("target"))),
            "source_label": None,
            "target_label": None,
        }
        for edge in graph.get("edges", [])
        if edge.get("type") == "path_patch" and edge.get("source") and edge.get("target")
    ]
    merged = []
    seen = set()
    for candidate in [*candidates, *edge_candidates]:
        pair = (candidate["source_feature_id"], candidate["target_feature_id"])
        if pair in seen:
            continue
        merged.append(candidate)
        seen.add(pair)
    return merged


def _validation_annotation(item: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "status",
        "claim_grade",
        "interpretation",
        "next_action",
        "reason_codes",
        "record_count",
        "control_record_count",
        "prompt_count",
        "mean_abs_target_activation_delta",
        "control_mean_abs_target_activation_delta",
        "path_specificity_score",
        "effect_control_ratio",
        "sign_consistency",
        "target_activation_delta_ci",
        "mean_score_delta",
        "control_mean_abs_score_delta",
    ]
    return {key: item.get(key) for key in keys if key in item}


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
    reason_codes = _validation_reason_codes(
        status=status,
        effect_count=len(effect_rows),
        control_count=len(control_rows),
        prompt_count=prompt_count,
        mean_abs_effect=mean_abs_effect,
        specificity=specificity,
        ratio=ratio,
        sign_consistency=sign_consistency,
        thresholds=thresholds,
    )
    claim_grade = _claim_grade(status, reason_codes)
    return {
        "source_feature_id": candidate["source_feature_id"],
        "target_feature_id": candidate["target_feature_id"],
        "source_label": candidate.get("source_label"),
        "target_label": candidate.get("target_label"),
        "status": status,
        "claim_grade": claim_grade,
        "reason_codes": reason_codes,
        "interpretation": _interpret_status(status, reason_codes),
        "next_action": _next_action(claim_grade, reason_codes),
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


def _validation_reason_codes(
    *,
    status: str,
    effect_count: int,
    control_count: int,
    prompt_count: int,
    mean_abs_effect: float,
    specificity: float,
    ratio: float | None,
    sign_consistency: float,
    thresholds: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if effect_count == 0:
        return ["no_effect_records"]
    if prompt_count < int(thresholds["min_prompt_count"]):
        reasons.append("prompt_count_below_threshold")
    if bool(thresholds["require_controls"]) and control_count == 0:
        reasons.append("missing_control_records")
    if control_count > 0:
        if specificity < float(thresholds["min_specificity"]):
            reasons.append("control_specificity_below_threshold")
        if ratio is not None and ratio < float(thresholds["min_effect_control_ratio"]):
            reasons.append("effect_control_ratio_below_threshold")
    if mean_abs_effect < float(thresholds["min_effect"]):
        reasons.append("effect_below_threshold")
    if sign_consistency < float(thresholds["min_sign_consistency"]):
        reasons.append("sign_consistency_below_threshold")
    if reasons:
        return reasons
    if status == "robust":
        return ["passed_effect_control_and_sign_thresholds"]
    if status == "suggestive":
        return ["passed_suggestive_effect_and_sign_thresholds"]
    return [f"classified_{status}"]


def _interpret_status(status: str, reason_codes: list[str] | None = None) -> str:
    reasons = set(reason_codes or [])
    if status == "robust":
        return "The path replicated with a target-latent effect that beat controls and kept a consistent sign."
    if status == "suggestive":
        return "The path has measurable evidence but should be repeated with more prompts or stronger controls."
    if status == "failed_control":
        if "missing_control_records" in reasons:
            return "Control rows were required for this run, and no matching controls were present."
        if (
            "control_specificity_below_threshold" in reasons
            or "effect_control_ratio_below_threshold" in reasons
        ):
            return "Control interventions produced comparable target-latent deltas, so this path needs better separation."
        return "The measured effect did not separate cleanly from control interventions."
    if "prompt_count_below_threshold" in reasons:
        return "The path has too few distinct prompts for the current validation threshold."
    if "effect_below_threshold" in reasons:
        return "The target-latent effect is below the current validation threshold."
    if "sign_consistency_below_threshold" in reasons:
        return "The target-latent effect changes sign across records more often than the current threshold allows."
    return "The available path records are weak for this candidate."


def _claim_grade(status: str, reason_codes: list[str]) -> str:
    reasons = set(reason_codes)
    if status == "robust":
        return "validated"
    if status == "suggestive":
        return "needs_replication"
    if status == "failed_control":
        if "missing_control_records" in reasons:
            return "needs_controls"
        return "control_failed"
    if "prompt_count_below_threshold" in reasons:
        return "underpowered"
    if "effect_below_threshold" in reasons:
        return "low_effect"
    return "insufficient_evidence"


def _next_action(claim_grade: str, reason_codes: list[str]) -> str:
    reasons = set(reason_codes)
    if claim_grade == "validated":
        return "Use as a causal path candidate and replicate on broader held-out prompts before broad claims."
    if claim_grade == "needs_replication":
        return "Increase prompt count, rerun controls, and check sign consistency on held-out prompts."
    if claim_grade == "needs_controls":
        return "Rerun with random-source or matched-source control rows."
    if claim_grade == "control_failed":
        return "Improve control separation with matched controls, more prompts, or a narrower source-target pair."
    if claim_grade == "underpowered":
        return "Add more distinct prompts before interpreting this path."
    if claim_grade == "low_effect" and "sign_consistency_below_threshold" in reasons:
        return "Try a richer strength sweep, then rerun only if the sign becomes stable."
    if claim_grade == "low_effect":
        return "Lower priority or rerun with a stronger intervention sweep."
    return "Collect more path records or deprioritize this candidate."


def _validation_run_assessment(validations: list[dict[str, Any]]) -> dict[str, str]:
    if not validations:
        return {
            "overall_claim_grade": "no_path_candidates",
            "summary": "No path-patch candidates were selected from the attribution graph.",
            "recommended_next_action": "Export or add measured path-patch records, then rerun graph validation.",
        }
    counts = _claim_grade_counts(validations)
    if counts.get("validated"):
        return {
            "overall_claim_grade": "validated_paths_present",
            "summary": f"{counts['validated']} path claim(s) passed the validation thresholds.",
            "recommended_next_action": "Replicate validated paths on a broader held-out prompt set and inspect the annotated graph.",
        }
    if counts.get("needs_replication"):
        return {
            "overall_claim_grade": "replication_needed",
            "summary": f"{counts['needs_replication']} path claim(s) have suggestive evidence.",
            "recommended_next_action": "Increase prompt coverage, rerun controls, and validate the same source-target pairs again.",
        }
    if counts.get("needs_controls"):
        return {
            "overall_claim_grade": "controls_needed",
            "summary": f"{counts['needs_controls']} path claim(s) need control records before interpretation.",
            "recommended_next_action": "Rerun path patching with random-source or matched-source controls.",
        }
    if counts.get("control_failed"):
        return {
            "overall_claim_grade": "control_failed",
            "summary": f"{counts['control_failed']} path claim(s) failed control separation.",
            "recommended_next_action": "Prioritize better controls, narrower path pairs, or more prompts before using these paths.",
        }
    if counts.get("underpowered"):
        return {
            "overall_claim_grade": "underpowered",
            "summary": f"{counts['underpowered']} path claim(s) had too few distinct prompts.",
            "recommended_next_action": "Collect more held-out prompts and rerun validation.",
        }
    if counts.get("low_effect"):
        return {
            "overall_claim_grade": "low_effect",
            "summary": f"{counts['low_effect']} path claim(s) had low target-latent effect size.",
            "recommended_next_action": "Lower priority or rerun with a richer source-steering strength sweep.",
        }
    return {
        "overall_claim_grade": "insufficient_evidence",
        "summary": "The selected path claims do not yet have enough validation evidence.",
        "recommended_next_action": "Collect more path records, add controls, or deprioritize these candidates.",
    }


def _validation_agent_next_actions(run_assessment: dict[str, str]) -> list[dict[str, str]]:
    grade = run_assessment["overall_claim_grade"]
    common_review = {
        "id": "inspect_validation_report",
        "title": "Review path-level claim grades and reason codes",
        "command": "python -c \"from pathlib import Path; print(Path('<validation.md>').read_text(encoding='utf-8'))\"",
    }
    if grade == "validated_paths_present":
        return [
            common_review,
            {
                "id": "replicate_validated_paths",
                "title": "Replicate validated paths on a broader held-out prompt set",
                "command": "interp-lab validate-hf-sae-paths --graph <graph.json> --model <model> --dataset <broader-heldout.jsonl> --source-sae <source-sae.json> --target-sae <target-sae.json> --path-records-out <paths.jsonl> --out <validation.json> --graph-out <validated-graph.json>",
            },
            {
                "id": "publish_validated_graph",
                "title": "Package the annotated graph and validation reports for review",
                "command": "interp-lab publish-hf-artifact --repo-id <user/repo> --path <run-directory> --dry-run",
            },
        ]
    if grade in {"control_failed", "controls_needed"}:
        return [
            common_review,
            {
                "id": "rerun_with_stronger_controls",
                "title": "Rerun path patching with stronger control coverage",
                "command": "interp-lab validate-hf-sae-paths --graph <graph.json> --model <model> --dataset <heldout.jsonl> --source-sae <source-sae.json> --target-sae <target-sae.json> --random-source-controls 4 --path-records-out <paths.jsonl> --out <validation.json> --graph-out <validated-graph.json>",
            },
            {
                "id": "narrow_path_pairs",
                "title": "Rerun exact source-target pairs that remain scientifically interesting",
                "command": "interp-lab export-hf-sae-paths --model <model> --dataset <heldout.jsonl> --source-sae <source-sae.json> --target-sae <target-sae.json> --path-pair SOURCE=TARGET --random-source-controls 4 --out <paths.jsonl>",
            },
        ]
    if grade in {"replication_needed", "underpowered"}:
        return [
            common_review,
            {
                "id": "increase_prompt_coverage",
                "title": "Collect a larger held-out prompt set and rerun validation",
                "command": "interp-lab build-prompts --positive <positive.txt> --negative <negative.txt> --out <heldout.jsonl>",
            },
            {
                "id": "rerun_validation",
                "title": "Validate the same graph paths with the expanded records",
                "command": "interp-lab validate-attribution-graph --graph <graph.json> --path-records <paths.jsonl> --out <validation.json> --graph-out <validated-graph.json>",
            },
        ]
    return [
        common_review,
        {
            "id": "collect_more_path_records",
            "title": "Collect more measured path-patching records before interpreting this graph",
            "command": "interp-lab export-hf-sae-paths --model <model> --dataset <heldout.jsonl> --source-sae <source-sae.json> --target-sae <target-sae.json> --out <paths.jsonl>",
        },
    ]


def _claim_grade_counts(validations: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for validation in validations:
        grade = str(validation.get("claim_grade", "insufficient_evidence"))
        counts[grade] = counts.get(grade, 0) + 1
    return counts


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
