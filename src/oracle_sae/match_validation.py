from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oracle_sae.reporting import load_match_report
from oracle_sae.schema import CandidateMatch, MatchReport


DEFAULT_MIN_SCORE = 0.75
DEFAULT_MIN_COMPONENT = 0.65
DEFAULT_MIN_CAUSAL_COMPONENT = 0.65
DEFAULT_MAX_SIGNED_EFFECT_DELTA = 0.15
DEFAULT_MIN_ABS_SIGNED_EFFECT = 0.02


@dataclass(frozen=True)
class MatchValidationWriteResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path


def export_match_validation_report(
    *,
    matches_path: str | Path,
    out_path: str | Path,
    markdown_out_path: str | Path | None = None,
    top_k: int | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    min_component: float = DEFAULT_MIN_COMPONENT,
    min_causal_component: float = DEFAULT_MIN_CAUSAL_COMPONENT,
    max_signed_effect_delta: float = DEFAULT_MAX_SIGNED_EFFECT_DELTA,
    min_abs_signed_effect: float = DEFAULT_MIN_ABS_SIGNED_EFFECT,
) -> MatchValidationWriteResult:
    match_file = Path(matches_path)
    report = build_match_validation_report(
        load_match_report(match_file),
        match_path=str(match_file),
        top_k=top_k,
        min_score=min_score,
        min_component=min_component,
        min_causal_component=min_causal_component,
        max_signed_effect_delta=max_signed_effect_delta,
        min_abs_signed_effect=min_abs_signed_effect,
    )
    json_path = Path(out_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = Path(markdown_out_path) if markdown_out_path is not None else json_path.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_match_validation_markdown(report), encoding="utf-8")
    return MatchValidationWriteResult(report=report, json_path=json_path, markdown_path=markdown_path)


def build_match_validation_report(
    report: MatchReport,
    *,
    match_path: str | None = None,
    top_k: int | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    min_component: float = DEFAULT_MIN_COMPONENT,
    min_causal_component: float = DEFAULT_MIN_CAUSAL_COMPONENT,
    max_signed_effect_delta: float = DEFAULT_MAX_SIGNED_EFFECT_DELTA,
    min_abs_signed_effect: float = DEFAULT_MIN_ABS_SIGNED_EFFECT,
) -> dict[str, Any]:
    thresholds = {
        "min_score": float(min_score),
        "min_component": float(min_component),
        "min_causal_component": float(min_causal_component),
        "max_signed_effect_delta": float(max_signed_effect_delta),
        "min_abs_signed_effect": float(min_abs_signed_effect),
    }
    matches = report.matches[:top_k] if top_k is not None else list(report.matches)
    validations = [_validate_match(match, thresholds=thresholds) for match in matches]
    status_counts = _counts(validations, "status")
    claim_grade_counts = _counts(validations, "claim_grade")
    run_assessment = _validation_run_assessment(validations)
    return {
        "schema_version": "interp-lab.match_validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "match_path": match_path,
        "left_model": report.left_model,
        "right_model": report.right_model,
        "thresholds": thresholds,
        "summary": {
            "match_count": len(matches),
            "validated_count": status_counts.get("validated", 0),
            "needs_causal_evidence_count": status_counts.get("needs_causal_evidence", 0),
            "plausible_count": status_counts.get("plausible", 0),
            "contradicted_count": status_counts.get("contradicted", 0),
            "weak_count": status_counts.get("weak", 0),
            "claim_grade_counts": dict(sorted(claim_grade_counts.items())),
            "overall_claim_grade": run_assessment["overall_claim_grade"],
            "recommended_next_action": run_assessment["recommended_next_action"],
            "status_counts": dict(sorted(status_counts.items())),
        },
        "run_assessment": run_assessment,
        "agent_next_actions": _validation_agent_next_actions(run_assessment),
        "validations": validations,
    }


def render_match_validation_markdown(report: dict[str, Any]) -> str:
    assessment = report.get("run_assessment", {})
    lines = [
        "# Cross-Model Match Validation",
        "",
        f"Left model: `{report.get('left_model', '')}`",
        f"Right model: `{report.get('right_model', '')}`",
        f"Matches checked: `{report['summary']['match_count']}`",
        f"Overall: `{assessment.get('overall_claim_grade', report['summary'].get('overall_claim_grade', ''))}`",
        f"Recommended next action: {assessment.get('recommended_next_action', report['summary'].get('recommended_next_action', ''))}",
        "",
        "| Status | Claim | Match | Score | Causal | Signed effect delta |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in report.get("validations", []):
        lines.append(
            "| "
            f"{item['status']} | "
            f"{item['claim_grade']} | "
            f"`{item['left_feature_id']} -> {item['right_feature_id']}` | "
            f"{_markdown_number(item['score'])} | "
            f"{_markdown_number(item.get('causal_component'))} | "
            f"{_markdown_number(item.get('signed_effect_delta'))} |"
        )
    lines.extend(["", "## Notes", ""])
    for item in report.get("validations", []):
        lines.append(
            f"- `{item['left_feature_id']} -> {item['right_feature_id']}`: "
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


def build_match_validation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate cross-model candidate feature matches.")
    parser.add_argument("--matches", required=True, help="Match report JSON from `interp-lab match`.")
    parser.add_argument("--out", required=True, help="Output validation JSON path.")
    parser.add_argument("--markdown-out", help="Output validation Markdown path. Defaults to --out with .md.")
    parser.add_argument("--top-k", type=int, help="Validate only the top K matches from the report.")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--min-component", type=float, default=DEFAULT_MIN_COMPONENT)
    parser.add_argument("--min-causal-component", type=float, default=DEFAULT_MIN_CAUSAL_COMPONENT)
    parser.add_argument("--max-signed-effect-delta", type=float, default=DEFAULT_MAX_SIGNED_EFFECT_DELTA)
    parser.add_argument("--min-abs-signed-effect", type=float, default=DEFAULT_MIN_ABS_SIGNED_EFFECT)
    return parser


def run_match_validation_from_args(args: argparse.Namespace) -> MatchValidationWriteResult:
    return export_match_validation_report(
        matches_path=args.matches,
        out_path=args.out,
        markdown_out_path=args.markdown_out,
        top_k=args.top_k,
        min_score=args.min_score,
        min_component=args.min_component,
        min_causal_component=args.min_causal_component,
        max_signed_effect_delta=args.max_signed_effect_delta,
        min_abs_signed_effect=args.min_abs_signed_effect,
    )


def _validate_match(match: CandidateMatch, *, thresholds: dict[str, float]) -> dict[str, Any]:
    components = dict(match.components)
    structural_scores = [
        float(components[name])
        for name in ("text", "activation", "decoder")
        if name in components
    ]
    structural_pass_count = sum(
        1 for value in structural_scores if value >= thresholds["min_component"]
    )
    causal_component = _optional_float(components.get("causal"))
    signed_component = _optional_float(components.get("signed_effect"))
    signed_effect_delta = _signed_effect_delta(match)
    same_effect_direction = _same_effect_direction(match, thresholds["min_abs_signed_effect"])
    strong_signed_effects = _has_strong_signed_effects(match, thresholds["min_abs_signed_effect"])
    direction_conflict = same_effect_direction is False
    causal_component_pass = (
        causal_component is not None
        and causal_component >= thresholds["min_causal_component"]
    )
    score_pass = match.score >= thresholds["min_score"]
    signed_delta_ok = (
        signed_effect_delta is not None
        and signed_effect_delta <= thresholds["max_signed_effect_delta"]
    )
    status = _match_status(
        score_pass=score_pass,
        direction_conflict=direction_conflict,
        structural_pass_count=structural_pass_count,
        causal_component_pass=causal_component_pass,
        strong_signed_effects=strong_signed_effects,
        same_effect_direction=same_effect_direction,
        signed_delta_ok=signed_delta_ok,
    )
    reason_codes = _match_reason_codes(
        match=match,
        status=status,
        score_pass=score_pass,
        structural_pass_count=structural_pass_count,
        causal_component=causal_component,
        causal_component_pass=causal_component_pass,
        strong_signed_effects=strong_signed_effects,
        same_effect_direction=same_effect_direction,
        signed_delta_ok=signed_delta_ok,
        thresholds=thresholds,
    )
    claim_grade = _claim_grade(status)
    return {
        "left_feature_id": match.left_feature_id,
        "right_feature_id": match.right_feature_id,
        "left_model": match.left_model,
        "right_model": match.right_model,
        "left_label": match.left_label,
        "right_label": match.right_label,
        "score": round(float(match.score), 6),
        "status": status,
        "claim_grade": claim_grade,
        "reason_codes": reason_codes,
        "interpretation": _interpret_status(status, reason_codes),
        "next_action": _next_action(claim_grade, reason_codes),
        "components": {key: round(float(value), 6) for key, value in sorted(components.items())},
        "text_component": _optional_round(components.get("text")),
        "activation_component": _optional_round(components.get("activation")),
        "decoder_component": _optional_round(components.get("decoder")),
        "causal_component": _optional_round(causal_component),
        "signed_effect_component": _optional_round(signed_component),
        "left_signed_effect": _optional_round(match.left_signed_effect),
        "right_signed_effect": _optional_round(match.right_signed_effect),
        "signed_effect_delta": _optional_round(signed_effect_delta),
        "same_effect_direction": same_effect_direction,
        "strong_signed_effects": strong_signed_effects,
        "structural_pass_count": structural_pass_count,
    }


def _match_status(
    *,
    score_pass: bool,
    direction_conflict: bool,
    structural_pass_count: int,
    causal_component_pass: bool,
    strong_signed_effects: bool,
    same_effect_direction: bool | None,
    signed_delta_ok: bool,
) -> str:
    if direction_conflict:
        return "contradicted"
    if not score_pass:
        return "weak"
    if (
        causal_component_pass
        and strong_signed_effects
        and same_effect_direction is True
        and signed_delta_ok
    ):
        return "validated"
    if structural_pass_count >= 2:
        return "needs_causal_evidence"
    if structural_pass_count >= 1:
        return "plausible"
    return "weak"


def _match_reason_codes(
    *,
    match: CandidateMatch,
    status: str,
    score_pass: bool,
    structural_pass_count: int,
    causal_component: float | None,
    causal_component_pass: bool,
    strong_signed_effects: bool,
    same_effect_direction: bool | None,
    signed_delta_ok: bool,
    thresholds: dict[str, float],
) -> list[str]:
    reasons: list[str] = []
    if not score_pass:
        reasons.append("score_below_threshold")
    if structural_pass_count < 2:
        reasons.append("structural_components_below_threshold")
    if causal_component is None:
        reasons.append("missing_causal_component")
    elif not causal_component_pass:
        if abs(causal_component - 0.5) <= 1e-9:
            reasons.append("causal_component_neutral")
        else:
            reasons.append("causal_component_below_threshold")
    if not strong_signed_effects:
        if match.left_signed_effect is None or match.right_signed_effect is None:
            reasons.append("missing_signed_effects")
        else:
            reasons.append("signed_effects_below_threshold")
    elif same_effect_direction is False:
        reasons.append("signed_effect_direction_conflict")
    elif not signed_delta_ok:
        reasons.append("signed_effect_delta_above_threshold")
    if reasons:
        return reasons
    if status == "validated":
        return ["passed_score_structural_causal_and_signed_effect_thresholds"]
    if status == "needs_causal_evidence":
        return ["passed_structural_thresholds_but_needs_causal_validation"]
    if status == "plausible":
        return ["passed_score_threshold_with_limited_component_support"]
    return [f"classified_{status}"]


def _claim_grade(status: str) -> str:
    if status == "validated":
        return "validated_equivalent"
    if status == "needs_causal_evidence":
        return "needs_more_evidence"
    if status == "plausible":
        return "plausible_equivalent"
    if status == "contradicted":
        return "contradicted_effect"
    return "weak_match"


def _interpret_status(status: str, reason_codes: list[str]) -> str:
    reasons = set(reason_codes)
    if status == "validated":
        return "The match preserves high fingerprint similarity and aligned measured signed effects under the current thresholds."
    if status == "needs_causal_evidence":
        return "The match has strong structural similarity, but needs aligned causal or signed-effect evidence before treating it as an equivalent feature."
    if status == "plausible":
        return "The match is plausible from the available fingerprints and should be prioritized after stronger candidates."
    if status == "contradicted":
        return "The features have opposite measured signed effects for this criterion."
    if "score_below_threshold" in reasons:
        return "The candidate does not clear the match-score threshold."
    return "The available fingerprint evidence is weak for this candidate."


def _next_action(claim_grade: str, reason_codes: list[str]) -> str:
    reasons = set(reason_codes)
    if claim_grade == "validated_equivalent":
        return "Replicate the match on held-out prompts, then include it in cross-model mechanism summaries."
    if claim_grade == "needs_more_evidence":
        if "missing_signed_effects" in reasons or "causal_component_neutral" in reasons:
            return "Run matched interventions or path-patching records for both features on the same criterion."
        if "signed_effect_delta_above_threshold" in reasons:
            return "Repeat interventions with more prompts and compare effect-size calibration."
        return "Collect causal evidence for this pair before using it as an equivalence claim."
    if claim_grade == "plausible_equivalent":
        return "Keep as a candidate and gather activation examples or interventions if the pair is scientifically useful."
    if claim_grade == "contradicted_effect":
        return "Inspect labels and intervention setup; treat this pair as a contrast unless new evidence resolves the direction conflict."
    return "Lower priority unless other evidence makes this pair important."


def _validation_run_assessment(validations: list[dict[str, Any]]) -> dict[str, str]:
    if not validations:
        return {
            "overall_claim_grade": "no_match_candidates",
            "summary": "No candidate matches were present in the match report.",
            "recommended_next_action": "Run `interp-lab match`, then validate the resulting matches.",
        }
    counts = _counts(validations, "status")
    if counts.get("validated"):
        return {
            "overall_claim_grade": "validated_matches_present",
            "summary": f"{counts['validated']} match claim(s) passed the validation thresholds.",
            "recommended_next_action": "Replicate validated matches on held-out prompts and include them in graph review.",
        }
    if counts.get("needs_causal_evidence"):
        return {
            "overall_claim_grade": "causal_evidence_needed",
            "summary": f"{counts['needs_causal_evidence']} match claim(s) have structural support and need causal evidence.",
            "recommended_next_action": "Run matched interventions or path patching for the highest-scoring pairs.",
        }
    if counts.get("contradicted"):
        return {
            "overall_claim_grade": "contradicted_matches_present",
            "summary": f"{counts['contradicted']} match claim(s) have opposite signed effects.",
            "recommended_next_action": "Review contradicted pairs as possible contrast features before publishing equivalence claims.",
        }
    if counts.get("plausible"):
        return {
            "overall_claim_grade": "plausible_matches_present",
            "summary": f"{counts['plausible']} match claim(s) have partial support.",
            "recommended_next_action": "Collect more examples, decoder evidence, or causal tests for the most useful pairs.",
        }
    return {
        "overall_claim_grade": "weak_matches_only",
        "summary": "The checked matches did not clear the current evidence thresholds.",
        "recommended_next_action": "Lower priority or rerun matching with richer feature fingerprints.",
    }


def _validation_agent_next_actions(run_assessment: dict[str, str]) -> list[dict[str, str]]:
    grade = run_assessment["overall_claim_grade"]
    common_review = {
        "id": "inspect_match_validation",
        "title": "Review match-level claim grades and reason codes",
        "command": "python -c \"from pathlib import Path; print(Path('<match-validation.md>').read_text(encoding='utf-8'))\"",
    }
    if grade == "validated_matches_present":
        return [
            common_review,
            {
                "id": "replicate_validated_matches",
                "title": "Replicate validated cross-model matches on held-out prompts",
                "command": "interp-lab validate-matches --matches <matches.json> --out <match-validation.json>",
            },
            {
                "id": "add_matches_to_graph_review",
                "title": "Use validated matches while reviewing attribution graphs",
                "command": "interp-lab export-attribution-graph --report <report.json> --out <graph.json> --html-out <graph.html>",
            },
        ]
    if grade == "causal_evidence_needed":
        return [
            common_review,
            {
                "id": "run_matched_interventions",
                "title": "Collect signed causal evidence for the highest-scoring feature pairs",
                "command": "interp-lab export-hf-interventions --model <model> --dataset <prompts.jsonl> --features <features.jsonl> --criterion <criterion> --out <interventions.jsonl>",
            },
        ]
    if grade == "contradicted_matches_present":
        return [
            common_review,
            {
                "id": "inspect_contrast_pairs",
                "title": "Inspect pairs with opposite signed effects before using them as equivalents",
                "command": "interp-lab validate-matches --matches <matches.json> --out <match-validation.json> --min-score 0.7",
            },
        ]
    return [
        common_review,
        {
            "id": "enrich_feature_fingerprints",
            "title": "Add examples, decoder signatures, or interventions before matching again",
            "command": "interp-lab inspect --model <model> --criterion <criterion> --backend records --records <records.jsonl> --interventions <interventions.jsonl> --out <report-dir>",
        },
    ]


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _has_strong_signed_effects(match: CandidateMatch, threshold: float) -> bool:
    if match.left_signed_effect is None or match.right_signed_effect is None:
        return False
    return abs(match.left_signed_effect) >= threshold and abs(match.right_signed_effect) >= threshold


def _same_effect_direction(match: CandidateMatch, threshold: float) -> bool | None:
    if not _has_strong_signed_effects(match, threshold):
        return None
    return (match.left_signed_effect or 0.0) * (match.right_signed_effect or 0.0) > 0.0


def _signed_effect_delta(match: CandidateMatch) -> float | None:
    if match.left_signed_effect is None or match.right_signed_effect is None:
        return None
    return abs(match.left_signed_effect - match.right_signed_effect)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_round(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _markdown_number(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"
