from __future__ import annotations

import argparse
import html
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oracle_sae.matching import match_feature_cards
from oracle_sae.math_utils import clamp, cosine
from oracle_sae.reporting import load_inspection_report
from oracle_sae.schema import FeatureCard, InspectionReport, utc_now_iso
from oracle_sae.text_vectors import content_tokens, hash_text_vector


EXPLANATION_CONSISTENCY_SCHEMA = "interp-lab.explanation_consistency.v1"
FEATURE_SEARCH_SCHEMA = "interp-lab.feature_search.v1"
MODEL_FAMILY_COMPARISON_SCHEMA = "interp-lab.model_family_comparison.v1"
TEXT_PIVOT_MATCH_SCHEMA = "interp-lab.text_pivot_match.v1"


@dataclass(frozen=True)
class WrittenJsonMarkdown:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path
    html_path: Path | None = None


def build_explanation_consistency_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", action="append", required=True, help="Inspection report JSON. Repeat for paraphrases.")
    parser.add_argument("--out", default="reports/explanation-consistency.json", help="Output JSON path.")
    parser.add_argument("--markdown-out", help="Output Markdown path. Defaults to --out with .md suffix.")
    parser.add_argument("--html-out", help="Optional self-contained HTML report path.")
    parser.add_argument("--min-similarity", type=float, default=0.72, help="Minimum explanation similarity for consistency.")
    parser.add_argument("--max-rank-span", type=int, default=5, help="Maximum rank span before a shared feature is marked as rank drift.")
    parser.add_argument("--top-k", type=int, help="Only compare the first N cards from each report.")
    return parser


def build_feature_search_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", action="append", required=True, help="Inspection report JSON. Repeat to search multiple reports.")
    parser.add_argument("--query", required=True, help="Natural-language feature description to search for.")
    parser.add_argument("--out", default="reports/feature-search.json", help="Output JSON path.")
    parser.add_argument("--markdown-out", help="Output Markdown path. Defaults to --out with .md suffix.")
    parser.add_argument("--html-out", help="Optional self-contained HTML report path.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of feature hits to keep.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum search score to include.")
    return parser


def build_model_family_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--member",
        action="append",
        required=True,
        help="Family/report pair as FAMILY=path/to/report.json. Repeat for every model report.",
    )
    parser.add_argument("--out", default="reports/model-family-comparison.json", help="Output JSON path.")
    parser.add_argument("--markdown-out", help="Output Markdown path. Defaults to --out with .md suffix.")
    parser.add_argument("--html-out", help="Optional self-contained HTML report path.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of pairwise feature matches to keep per model pair.")
    parser.add_argument("--min-score", type=float, default=0.65, help="Score threshold for strong pairwise matches.")
    return parser


def build_text_pivot_match_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--left", action="append", required=True, help="Left inspection report JSON or report directory. Repeat to merge reports.")
    parser.add_argument("--right", action="append", required=True, help="Right inspection report JSON or report directory. Repeat to merge reports.")
    parser.add_argument("--out", default="reports/text-pivot-matches.json", help="Output JSON path.")
    parser.add_argument("--markdown-out", help="Output Markdown path. Defaults to --out with .md suffix.")
    parser.add_argument("--html-out", help="Optional self-contained HTML report path.")
    parser.add_argument("--top-k", type=int, default=20, help="Total number of candidate matches to keep.")
    parser.add_argument("--per-left", type=int, default=3, help="Candidate matches to keep for each left feature.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum combined match score.")
    parser.add_argument("--min-text-score", type=float, default=0.55, help="Minimum text-pivot score before a match is considered text-supported.")
    return parser


def run_explanation_consistency_from_args(args: argparse.Namespace) -> WrittenJsonMarkdown:
    return export_explanation_consistency_report(
        reports=args.report,
        out=args.out,
        markdown_out=args.markdown_out,
        html_out=args.html_out,
        min_similarity=args.min_similarity,
        max_rank_span=args.max_rank_span,
        top_k=args.top_k,
    )


def run_feature_search_from_args(args: argparse.Namespace) -> WrittenJsonMarkdown:
    return export_feature_search_report(
        reports=args.report,
        query=args.query,
        out=args.out,
        markdown_out=args.markdown_out,
        html_out=args.html_out,
        top_k=args.top_k,
        min_score=args.min_score,
    )


def run_model_family_from_args(args: argparse.Namespace) -> WrittenJsonMarkdown:
    return export_model_family_comparison_report(
        members=[parse_family_member(item) for item in args.member],
        out=args.out,
        markdown_out=args.markdown_out,
        html_out=args.html_out,
        top_k=args.top_k,
        min_score=args.min_score,
    )


def run_text_pivot_match_from_args(args: argparse.Namespace) -> WrittenJsonMarkdown:
    return export_text_pivot_match_report(
        left_reports=args.left,
        right_reports=args.right,
        out=args.out,
        markdown_out=args.markdown_out,
        html_out=args.html_out,
        top_k=args.top_k,
        per_left=args.per_left,
        min_score=args.min_score,
        min_text_score=args.min_text_score,
    )


def export_explanation_consistency_report(
    *,
    reports: list[str | Path],
    out: str | Path,
    markdown_out: str | Path | None = None,
    html_out: str | Path | None = None,
    min_similarity: float = 0.72,
    max_rank_span: int = 5,
    top_k: int | None = None,
) -> WrittenJsonMarkdown:
    report = build_explanation_consistency_report(
        reports=reports,
        min_similarity=min_similarity,
        max_rank_span=max_rank_span,
        top_k=top_k,
    )
    return _write_json_markdown(
        report,
        out,
        markdown_out,
        render_explanation_consistency_markdown(report),
        html_out=html_out,
        html=render_explanation_consistency_html(report),
    )


def build_explanation_consistency_report(
    *,
    reports: list[str | Path],
    min_similarity: float = 0.72,
    max_rank_span: int = 5,
    top_k: int | None = None,
) -> dict[str, Any]:
    loaded = _load_reports(reports)
    feature_groups: dict[str, list[tuple[int, FeatureCard]]] = {}
    for report_index, item in enumerate(loaded):
        cards = item["report"].cards[:top_k] if top_k is not None else item["report"].cards
        for rank, card in enumerate(cards, start=1):
            feature_groups.setdefault(card.feature_id, []).append((rank, card))
    checks = []
    for feature_id, occurrences in sorted(feature_groups.items()):
        ranks = [rank for rank, _ in occurrences]
        cards = [card for _, card in occurrences]
        similarities = [
            _text_similarity(_explanation_text(left), _explanation_text(right))
            for left, right in itertools.combinations(cards, 2)
        ]
        mean_similarity = sum(similarities) / len(similarities) if similarities else 1.0
        rank_span = max(ranks) - min(ranks) if ranks else 0
        if len(occurrences) < len(loaded):
            status = "missing_in_some_reports"
        elif mean_similarity < min_similarity:
            status = "explanation_drift"
        elif rank_span > max_rank_span:
            status = "rank_drift"
        else:
            status = "consistent"
        checks.append(
            {
                "feature_id": feature_id,
                "status": status,
                "occurrence_count": len(occurrences),
                "report_count": len(loaded),
                "mean_explanation_similarity": round(mean_similarity, 6),
                "rank_span": rank_span,
                "importance_mean": round(sum(card.importance for card in cards) / len(cards), 6),
                "labels": sorted({card.label for card in cards if card.label}),
                "reports": [
                    {
                        "path": loaded[index]["path"],
                        "model": loaded[index]["report"].model,
                        "criterion": loaded[index]["report"].criterion.text,
                        "rank": rank,
                        "importance": card.importance,
                        "explanation": card.explanation,
                    }
                    for index, (rank, card) in _occurrences_with_report_indexes(loaded, occurrences)
                ],
            }
        )
    checks.sort(key=lambda item: (_status_sort_key(item["status"]), -item["occurrence_count"], -item["importance_mean"]))
    summary = {
        "report_count": len(loaded),
        "feature_count": len(checks),
        "consistent_count": sum(1 for item in checks if item["status"] == "consistent"),
        "explanation_drift_count": sum(1 for item in checks if item["status"] == "explanation_drift"),
        "rank_drift_count": sum(1 for item in checks if item["status"] == "rank_drift"),
        "missing_count": sum(1 for item in checks if item["status"] == "missing_in_some_reports"),
        "shared_feature_count": sum(1 for item in checks if item["occurrence_count"] == len(loaded)),
    }
    summary["shared_feature_fraction"] = round(
        summary["shared_feature_count"] / summary["feature_count"],
        6,
    ) if summary["feature_count"] else 0.0
    return {
        "schema_version": EXPLANATION_CONSISTENCY_SCHEMA,
        "created_at": utc_now_iso(),
        "summary": summary,
        "reports": [
            {"path": item["path"], "model": item["report"].model, "criterion": item["report"].criterion.text}
            for item in loaded
        ],
        "thresholds": {"min_similarity": min_similarity, "max_rank_span": max_rank_span, "top_k": top_k},
        "checks": checks,
        "agent_next_actions": _consistency_actions(summary),
    }


def export_feature_search_report(
    *,
    reports: list[str | Path],
    query: str,
    out: str | Path,
    markdown_out: str | Path | None = None,
    html_out: str | Path | None = None,
    top_k: int = 10,
    min_score: float = 0.0,
) -> WrittenJsonMarkdown:
    report = build_feature_search_report(reports=reports, query=query, top_k=top_k, min_score=min_score)
    return _write_json_markdown(
        report,
        out,
        markdown_out,
        render_feature_search_markdown(report),
        html_out=html_out,
        html=render_feature_search_html(report),
    )


def build_feature_search_report(
    *,
    reports: list[str | Path],
    query: str,
    top_k: int = 10,
    min_score: float = 0.0,
) -> dict[str, Any]:
    loaded = _load_reports(reports)
    query_vector = hash_text_vector(query)
    results = []
    for item in loaded:
        report: InspectionReport = item["report"]
        for rank, card in enumerate(report.cards, start=1):
            components = _search_components(query, query_vector, card)
            semantic = max(components["label"], components["explanation"], components["fingerprint"])
            score = round(
                0.45 * semantic
                + 0.15 * components["examples"]
                + 0.25 * card.importance
                + 0.15 * abs(card.association),
                6,
            )
            if score < min_score:
                continue
            results.append(
                {
                    "feature_id": card.feature_id,
                    "model": card.model,
                    "layer": card.layer,
                    "report_path": item["path"],
                    "criterion": report.criterion.text,
                    "rank": rank,
                    "score": score,
                    "components": components,
                    "label": card.label,
                    "explanation": card.explanation,
                    "importance": card.importance,
                    "association": card.association,
                    "causal_effect": card.causal_effect,
                    "source": card.source,
                    "matched_terms": sorted(set(content_tokens(query)) & set(content_tokens(_card_search_text(card)))),
                    "agent_next_action": (
                        f"Run interp-lab intervene --report {item['path']} --feature {card.feature_id} "
                        "to test amplification or suppression."
                    ),
                }
            )
    results.sort(key=lambda item: item["score"], reverse=True)
    kept = results[:top_k]
    return {
        "schema_version": FEATURE_SEARCH_SCHEMA,
        "created_at": utc_now_iso(),
        "query": query,
        "summary": {
            "report_count": len(loaded),
            "searched_feature_count": sum(len(item["report"].cards) for item in loaded),
            "result_count": len(kept),
            "min_score": min_score,
        },
        "results": kept,
        "agent_next_actions": [
            {
                "id": "inspect_top_search_hits",
                "description": "Review high scoring explanation matches, then run causal interventions before treating a hit as behaviorally responsible.",
            }
        ],
    }


def export_model_family_comparison_report(
    *,
    members: list[dict[str, str]],
    out: str | Path,
    markdown_out: str | Path | None = None,
    html_out: str | Path | None = None,
    top_k: int = 5,
    min_score: float = 0.65,
) -> WrittenJsonMarkdown:
    report = build_model_family_comparison_report(members=members, top_k=top_k, min_score=min_score)
    return _write_json_markdown(
        report,
        out,
        markdown_out,
        render_model_family_markdown(report),
        html_out=html_out,
        html=render_model_family_html(report),
    )


def export_text_pivot_match_report(
    *,
    left_reports: list[str | Path],
    right_reports: list[str | Path],
    out: str | Path,
    markdown_out: str | Path | None = None,
    html_out: str | Path | None = None,
    top_k: int = 20,
    per_left: int = 3,
    min_score: float = 0.0,
    min_text_score: float = 0.55,
) -> WrittenJsonMarkdown:
    report = build_text_pivot_match_report(
        left_reports=left_reports,
        right_reports=right_reports,
        top_k=top_k,
        per_left=per_left,
        min_score=min_score,
        min_text_score=min_text_score,
    )
    return _write_json_markdown(
        report,
        out,
        markdown_out,
        render_text_pivot_match_markdown(report),
        html_out=html_out,
        html=render_text_pivot_match_html(report),
    )


def build_text_pivot_match_report(
    *,
    left_reports: list[str | Path],
    right_reports: list[str | Path],
    top_k: int = 20,
    per_left: int = 3,
    min_score: float = 0.0,
    min_text_score: float = 0.55,
) -> dict[str, Any]:
    left_loaded = _load_reports(left_reports)
    right_loaded = _load_reports(right_reports)
    left_cards = _ranked_cards(left_loaded)
    right_cards = _ranked_cards(right_loaded)
    grouped: list[dict[str, Any]] = []
    all_matches: list[dict[str, Any]] = []
    for left_item in left_cards:
        candidates = []
        for right_item in right_cards:
            match = _text_pivot_match(left_item, right_item, min_text_score=min_text_score)
            if match["score"] >= min_score:
                candidates.append(match)
        candidates.sort(key=lambda item: item["score"], reverse=True)
        kept = candidates[: max(1, per_left)]
        all_matches.extend(kept)
        if kept:
            grouped.append(
                {
                    "left_feature_id": left_item["card"].feature_id,
                    "left_model": left_item["report"].model,
                    "left_report": left_item["path"],
                    "left_rank": left_item["rank"],
                    "left_label": left_item["card"].label,
                    "candidates": kept,
                }
            )
    all_matches.sort(key=lambda item: item["score"], reverse=True)
    top_matches = all_matches[:top_k]
    summary = {
        "left_report_count": len(left_loaded),
        "right_report_count": len(right_loaded),
        "left_feature_count": len(left_cards),
        "right_feature_count": len(right_cards),
        "candidate_count": len(top_matches),
        "text_supported_count": sum(1 for item in top_matches if item["components"]["text_pivot"] >= min_text_score),
        "causal_supported_count": sum(1 for item in top_matches if item["evidence_grade"] == "text_pivot_with_causal_support"),
        "strongest_score": top_matches[0]["score"] if top_matches else 0.0,
    }
    return {
        "schema_version": TEXT_PIVOT_MATCH_SCHEMA,
        "created_at": utc_now_iso(),
        "summary": summary,
        "thresholds": {
            "top_k": top_k,
            "per_left": per_left,
            "min_score": min_score,
            "min_text_score": min_text_score,
        },
        "left_reports": _report_descriptors(left_loaded),
        "right_reports": _report_descriptors(right_loaded),
        "matches": top_matches,
        "by_left_feature": grouped,
        "agent_next_actions": _text_pivot_actions(top_matches),
    }


def build_model_family_comparison_report(
    *,
    members: list[dict[str, str]],
    top_k: int = 5,
    min_score: float = 0.65,
) -> dict[str, Any]:
    loaded = []
    for member in members:
        report_path = _resolve_report_path(member["report"])
        report = load_inspection_report(report_path)
        loaded.append({"family": member["family"], "path": str(report_path), "report": report})
    pairwise = []
    for left, right in itertools.combinations(loaded, 2):
        matches = match_feature_cards(left["report"].cards, right["report"].cards, top_k=top_k, min_score=0.0)
        strong = [match for match in matches if match.score >= min_score]
        top_score = matches[0].score if matches else 0.0
        pairwise.append(
            {
                "left_family": left["family"],
                "right_family": right["family"],
                "left_model": left["report"].model,
                "right_model": right["report"].model,
                "left_report": left["path"],
                "right_report": right["path"],
                "relation": "within_family" if left["family"] == right["family"] else "cross_family",
                "top_score": top_score,
                "strong_match_count": len(strong),
                "matches": [match.to_dict() for match in matches],
            }
        )
    family_rows = []
    for family in sorted({item["family"] for item in loaded}):
        reports = [item for item in loaded if item["family"] == family]
        family_rows.append(
            {
                "family": family,
                "report_count": len(reports),
                "models": [item["report"].model for item in reports],
                "feature_count": sum(len(item["report"].cards) for item in reports),
            }
        )
    cross_pairs = [item for item in pairwise if item["relation"] == "cross_family"]
    within_pairs = [item for item in pairwise if item["relation"] == "within_family"]
    family_pairs = _family_pair_summaries(pairwise)
    summary = {
        "family_count": len(family_rows),
        "report_count": len(loaded),
        "pair_count": len(pairwise),
        "cross_family_pair_count": len(cross_pairs),
        "within_family_pair_count": len(within_pairs),
        "mean_cross_family_top_score": round(_mean([item["top_score"] for item in cross_pairs]), 6),
        "mean_within_family_top_score": round(_mean([item["top_score"] for item in within_pairs]), 6),
        "strong_cross_family_pair_count": sum(1 for item in cross_pairs if item["strong_match_count"] > 0),
    }
    return {
        "schema_version": MODEL_FAMILY_COMPARISON_SCHEMA,
        "created_at": utc_now_iso(),
        "summary": summary,
        "families": family_rows,
        "family_pairs": family_pairs,
        "pairwise": pairwise,
        "thresholds": {"top_k": top_k, "min_score": min_score},
        "agent_next_actions": _family_actions(pairwise),
    }


def parse_family_member(value: str) -> dict[str, str]:
    if "=" not in value:
        raise ValueError("--member must be formatted as FAMILY=path/to/report.json")
    family, report = value.split("=", maxsplit=1)
    family = family.strip()
    report = report.strip()
    if not family or not report:
        raise ValueError("--member must include a non-empty family and report path")
    return {"family": family, "report": report}


def render_explanation_consistency_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Explanation Consistency",
        "",
        f"Reports: {report['summary']['report_count']}",
        f"Consistent features: {report['summary']['consistent_count']}",
        f"Shared feature fraction: {report['summary']['shared_feature_fraction']:.3f}",
        f"Explanation drift: {report['summary']['explanation_drift_count']}",
        f"Rank drift: {report['summary']['rank_drift_count']}",
        f"Missing in some reports: {report['summary']['missing_count']}",
        "",
        "## Checks",
    ]
    for item in report["checks"][:25]:
        lines.extend(
            [
                "",
                f"### {item['feature_id']}",
                f"Status: {item['status']}",
                f"Mean explanation similarity: {item['mean_explanation_similarity']:.3f}",
                f"Rank span: {item['rank_span']}",
                f"Labels: {', '.join(item['labels']) if item['labels'] else 'none'}",
            ]
        )
    lines.extend(_actions_markdown(report.get("agent_next_actions", [])))
    return "\n".join(lines) + "\n"


def render_feature_search_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Feature Search",
        "",
        f"Query: {report['query']}",
        f"Results: {report['summary']['result_count']} of {report['summary']['searched_feature_count']} searched features",
        "",
        "## Results",
    ]
    for result in report["results"]:
        terms = ", ".join(result["matched_terms"]) if result["matched_terms"] else "semantic match"
        lines.extend(
            [
                "",
                f"### {result['feature_id']} ({result['model']})",
                f"Score: {result['score']:.3f}",
                f"Label: {result['label']}",
                f"Matched terms: {terms}",
                f"Importance: {result['importance']:.3f}; causal effect: {result['causal_effect']:.3f}",
                result["explanation"],
            ]
        )
    lines.extend(_actions_markdown(report.get("agent_next_actions", [])))
    return "\n".join(lines) + "\n"


def render_model_family_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Model-Family Comparison",
        "",
        f"Families: {report['summary']['family_count']}",
        f"Reports: {report['summary']['report_count']}",
        f"Mean cross-family top score: {report['summary']['mean_cross_family_top_score']:.3f}",
        f"Strong cross-family pairs: {report['summary']['strong_cross_family_pair_count']}",
        "",
        "## Families",
    ]
    for family in report["families"]:
        lines.append(f"- {family['family']}: {family['report_count']} report(s), {family['feature_count']} feature(s)")
    if report.get("family_pairs"):
        lines.append("")
        lines.append("## Family Pairs")
        for pair in report["family_pairs"]:
            lines.append(
                f"- {pair['left_family']} -> {pair['right_family']}: "
                f"mean top score {pair['mean_top_score']:.3f}, strong pairs {pair['strong_pair_count']}/{pair['pair_count']}"
            )
    lines.append("")
    lines.append("## Pairwise")
    for pair in report["pairwise"]:
        lines.extend(
            [
                "",
                f"### {pair['left_model']} -> {pair['right_model']}",
                f"Families: {pair['left_family']} -> {pair['right_family']} ({pair['relation']})",
                f"Top score: {pair['top_score']:.3f}; strong matches: {pair['strong_match_count']}",
            ]
        )
        for match in pair["matches"][:3]:
            lines.append(f"- {match['left_feature_id']} -> {match['right_feature_id']}: {match['score']:.3f}")
    lines.extend(_actions_markdown(report.get("agent_next_actions", [])))
    return "\n".join(lines) + "\n"


def render_text_pivot_match_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Text-Pivot Matches",
        "",
        f"Left reports: {report['summary']['left_report_count']}",
        f"Right reports: {report['summary']['right_report_count']}",
        f"Candidates: {report['summary']['candidate_count']}",
        f"Text-supported: {report['summary']['text_supported_count']}",
        f"Causal-supported: {report['summary']['causal_supported_count']}",
        "",
        "## Matches",
    ]
    for item in report["matches"]:
        lines.extend(
            [
                "",
                f"### {item['left_feature_id']} -> {item['right_feature_id']}",
                f"Score: {item['score']:.3f}",
                f"Evidence grade: {item['evidence_grade']}",
                f"Models: {item['left_model']} -> {item['right_model']}",
                f"Labels: {item['left_label']} -> {item['right_label']}",
                "Components: "
                + ", ".join(f"{key}={value:.3f}" for key, value in sorted(item["components"].items())),
            ]
        )
    lines.extend(_actions_markdown(report.get("agent_next_actions", [])))
    return "\n".join(lines) + "\n"


def render_explanation_consistency_html(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = "".join(
        "<tr>"
        f"<td>{_h(item['feature_id'])}</td>"
        f"<td><span class=\"pill {item['status']}\">{_h(item['status'])}</span></td>"
        f"<td>{item['mean_explanation_similarity']:.3f}</td>"
        f"<td>{item['rank_span']}</td>"
        f"<td>{item['occurrence_count']}/{item['report_count']}</td>"
        f"<td>{_h(', '.join(item['labels']))}</td>"
        "</tr>"
        for item in report["checks"][:200]
    )
    body = f"""
      <section class="metrics">
        {_metric('Reports', summary['report_count'])}
        {_metric('Features', summary['feature_count'])}
        {_metric('Consistent', summary['consistent_count'])}
        {_metric('Shared Fraction', f"{summary['shared_feature_fraction']:.3f}")}
      </section>
      <table>
        <thead><tr><th>Feature</th><th>Status</th><th>Similarity</th><th>Rank Span</th><th>Seen</th><th>Labels</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    """
    return _analysis_html("Explanation Consistency", body, report)


def render_feature_search_html(report: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{_h(item['feature_id'])}</td>"
        f"<td>{_h(item['model'])}</td>"
        f"<td>{item['score']:.3f}</td>"
        f"<td>{item['importance']:.3f}</td>"
        f"<td>{item['causal_effect']:.3f}</td>"
        f"<td>{_h(item['label'])}</td>"
        f"<td>{_h(', '.join(item['matched_terms']) or 'semantic match')}</td>"
        "</tr>"
        for item in report["results"]
    )
    summary = report["summary"]
    body = f"""
      <p class="lede">Query: <strong>{_h(report['query'])}</strong></p>
      <section class="metrics">
        {_metric('Reports', summary['report_count'])}
        {_metric('Searched Features', summary['searched_feature_count'])}
        {_metric('Results', summary['result_count'])}
      </section>
      <table>
        <thead><tr><th>Feature</th><th>Model</th><th>Score</th><th>Importance</th><th>Effect</th><th>Label</th><th>Terms</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    """
    return _analysis_html("Feature Search", body, report)


def render_model_family_html(report: dict[str, Any]) -> str:
    summary = report["summary"]
    family_rows = "".join(
        "<tr>"
        f"<td>{_h(item['family'])}</td>"
        f"<td>{item['report_count']}</td>"
        f"<td>{item['feature_count']}</td>"
        f"<td>{_h(', '.join(item['models']))}</td>"
        "</tr>"
        for item in report["families"]
    )
    pair_rows = "".join(
        "<tr>"
        f"<td>{_h(item['left_family'])} -> {_h(item['right_family'])}</td>"
        f"<td>{item['pair_count']}</td>"
        f"<td>{item['mean_top_score']:.3f}</td>"
        f"<td>{item['strong_pair_count']}</td>"
        "</tr>"
        for item in report.get("family_pairs", [])
    )
    body = f"""
      <section class="metrics">
        {_metric('Families', summary['family_count'])}
        {_metric('Reports', summary['report_count'])}
        {_metric('Cross-Family Pairs', summary['cross_family_pair_count'])}
        {_metric('Mean Cross Top', f"{summary['mean_cross_family_top_score']:.3f}")}
      </section>
      <h2>Families</h2>
      <table>
        <thead><tr><th>Family</th><th>Reports</th><th>Features</th><th>Models</th></tr></thead>
        <tbody>{family_rows}</tbody>
      </table>
      <h2>Family Pairs</h2>
      <table>
        <thead><tr><th>Families</th><th>Pairs</th><th>Mean Top Score</th><th>Strong Pairs</th></tr></thead>
        <tbody>{pair_rows}</tbody>
      </table>
    """
    return _analysis_html("Model-Family Comparison", body, report)


def render_text_pivot_match_html(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = "".join(
        "<tr>"
        f"<td>{_h(item['left_feature_id'])}<br><span>{_h(item['left_model'])}</span></td>"
        f"<td>{_h(item['right_feature_id'])}<br><span>{_h(item['right_model'])}</span></td>"
        f"<td>{item['score']:.3f}</td>"
        f"<td>{item['components']['text_pivot']:.3f}</td>"
        f"<td>{item['components']['causal']:.3f}</td>"
        f"<td><span class=\"pill {item['evidence_grade']}\">{_h(item['evidence_grade'])}</span></td>"
        f"<td>{_h(item['left_label'])} -> {_h(item['right_label'])}</td>"
        "</tr>"
        for item in report["matches"]
    )
    body = f"""
      <section class="metrics">
        {_metric('Candidates', summary['candidate_count'])}
        {_metric('Text Supported', summary['text_supported_count'])}
        {_metric('Causal Supported', summary['causal_supported_count'])}
        {_metric('Strongest Score', f"{summary['strongest_score']:.3f}")}
      </section>
      <table>
        <thead><tr><th>Left</th><th>Right</th><th>Score</th><th>Text</th><th>Causal</th><th>Grade</th><th>Labels</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    """
    return _analysis_html("Text-Pivot Matches", body, report)


def _load_reports(paths: list[str | Path]) -> list[dict[str, Any]]:
    loaded = []
    for path in paths:
        report_path = _resolve_report_path(path)
        loaded.append({"path": str(report_path), "report": load_inspection_report(report_path)})
    return loaded


def _report_descriptors(loaded: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"path": item["path"], "model": item["report"].model, "criterion": item["report"].criterion.text}
        for item in loaded
    ]


def _ranked_cards(loaded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for item in loaded:
        total = len(item["report"].cards)
        for rank, card in enumerate(item["report"].cards, start=1):
            ranked.append({"path": item["path"], "report": item["report"], "card": card, "rank": rank, "total": total})
    return ranked


def _occurrences_with_report_indexes(
    loaded: list[dict[str, Any]],
    occurrences: list[tuple[int, FeatureCard]],
) -> list[tuple[int, tuple[int, FeatureCard]]]:
    output = []
    for rank, card in occurrences:
        for index, item in enumerate(loaded):
            if any(existing.feature_id == card.feature_id and existing is card for existing in item["report"].cards):
                output.append((index, (rank, card)))
                break
    return output


def _search_components(query: str, query_vector: list[float], card: FeatureCard) -> dict[str, float]:
    return {
        "label": round(_text_similarity(query, card.label), 6),
        "explanation": round(_text_similarity(query, card.explanation), 6),
        "examples": round(_text_similarity(query, " ".join(card.examples[:5])), 6),
        "fingerprint": round((cosine(query_vector, card.fingerprint.text_vector) + 1.0) / 2.0, 6),
    }


def _text_pivot_match(left_item: dict[str, Any], right_item: dict[str, Any], *, min_text_score: float) -> dict[str, Any]:
    left = left_item["card"]
    right = right_item["card"]
    explanation = _text_similarity(_explanation_text(left), _explanation_text(right))
    label = _text_similarity(left.label, right.label)
    examples = _text_similarity(" ".join(left.examples[:5]), " ".join(right.examples[:5]))
    fingerprint_text = _unit_cosine(left.fingerprint.text_vector, right.fingerprint.text_vector)
    activation = _unit_cosine(left.fingerprint.activation_signature, right.fingerprint.activation_signature, neutral=0.5)
    decoder = _unit_cosine(left.fingerprint.decoder_signature, right.fingerprint.decoder_signature, neutral=0.5)
    causal = _causal_component(left, right)
    rank = _rank_component(left_item["rank"], right_item["rank"], left_item["total"], right_item["total"])
    signed = _signed_component(left, right)
    text_pivot = max(explanation, label, examples, fingerprint_text)
    score = clamp(
        0.38 * text_pivot
        + 0.17 * explanation
        + 0.13 * causal
        + 0.10 * activation
        + 0.07 * decoder
        + 0.10 * rank
        + 0.05 * signed
    )
    if _opposite_signed_effects(left, right):
        score = min(score, 0.49)
    components = {
        "text_pivot": round(text_pivot, 6),
        "explanation": round(explanation, 6),
        "label": round(label, 6),
        "examples": round(examples, 6),
        "fingerprint_text": round(fingerprint_text, 6),
        "activation": round(activation, 6),
        "decoder": round(decoder, 6),
        "causal": round(causal, 6),
        "rank": round(rank, 6),
        "signed_effect": round(signed, 6),
    }
    return {
        "left_feature_id": left.feature_id,
        "right_feature_id": right.feature_id,
        "left_model": left.model,
        "right_model": right.model,
        "left_report": left_item["path"],
        "right_report": right_item["path"],
        "left_rank": left_item["rank"],
        "right_rank": right_item["rank"],
        "left_label": left.label,
        "right_label": right.label,
        "score": round(score, 6),
        "components": components,
        "evidence_grade": _text_pivot_grade(components, min_text_score=min_text_score),
        "matched_terms": sorted(set(content_tokens(_card_search_text(left))) & set(content_tokens(_card_search_text(right)))),
        "agent_next_action": (
            f"Validate with interp-lab match --left {left_item['path']} --right {right_item['path']} "
            "followed by interp-lab validate-matches."
        ),
    }


def _card_search_text(card: FeatureCard) -> str:
    return " ".join([card.label, card.explanation, " ".join(card.examples), card.fingerprint.text])


def _explanation_text(card: FeatureCard) -> str:
    return card.explanation or card.label


def _text_similarity(left: str, right: str) -> float:
    if not left.strip() and not right.strip():
        return 1.0
    if not left.strip() or not right.strip():
        return 0.0
    return (cosine(hash_text_vector(left), hash_text_vector(right)) + 1.0) / 2.0


def _unit_cosine(left: list[float], right: list[float], *, neutral: float = 0.0) -> float:
    if not left or not right:
        return neutral
    return clamp((cosine(left, right) + 1.0) / 2.0)


def _causal_component(left: FeatureCard, right: FeatureCard) -> float:
    if not left.fingerprint.causal_vector or not right.fingerprint.causal_vector:
        return 0.5
    direction = _unit_cosine(left.fingerprint.causal_vector, right.fingerprint.causal_vector, neutral=0.5)
    strength = min(1.0, _norm(left.fingerprint.causal_vector)) * min(1.0, _norm(right.fingerprint.causal_vector))
    return clamp(direction * strength)


def _rank_component(left_rank: int, right_rank: int, left_total: int, right_total: int) -> float:
    left_position = (left_rank - 1) / max(1, left_total - 1)
    right_position = (right_rank - 1) / max(1, right_total - 1)
    return clamp(1.0 - abs(left_position - right_position))


def _signed_component(left: FeatureCard, right: FeatureCard) -> float:
    left_signed = _signed_effect(left)
    right_signed = _signed_effect(right)
    if left_signed is None or right_signed is None:
        return 0.5
    return clamp(1.0 - abs(left_signed - right_signed) / 2.0)


def _opposite_signed_effects(left: FeatureCard, right: FeatureCard) -> bool:
    left_signed = _signed_effect(left)
    right_signed = _signed_effect(right)
    if left_signed is None or right_signed is None:
        return False
    return left_signed * right_signed < 0 and min(abs(left_signed), abs(right_signed)) >= 0.05


def _signed_effect(card: FeatureCard) -> float | None:
    for key in ("signed_causal_effect", "signed_association"):
        if key in card.causal_effects:
            return float(card.causal_effects[key])
    return None


def _norm(values: list[float]) -> float:
    return sum(value * value for value in values) ** 0.5


def _text_pivot_grade(components: dict[str, float], *, min_text_score: float) -> str:
    if components["text_pivot"] >= min_text_score and components["causal"] >= 0.65 and components["signed_effect"] >= 0.65:
        return "text_pivot_with_causal_support"
    if components["text_pivot"] >= min_text_score and components["activation"] >= 0.65:
        return "text_pivot_with_activation_support"
    if components["text_pivot"] >= min_text_score:
        return "text_pivot_candidate"
    return "weak_text_pivot"


def _status_sort_key(status: str) -> int:
    return {
        "explanation_drift": 0,
        "rank_drift": 1,
        "missing_in_some_reports": 2,
        "consistent": 3,
    }.get(status, 4)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _family_pair_summaries(pairwise: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in pairwise:
        key = (item["left_family"], item["right_family"])
        if item["relation"] == "within_family":
            key = (item["left_family"], item["left_family"])
        buckets.setdefault(key, []).append(item)
    rows = []
    for (left_family, right_family), items in sorted(buckets.items()):
        rows.append(
            {
                "left_family": left_family,
                "right_family": right_family,
                "relation": "within_family" if left_family == right_family else "cross_family",
                "pair_count": len(items),
                "mean_top_score": round(_mean([item["top_score"] for item in items]), 6),
                "strong_pair_count": sum(1 for item in items if item["strong_match_count"] > 0),
            }
        )
    return rows


def _consistency_actions(summary: dict[str, Any]) -> list[dict[str, str]]:
    if summary["explanation_drift_count"] or summary["rank_drift_count"]:
        return [
            {
                "id": "review_drifting_explanations",
                "description": "Inspect drifted features, then add paraphrase reports or causal interventions for features that remain important.",
            }
        ]
    return [
        {
            "id": "validate_consistent_features",
            "description": "Use the consistent shared features as candidates for intervention and cross-model validation.",
        }
    ]


def _family_actions(pairwise: list[dict[str, Any]]) -> list[dict[str, str]]:
    strong = [pair for pair in pairwise if pair["strong_match_count"] > 0 and pair["relation"] == "cross_family"]
    if strong:
        pair = strong[0]
        return [
            {
                "id": "validate_cross_family_matches",
                "description": (
                    "Run interp-lab match and validate-matches on the strongest cross-family pair: "
                    f"{pair['left_report']} and {pair['right_report']}."
                ),
            }
        ]
    return [
        {
            "id": "collect_more_family_evidence",
            "description": "Add intervention-backed reports or aligned criteria before claiming model-family transfer.",
        }
    ]


def _text_pivot_actions(matches: list[dict[str, Any]]) -> list[dict[str, str]]:
    if any(item["evidence_grade"] == "text_pivot_with_causal_support" for item in matches):
        return [
            {
                "id": "validate_text_pivot_matches",
                "description": "Run match validation and targeted interventions for the strongest text-pivot pairs before treating them as equivalent features.",
            }
        ]
    if matches:
        return [
            {
                "id": "collect_causal_support_for_text_matches",
                "description": "Use text-pivot candidates to choose features for intervention records, then rerun the report with causal evidence.",
            }
        ]
    return [
        {
            "id": "broaden_text_pivot_inputs",
            "description": "Add richer NLA explanations, activation records, or related reports before repeating text-pivot matching.",
        }
    ]


def _actions_markdown(actions: list[dict[str, str]]) -> list[str]:
    if not actions:
        return []
    lines = ["", "## Agent Next Actions"]
    for action in actions:
        lines.append(f"- {action['id']}: {action['description']}")
    return lines


def _analysis_html(title: str, body: str, report: dict[str, Any]) -> str:
    actions = "".join(
        f"<li><strong>{_h(action['id'])}</strong>: {_h(action['description'])}</li>"
        for action in report.get("agent_next_actions", [])
    )
    payload = _h(json.dumps(report, indent=2, sort_keys=True))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_h(title)}</title>
  <style>
    body {{ margin: 0; background: #f7f7f4; color: #1c2528; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    .lede {{ color: #4b585d; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 20px 0; }}
    .metric {{ background: white; border: 1px solid #d9dedb; border-radius: 8px; padding: 14px; }}
    .metric span {{ display: block; color: #5c686d; font-size: 12px; text-transform: uppercase; letter-spacing: 0; }}
    .metric strong {{ display: block; margin-top: 6px; font-size: 22px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9dedb; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e4e8e5; text-align: left; vertical-align: top; font-size: 14px; }}
    th {{ background: #edf3f0; color: #314045; }}
    .pill {{ display: inline-block; padding: 3px 8px; border-radius: 999px; background: #e7f5f1; color: #0f766e; font-size: 12px; }}
    .explanation_drift {{ background: #fff1f2; color: #be123c; }}
    .rank_drift {{ background: #fff7ed; color: #c2410c; }}
    .missing_in_some_reports {{ background: #f1f5f9; color: #475569; }}
    pre {{ overflow: auto; background: #101820; color: #e5f2ee; border-radius: 8px; padding: 16px; }}
  </style>
</head>
<body>
  <main>
    <h1>{_h(title)}</h1>
    {body}
    <h2>Agent Next Actions</h2>
    <ul>{actions}</ul>
    <h2>JSON</h2>
    <pre>{payload}</pre>
  </main>
</body>
</html>
"""


def _metric(label: str, value: object) -> str:
    return f"<div class=\"metric\"><span>{_h(label)}</span><strong>{_h(str(value))}</strong></div>"


def _h(value: object) -> str:
    return html.escape(str(value), quote=True)


def _resolve_report_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate / "report.json"
    return candidate


def _write_json_markdown(
    report: dict[str, Any],
    out: str | Path,
    markdown_out: str | Path | None,
    markdown: str,
    *,
    html_out: str | Path | None = None,
    html: str | None = None,
) -> WrittenJsonMarkdown:
    json_path = Path(out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = Path(markdown_out) if markdown_out is not None else json_path.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path = None
    if html_out is not None:
        html_path = Path(html_out)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html or _analysis_html("interp-lab Analysis", f"<pre>{_h(markdown)}</pre>", report), encoding="utf-8")
    return WrittenJsonMarkdown(report=report, json_path=json_path, markdown_path=markdown_path, html_path=html_path)
