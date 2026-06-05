"""Diff two inspection reports.

Re-running an inspection with a new seed, a new model checkpoint, or a tweaked
criterion should not silently change your conclusions. ``compare-runs`` makes the
change visible: which features moved in the ranking, how much each score shifted,
and what appeared or disappeared between a baseline (``--left``) and a candidate
(``--right``). It is pure-Python and reads the same ``report.json`` files
``inspect`` already writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from interp_lab.explanation_reports import WrittenJsonMarkdown
from interp_lab.reporting import load_inspection_report
from interp_lab.schema import FeatureCard, InspectionReport

RUN_DIFF_SCHEMA = "interp-lab.run_diff.v1"

_SCORE_KEYS = ("association", "causal_effect", "specificity", "stability")


def _strong(card: FeatureCard) -> float:
    return float(card.causal_effects.get("strong_causal_score", 0.0))


def _side_summary(report: InspectionReport) -> dict[str, Any]:
    tool = report.metadata.get("tool") if isinstance(report.metadata, dict) else None
    return {
        "model": report.model,
        "criterion": report.criterion.text,
        "feature_count": len(report.cards),
        "created_at": report.created_at,
        "tool_version": tool.get("version") if isinstance(tool, dict) else None,
    }


def build_run_diff_report(left: InspectionReport, right: InspectionReport) -> dict[str, Any]:
    left_by_id = {card.feature_id: (rank, card) for rank, card in enumerate(left.cards, start=1)}
    right_by_id = {card.feature_id: (rank, card) for rank, card in enumerate(right.cards, start=1)}

    common_ids = [fid for fid in left_by_id if fid in right_by_id]
    changed: list[dict[str, Any]] = []
    rank_unchanged = 0
    for fid in common_ids:
        left_rank, left_card = left_by_id[fid]
        right_rank, right_card = right_by_id[fid]
        if left_rank == right_rank:
            rank_unchanged += 1
        deltas = {key: round(getattr(right_card, key) - getattr(left_card, key), 6) for key in _SCORE_KEYS}
        deltas["strong_causal_score"] = round(_strong(right_card) - _strong(left_card), 6)
        changed.append(
            {
                "feature_id": fid,
                "label": right_card.label or left_card.label,
                "left_rank": left_rank,
                "right_rank": right_rank,
                # Positive rank_delta == moved UP (toward rank 1) in the candidate run.
                "rank_delta": left_rank - right_rank,
                "importance_left": round(left_card.importance, 6),
                "importance_right": round(right_card.importance, 6),
                "importance_delta": round(right_card.importance - left_card.importance, 6),
                "score_deltas": deltas,
            }
        )
    changed.sort(key=lambda item: abs(item["importance_delta"]), reverse=True)

    added = [
        {"feature_id": fid, "rank": rank, "importance": round(card.importance, 6), "label": card.label}
        for fid, (rank, card) in sorted(right_by_id.items(), key=lambda kv: kv[1][0])
        if fid not in left_by_id
    ]
    dropped = [
        {"feature_id": fid, "rank": rank, "importance": round(card.importance, 6), "label": card.label}
        for fid, (rank, card) in sorted(left_by_id.items(), key=lambda kv: kv[1][0])
        if fid not in right_by_id
    ]

    importance_deltas = [abs(item["importance_delta"]) for item in changed]
    mean_abs = round(sum(importance_deltas) / len(importance_deltas), 6) if importance_deltas else 0.0
    max_abs = round(max(importance_deltas), 6) if importance_deltas else 0.0
    rank_stability = round(rank_unchanged / len(common_ids), 6) if common_ids else 1.0

    criterion_match = left.criterion.text.strip() == right.criterion.text.strip()
    model_match = left.model == right.model
    summary = {
        "criterion_match": criterion_match,
        "model_match": model_match,
        "common_count": len(common_ids),
        "added_count": len(added),
        "dropped_count": len(dropped),
        "mean_abs_importance_delta": mean_abs,
        "max_abs_importance_delta": max_abs,
        "rank_stability": rank_stability,
    }

    report = {
        "schema_version": RUN_DIFF_SCHEMA,
        "left": _side_summary(left),
        "right": _side_summary(right),
        "summary": summary,
        "changed_features": changed,
        "added_features": added,
        "dropped_features": dropped,
        "interpretation": _interpret(summary, criterion_match, model_match),
        "agent_next_actions": _next_actions(added, dropped),
    }
    return report


def _join_sentences(parts: list[str]) -> str:
    # Capitalize the first letter of each sentence without lowercasing the rest
    # (str.capitalize would erase the intentional all-caps emphasis below).
    sentences = [part[0].upper() + part[1:] if part else part for part in parts]
    return ". ".join(sentences) + "." if sentences else ""


def _interpret(summary: dict[str, Any], criterion_match: bool, model_match: bool) -> str:
    parts: list[str] = []
    if not criterion_match:
        parts.append("the two runs used DIFFERENT criteria, so differences are expected")
    if not model_match:
        parts.append("the two runs are different models, so this is a cross-model comparison, not a reproducibility check")
    if summary["common_count"] == 0:
        parts.append("the runs share no feature ids, so only the added/dropped lists apply")
        return _join_sentences(parts)
    if summary["rank_stability"] >= 0.9 and summary["max_abs_importance_delta"] < 0.05 and not summary["added_count"] and not summary["dropped_count"]:
        parts.append("the runs are essentially identical (stable ranking, tiny score drift)")
    elif summary["rank_stability"] >= 0.6:
        parts.append("the ranking is broadly stable with some movement; review the top movers")
    else:
        parts.append("the ranking changed substantially; treat the conclusions as run-dependent")
    if summary["added_count"] or summary["dropped_count"]:
        parts.append(f"{summary['added_count']} feature(s) appeared and {summary['dropped_count']} dropped out")
    return _join_sentences(parts)


def _next_actions(added: list[dict[str, Any]], dropped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions = [
        {
            "id": "inspect_top_movers",
            "title": "Open the markdown diff to read the biggest movers and added/dropped features",
            "command": "python -c \"from pathlib import Path; print(Path('<run-diff.md>').read_text(encoding='utf-8'))\"",
        }
    ]
    if added or dropped:
        actions.append(
            {
                "id": "stabilize_run",
                "title": "If this was a reproducibility check, broaden prompts or fix the seed to stabilize the feature set",
                "command": "interp-lab inspect --model <model> --criterion <criterion> --backend <backend> --out <out>",
            }
        )
    return actions


def render_run_diff_markdown(report: dict[str, Any]) -> str:
    left = report["left"]
    right = report["right"]
    summary = report["summary"]
    lines = [
        "# interp-lab Run Diff",
        "",
        f"Left (baseline):  `{left['model']}`  -- {left['feature_count']} features"
        + (f"  ({left['tool_version']})" if left.get("tool_version") else ""),
        f"Right (candidate): `{right['model']}`  -- {right['feature_count']} features"
        + (f"  ({right['tool_version']})" if right.get("tool_version") else ""),
        "",
        f"Interpretation: {report['interpretation']}",
        "",
        "## Summary",
        "",
        f"- Criterion match: `{summary['criterion_match']}`  |  Model match: `{summary['model_match']}`",
        f"- Shared features: `{summary['common_count']}`  |  added: `{summary['added_count']}`  |  dropped: `{summary['dropped_count']}`",
        f"- Rank stability (shared features at the same rank): `{summary['rank_stability']:.3f}`",
        f"- Importance drift: mean `|Δ|`=`{summary['mean_abs_importance_delta']:.3f}`, max `|Δ|`=`{summary['max_abs_importance_delta']:.3f}`",
        "",
    ]
    movers = report["changed_features"]
    if movers:
        lines.extend(
            [
                "## Biggest movers (shared features)",
                "",
                "| Feature | Label | Rank L→R | ΔRank | Importance L→R | ΔImportance |",
                "| --- | --- | --- | ---: | --- | ---: |",
            ]
        )
        for item in movers[:15]:
            arrow = _rank_arrow(item["rank_delta"])
            lines.append(
                f"| `{item['feature_id']}` | {item['label']} | {item['left_rank']}→{item['right_rank']} "
                f"| {arrow}{abs(item['rank_delta'])} | {item['importance_left']:.3f}→{item['importance_right']:.3f} "
                f"| {item['importance_delta']:+.3f} |"
            )
        lines.append("")
    if report["added_features"]:
        lines.extend(["## Added in the candidate run", ""])
        for item in report["added_features"][:15]:
            lines.append(f"- `{item['feature_id']}` (rank {item['rank']}, importance {item['importance']:.3f}) — {item['label']}")
        lines.append("")
    if report["dropped_features"]:
        lines.extend(["## Dropped from the baseline run", ""])
        for item in report["dropped_features"][:15]:
            lines.append(f"- `{item['feature_id']}` (rank {item['rank']}, importance {item['importance']:.3f}) — {item['label']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _rank_arrow(rank_delta: int) -> str:
    if rank_delta > 0:
        return "▲"  # moved up toward rank 1
    if rank_delta < 0:
        return "▼"
    return "="


def export_run_diff_report(
    *,
    left: str | Path,
    right: str | Path,
    out: str | Path | None = None,
    markdown_out: str | Path | None = None,
) -> dict[str, Any] | WrittenJsonMarkdown:
    left_report = load_inspection_report(left)
    right_report = load_inspection_report(right)
    report = build_run_diff_report(left_report, right_report)
    if out is None:
        return report
    json_path = Path(out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = Path(markdown_out) if markdown_out is not None else json_path.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_run_diff_markdown(report), encoding="utf-8")
    return WrittenJsonMarkdown(report=report, json_path=json_path, markdown_path=markdown_path)


def build_compare_runs_parser() -> argparse.ArgumentParser:
    # Default add_help=True so `compare-runs --help` works; the subparser that adopts
    # this as a parent passes add_help=False to avoid a duplicate -h (the validate-
    # matches pattern). add_help=False here would make --help an "unrecognized argument".
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, help="Baseline report.json.")
    parser.add_argument("--right", required=True, help="Candidate report.json to compare against the baseline.")
    parser.add_argument("--out", help="Output diff JSON path (also writes a sibling .md). Omit to print JSON.")
    parser.add_argument("--markdown-out", help="Optional explicit markdown path.")
    return parser


def run_compare_runs_from_args(args: argparse.Namespace) -> dict[str, Any] | WrittenJsonMarkdown:
    return export_run_diff_report(
        left=args.left,
        right=args.right,
        out=args.out,
        markdown_out=args.markdown_out,
    )
