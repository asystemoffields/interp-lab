from __future__ import annotations

import argparse
import html as html_lib
import json
import re
from pathlib import Path
from typing import Any

from oracle_sae.math_utils import cosine, pearson
from oracle_sae.reporting import load_inspection_report
from oracle_sae.schema import FeatureCard, InspectionReport

DEFAULT_COACTIVATION_THRESHOLD = 0.65
DEFAULT_STRONG_CAUSAL_THRESHOLD = 0.05
TOKEN_PATTERN = re.compile(r"token\[\d+\]=(['\"])(?P<token>.*?)(?<!\\)\1")
GENERIC_LABEL_PREFIXES = ("trained sae latent", "latent", "feature")


def export_attribution_graph(
    *,
    report_path: str | Path | list[str | Path],
    out_path: str | Path,
    markdown_out_path: str | Path | None = None,
    html_out_path: str | Path | None = None,
    include_similarity_edges: bool = False,
    similarity_threshold: float = 0.9,
    include_coactivation_edges: bool = True,
    coactivation_threshold: float = DEFAULT_COACTIVATION_THRESHOLD,
    include_supernodes: bool = True,
    strong_causal_threshold: float = DEFAULT_STRONG_CAUSAL_THRESHOLD,
    path_records_path: str | Path | list[str | Path] | None = None,
) -> Path:
    report = load_graph_report(report_path)
    path_records = load_path_patch_records(path_records_path) if path_records_path is not None else []
    graph = build_attribution_graph(
        report,
        include_similarity_edges=include_similarity_edges,
        similarity_threshold=similarity_threshold,
        include_coactivation_edges=include_coactivation_edges,
        coactivation_threshold=coactivation_threshold,
        include_supernodes=include_supernodes,
        strong_causal_threshold=strong_causal_threshold,
        path_records=path_records,
    )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out_path is not None:
        write_attribution_graph_markdown(graph, markdown_out_path)
    if html_out_path is not None:
        write_attribution_graph_html(graph, html_out_path)
    return path


def write_attribution_graph_markdown(graph: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_attribution_graph_markdown(graph), encoding="utf-8")
    return path


def write_attribution_graph_html(graph: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_attribution_graph_html(graph), encoding="utf-8")
    return path


def render_attribution_graph_html(graph: dict[str, Any]) -> str:
    summary = summarize_attribution_graph(graph)
    graph_payload = _json_script_payload(graph)
    summary_payload = _json_script_payload(summary)
    title = _html_text(_raw_criterion_text(graph) or "Attribution graph")
    model = _html_text(graph.get("model", ""))
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>interp-lab attribution graph - {title}</title>",
            "<style>",
            _GRAPH_HTML_CSS,
            "</style>",
            "</head>",
            "<body>",
            "<header>",
            '<div class="kicker">interp-lab attribution graph</div>',
            f"<h1>{title}</h1>",
            f'<p class="lede">Model: <code>{model}</code></p>',
            "</header>",
            "<main>",
            '<section class="toolbar" aria-label="Graph controls">',
            '<label for="feature-search">Filter features</label>',
            '<input id="feature-search" type="search" placeholder="Search labels, IDs, roles, tokens">',
            '<label for="role-filter">Role</label>',
            '<select id="role-filter"><option value="">All roles</option></select>',
            '<label for="status-filter">Path status</label>',
            '<select id="status-filter"><option value="">All path statuses</option></select>',
            '<span id="filter-count" class="muted"></span>',
            "</section>",
            '<section id="metrics" class="metrics" aria-label="Summary metrics"></section>',
            '<section id="graph-brief" class="panel" aria-label="Evidence summary"></section>',
            '<section class="panel graph-panel">',
            "<h2>Graph</h2>",
            '<svg id="graph-svg" role="img" aria-label="Attribution graph visualization"></svg>',
            "</section>",
            '<section class="grid">',
            '<article class="panel">',
            "<h2>Candidate Paths</h2>",
            '<div id="candidate-paths"></div>',
            "</article>",
            '<article class="panel">',
            "<h2>Strong Features</h2>",
            '<div id="strong-features"></div>',
            "</article>",
            "</section>",
            '<section class="grid">',
            '<article class="panel">',
            "<h2>Feature Cards</h2>",
            '<div id="feature-cards"></div>',
            "</article>",
            '<article class="panel">',
            "<h2>Agent Next Actions</h2>",
            '<div id="agent-actions"></div>',
            "</article>",
            "</section>",
            "</main>",
            f'<script id="graph-data" type="application/json">{graph_payload}</script>',
            f'<script id="summary-data" type="application/json">{summary_payload}</script>',
            "<script>",
            _GRAPH_HTML_JS,
            "</script>",
            "</body>",
            "</html>",
        ]
    )


def export_attribution_graph_summary(*, graph_path: str | Path, out_path: str | Path) -> Path:
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summarize_attribution_graph(graph), indent=2, sort_keys=True), encoding="utf-8")
    return path


def summarize_attribution_graph(graph: dict[str, Any]) -> dict[str, Any]:
    summary = graph.get("mechanism_summary", {}) if isinstance(graph.get("mechanism_summary"), dict) else {}
    validation = _graph_validation_metadata(graph)
    run_assessment = validation.get("run_assessment", {}) if isinstance(validation.get("run_assessment"), dict) else {}
    validation_summary = validation.get("summary", {}) if isinstance(validation.get("summary"), dict) else {}
    candidate_paths = summary.get("candidate_paths", []) if isinstance(summary.get("candidate_paths"), list) else []
    nodes = graph.get("nodes", []) if isinstance(graph.get("nodes"), list) else []
    edges = graph.get("edges", []) if isinstance(graph.get("edges"), list) else []
    return {
        "schema_version": "interp-lab.attribution_graph_summary.v1",
        "model": graph.get("model"),
        "criterion": _raw_criterion_text(graph),
        "counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "features": sum(1 for node in nodes if isinstance(node, dict) and node.get("type") == "feature"),
            "path_patch_edges": sum(
                1 for edge in edges if isinstance(edge, dict) and edge.get("type") == "path_patch"
            ),
            "candidate_paths": len(candidate_paths),
        },
        "validation": {
            "status_counts": summary.get("path_validation_status_counts", {}),
            "claim_grade_counts": validation_summary.get("claim_grade_counts", {}),
            "overall_claim_grade": run_assessment.get("overall_claim_grade"),
            "recommended_next_action": run_assessment.get("recommended_next_action"),
        },
        "strongest_features": _summary_strong_features(summary.get("strong_causal_features", [])),
        "candidate_paths": _summary_candidate_paths(candidate_paths),
        "agent_next_actions": validation.get("agent_next_actions", []) or _graph_summary_agent_next_actions(candidate_paths),
    }


def render_attribution_graph_markdown(graph: dict[str, Any]) -> str:
    summary = graph.get("mechanism_summary", {})
    lines = [
        "# Attribution Graph",
        "",
        f"Model: `{graph.get('model', '')}`",
        f"Criterion: {_criterion_text(graph)}",
        f"Nodes: `{len(graph.get('nodes', []))}`",
        f"Edges: `{len(graph.get('edges', []))}`",
        "",
    ]
    status_counts = summary.get("path_validation_status_counts")
    if isinstance(status_counts, dict) and status_counts:
        rendered_counts = ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
        lines.extend([f"Path validation: `{rendered_counts}`", ""])
    run_assessment = _graph_validation_run_assessment(graph)
    if run_assessment:
        lines.extend(
            [
                f"Overall validation: `{_cell(run_assessment.get('overall_claim_grade'))}`",
                f"Recommended next action: {_cell(run_assessment.get('recommended_next_action'))}",
                "",
            ]
        )
    lines.extend(_strong_feature_markdown(summary.get("strong_causal_features", [])))
    lines.extend(_candidate_path_markdown(summary.get("candidate_paths", [])))
    lines.extend(_feature_group_markdown(summary.get("candidate_feature_groups", [])))
    lines.extend(_validation_plan_markdown(summary.get("validation_plan", [])))
    return "\n".join(lines).rstrip() + "\n"


def build_attribution_graph(
    report: InspectionReport,
    *,
    include_similarity_edges: bool = False,
    similarity_threshold: float = 0.9,
    include_coactivation_edges: bool = True,
    coactivation_threshold: float = DEFAULT_COACTIVATION_THRESHOLD,
    include_supernodes: bool = True,
    strong_causal_threshold: float = DEFAULT_STRONG_CAUSAL_THRESHOLD,
    path_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    criterion_id = "criterion"
    nodes = [
        {
            "id": criterion_id,
            "type": "criterion",
            "label": report.criterion.text,
            "model": report.model,
        }
    ]
    edges = []
    for card in report.cards:
        nodes.append(_feature_node(card, strong_causal_threshold=strong_causal_threshold))
        edge = _criterion_edge(card, criterion_id=criterion_id)
        if edge is not None:
            edges.append(edge)
    supernodes: list[dict[str, Any]] = []
    if include_supernodes:
        supernodes, supernode_edges = _supernodes(
            report.cards,
            criterion_id=criterion_id,
            strong_causal_threshold=strong_causal_threshold,
        )
        nodes.extend(supernodes)
        edges.extend(supernode_edges)
    coactivation_edges: list[dict[str, Any]] = []
    if include_coactivation_edges:
        coactivation_edges = _coactivation_edges(
            report.cards,
            threshold=coactivation_threshold,
            strong_causal_threshold=strong_causal_threshold,
        )
        edges.extend(coactivation_edges)
    path_patch_edges = _path_patch_edges(path_records or [], report.cards)
    edges.extend(path_patch_edges)
    if include_similarity_edges:
        edges.extend(_similarity_edges(report.cards, threshold=similarity_threshold))
    return {
        "schema_version": "interp-lab.attribution_graph.v1",
        "model": report.model,
        "criterion": report.criterion.to_dict(),
        "nodes": nodes,
        "edges": edges,
        "mechanism_summary": _mechanism_summary(
            report.cards,
            supernodes=supernodes,
            coactivation_edges=coactivation_edges,
            path_patch_edges=path_patch_edges,
            strong_causal_threshold=strong_causal_threshold,
        ),
        "metadata": {
            "source_report_created_at": report.created_at,
            "feature_count": len(report.cards),
            "coactivation_threshold": coactivation_threshold,
            "strong_causal_threshold": strong_causal_threshold,
            "path_patch_record_count": len(path_records or []),
        },
    }


def build_graph_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an interp-lab report as an attribution graph JSON.")
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Inspection report JSON. Repeat to fuse multiple layer reports into one graph.",
    )
    parser.add_argument("--out", required=True, help="Output graph JSON path.")
    parser.add_argument("--markdown-out", help="Optional output graph Markdown summary path.")
    parser.add_argument("--html-out", help="Optional output self-contained HTML graph viewer path.")
    parser.add_argument(
        "--include-similarity-edges",
        action="store_true",
        help="Add feature-to-feature edges from fingerprint similarity.",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    parser.add_argument(
        "--no-coactivation-edges",
        dest="include_coactivation_edges",
        action="store_false",
        help="Skip feature-to-feature edges from activation-signature correlation.",
    )
    parser.add_argument("--coactivation-threshold", type=float, default=DEFAULT_COACTIVATION_THRESHOLD)
    parser.add_argument(
        "--path-records",
        action="append",
        help="Path-patching JSONL from export-hf-sae-paths. Repeatable.",
    )
    parser.add_argument(
        "--no-supernodes",
        dest="include_supernodes",
        action="store_false",
        help="Skip candidate feature-group supernodes.",
    )
    parser.add_argument("--strong-causal-threshold", type=float, default=DEFAULT_STRONG_CAUSAL_THRESHOLD)
    parser.set_defaults(include_coactivation_edges=True, include_supernodes=True)
    return parser


def build_graph_summary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a compact attribution graph summary JSON for agents.")
    parser.add_argument("--graph", required=True, help="Attribution graph JSON.")
    parser.add_argument("--out", required=True, help="Output graph summary JSON path.")
    return parser


def run_graph_export_from_args(args: argparse.Namespace) -> Path:
    return export_attribution_graph(
        report_path=args.report,
        out_path=args.out,
        markdown_out_path=args.markdown_out,
        html_out_path=args.html_out,
        include_similarity_edges=args.include_similarity_edges,
        similarity_threshold=args.similarity_threshold,
        include_coactivation_edges=args.include_coactivation_edges,
        coactivation_threshold=args.coactivation_threshold,
        include_supernodes=args.include_supernodes,
        strong_causal_threshold=args.strong_causal_threshold,
        path_records_path=args.path_records,
    )


def run_graph_summary_from_args(args: argparse.Namespace) -> Path:
    return export_attribution_graph_summary(graph_path=args.graph, out_path=args.out)


def _feature_node(card: FeatureCard, *, strong_causal_threshold: float) -> dict[str, Any]:
    return {
        "id": _feature_node_id(card),
        "type": "feature",
        "feature_id": card.feature_id,
        "model": card.model,
        "layer": card.layer,
        "label": card.label,
        "role": _feature_role(card, strong_causal_threshold=strong_causal_threshold),
        "top_tokens": _top_tokens(card),
        "source": card.source,
        "importance": card.importance,
        "association": card.association,
        "causal_effect": card.causal_effect,
        "specificity": card.specificity,
        "stability": card.stability,
    }


def _feature_node_id(card: FeatureCard) -> str:
    return f"feature:{card.model}:{card.feature_id}"


def _criterion_edge(card: FeatureCard, *, criterion_id: str) -> dict[str, Any] | None:
    measured = _has_measured_causal_evidence(card)
    if measured:
        signed = card.causal_effects.get(
            "signed_causal_effect",
            card.causal_effects.get("signed_association"),
        )
        effect = card.causal_effects.get("criterion", card.causal_effect)
        edge_type = "causal_effect"
        evidence = "measured_intervention"
    else:
        signed = card.causal_effects.get("signed_association", card.metadata.get("signed_association"))
        effect = card.causal_effects.get("criterion", abs(float(signed)) if signed is not None else card.association)
        edge_type = "criterion_association"
        evidence = "activation_criterion_association"
    return {
        "source": _feature_node_id(card),
        "target": criterion_id,
        "type": edge_type,
        "evidence": evidence,
        "source_feature_id": card.feature_id,
        "effect": float(effect),
        "signed_effect": float(signed) if signed is not None else None,
        "specificity": card.causal_effects.get("specificity", card.specificity),
        "side_effect": card.causal_effects.get("side_effect"),
        "strong_causal_score": card.causal_effects.get("strong_causal_score"),
        "confidence_interval": _confidence_interval(card),
        "record_count": card.causal_effects.get("intervention_record_count"),
    }


def _confidence_interval(card: FeatureCard) -> dict[str, float] | None:
    low = card.causal_effects.get("criterion_ci_low")
    high = card.causal_effects.get("criterion_ci_high")
    if low is None or high is None:
        return None
    return {"low": low, "high": high}


def _has_measured_causal_evidence(card: FeatureCard) -> bool:
    if "signed_causal_effect" in card.causal_effects:
        return True
    if float(card.causal_effects.get("intervention_record_count", 0.0) or 0.0) > 0.0:
        return True
    interventions = card.metadata.get("interventions")
    if isinstance(interventions, dict):
        try:
            return float(interventions.get("count", 0.0) or 0.0) > 0.0
        except (TypeError, ValueError):
            return False
    return False


def _feature_node_lookup(cards: list[FeatureCard]) -> dict[tuple[str | None, str], str]:
    lookup: dict[tuple[str | None, str], str] = {}
    counts: dict[str, int] = {}
    for card in cards:
        counts[card.feature_id] = counts.get(card.feature_id, 0) + 1
    for card in cards:
        lookup[(card.model, card.feature_id)] = _feature_node_id(card)
        if counts[card.feature_id] == 1:
            lookup[(None, card.feature_id)] = _feature_node_id(card)
    return lookup


def _lookup_feature_node_id(
    lookup: dict[tuple[str | None, str], str],
    feature_id: str,
    model: str | None,
) -> str:
    return lookup.get((model, feature_id)) or lookup.get((None, feature_id)) or feature_id


def _path_edge_model(rows: list[dict[str, Any]]) -> str | None:
    models = {str(row.get("model", "")) for row in rows if row.get("model")}
    if len(models) == 1:
        return next(iter(models))
    return None


def _similarity_edges(cards: list[FeatureCard], *, threshold: float) -> list[dict[str, Any]]:
    edges = []
    for left_index, left in enumerate(cards):
        for right in cards[left_index + 1 :]:
            score = max(
                cosine(left.fingerprint.activation_signature, right.fingerprint.activation_signature),
                cosine(left.fingerprint.decoder_signature, right.fingerprint.decoder_signature),
            )
            if score >= threshold:
                edges.append(
                    {
                        "source": _feature_node_id(left),
                        "target": _feature_node_id(right),
                        "type": "fingerprint_similarity",
                        "source_feature_id": left.feature_id,
                        "target_feature_id": right.feature_id,
                        "score": round(score, 6),
                    }
                )
    return edges


def load_path_patch_records(path_value: str | Path | list[str | Path]) -> list[dict[str, Any]]:
    paths = path_value if isinstance(path_value, list) else [path_value]
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                records.append(record)
    return records


def load_graph_report(report_path: str | Path | list[str | Path]) -> InspectionReport:
    if isinstance(report_path, list):
        reports = [load_inspection_report(path) for path in report_path]
    else:
        reports = [load_inspection_report(report_path)]
    if not reports:
        raise ValueError("at least one report is required")
    if len(reports) == 1:
        return reports[0]
    first = reports[0]
    cards = []
    for report in reports:
        cards.extend(report.cards)
    model_names = sorted({report.model for report in reports})
    criterion_texts = sorted({report.criterion.text for report in reports})
    if len(criterion_texts) != 1:
        raise ValueError(
            "cannot fuse reports with different criteria: "
            + ", ".join(repr(text) for text in criterion_texts)
        )
    metadata = dict(first.metadata)
    metadata.update(
        {
            "source_report_count": len(reports),
            "source_report_models": model_names,
            "source_report_criteria": criterion_texts,
        }
    )
    return InspectionReport(
        model=model_names[0] if len(model_names) == 1 else " + ".join(model_names),
        criterion=first.criterion,
        cards=cards,
        created_at=first.created_at,
        metadata=metadata,
    )


def _path_patch_edges(records: list[dict[str, Any]], cards: list[FeatureCard]) -> list[dict[str, Any]]:
    node_lookup = _feature_node_lookup(cards)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        source = str(record.get("source_feature_id", ""))
        target = str(record.get("target_feature_id", ""))
        if source and target:
            grouped.setdefault((source, target), []).append(record)
    edges = []
    for (source, target), rows in sorted(grouped.items()):
        effect_rows = [row for row in rows if not _is_path_control(row)]
        control_rows = [row for row in rows if _is_path_control(row)]
        deltas = [_float(row.get("target_activation_delta")) for row in effect_rows]
        score_deltas = [
            _float(row.get("score_delta"))
            for row in effect_rows
            if row.get("score_delta") is not None
        ]
        control_deltas = [_float(row.get("target_activation_delta")) for row in control_rows]
        control_score_deltas = [
            _float(row.get("score_delta"))
            for row in control_rows
            if row.get("score_delta") is not None
        ]
        mean_abs_delta = _mean([abs(value) for value in deltas])
        control_mean_abs_delta = _mean([abs(value) for value in control_deltas])
        by_strength = _path_strength_summary(effect_rows, control_rows)
        best_strength = max(
            by_strength,
            key=lambda item: abs(float(item["mean_target_activation_delta"])),
            default=None,
        )
        prompts = {str(row.get("prompt_id", "")) for row in effect_rows}
        model = _path_edge_model(rows)
        edges.append(
            {
                "source": _lookup_feature_node_id(node_lookup, source, model),
                "target": _lookup_feature_node_id(node_lookup, target, model),
                "type": "path_patch",
                "source_feature_id": source,
                "target_feature_id": target,
                "evidence": "source_sae_latent_steering",
                "mean_target_activation_delta": round(_mean(deltas), 6),
                "mean_abs_target_activation_delta": round(mean_abs_delta, 6),
                "mean_score_delta": round(_mean(score_deltas), 6) if score_deltas else None,
                "control_mean_abs_target_activation_delta": round(control_mean_abs_delta, 6)
                if control_rows
                else None,
                "control_mean_abs_score_delta": round(_mean([abs(value) for value in control_score_deltas]), 6)
                if control_score_deltas
                else None,
                "path_specificity_score": round(max(0.0, mean_abs_delta - control_mean_abs_delta), 6)
                if control_rows
                else None,
                "best_strength": best_strength,
                "by_strength": by_strength,
                "record_count": len(effect_rows),
                "control_record_count": len(control_rows),
                "prompt_count": len(prompts),
                "strengths": [item["strength"] for item in by_strength],
            }
        )
    edges.sort(
        key=lambda edge: (
            edge["path_specificity_score"]
            if edge.get("path_specificity_score") is not None
            else edge["mean_abs_target_activation_delta"]
        ),
        reverse=True,
    )
    return edges


def _coactivation_edges(
    cards: list[FeatureCard],
    *,
    threshold: float,
    strong_causal_threshold: float,
) -> list[dict[str, Any]]:
    edges = []
    for left_index, left in enumerate(cards):
        for right in cards[left_index + 1 :]:
            score = pearson(left.fingerprint.activation_signature, right.fingerprint.activation_signature)
            if abs(score) < threshold:
                continue
            source, target = _ordered_pair(left, right)
            edge = {
                "source": _feature_node_id(source),
                "target": _feature_node_id(target),
                "type": "coactivation",
                "source_feature_id": source.feature_id,
                "target_feature_id": target.feature_id,
                "correlation": round(score, 6),
                "abs_correlation": round(abs(score), 6),
                "evidence": "activation_signature_correlation",
                "candidate_path": _is_candidate_path(source, target, strong_causal_threshold=strong_causal_threshold),
            }
            if source.layer is not None and target.layer is not None and source.layer != target.layer:
                edge["direction_hint"] = "earlier_to_later_layer"
            edges.append(edge)
    edges.sort(key=lambda edge: edge["abs_correlation"], reverse=True)
    return edges[:128]


def _ordered_pair(left: FeatureCard, right: FeatureCard) -> tuple[FeatureCard, FeatureCard]:
    if left.layer is None or right.layer is None:
        return (left, right) if left.feature_id <= right.feature_id else (right, left)
    if left.layer == right.layer:
        return (left, right) if left.feature_id <= right.feature_id else (right, left)
    return (left, right) if left.layer < right.layer else (right, left)


def _is_candidate_path(source: FeatureCard, target: FeatureCard, *, strong_causal_threshold: float) -> bool:
    if target.layer is not None and source.layer is not None and target.layer < source.layer:
        return False
    return _strong_causal_score(target) >= strong_causal_threshold


def _supernodes(
    cards: list[FeatureCard],
    *,
    criterion_id: str,
    strong_causal_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[FeatureCard]] = {}
    for card in cards:
        grouped.setdefault(_role_supernode_id(card, strong_causal_threshold=strong_causal_threshold), []).append(card)
    for theme, members in _theme_groups(cards).items():
        grouped[f"theme:{theme}"] = members

    nodes = []
    edges = []
    for group_id, members in sorted(grouped.items()):
        if not members:
            continue
        node_id = f"supernode:{group_id}"
        signed = _mean([_signed_effect(member) for member in members])
        strong = _mean([_strong_causal_score(member) for member in members])
        nodes.append(
            {
                "id": node_id,
                "type": "supernode",
                "label": _supernode_label(group_id, members),
                "member_count": len(members),
                "role": _feature_role(
                    max(members, key=_strong_causal_score),
                    strong_causal_threshold=strong_causal_threshold,
                ),
                "mean_signed_effect": round(signed, 6),
                "mean_strong_causal_score": round(strong, 6),
                "members": [_feature_node_id(member) for member in members],
                "member_feature_ids": [member.feature_id for member in members],
            }
        )
        for member in members:
            edges.append(
                {
                    "source": _feature_node_id(member),
                    "target": node_id,
                    "type": "member_of",
                    "evidence": "automatic_role_or_theme_grouping",
                    "source_feature_id": member.feature_id,
                }
            )
        edges.append(
            {
                "source": node_id,
                "target": criterion_id,
                "type": "aggregate_causal_effect"
                if any(_has_measured_causal_evidence(member) for member in members)
                else "aggregate_criterion_association",
                "signed_effect": round(signed, 6),
                "strong_causal_score": round(strong, 6),
                "member_count": len(members),
            }
        )
    return nodes, edges


def _role_supernode_id(card: FeatureCard, *, strong_causal_threshold: float) -> str:
    layer = "unknown-layer" if card.layer is None else f"layer-{card.layer}"
    return f"{layer}:{_feature_role(card, strong_causal_threshold=strong_causal_threshold)}"


def _feature_role(card: FeatureCard, *, strong_causal_threshold: float = DEFAULT_STRONG_CAUSAL_THRESHOLD) -> str:
    strong = _strong_causal_score(card)
    signed = _signed_effect(card)
    if strong >= strong_causal_threshold and signed > 0:
        return "criterion_promoter"
    if strong >= strong_causal_threshold and signed < 0:
        return "criterion_suppressor"
    if abs(float(card.association)) >= 0.2:
        return "associated_detector"
    return "observed_feature"


def _theme_groups(cards: list[FeatureCard]) -> dict[str, list[FeatureCard]]:
    by_theme: dict[str, list[FeatureCard]] = {}
    for card in cards:
        for token in _top_tokens(card)[:2]:
            key = _theme_key(token)
            if key:
                by_theme.setdefault(key, []).append(card)
    return {theme: members for theme, members in by_theme.items() if len(members) >= 2}


def _top_tokens(card: FeatureCard) -> list[str]:
    tokens = []
    for example in card.examples[:3]:
        match = TOKEN_PATTERN.search(str(example))
        if not match:
            continue
        token = match.group("token").replace("\\n", "\\n").strip()
        if token and len(token) <= 32 and token not in tokens:
            tokens.append(token)
    return tokens


def _theme_key(token: str) -> str:
    clean = str(token).strip().lower()
    if len(clean) < 2:
        return ""
    if clean in {",", ".", ":", ";", "and", "the", "a", "an", "of", "to", "in"}:
        return ""
    return re.sub(r"[^a-z0-9_+-]+", "-", clean).strip("-")


def _supernode_label(group_id: str, members: list[FeatureCard]) -> str:
    if group_id.startswith("theme:"):
        theme = group_id.split(":", 1)[1].replace("-", " ")
        return f"theme: {theme}"
    layer_part, role = group_id.split(":", 1)
    layer_label = layer_part.replace("-", " ")
    role_label = role.replace("_", " ")
    labels = [_display_label(member) for member in members]
    example = f": {labels[0]}" if labels else ""
    return f"{layer_label} {role_label}{example}"


def _display_label(card: FeatureCard) -> str:
    label = str(card.label).strip()
    if label and not label.lower().startswith(GENERIC_LABEL_PREFIXES):
        return label
    tokens = _top_tokens(card)
    if tokens:
        return ", ".join(_display_token(token) for token in tokens[:3])
    return card.feature_id


def _mechanism_summary(
    cards: list[FeatureCard],
    *,
    supernodes: list[dict[str, Any]],
    coactivation_edges: list[dict[str, Any]],
    path_patch_edges: list[dict[str, Any]],
    strong_causal_threshold: float,
) -> dict[str, Any]:
    strong_cards = [
        card
        for card in sorted(cards, key=_strong_causal_score, reverse=True)
        if _strong_causal_score(card) >= strong_causal_threshold
    ]
    candidate_paths = _candidate_path_summaries(cards, path_patch_edges, coactivation_edges, limit=8)
    return {
        "strong_causal_features": [
            {
                "feature_id": card.feature_id,
                "label": _display_label(card),
                "layer": card.layer,
                "signed_effect": round(_signed_effect(card), 6),
                "strong_causal_score": round(_strong_causal_score(card), 6),
                "role": _feature_role(card, strong_causal_threshold=strong_causal_threshold),
            }
            for card in strong_cards[:8]
        ],
        "candidate_feature_groups": [
            {
                "id": node["id"],
                "label": node["label"],
                "member_count": node["member_count"],
                "mean_strong_causal_score": node["mean_strong_causal_score"],
            }
            for node in sorted(
                supernodes,
                key=lambda node: (float(node.get("mean_strong_causal_score", 0.0)), int(node.get("member_count", 0))),
                reverse=True,
            )[:8]
        ],
        "candidate_paths": candidate_paths,
        "validation_plan": _validation_plan(strong_cards, candidate_paths),
    }


def _candidate_path_summaries(
    cards: list[FeatureCard],
    path_patch_edges: list[dict[str, Any]],
    coactivation_edges: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    by_id = {card.feature_id: card for card in cards}
    paths = []
    for edge in path_patch_edges:
        source_feature_id = str(edge.get("source_feature_id", edge["source"]))
        target_feature_id = str(edge.get("target_feature_id", edge["target"]))
        source = by_id.get(source_feature_id)
        target = by_id.get(target_feature_id)
        paths.append(
            {
                "source_feature_id": source_feature_id,
                "source_label": _display_label(source) if source is not None else str(edge["source"]),
                "source_layer": source.layer if source is not None else None,
                "target_feature_id": target_feature_id,
                "target_label": _display_label(target) if target is not None else str(edge["target"]),
                "target_layer": target.layer if target is not None else None,
                "evidence": "path_patch",
                "mean_target_activation_delta": edge["mean_target_activation_delta"],
                "mean_abs_target_activation_delta": edge["mean_abs_target_activation_delta"],
                "mean_score_delta": edge.get("mean_score_delta"),
                "control_mean_abs_target_activation_delta": edge.get("control_mean_abs_target_activation_delta"),
                "path_specificity_score": edge.get("path_specificity_score"),
                "best_strength": edge.get("best_strength"),
                "record_count": edge["record_count"],
                "control_record_count": edge.get("control_record_count", 0),
            }
        )
    for edge in coactivation_edges:
        if not edge.get("candidate_path"):
            continue
        source_feature_id = str(edge.get("source_feature_id", edge["source"]))
        target_feature_id = str(edge.get("target_feature_id", edge["target"]))
        source = by_id.get(source_feature_id)
        target = by_id.get(target_feature_id)
        if source is None or target is None:
            continue
        paths.append(
            {
                "source_feature_id": source.feature_id,
                "source_label": _display_label(source),
                "source_layer": source.layer,
                "target_feature_id": target.feature_id,
                "target_label": _display_label(target),
                "target_layer": target.layer,
                "evidence": "coactivation",
                "correlation": edge["correlation"],
                "target_signed_effect": round(_signed_effect(target), 6),
                "target_strong_causal_score": round(_strong_causal_score(target), 6),
            }
        )
    return paths[:limit]


def _validation_plan(strong_cards: list[FeatureCard], candidate_paths: list[dict[str, Any]]) -> list[str]:
    plan = []
    for path in candidate_paths[:3]:
        if path.get("evidence") == "path_patch":
            plan.append(
                f"Replicate the measured path {path['source_feature_id']} -> {path['target_feature_id']} "
                "on held-out prompts and compare target-latent delta against random-source controls."
            )
        else:
            direction = "promotes" if float(path["target_signed_effect"]) >= 0 else "suppresses"
            plan.append(
                "Inhibit "
                f"{path['source_feature_id']} and measure {path['target_feature_id']} plus the criterion score; "
                f"the candidate path predicts a change in a feature that {direction} the criterion."
            )
    for card in strong_cards[:3]:
        direction = "promotes" if _signed_effect(card) >= 0 else "suppresses"
        plan.append(
            f"Sweep steering strengths for {card.feature_id}; it currently {direction} the criterion "
            f"with strong causal score {_strong_causal_score(card):.3f}."
        )
    return plan[:6]


def _strong_feature_markdown(features: list[dict[str, Any]]) -> list[str]:
    if not features:
        return ["## Strong Causal Features", "", "No strong causal features met the current threshold.", ""]
    lines = [
        "## Strong Causal Features",
        "",
        "| Feature | Label | Layer | Role | Signed | Strong |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for feature in features[:8]:
        lines.append(
            "| "
            f"`{_cell(feature.get('feature_id'))}` | "
            f"{_cell(feature.get('label'))} | "
            f"{_cell(feature.get('layer'))} | "
            f"{_cell(feature.get('role'))} | "
            f"{_number(feature.get('signed_effect'))} | "
            f"{_number(feature.get('strong_causal_score'))} |"
        )
    lines.append("")
    return lines


def _candidate_path_markdown(paths: list[dict[str, Any]]) -> list[str]:
    if not paths:
        return ["## Candidate Paths", "", "No candidate paths are currently present.", ""]
    lines = [
        "## Candidate Paths",
        "",
        "| Status | Claim | Path | Evidence | Effect | Control | Specificity | Records |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for path in paths[:12]:
        validation = path.get("validation") if isinstance(path.get("validation"), dict) else {}
        status = validation.get("status", "")
        claim_grade = validation.get("claim_grade", "")
        effect = path.get("mean_abs_target_activation_delta", validation.get("mean_abs_target_activation_delta"))
        control = path.get(
            "control_mean_abs_target_activation_delta",
            validation.get("control_mean_abs_target_activation_delta"),
        )
        specificity = path.get("path_specificity_score", validation.get("path_specificity_score"))
        records = path.get("record_count", validation.get("record_count", ""))
        lines.append(
            "| "
            f"{_cell(status)} | "
            f"{_cell(claim_grade)} | "
            f"`{_cell(path.get('source_feature_id'))} -> {_cell(path.get('target_feature_id'))}` | "
            f"{_cell(path.get('evidence'))} | "
            f"{_number(effect)} | "
            f"{_number(control)} | "
            f"{_number(specificity)} | "
            f"{_cell(records)} |"
        )
    lines.append("")
    notes = _candidate_path_note_markdown(paths[:12])
    if notes:
        lines.extend(notes)
    return lines


def _candidate_path_note_markdown(paths: list[dict[str, Any]]) -> list[str]:
    notes = []
    for path in paths:
        validation = path.get("validation") if isinstance(path.get("validation"), dict) else {}
        interpretation = validation.get("interpretation")
        if not interpretation:
            continue
        source = _cell(path.get("source_feature_id"))
        target = _cell(path.get("target_feature_id"))
        status = _cell(validation.get("status"))
        reason_codes = validation.get("reason_codes")
        next_action = validation.get("next_action")
        reason_text = ""
        if isinstance(reason_codes, list) and reason_codes:
            reason_text = f" Reasons: `{', '.join(_cell(reason) for reason in reason_codes)}`."
        next_text = f" Next: {_cell(next_action)}" if next_action else ""
        notes.append(f"- `{source} -> {target}` {status}: {_cell(interpretation)}{next_text}{reason_text}")
    if not notes:
        return []
    return ["### Path Notes", "", *notes, ""]


def _feature_group_markdown(groups: list[dict[str, Any]]) -> list[str]:
    if not groups:
        return []
    lines = [
        "## Candidate Feature Groups",
        "",
        "| Group | Label | Members | Mean Strong |",
        "| --- | --- | ---: | ---: |",
    ]
    for group in groups[:8]:
        lines.append(
            "| "
            f"`{_cell(group.get('id'))}` | "
            f"{_cell(group.get('label'))} | "
            f"{_cell(group.get('member_count'))} | "
            f"{_number(group.get('mean_strong_causal_score'))} |"
        )
    lines.append("")
    return lines


def _validation_plan_markdown(plan: list[str]) -> list[str]:
    if not plan:
        return []
    lines = ["## Validation Plan", ""]
    for item in plan[:8]:
        lines.append(f"- {_cell(item)}")
    lines.append("")
    return lines


def _criterion_text(graph: dict[str, Any]) -> str:
    criterion = graph.get("criterion")
    if isinstance(criterion, dict):
        return _cell(criterion.get("text", ""))
    return _cell(criterion)


def _raw_criterion_text(graph: dict[str, Any]) -> str:
    criterion = graph.get("criterion")
    if isinstance(criterion, dict):
        return str(criterion.get("text", ""))
    if criterion is None:
        return ""
    return str(criterion)


def _graph_validation_run_assessment(graph: dict[str, Any]) -> dict[str, Any]:
    graph_validation = _graph_validation_metadata(graph)
    run_assessment = graph_validation.get("run_assessment")
    return run_assessment if isinstance(run_assessment, dict) else {}


def _graph_validation_metadata(graph: dict[str, Any]) -> dict[str, Any]:
    metadata = graph.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    graph_validation = metadata.get("graph_validation")
    return graph_validation if isinstance(graph_validation, dict) else {}


def _summary_strong_features(features: Any) -> list[dict[str, Any]]:
    if not isinstance(features, list):
        return []
    fields = ["feature_id", "label", "layer", "role", "signed_effect", "strong_causal_score"]
    return [
        {field: feature.get(field) for field in fields if field in feature}
        for feature in features[:8]
        if isinstance(feature, dict)
    ]


def _summary_candidate_paths(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for path in paths[:12]:
        if not isinstance(path, dict):
            continue
        validation = path.get("validation") if isinstance(path.get("validation"), dict) else {}
        rows.append(
            {
                "source_feature_id": path.get("source_feature_id"),
                "target_feature_id": path.get("target_feature_id"),
                "evidence": path.get("evidence"),
                "status": validation.get("status"),
                "claim_grade": validation.get("claim_grade"),
                "mean_abs_target_activation_delta": path.get(
                    "mean_abs_target_activation_delta",
                    validation.get("mean_abs_target_activation_delta"),
                ),
                "control_mean_abs_target_activation_delta": path.get(
                    "control_mean_abs_target_activation_delta",
                    validation.get("control_mean_abs_target_activation_delta"),
                ),
                "path_specificity_score": path.get("path_specificity_score", validation.get("path_specificity_score")),
                "record_count": path.get("record_count", validation.get("record_count")),
                "reason_codes": validation.get("reason_codes", []),
                "next_action": validation.get("next_action"),
            }
        )
    return rows


def _graph_summary_agent_next_actions(candidate_paths: list[dict[str, Any]]) -> list[dict[str, str]]:
    if candidate_paths:
        return [
            {
                "id": "validate_candidate_paths",
                "title": "Validate candidate paths with held-out path-patching records",
                "command": "interp-lab validate-attribution-graph --graph <graph.json> --path-records <paths.jsonl> --out <validation.json> --graph-out <validated-graph.json>",
            }
        ]
    return [
        {
            "id": "measure_path_records",
            "title": "Measure path-patching records before validating this graph",
            "command": "interp-lab export-hf-sae-paths --model <model> --dataset <heldout.jsonl> --source-sae <source-sae.json> --target-sae <target-sae.json> --out <paths.jsonl>",
        }
    ]


def _json_script_payload(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True).replace("</", "<\\/")


def _html_text(value: Any) -> str:
    if value is None:
        return ""
    return html_lib.escape(str(value), quote=True)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _number(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return _cell(value)


def _signed_effect(card: FeatureCard) -> float:
    value = card.causal_effects.get("signed_causal_effect")
    if value is None:
        value = card.causal_effects.get("signed_association", card.metadata.get("signed_association", 0.0))
    return float(value)


def _strong_causal_score(card: FeatureCard) -> float:
    return float(card.causal_effects.get("strong_causal_score", 0.0))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _path_strength_summary(
    rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_float(row.get("strength")), []).append(row)
    control_grouped: dict[float, list[dict[str, Any]]] = {}
    for row in control_rows or []:
        control_grouped.setdefault(_float(row.get("strength")), []).append(row)
    summary = []
    for strength in sorted(set(grouped) | set(control_grouped)):
        strength_rows = grouped.get(strength, [])
        strength_control_rows = control_grouped.get(strength, [])
        deltas = [_float(row.get("target_activation_delta")) for row in strength_rows]
        score_deltas = [
            _float(row.get("score_delta"))
            for row in strength_rows
            if row.get("score_delta") is not None
        ]
        control_deltas = [_float(row.get("target_activation_delta")) for row in strength_control_rows]
        control_score_deltas = [
            _float(row.get("score_delta"))
            for row in strength_control_rows
            if row.get("score_delta") is not None
        ]
        mean_abs_delta = _mean([abs(value) for value in deltas])
        control_mean_abs_delta = _mean([abs(value) for value in control_deltas])
        summary.append(
            {
                "strength": round(strength, 6),
                "record_count": len(strength_rows),
                "mean_target_activation_delta": round(_mean(deltas), 6),
                "mean_abs_target_activation_delta": round(mean_abs_delta, 6),
                "mean_score_delta": round(_mean(score_deltas), 6) if score_deltas else None,
                "control_record_count": len(strength_control_rows),
                "control_mean_abs_target_activation_delta": round(control_mean_abs_delta, 6)
                if strength_control_rows
                else None,
                "control_mean_abs_score_delta": round(_mean([abs(value) for value in control_score_deltas]), 6)
                if control_score_deltas
                else None,
                "path_specificity_score": round(max(0.0, mean_abs_delta - control_mean_abs_delta), 6)
                if strength_control_rows
                else None,
            }
        )
    return summary


def _is_path_control(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata", {})
    return isinstance(metadata, dict) and bool(metadata.get("control_type"))


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _display_token(token: str) -> str:
    if len(token) == 1 and not token.isalnum():
        return repr(token)
    return token


_GRAPH_HTML_CSS = r"""
:root {
  color-scheme: light;
  --bg: #f7f8fb;
  --panel: #ffffff;
  --ink: #152033;
  --muted: #5c6b80;
  --line: #dbe2ea;
  --blue: #2563eb;
  --green: #15803d;
  --amber: #b45309;
  --violet: #7c3aed;
  --red: #b91c1c;
  --shadow: 0 10px 30px rgba(21, 32, 51, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header, main { max-width: 1240px; margin: 0 auto; padding: 24px; }
header { padding-top: 32px; padding-bottom: 12px; }
.kicker {
  color: var(--blue);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
h1 { margin: 6px 0 4px; font-size: 30px; line-height: 1.15; letter-spacing: 0; }
h2 { margin: 0 0 14px; font-size: 16px; letter-spacing: 0; }
code { background: #eef2f7; border: 1px solid var(--line); border-radius: 4px; padding: 1px 5px; }
.lede, .muted { color: var(--muted); }
.toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
}
.toolbar label { font-weight: 700; }
.toolbar input {
  min-width: min(420px, 100%);
  flex: 1;
}
.toolbar select {
  min-width: 160px;
}
.toolbar input, .toolbar select {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px 12px;
  font: inherit;
  background: #fff;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.metric, .panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}
.metric { padding: 14px; }
.metric .value { display: block; font-size: 24px; font-weight: 800; }
.metric .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }
.panel { padding: 16px; margin-bottom: 16px; }
.brief-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
}
.brief-item {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  background: #fbfcfe;
}
.brief-item strong { display: block; margin-bottom: 4px; }
.graph-panel { overflow-x: auto; }
#graph-svg { width: 100%; min-width: 900px; height: 560px; border: 1px solid var(--line); border-radius: 6px; background: #fbfcfe; }
.grid { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(280px, 0.85fr); gap: 16px; }
table { width: 100%; border-collapse: collapse; }
th, td { border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
tr:last-child td { border-bottom: 0; }
.pill {
  display: inline-block;
  border-radius: 999px;
  border: 1px solid var(--line);
  padding: 2px 7px;
  font-size: 12px;
  background: #f8fafc;
  white-space: nowrap;
}
.pill.robust, .pill.validated, .pill.criterion_promoter { color: var(--green); border-color: #bbf7d0; background: #f0fdf4; }
.pill.failed_control, .pill.control_failed, .pill.criterion_suppressor { color: var(--red); border-color: #fecaca; background: #fef2f2; }
.pill.path_patch { color: var(--amber); border-color: #fed7aa; background: #fff7ed; }
.pill.coactivation { color: var(--violet); border-color: #ddd6fe; background: #f5f3ff; }
.node-label { font-size: 11px; fill: #1f2937; pointer-events: none; }
.node.dimmed, .edge.dimmed { opacity: 0.12; }
.feature-row.hidden { display: none; }
.path-row.hidden { display: none; }
.command-cell {
  display: grid;
  gap: 6px;
}
.copy-command {
  justify-self: start;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  padding: 5px 8px;
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.copy-command:focus-visible {
  outline: 2px solid var(--blue);
  outline-offset: 2px;
}
.empty { color: var(--muted); font-style: italic; }
@media (max-width: 760px) {
  header, main { padding: 18px; }
  h1 { font-size: 24px; }
  .grid { grid-template-columns: 1fr; }
  #graph-svg { min-width: 720px; height: 520px; }
}
"""


_GRAPH_HTML_JS = r"""
const graph = JSON.parse(document.getElementById("graph-data").textContent);
const summary = JSON.parse(document.getElementById("summary-data").textContent);
const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
const edges = Array.isArray(graph.edges) ? graph.edges : [];
const positions = new Map();

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function number(value) {
  if (value === null || value === undefined || value === "") return "";
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(4) : escapeHtml(value);
}

function pill(value) {
  if (!value) return "";
  const css = String(value).replaceAll(/[^a-zA-Z0-9_-]/g, "_");
  return `<span class="pill ${css}">${escapeHtml(value)}</span>`;
}

function compactText(value, fallback = "") {
  const text = String(value ?? "").trim();
  return text || fallback;
}

function copyButton(command) {
  if (!command) return "";
  return `<button type="button" class="copy-command" data-command="${escapeHtml(command)}">Copy</button>`;
}

function cssEscape(value) {
  if (globalThis.CSS && typeof CSS.escape === "function") return CSS.escape(value);
  return String(value).replaceAll("\\", "\\\\").replaceAll('"', '\\"');
}

function nodeLabel(node) {
  return node.label || node.id || "";
}

function searchableText(node) {
  return [
    node.id,
    node.label,
    node.role,
    node.type,
    node.layer,
    Array.isArray(node.top_tokens) ? node.top_tokens.join(" ") : "",
  ].join(" ").toLowerCase();
}

function edgeColor(typeOrRole) {
  if (typeOrRole === "path_patch") return "#b45309";
  if (typeOrRole === "causal_effect" || typeOrRole === "aggregate_causal_effect") return "#2563eb";
  if (typeOrRole === "coactivation") return "#7c3aed";
  if (typeOrRole === "fingerprint_similarity") return "#15803d";
  if (typeOrRole === "criterion_promoter") return "#15803d";
  if (typeOrRole === "criterion_suppressor") return "#b91c1c";
  return "#94a3b8";
}

function renderMetrics() {
  const counts = summary.counts || {};
  const validation = summary.validation || {};
  const items = [
    ["Features", counts.features],
    ["Edges", counts.edges],
    ["Path edges", counts.path_patch_edges],
    ["Candidate paths", counts.candidate_paths],
    ["Claim grade", validation.overall_claim_grade || "unvalidated"],
  ];
  document.getElementById("metrics").innerHTML = items.map(([label, value]) => `
    <div class="metric"><span class="value">${escapeHtml(value ?? 0)}</span><span class="label">${escapeHtml(label)}</span></div>
  `).join("");
}

function renderBrief() {
  const validation = summary.validation || {};
  const strongest = Array.isArray(summary.strongest_features) ? summary.strongest_features[0] : null;
  const paths = Array.isArray(summary.candidate_paths) ? summary.candidate_paths : [];
  const bestPath = paths[0] || null;
  const nextAction = Array.isArray(summary.agent_next_actions) ? summary.agent_next_actions[0] : null;
  const items = [
    ["Evidence grade", compactText(validation.overall_claim_grade, "unvalidated")],
    ["Strongest feature", strongest ? `${compactText(strongest.feature_id)} ${number(strongest.strong_causal_score)}` : "none"],
    ["Top path", bestPath ? `${compactText(bestPath.source_feature_id)} -> ${compactText(bestPath.target_feature_id)}` : "none measured"],
    ["Next action", compactText(validation.recommended_next_action || (nextAction && nextAction.title), "measure or validate candidate paths")],
  ];
  document.getElementById("graph-brief").innerHTML = `
    <h2>Evidence Summary</h2>
    <div class="brief-grid">
      ${items.map(([label, value]) => `<div class="brief-item"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(value)}</span></div>`).join("")}
    </div>
  `;
}

function populateFilters() {
  const roleFilter = document.getElementById("role-filter");
  const statusFilter = document.getElementById("status-filter");
  const roles = [...new Set(nodes.map((node) => node.role).filter(Boolean))].sort();
  const statuses = [...new Set((Array.isArray(summary.candidate_paths) ? summary.candidate_paths : [])
    .map((row) => row.status || row.claim_grade || row.evidence)
    .filter(Boolean))].sort();
  roleFilter.innerHTML = '<option value="">All roles</option>' + roles.map((role) => `<option value="${escapeHtml(role)}">${escapeHtml(role)}</option>`).join("");
  statusFilter.innerHTML = '<option value="">All path statuses</option>' + statuses.map((status) => `<option value="${escapeHtml(status)}">${escapeHtml(status)}</option>`).join("");
}

function renderGraph() {
  const svg = document.getElementById("graph-svg");
  const width = 1180;
  const featureNodes = nodes.filter((node) => node.type === "feature");
  const layers = [...new Set(featureNodes.map((node) => node.layer ?? "unknown"))].sort((a, b) => Number(a) - Number(b));
  const columns = new Map();
  for (const node of nodes) {
    let column = "features";
    if (node.type === "criterion") column = "criterion";
    else if (node.type === "supernode") column = "groups";
    else column = `layer:${node.layer ?? "unknown"}`;
    if (!columns.has(column)) columns.set(column, []);
    columns.get(column).push(node);
  }
  const orderedColumns = layers.map((layer) => `layer:${layer}`).filter((key) => columns.has(key));
  if (columns.has("groups")) orderedColumns.push("groups");
  if (columns.has("criterion")) orderedColumns.push("criterion");
  if (!orderedColumns.length) orderedColumns.push("features");
  const height = Math.max(520, Math.max(...[...columns.values()].map((items) => items.length), 1) * 76 + 90);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `
    <defs>
      <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0,0 L0,6 L6,3 z" fill="#64748b"></path>
      </marker>
    </defs>
  `;
  orderedColumns.forEach((column, columnIndex) => {
    const items = columns.get(column) || [];
    const x = 58 + columnIndex * ((width - 116) / Math.max(1, orderedColumns.length - 1));
    items.forEach((node, rowIndex) => {
      const y = 60 + rowIndex * ((height - 120) / Math.max(1, items.length - 1));
      positions.set(String(node.id), {x, y});
    });
  });
  for (const edge of edges) {
    const source = positions.get(String(edge.source));
    const target = positions.get(String(edge.target));
    if (!source || !target) continue;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const dx = Math.max(30, Math.abs(target.x - source.x) * 0.45);
    line.setAttribute("d", `M ${source.x} ${source.y} C ${source.x + dx} ${source.y}, ${target.x - dx} ${target.y}, ${target.x} ${target.y}`);
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", edgeColor(edge.type));
    line.setAttribute("stroke-width", edge.type === "path_patch" ? "2.7" : "1.6");
    line.setAttribute("opacity", edge.type === "member_of" ? "0.32" : "0.72");
    line.setAttribute("marker-end", "url(#arrow)");
    line.classList.add("edge");
    line.dataset.source = String(edge.source);
    line.dataset.target = String(edge.target);
    svg.appendChild(line);
  }
  for (const node of nodes) {
    const pos = positions.get(String(node.id));
    if (!pos) continue;
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.classList.add("node");
    group.dataset.nodeId = String(node.id);
    group.dataset.search = searchableText(node);
    group.dataset.role = String(node.role || "");
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", pos.x);
    circle.setAttribute("cy", pos.y);
    circle.setAttribute("r", node.type === "criterion" ? "19" : node.type === "supernode" ? "15" : "12");
    circle.setAttribute("fill", node.type === "criterion" ? "#2563eb" : node.type === "supernode" ? "#0f766e" : "#ffffff");
    circle.setAttribute("stroke", node.type === "feature" ? edgeColor(node.role) : "#1f2937");
    circle.setAttribute("stroke-width", "2");
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${node.id}\n${nodeLabel(node)}\n${node.role || node.type || ""}`;
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", pos.x + 18);
    label.setAttribute("y", pos.y + 4);
    label.setAttribute("class", "node-label");
    label.textContent = String(nodeLabel(node)).slice(0, 42);
    group.appendChild(title);
    group.appendChild(circle);
    group.appendChild(label);
    svg.appendChild(group);
  }
}

function renderTable(targetId, rows, columns) {
  const target = document.getElementById(targetId);
  if (!rows.length) {
    target.innerHTML = '<p class="empty">No rows in this graph.</p>';
    return;
  }
  target.innerHTML = `
    <table>
      <thead><tr>${columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("")}</tr></thead>
      <tbody>
        ${rows.map((row) => `<tr class="${row._class || ""}" data-search="${escapeHtml(row._search || "")}" data-role="${escapeHtml(row._role || row.role || "")}" data-status="${escapeHtml(row._status || row.status || row.claim_grade || row.evidence || "")}">
          ${columns.map((column) => `<td>${column.render(row)}</td>`).join("")}
        </tr>`).join("")}
      </tbody>
    </table>
  `;
}

function renderDetails() {
  const paths = (Array.isArray(summary.candidate_paths) ? summary.candidate_paths : []).map((row) => ({
    ...row,
    _class: "path-row",
    _status: row.status || row.claim_grade || row.evidence || "",
    _search: [row.source_feature_id, row.target_feature_id, row.status, row.claim_grade, row.evidence, row.next_action].join(" ").toLowerCase(),
  }));
  renderTable("candidate-paths", paths, [
    {label: "Status", render: (row) => pill(row.status || row.claim_grade || row.evidence)},
    {label: "Path", render: (row) => `<code>${escapeHtml(row.source_feature_id)} -> ${escapeHtml(row.target_feature_id)}</code>`},
    {label: "Effect", render: (row) => number(row.mean_abs_target_activation_delta)},
    {label: "Specificity", render: (row) => number(row.path_specificity_score)},
    {label: "Records", render: (row) => escapeHtml(row.record_count ?? "")},
  ]);
  const strong = Array.isArray(summary.strongest_features) ? summary.strongest_features : [];
  renderTable("strong-features", strong, [
    {label: "Feature", render: (row) => `<code>${escapeHtml(row.feature_id)}</code>`},
    {label: "Label", render: (row) => escapeHtml(row.label || "")},
    {label: "Role", render: (row) => pill(row.role)},
    {label: "Strong", render: (row) => number(row.strong_causal_score)},
  ]);
  const featureRows = nodes.filter((node) => node.type === "feature").map((node) => ({
    ...node,
    _class: "feature-row",
    _role: node.role || "",
    _search: searchableText(node),
  }));
  renderTable("feature-cards", featureRows, [
    {label: "Feature", render: (row) => `<code>${escapeHtml(row.id)}</code>`},
    {label: "Label", render: (row) => escapeHtml(row.label || "")},
    {label: "Role", render: (row) => pill(row.role)},
    {label: "Layer", render: (row) => escapeHtml(row.layer ?? "")},
    {label: "Tokens", render: (row) => escapeHtml(Array.isArray(row.top_tokens) ? row.top_tokens.join(", ") : "")},
  ]);
  const actions = Array.isArray(summary.agent_next_actions) ? summary.agent_next_actions : [];
  renderTable("agent-actions", actions, [
    {label: "Action", render: (row) => `<code>${escapeHtml(row.id || "")}</code>`},
    {label: "Title", render: (row) => escapeHtml(row.title || "")},
    {label: "Command", render: (row) => `<span class="command-cell"><code>${escapeHtml(row.command || "")}</code>${copyButton(row.command || "")}</span>`},
  ]);
}

function applyFilter() {
  const query = document.getElementById("feature-search").value.trim().toLowerCase();
  const role = document.getElementById("role-filter").value;
  const status = document.getElementById("status-filter").value;
  const filteredRows = [...document.querySelectorAll(".feature-row, .path-row")];
  let visible = 0;
  filteredRows.forEach((row) => {
    const isPath = row.classList.contains("path-row");
    const queryMatch = !query || row.dataset.search.includes(query);
    const roleMatch = !role || row.dataset.role === role;
    const statusMatch = !status || (isPath ? row.dataset.status === status : true);
    const match = queryMatch && roleMatch && statusMatch;
    row.classList.toggle("hidden", !match);
    if (match && row.classList.contains("feature-row")) visible += 1;
  });
  document.querySelectorAll(".node").forEach((node) => {
    const queryMatch = !query || node.dataset.search.includes(query) || !node.dataset.search;
    const roleMatch = !role || node.dataset.role === role;
    const match = queryMatch && roleMatch;
    node.classList.toggle("dimmed", !match);
  });
  document.querySelectorAll(".edge").forEach((edge) => {
    if (!query && !role) {
      edge.classList.remove("dimmed");
      return;
    }
    const source = document.querySelector(`.node[data-node-id="${cssEscape(edge.dataset.source)}"]`);
    const target = document.querySelector(`.node[data-node-id="${cssEscape(edge.dataset.target)}"]`);
    const match = (source && !source.classList.contains("dimmed")) || (target && !target.classList.contains("dimmed"));
    edge.classList.toggle("dimmed", !match);
  });
  document.getElementById("filter-count").textContent = (query || role || status) ? `${visible} matching feature rows` : "";
}

async function copyCommand(button) {
  const command = button.dataset.command || "";
  try {
    await navigator.clipboard.writeText(command);
  } catch (error) {
    const area = document.createElement("textarea");
    area.value = command;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.left = "-9999px";
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  const original = button.textContent;
  button.textContent = "Copied";
  window.setTimeout(() => { button.textContent = original; }, 1200);
}

renderMetrics();
renderBrief();
populateFilters();
renderGraph();
renderDetails();
document.getElementById("feature-search").addEventListener("input", applyFilter);
document.getElementById("role-filter").addEventListener("change", applyFilter);
document.getElementById("status-filter").addEventListener("change", applyFilter);
document.addEventListener("click", (event) => {
  const button = event.target.closest(".copy-command");
  if (button) copyCommand(button);
});
"""
