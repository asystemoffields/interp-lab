from __future__ import annotations

import argparse
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
    return path


def write_attribution_graph_markdown(graph: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_attribution_graph_markdown(graph), encoding="utf-8")
    return path


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
        nodes.append(_feature_node(card))
        edge = _criterion_edge(card, criterion_id=criterion_id)
        if edge is not None:
            edges.append(edge)
    supernodes: list[dict[str, Any]] = []
    if include_supernodes:
        supernodes, supernode_edges = _supernodes(report.cards, criterion_id=criterion_id)
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
    path_patch_edges = _path_patch_edges(path_records or [])
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


def run_graph_export_from_args(args: argparse.Namespace) -> Path:
    return export_attribution_graph(
        report_path=args.report,
        out_path=args.out,
        markdown_out_path=args.markdown_out,
        include_similarity_edges=args.include_similarity_edges,
        similarity_threshold=args.similarity_threshold,
        include_coactivation_edges=args.include_coactivation_edges,
        coactivation_threshold=args.coactivation_threshold,
        include_supernodes=args.include_supernodes,
        strong_causal_threshold=args.strong_causal_threshold,
        path_records_path=args.path_records,
    )


def _feature_node(card: FeatureCard) -> dict[str, Any]:
    return {
        "id": card.feature_id,
        "type": "feature",
        "model": card.model,
        "layer": card.layer,
        "label": card.label,
        "role": _feature_role(card),
        "top_tokens": _top_tokens(card),
        "source": card.source,
        "importance": card.importance,
        "association": card.association,
        "causal_effect": card.causal_effect,
        "specificity": card.specificity,
        "stability": card.stability,
    }


def _criterion_edge(card: FeatureCard, *, criterion_id: str) -> dict[str, Any] | None:
    signed = card.causal_effects.get(
        "signed_causal_effect",
        card.causal_effects.get("signed_association"),
    )
    effect = card.causal_effects.get("criterion", card.causal_effect)
    if effect is None:
        return None
    return {
        "source": card.feature_id,
        "target": criterion_id,
        "type": "causal_effect",
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
                        "source": left.feature_id,
                        "target": right.feature_id,
                        "type": "fingerprint_similarity",
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


def _path_patch_edges(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        edges.append(
            {
                "source": source,
                "target": target,
                "type": "path_patch",
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
                "source": source.feature_id,
                "target": target.feature_id,
                "type": "coactivation",
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


def _supernodes(cards: list[FeatureCard], *, criterion_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[FeatureCard]] = {}
    for card in cards:
        grouped.setdefault(_role_supernode_id(card), []).append(card)
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
                "role": _feature_role(max(members, key=_strong_causal_score)),
                "mean_signed_effect": round(signed, 6),
                "mean_strong_causal_score": round(strong, 6),
                "members": [member.feature_id for member in members],
            }
        )
        for member in members:
            edges.append(
                {
                    "source": member.feature_id,
                    "target": node_id,
                    "type": "member_of",
                    "evidence": "automatic_role_or_theme_grouping",
                }
            )
        edges.append(
            {
                "source": node_id,
                "target": criterion_id,
                "type": "aggregate_causal_effect",
                "signed_effect": round(signed, 6),
                "strong_causal_score": round(strong, 6),
                "member_count": len(members),
            }
        )
    return nodes, edges


def _role_supernode_id(card: FeatureCard) -> str:
    layer = "unknown-layer" if card.layer is None else f"layer-{card.layer}"
    return f"{layer}:{_feature_role(card)}"


def _feature_role(card: FeatureCard) -> str:
    strong = _strong_causal_score(card)
    signed = _signed_effect(card)
    if strong >= DEFAULT_STRONG_CAUSAL_THRESHOLD and signed > 0:
        return "criterion_promoter"
    if strong >= DEFAULT_STRONG_CAUSAL_THRESHOLD and signed < 0:
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
                "role": _feature_role(card),
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
        source = by_id.get(str(edge["source"]))
        target = by_id.get(str(edge["target"]))
        paths.append(
            {
                "source_feature_id": str(edge["source"]),
                "source_label": _display_label(source) if source is not None else str(edge["source"]),
                "source_layer": source.layer if source is not None else None,
                "target_feature_id": str(edge["target"]),
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
        source = by_id.get(str(edge["source"]))
        target = by_id.get(str(edge["target"]))
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


def _graph_validation_run_assessment(graph: dict[str, Any]) -> dict[str, Any]:
    metadata = graph.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    graph_validation = metadata.get("graph_validation")
    if not isinstance(graph_validation, dict):
        return {}
    run_assessment = graph_validation.get("run_assessment")
    return run_assessment if isinstance(run_assessment, dict) else {}


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
