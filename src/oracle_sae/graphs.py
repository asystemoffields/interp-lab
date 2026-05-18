from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from oracle_sae.math_utils import cosine
from oracle_sae.reporting import load_inspection_report
from oracle_sae.schema import FeatureCard, InspectionReport


def export_attribution_graph(
    *,
    report_path: str | Path,
    out_path: str | Path,
    include_similarity_edges: bool = False,
    similarity_threshold: float = 0.9,
) -> Path:
    report = load_inspection_report(report_path)
    graph = build_attribution_graph(
        report,
        include_similarity_edges=include_similarity_edges,
        similarity_threshold=similarity_threshold,
    )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_attribution_graph(
    report: InspectionReport,
    *,
    include_similarity_edges: bool = False,
    similarity_threshold: float = 0.9,
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
    if include_similarity_edges:
        edges.extend(_similarity_edges(report.cards, threshold=similarity_threshold))
    return {
        "schema_version": "interp-lab.attribution_graph.v1",
        "model": report.model,
        "criterion": report.criterion.to_dict(),
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "source_report_created_at": report.created_at,
            "feature_count": len(report.cards),
        },
    }


def build_graph_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an interp-lab report as an attribution graph JSON.")
    parser.add_argument("--report", required=True, help="Inspection report JSON.")
    parser.add_argument("--out", required=True, help="Output graph JSON path.")
    parser.add_argument(
        "--include-similarity-edges",
        action="store_true",
        help="Add feature-to-feature edges from fingerprint similarity.",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    return parser


def run_graph_export_from_args(args: argparse.Namespace) -> Path:
    return export_attribution_graph(
        report_path=args.report,
        out_path=args.out,
        include_similarity_edges=args.include_similarity_edges,
        similarity_threshold=args.similarity_threshold,
    )


def _feature_node(card: FeatureCard) -> dict[str, Any]:
    return {
        "id": card.feature_id,
        "type": "feature",
        "model": card.model,
        "layer": card.layer,
        "label": card.label,
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
