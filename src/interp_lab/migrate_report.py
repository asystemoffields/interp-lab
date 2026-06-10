"""Migrate older inspection reports to current scoring semantics.

Pre-2.3 reports could count a correlational ``causal_effects["criterion"]``
score as a causal effect (and let importance double-count it). ``migrate-report``
re-runs :func:`interp_lab.scoring.score_feature` over each card's STORED
evidence -- serialized cards retain everything score_feature consumes
(``causal_effects``, ``metadata`` for the provenance gate, label/examples for
the association text fallback, and the fingerprint's ``activation_signature``
for stability) -- then re-ranks by the new importance and stamps
``metadata.migration`` with the old->new score deltas.

The schema stays ``inspection_report.v1``: migration changes scores under the
current semantics, not the serialization. Migrating an already-migrated report
is a no-op apart from a refreshed migration stamp (empty deltas, no reorder).
Anything that cannot be recovered exactly is recorded per card in
``metadata.migration_notes`` instead of being silently re-derived.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from interp_lab import __version__
from interp_lab.matching import has_intervention_provenance
from interp_lab.reporting import load_inspection_report, write_inspection_report
from interp_lab.schema import FeatureCard, FeatureEvidence, InspectionReport, utc_now_iso
from interp_lab.scoring import score_feature

_SCORE_FIELDS = ("importance", "association", "specificity", "causal_effect", "stability")


def migrate_inspection_report(
    report_or_path: InspectionReport | dict | str | Path,
    *,
    out: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Re-score an inspection report under current semantics and re-rank its cards.

    ``report_or_path`` may be an ``InspectionReport``, a report dict, or a path
    to a ``report.json``. With ``out`` set, the migrated report is written via
    :func:`interp_lab.reporting.write_inspection_report` (``report.json`` +
    ``report.md`` in that directory). Returns the migrated report dict; the
    migration stamp lives at ``metadata["migration"]``.
    """
    report = _resolve_report(report_or_path)
    old_order = [card.feature_id for card in report.cards]
    migrated_cards: list[FeatureCard] = []
    score_deltas: dict[str, dict[str, list[float]]] = {}
    for card in report.cards:
        scores = score_feature(_evidence_from_card(card), report.criterion)
        deltas = {
            field: [round(float(getattr(card, field)), 6), scores[field]]
            for field in _SCORE_FIELDS
            if round(float(getattr(card, field)), 6) != scores[field]
        }
        if deltas:
            score_deltas[card.feature_id] = deltas
        metadata = dict(card.metadata)
        notes = _migration_notes(card)
        if notes:
            metadata["migration_notes"] = notes
        else:
            metadata.pop("migration_notes", None)
        migrated_cards.append(
            replace(
                card,
                importance=scores["importance"],
                association=scores["association"],
                specificity=scores["specificity"],
                causal_effect=scores["causal_effect"],
                stability=scores["stability"],
                metadata=metadata,
            )
        )
    # Stable sort: re-rank by the new importance, preserving the stored order on
    # ties (the same ordering rule pipeline.inspect_model uses).
    migrated_cards.sort(key=lambda card: card.importance, reverse=True)
    new_order = [card.feature_id for card in migrated_cards]
    metadata = dict(report.metadata)
    metadata["migration"] = {
        "from_tool_version": _from_tool_version(report.metadata),
        "to_tool_version": __version__,
        "migrated_at": _timestamp(now),
        "changes": {
            "features_reordered": new_order != old_order,
            "score_deltas": score_deltas,
        },
    }
    migrated = InspectionReport(
        model=report.model,
        criterion=report.criterion,
        cards=migrated_cards,
        created_at=report.created_at,
        metadata=metadata,
    )
    if out is not None:
        write_inspection_report(migrated, out)
    return migrated.to_dict()


def build_migrate_report_parser() -> argparse.ArgumentParser:
    # Default add_help=True so `migrate-report --help` works standalone; the cli
    # subparser adopting this as a parent passes add_help=False.
    parser = argparse.ArgumentParser(
        description="Re-score an older inspection report under current scoring semantics."
    )
    parser.add_argument("--report", required=True, help="Inspection report.json to migrate.")
    parser.add_argument(
        "--out",
        help="Output report directory (writes report.json + report.md). Omit to print JSON only.",
    )
    parser.add_argument("--json", action="store_true", help="Print the migrated report as JSON.")
    return parser


def run_migrate_report_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return migrate_inspection_report(args.report, out=args.out)


def _resolve_report(report_or_path: InspectionReport | dict | str | Path) -> InspectionReport:
    if isinstance(report_or_path, InspectionReport):
        return report_or_path
    if isinstance(report_or_path, dict):
        return InspectionReport.from_dict(report_or_path)
    return load_inspection_report(Path(report_or_path))


def _evidence_from_card(card: FeatureCard) -> FeatureEvidence:
    """Rebuild the FeatureEvidence score_feature consumes from a serialized card.

    Every consumed field survives serialization: ``causal_effects`` (criterion /
    strong_causal_score / specificity / side_effect / signed_association),
    ``metadata`` (intervention provenance), ``label``/``examples`` (association
    text fallback), and ``fingerprint.activation_signature`` (stability).
    """
    return FeatureEvidence(
        feature_id=card.feature_id,
        model=card.model,
        layer=card.layer,
        label=card.label,
        examples=list(card.examples),
        activation_signature=list(card.fingerprint.activation_signature),
        decoder_signature=list(card.fingerprint.decoder_signature),
        causal_effects=dict(card.causal_effects),
        source=card.source,
        metadata=dict(card.metadata),
    )


def _migration_notes(card: FeatureCard) -> list[str]:
    """Per-card notes for evidence the migration cannot recover as a measurement."""
    notes: list[str] = []
    if "criterion" in card.causal_effects and not has_intervention_provenance(
        card.causal_effects, card.metadata
    ):
        notes.append(
            "causal_effects['criterion'] carries no intervention provenance; under current "
            "semantics it is correlational, so causal_effect re-scores to 0.0."
        )
    if (
        "signed_association" not in card.causal_effects
        and card.metadata.get("signed_association") is None
    ):
        notes.append(
            "no measured signed_association was retained; association is re-derived from "
            "criterion/label text similarity, not a measurement."
        )
    if not card.fingerprint.activation_signature:
        notes.append(
            "no activation_signature was retained on the fingerprint; stability re-scores "
            "from the no-signal baseline."
        )
    return notes


def _from_tool_version(metadata: dict[str, Any]) -> str:
    tool = metadata.get("tool")
    if isinstance(tool, dict) and tool.get("version"):
        return str(tool["version"])
    return "unknown"


def _timestamp(now: datetime | None) -> str:
    if now is None:
        return utc_now_iso()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.isoformat()
