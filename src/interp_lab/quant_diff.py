"""Which validated features does quantization break?

``quant-diff`` compares two inspection reports of the SAME criterion produced
from two precision variants of one model family -- e.g. ``llama-3-8b-f16``
(baseline) against ``llama-3-8b-q4_k_m`` (variant) -- and answers the question
mixed-precision quantization work actually cares about: which features
survived, which degraded, which disappeared, and which appeared from nowhere.

It is a thin, opinionated layer over the existing match + match-validation
machinery: features are paired with :func:`interp_lab.matching.match_feature_cards`,
each pair is graded by :func:`interp_lab.match_validation.build_match_validation_report`,
and this module turns the pairing into per-feature verdicts plus a headline a
human (or agent) can act on.

Verdicts
--------
- ``preserved``                 The baseline feature was intervention-validated and the
                                variant keeps comparable, same-direction causal evidence
                                within the verdict thresholds.
- ``degraded``                  The baseline feature was intervention-validated but the
                                variant flipped its signed effect, lost its intervention
                                evidence, washed the effect out, or dropped sharply in
                                importance. These are the features quantization broke.
- ``preserved_correlational``   The pair matches, but the baseline side only ever had
                                correlational (association) evidence -- there was no
                                validated causal claim to break, so it can NEVER appear
                                in ``degraded_validated``.
- ``changed_correlational``     Same correlational-only caveat, but the association
                                evidence moved (sign flip or large drop). A lead, not a
                                broken validated feature.
- ``lost``                      No acceptable match for the baseline feature in the
                                variant report.
- ``emerged``                   A variant-only feature with no acceptable baseline match.

Verdict thresholds (``VERDICT_THRESHOLDS``)
-------------------------------------------
- ``min_match_score`` (0.40)  Minimum match score for a pair to count as the "same"
  feature. Deliberately BELOW matching's 0.49 opposite-direction cap
  (``matching._score_with_signed_effect``): a sign-flipped pair must classify as
  *degraded*, not silently fall out of the pairing and read as *lost*.
- ``min_structural_component`` (0.65) / ``min_structural_components`` (1)  At least
  one structural axis (text / activation / decoder) must clear the match-validation
  component bar. Without this, two unrelated features with orthogonal fingerprints
  pair at the free-0.5 cosine floor and a genuinely lost feature would masquerade
  as a surviving match.
- ``max_importance_drop`` (0.15)  An importance fall larger than this (baseline minus
  variant) marks an intervention-validated feature degraded even when its signed
  effect still agrees.
- ``max_signed_effect_drop`` (0.15)  Maximum tolerated loss of same-provenance
  signed-effect magnitude (``abs(left) - abs(right)``) before the pair is degraded.
  Matches ``match_validation.DEFAULT_MAX_SIGNED_EFFECT_DELTA`` so the two layers
  agree on what a meaningful effect change is.
- ``min_abs_signed_effect`` (0.02)  Magnitude floor below which a signed effect is
  noise (same floor as matching/match-validation). A baseline effect above the floor
  that lands below it in the variant is "washed out".

Provenance discipline: signed effects are only compared when both sides carry the
SAME provenance (intervention vs association), via matching's shared accessors.
Mixed-provenance pairs are labeled ``mixed_provenance_not_compared`` and the
intervention side losing its evidence is itself a degradation signal
(``right_lost_intervention_evidence``).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from interp_lab.agent_actions import next_action
from interp_lab.match_validation import (
    DEFAULT_MAX_SIGNED_EFFECT_DELTA,
    DEFAULT_MIN_ABS_SIGNED_EFFECT,
    DEFAULT_MIN_COMPONENT,
    build_match_validation_report,
)
from interp_lab.matching import has_intervention_provenance, signed_effect_with_provenance
from interp_lab.pipeline import match_reports
from interp_lab.reporting import load_inspection_report, load_match_report
from interp_lab.schema import FeatureCard, InspectionReport, MatchReport

QUANT_DIFF_SCHEMA = "interp-lab.quant_diff.v1"

DEFAULT_MIN_MATCH_SCORE = 0.40
DEFAULT_MIN_STRUCTURAL_COMPONENT = DEFAULT_MIN_COMPONENT
DEFAULT_MIN_STRUCTURAL_COMPONENTS = 1
DEFAULT_MAX_IMPORTANCE_DROP = 0.15
DEFAULT_MAX_SIGNED_EFFECT_DROP = DEFAULT_MAX_SIGNED_EFFECT_DELTA

_STRUCTURAL_COMPONENTS = ("text", "activation", "decoder")

_THRESHOLD_DOCS = {
    "min_match_score": (
        "Minimum match score for a baseline/variant pair to count as the same feature; "
        "kept below matching's 0.49 opposite-direction cap so sign-flipped features "
        "classify as degraded instead of lost."
    ),
    "min_structural_component": (
        "Component bar a structural axis (text/activation/decoder) must clear for the "
        "pair to be acceptable; blocks free-0.5 cosine pairings of unrelated features."
    ),
    "min_structural_components": "How many structural axes must clear the component bar.",
    "max_importance_drop": (
        "Importance fall (baseline minus variant) beyond which an intervention-validated "
        "feature is degraded even with an agreeing signed effect."
    ),
    "max_signed_effect_drop": (
        "Maximum tolerated loss of same-provenance signed-effect magnitude "
        "(abs(baseline) - abs(variant)) before the pair is degraded."
    ),
    "min_abs_signed_effect": (
        "Magnitude floor below which a signed effect is treated as noise; a baseline "
        "effect above the floor that lands below it in the variant is washed out."
    ),
}


def build_quant_diff(
    left_report: InspectionReport,
    right_report: InspectionReport,
    *,
    matches: MatchReport | dict[str, Any] | None = None,
    match_validation: dict[str, Any] | None = None,
    left_label: str = "baseline",
    right_label: str = "variant",
    min_match_score: float = DEFAULT_MIN_MATCH_SCORE,
    min_structural_component: float = DEFAULT_MIN_STRUCTURAL_COMPONENT,
    min_structural_components: int = DEFAULT_MIN_STRUCTURAL_COMPONENTS,
    max_importance_drop: float = DEFAULT_MAX_IMPORTANCE_DROP,
    max_signed_effect_drop: float = DEFAULT_MAX_SIGNED_EFFECT_DROP,
    min_abs_signed_effect: float = DEFAULT_MIN_ABS_SIGNED_EFFECT,
) -> dict[str, Any]:
    """Build the quantization variant-comparison report.

    ``left_report`` is the higher-precision baseline, ``right_report`` the quantized
    variant. Both must inspect the SAME criterion; model ids may differ (that is the
    point: ``llama-3-8b-f16`` vs ``llama-3-8b-q4_k_m``). When ``matches`` /
    ``match_validation`` artifacts are not supplied they are computed in-process with
    the real matching and match-validation functions over every card pair.
    """
    if left_report.criterion.text.strip() != right_report.criterion.text.strip():
        raise ValueError(
            "quant-diff requires both reports to inspect the SAME criterion "
            f"(left: {left_report.criterion.text!r}, right: {right_report.criterion.text!r}); "
            "a cross-criterion diff would report every feature as broken."
        )
    thresholds = {
        "min_match_score": float(min_match_score),
        "min_structural_component": float(min_structural_component),
        "min_structural_components": int(min_structural_components),
        "max_importance_drop": float(max_importance_drop),
        "max_signed_effect_drop": float(max_signed_effect_drop),
        "min_abs_signed_effect": float(min_abs_signed_effect),
    }
    match_report = _coerce_match_report(matches, left_report, right_report)
    if match_validation is None:
        match_validation = build_match_validation_report(match_report)
    validation_by_pair = {
        (str(item.get("left_feature_id")), str(item.get("right_feature_id"))): item
        for item in match_validation.get("validations", [])
    }

    pairs = _assign_pairs(match_report, thresholds)
    left_by_id = {card.feature_id: card for card in left_report.cards}
    right_by_id = {card.feature_id: card for card in right_report.cards}

    features: list[dict[str, Any]] = []
    for left_id, right_id, score in pairs:
        left_card = left_by_id.get(left_id)
        right_card = right_by_id.get(right_id)
        if left_card is None or right_card is None:
            # The match artifact references features the reports no longer carry
            # (e.g. matches built from a wider top_k). Skip rather than invent cards.
            continue
        validation = validation_by_pair.get((left_id, right_id))
        features.append(
            _matched_entry(
                left_card,
                right_card,
                score=score,
                validation=validation,
                thresholds=thresholds,
            )
        )
    matched_left = {entry["left_feature_id"] for entry in features}
    matched_right = {entry["right_feature_id"] for entry in features}
    lost_features = [
        _unmatched_entry(card, verdict="lost", side="left")
        for card in left_report.cards
        if card.feature_id not in matched_left
    ]
    emerged_features = [
        _unmatched_entry(card, verdict="emerged", side="right")
        for card in right_report.cards
        if card.feature_id not in matched_right
    ]

    summary = _summary(
        features,
        lost_features,
        emerged_features,
        left_report=left_report,
        right_report=right_report,
        thresholds=thresholds,
    )
    report = {
        "schema_version": QUANT_DIFF_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "criterion": left_report.criterion.text,
        "left": _side_summary(left_report, left_label),
        "right": _side_summary(right_report, right_label),
        "summary": summary,
        "features": features,
        "lost_features": lost_features,
        "emerged_features": emerged_features,
        "interpretation": _interpret(summary, left_label, right_label),
        "agent_next_actions": _quant_diff_next_actions(summary),
    }
    return report


def export_quant_diff(
    left: str | Path,
    right: str | Path,
    out: str | Path,
    *,
    markdown_out: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the quant diff from report paths and write JSON + Markdown.

    ``matches`` / ``match_validation`` keyword arguments may be paths to the
    ``match`` / ``validate-matches`` artifacts; remaining keywords pass through to
    :func:`build_quant_diff`. Returns the report dict.
    """
    matches = kwargs.pop("matches", None)
    if isinstance(matches, (str, Path)):
        matches = load_match_report(matches)
    match_validation = kwargs.pop("match_validation", None)
    if isinstance(match_validation, (str, Path)):
        match_validation = json.loads(Path(match_validation).read_text(encoding="utf-8"))
    report = build_quant_diff(
        load_inspection_report(left),
        load_inspection_report(right),
        matches=matches,
        match_validation=match_validation,
        **kwargs,
    )
    json_path = Path(out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = Path(markdown_out) if markdown_out is not None else json_path.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_quant_diff_markdown(report), encoding="utf-8")
    return report


def render_quant_diff_markdown(report: dict[str, Any]) -> str:
    left = report["left"]
    right = report["right"]
    summary = report["summary"]
    lines = [
        "# Quantization Feature Diff",
        "",
        f"Baseline ({left['label']}): `{left['model']}`  -- {left['feature_count']} features",
        f"Variant ({right['label']}):  `{right['model']}`  -- {right['feature_count']} features",
        f"Criterion: {report['criterion']}",
        "",
        f"Interpretation: {report['interpretation']}",
        "",
        "## Features broken by quantization",
        "",
    ]
    broken = list(summary["degraded_validated"])
    validated_lost = list(summary["validated_lost"])
    if broken or validated_lost:
        lines.extend(
            [
                "| Baseline feature | Variant feature | Verdict | Signed effect L→R | ΔImportance | Why |",
                "| --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for item in broken:
            lines.append(
                f"| `{item['left_feature_id']}` {item['left_label']} "
                f"| `{item['right_feature_id']}` {item['right_label']} "
                f"| degraded | {_signed_arrow(item['signed_effect_comparison'])} "
                f"| {item['deltas']['importance_delta']:+.3f} "
                f"| {', '.join(item['reasons'])} |"
            )
        for item in validated_lost:
            lines.append(
                f"| `{item['feature_id']}` {item['label']} | -- | lost "
                f"| {_format_signed(item.get('signed_effect'))} → -- | -- | no_acceptable_match_in_variant |"
            )
    else:
        lines.append(
            "None -- no intervention-validated baseline feature degraded or vanished "
            "under the current verdict thresholds."
        )
    lines.append("")
    if report["features"]:
        lines.extend(
            [
                "## All matched features",
                "",
                "| Verdict | Match | Score | Validation | ΔImportance | ΔCausal effect |",
                "| --- | --- | ---: | --- | ---: | ---: |",
            ]
        )
        for item in report["features"]:
            lines.append(
                f"| {item['verdict']} | `{item['left_feature_id']} -> {item['right_feature_id']}` "
                f"| {item['match_score']:.3f} | {item['validation_status'] or 'unvalidated'} "
                f"| {item['deltas']['importance_delta']:+.3f} "
                f"| {item['deltas']['causal_effect_delta']:+.3f} |"
            )
        lines.append("")
    if report["lost_features"]:
        lines.extend(["## Lost (no acceptable match in the variant)", ""])
        for item in report["lost_features"]:
            validated = " [was intervention-validated]" if item["intervention_validated"] else ""
            lines.append(
                f"- `{item['feature_id']}` (importance {item['importance']:.3f}) -- {item['label']}{validated}"
            )
        lines.append("")
    if report["emerged_features"]:
        lines.extend(["## Emerged (variant-only)", ""])
        for item in report["emerged_features"]:
            lines.append(f"- `{item['feature_id']}` (importance {item['importance']:.3f}) -- {item['label']}")
        lines.append("")
    lines.extend(["## Verdict thresholds", ""])
    for key, value in summary["verdict_thresholds"].items():
        lines.append(f"- `{key}` = `{value}`: {_THRESHOLD_DOCS.get(key, '')}")
    lines.append("")
    actions = report.get("agent_next_actions", [])
    if actions:
        lines.extend(["## Agent Next Actions", ""])
        for action in actions:
            if "command" in action:
                lines.append(f"- `{action['id']}`: {action['title']}: `{action['command']}`")
            else:
                lines.append(f"- `{action['id']}`: {action['title']}: {action['instruction']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_quant_diff_parser() -> argparse.ArgumentParser:
    # Default add_help=True so `quant-diff --help` works; the cli subparser adopting
    # this as a parent passes add_help=False (the compare-runs pattern).
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-report", required=True, help="Baseline (higher-precision) report.json.")
    parser.add_argument("--right-report", required=True, help="Variant (quantized) report.json.")
    parser.add_argument("--matches", help="Optional match artifact from `interp-lab match`; computed in-process if omitted.")
    parser.add_argument(
        "--match-validation",
        help="Optional validation artifact from `interp-lab validate-matches`; computed in-process if omitted.",
    )
    parser.add_argument("--out", help="Output quant-diff JSON path (also writes a sibling .md). Omit to print JSON.")
    parser.add_argument("--markdown-out", help="Optional explicit markdown path.")
    parser.add_argument("--left-label", default="baseline", help="Label for the baseline side, e.g. f16.")
    parser.add_argument("--right-label", default="variant", help="Label for the quantized side, e.g. q4_k_m.")
    parser.add_argument("--json", action="store_true", help="Print the quant diff as JSON.")
    return parser


def run_quant_diff_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.out is None:
        matches = load_match_report(args.matches) if args.matches else None
        match_validation = (
            json.loads(Path(args.match_validation).read_text(encoding="utf-8"))
            if args.match_validation
            else None
        )
        report = build_quant_diff(
            load_inspection_report(args.left_report),
            load_inspection_report(args.right_report),
            matches=matches,
            match_validation=match_validation,
            left_label=args.left_label,
            right_label=args.right_label,
        )
        if args.markdown_out is not None:
            markdown_path = Path(args.markdown_out)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(render_quant_diff_markdown(report), encoding="utf-8")
        return report
    return export_quant_diff(
        args.left_report,
        args.right_report,
        args.out,
        markdown_out=args.markdown_out,
        matches=args.matches,
        match_validation=args.match_validation,
        left_label=args.left_label,
        right_label=args.right_label,
    )


def _coerce_match_report(
    matches: MatchReport | dict[str, Any] | None,
    left_report: InspectionReport,
    right_report: InspectionReport,
) -> MatchReport:
    if matches is None:
        # Keep every candidate pair: the per-left best acceptable match is selected
        # here, so a CLI-style top-k cut would turn real survivors into "lost".
        pair_count = max(1, len(left_report.cards) * len(right_report.cards))
        return match_reports(left_report, right_report, top_k=pair_count, min_score=0.0)
    if isinstance(matches, MatchReport):
        return matches
    return MatchReport.from_dict(matches)


def _assign_pairs(
    match_report: MatchReport,
    thresholds: dict[str, float],
) -> list[tuple[str, str, float]]:
    """Greedy one-to-one assignment of acceptable matches, best score first."""
    ordered = sorted(
        match_report.matches,
        key=lambda match: (-match.score, match.left_feature_id, match.right_feature_id),
    )
    used_left: set[str] = set()
    used_right: set[str] = set()
    pairs: list[tuple[str, str, float]] = []
    for match in ordered:
        if match.left_feature_id in used_left or match.right_feature_id in used_right:
            continue
        if match.score < thresholds["min_match_score"]:
            continue
        structural_passes = sum(
            1
            for name in _STRUCTURAL_COMPONENTS
            if float(match.components.get(name, 0.0)) >= thresholds["min_structural_component"]
        )
        if structural_passes < thresholds["min_structural_components"]:
            continue
        used_left.add(match.left_feature_id)
        used_right.add(match.right_feature_id)
        pairs.append((match.left_feature_id, match.right_feature_id, float(match.score)))
    return pairs


def _matched_entry(
    left_card: FeatureCard,
    right_card: FeatureCard,
    *,
    score: float,
    validation: dict[str, Any] | None,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    signed_comparison = _signed_effect_comparison(left_card, right_card, thresholds)
    deltas = {
        "importance_delta": round(right_card.importance - left_card.importance, 6),
        "association_delta": round(right_card.association - left_card.association, 6),
        "causal_effect_delta": round(right_card.causal_effect - left_card.causal_effect, 6),
    }
    left_validated = has_intervention_provenance(left_card.causal_effects, left_card.metadata)
    right_validated = has_intervention_provenance(right_card.causal_effects, right_card.metadata)
    verdict, reasons = _verdict(
        left_validated=left_validated,
        right_validated=right_validated,
        validation_status=str(validation["status"]) if validation else None,
        signed_comparison=signed_comparison,
        importance_drop=left_card.importance - right_card.importance,
        thresholds=thresholds,
    )
    return {
        "verdict": verdict,
        "reasons": reasons,
        "left_feature_id": left_card.feature_id,
        "right_feature_id": right_card.feature_id,
        "left_label": left_card.label,
        "right_label": right_card.label,
        "match_score": round(float(score), 6),
        "validation_status": str(validation["status"]) if validation else None,
        "claim_grade": str(validation["claim_grade"]) if validation else None,
        "left_intervention_validated": left_validated,
        "right_intervention_validated": right_validated,
        "left_scores": _card_scores(left_card),
        "right_scores": _card_scores(right_card),
        "deltas": deltas,
        "signed_effect_comparison": signed_comparison,
    }


def _verdict(
    *,
    left_validated: bool,
    right_validated: bool,
    validation_status: str | None,
    signed_comparison: dict[str, Any],
    importance_drop: float,
    thresholds: dict[str, float],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    direction_flip = signed_comparison.get("direction_flip") is True
    magnitude_drop = signed_comparison.get("magnitude_drop")
    washed_out = bool(signed_comparison.get("washed_out"))
    if left_validated:
        if validation_status == "contradicted" or direction_flip:
            reasons.append("signed_effect_direction_flipped")
        if not right_validated:
            reasons.append("right_lost_intervention_evidence")
        if (
            signed_comparison.get("compared")
            and signed_comparison.get("provenance") == "intervention"
        ):
            if magnitude_drop is not None and magnitude_drop > thresholds["max_signed_effect_drop"]:
                reasons.append("signed_effect_magnitude_dropped")
            if washed_out:
                reasons.append("signed_effect_washed_out")
        if importance_drop > thresholds["max_importance_drop"]:
            reasons.append("importance_dropped")
        if reasons:
            return "degraded", reasons
        return "preserved", ["intervention_evidence_preserved_within_thresholds"]
    # Correlational baseline: there was no validated causal claim to break, so these
    # verdicts are leads -- they must never land in degraded_validated.
    if validation_status == "contradicted" or direction_flip:
        reasons.append("association_direction_flipped")
    if (
        signed_comparison.get("compared")
        and magnitude_drop is not None
        and magnitude_drop > thresholds["max_signed_effect_drop"]
    ):
        reasons.append("association_magnitude_dropped")
    if importance_drop > thresholds["max_importance_drop"]:
        reasons.append("importance_dropped")
    if reasons:
        return "changed_correlational", reasons
    return "preserved_correlational", ["correlational_evidence_only_on_baseline"]


def _signed_effect_comparison(
    left_card: FeatureCard,
    right_card: FeatureCard,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    left_signed, left_provenance = signed_effect_with_provenance(
        left_card.causal_effects, left_card.metadata
    )
    right_signed, right_provenance = signed_effect_with_provenance(
        right_card.causal_effects, right_card.metadata
    )
    compared = (
        left_signed is not None
        and right_signed is not None
        and left_provenance == right_provenance
        and left_provenance != "none"
    )
    comparison: dict[str, Any] = {
        "left_signed_effect": _optional_round(left_signed),
        "left_provenance": left_provenance,
        "right_signed_effect": _optional_round(right_signed),
        "right_provenance": right_provenance,
        "compared": compared,
        "provenance": left_provenance if compared else None,
        "delta": None,
        "direction_flip": None,
        "magnitude_drop": None,
        "washed_out": False,
    }
    if not compared:
        if (
            left_signed is not None
            and right_signed is not None
            and left_provenance != right_provenance
        ):
            # A measured causal effect must never be compared against a
            # correlational proxy -- label the pair instead of publishing a delta.
            comparison["note"] = "mixed_provenance_not_compared"
        return comparison
    floor = thresholds["min_abs_signed_effect"]
    comparison["delta"] = round(right_signed - left_signed, 6)
    comparison["magnitude_drop"] = round(abs(left_signed) - abs(right_signed), 6)
    comparison["direction_flip"] = (
        abs(left_signed) >= floor and abs(right_signed) >= floor and left_signed * right_signed < 0
    )
    comparison["washed_out"] = abs(left_signed) >= floor and abs(right_signed) < floor
    return comparison


def _unmatched_entry(card: FeatureCard, *, verdict: str, side: str) -> dict[str, Any]:
    signed, provenance = signed_effect_with_provenance(card.causal_effects, card.metadata)
    return {
        "verdict": verdict,
        "side": side,
        "feature_id": card.feature_id,
        "label": card.label,
        "layer": card.layer,
        "importance": round(card.importance, 6),
        "signed_effect": _optional_round(signed),
        "signed_effect_provenance": provenance,
        "intervention_validated": has_intervention_provenance(card.causal_effects, card.metadata),
    }


def _card_scores(card: FeatureCard) -> dict[str, Any]:
    return {
        "importance": round(card.importance, 6),
        "association": round(card.association, 6),
        "causal_effect": round(card.causal_effect, 6),
        "layer": card.layer,
    }


def _side_summary(report: InspectionReport, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "model": report.model,
        "feature_count": len(report.cards),
        "created_at": report.created_at,
    }


def _summary(
    features: list[dict[str, Any]],
    lost_features: list[dict[str, Any]],
    emerged_features: list[dict[str, Any]],
    *,
    left_report: InspectionReport,
    right_report: InspectionReport,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    verdict_counts: dict[str, int] = {}
    for entry in features:
        verdict_counts[entry["verdict"]] = verdict_counts.get(entry["verdict"], 0) + 1
    degraded_validated = [entry for entry in features if entry["verdict"] == "degraded"]
    validated_lost = [entry for entry in lost_features if entry["intervention_validated"]]
    return {
        "features_compared": len(features),
        "left_feature_count": len(left_report.cards),
        "right_feature_count": len(right_report.cards),
        "preserved_count": verdict_counts.get("preserved", 0),
        "preserved_correlational_count": verdict_counts.get("preserved_correlational", 0),
        "degraded_count": verdict_counts.get("degraded", 0),
        "changed_correlational_count": verdict_counts.get("changed_correlational", 0),
        "lost_count": len(lost_features),
        "emerged_count": len(emerged_features),
        "validated_lost_count": len(validated_lost),
        # The list that matters: intervention-validated baseline features whose causal
        # claim the variant no longer supports. Correlational-only pairs cannot appear
        # here by construction (their verdicts are *_correlational).
        "degraded_validated": degraded_validated,
        "validated_lost": validated_lost,
        "verdict_thresholds": dict(thresholds),
    }


def _interpret(summary: dict[str, Any], left_label: str, right_label: str) -> str:
    broken = summary["degraded_count"]
    validated_lost = summary["validated_lost_count"]
    parts: list[str] = []
    if broken:
        ids = ", ".join(item["left_feature_id"] for item in summary["degraded_validated"])
        parts.append(
            f"quantization degraded {broken} intervention-validated feature(s) ({ids}); "
            "start with the 'Features broken by quantization' table"
        )
    if validated_lost:
        ids = ", ".join(item["feature_id"] for item in summary["validated_lost"])
        parts.append(
            f"{validated_lost} intervention-validated baseline feature(s) have no acceptable "
            f"match in the {right_label} report ({ids})"
        )
    if not parts:
        if summary["preserved_count"]:
            parts.append(
                f"every intervention-validated {left_label} feature survives in {right_label} "
                "within the verdict thresholds"
            )
        else:
            parts.append(
                "no intervention-validated baseline features were available; the comparison is "
                "correlational only, so treat survivals as leads rather than causal claims"
            )
    if summary["changed_correlational_count"]:
        parts.append(
            f"{summary['changed_correlational_count']} correlational-only pair(s) moved; they are "
            "leads, not broken validated features"
        )
    sentences = [part[0].upper() + part[1:] if part else part for part in parts]
    return ". ".join(sentences) + "."


def _quant_diff_next_actions(summary: dict[str, Any]) -> list[dict[str, Any]]:
    actions = [
        next_action(
            action_id="review_quant_diff",
            title="Read the quant-diff markdown, starting with the 'Features broken by quantization' table",
            instruction=(
                "Read the markdown written next to this JSON (<quant-diff.md>); the "
                "'Features broken by quantization' table lists the intervention-validated "
                "features the quantized variant no longer supports."
            ),
            requires=["quant diff markdown"],
        )
    ]
    if summary["degraded_count"] or summary["validated_lost_count"]:
        actions.append(
            next_action(
                action_id="plan_evidence_for_degraded",
                title="Plan the cheapest evidence upgrades for the degraded features on the variant report",
                argv=[
                    "interp-lab",
                    "plan-evidence",
                    "--report",
                    "<variant-report.json>",
                    "--out",
                    "<evidence-plan.json>",
                    "--json",
                ],
                requires=["variant inspection report"],
            )
        )
        actions.append(
            next_action(
                action_id="reexport_records_at_higher_precision",
                title="Re-export records at higher precision for the degraded layers and re-run the quant diff",
                argv=[
                    "interp-lab",
                    "export-gguf-records",
                    "--model",
                    "<higher-precision.gguf>",
                    "--dataset",
                    "<prompts.jsonl>",
                    "--out",
                    "<records.jsonl>",
                ],
                requires=["llama.cpp runtime", "higher-precision GGUF (or PMRA mixed-precision re-quant)"],
            )
        )
    return actions


def _signed_arrow(comparison: dict[str, Any]) -> str:
    left = _format_signed(comparison.get("left_signed_effect"))
    right = _format_signed(comparison.get("right_signed_effect"))
    if comparison.get("note") == "mixed_provenance_not_compared":
        return f"{left} → {right} (mixed provenance, not compared)"
    return f"{left} → {right}"


def _format_signed(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):+.3f}"


def _optional_round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)
