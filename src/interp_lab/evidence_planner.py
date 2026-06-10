"""Plan the cheapest evidence-gathering path from an inspection report.

``plan-evidence`` is the directed-investigation brain for agents: given a
report, it diagnoses each feature card's evidence gaps against the SAME grading
semantics the validation layers use (provenance discipline from ``matching``/
``scoring``, control requirements from ``graph_validation``, the
underpowered/needs-controls claim ladder), then ranks the interventions that
would most cheaply move each card's claim grade toward ``validated``.

Power-analysis honesty: when a card carries only a correlational
``signed_association``, that value may seed the power calculation as a PRIOR
effect-size estimate, but it is labeled ``effect_size_source:
"association_prior"`` everywhere it appears -- it is never presented as a
measured effect. The sample-size bound uses :func:`interp_lab.stats.t_critical`
(the same Student-t machinery as every CI this toolkit reports): the smallest
``n`` whose two-sided CI half-width at ``confidence`` is below the effect
estimate, i.e. ``t_critical(n - 1, confidence) * sd / sqrt(n) < |effect|``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from interp_lab.agent_actions import next_action
from interp_lab.explanation_reports import WrittenJsonMarkdown
from interp_lab.matching import SIGNED_EFFECT_DIRECTION_MIN, has_intervention_provenance, signed_effect_with_provenance
from interp_lab.schema import FeatureCard, InspectionReport, utc_now_iso
from interp_lab.stats import t_critical

EVIDENCE_PLAN_SCHEMA = "interp-lab.evidence_plan.v1"

# Smallest effect the validation layer treats as meaningful (kept equal to
# graph_validation.DEFAULT_MIN_EFFECT). Used as the last-resort planning effect
# when a card carries no measured effect, no association prior, and the caller
# supplied no assumed_effect -- labeled "default_minimum_effect", never measured.
DEFAULT_MINIMUM_EFFECT = 0.05

# Conservative default spread for directed effects when the card carries neither
# a sample stdev nor a recoverable CI. Criterion scores live in [0, 1], so
# directed effects live in [-1, 1]; 0.25 (a quarter of the score range) is wider
# than any observed toy/HF intervention spread in this repo's fixtures and makes
# the recommended n err on the large side rather than promising cheap certainty.
DEFAULT_EFFECT_SD = 0.25

# Cap for the iterative power solve. Past this many intervention prompts the
# effect is too small to be worth chasing with this tooling; the gap is flagged
# "effect_likely_too_small" instead of recommending an unbounded run.
MAX_RECOMMENDED_INTERVENTIONS = 256

# Control vocabulary from adapters/interventions.py CONTROL_TYPES: the two types
# the grading layer credits that a planner can always construct (a random
# unrelated feature, and a frequency-matched feature).
RECOMMENDED_CONTROL_TYPES = ("random_feature", "matched_frequency")

# |mean| / mean(|x|) over directed effects: 1.0 when every record agrees in
# sign, -> 0 when signs cancel. Below 0.5 most of the magnitude cancels, so the
# direction claim is unstable.
SIGN_ALIGNMENT_MIN = 0.5

# Minimum total control records recommended (2 per recommended control type),
# matching the spirit of graph_validation's --random-source-controls 4 default.
MIN_CONTROL_RECORDS = 4

_SAE_LATENT_PATTERN = re.compile(r"^SAE:(?:L\d+:)?F\d+$")

# Card-level claim ladder (distance to "validated"). sign_unstable sits with
# unsigned evidence: an unstable direction is no direction.
_GRADE_RANK = {
    "correlational_only": 0,
    "unsigned_causal_evidence": 1,
    "sign_unstable": 1,
    "underpowered": 2,
    "needs_controls": 3,
    "validated": 4,
}

_GAP_TO_GRADE = {
    "no_causal_evidence": "correlational_only",
    "no_signed_effect": "unsigned_causal_evidence",
    "sign_inconsistency": "sign_unstable",
    "insufficient_power": "underpowered",
    "no_controls": "needs_controls",
}

# Gap precedence: the most fundamental missing evidence defines the grade.
_GAP_ORDER = ("no_causal_evidence", "no_signed_effect", "sign_inconsistency", "insufficient_power", "no_controls")


def recommended_intervention_count(
    effect: float,
    sd: float,
    *,
    confidence: float = 0.95,
    max_n: int = MAX_RECOMMENDED_INTERVENTIONS,
) -> tuple[int, bool]:
    """Smallest n whose two-sided Student-t CI at ``confidence`` excludes zero.

    Solves iteratively for the smallest ``n >= 2`` with
    ``t_critical(n - 1, confidence) * sd / sqrt(n) < |effect|``. Returns
    ``(n, capped)``; when even ``max_n`` records cannot exclude zero the count is
    capped there and ``capped`` is True (callers flag ``effect_likely_too_small``).
    """
    if effect <= 0.0:
        raise ValueError("effect must be positive for a power calculation")
    if sd < 0.0:
        raise ValueError("sd must be non-negative")
    if sd == 0.0:
        return 2, False  # any CI excludes zero with no spread; 2 is the t minimum
    for n in range(2, max_n + 1):
        if t_critical(n - 1, confidence) * sd / math.sqrt(n) < effect:
            return n, False
    return max_n, True


def build_evidence_plan(
    report: InspectionReport | dict,
    *,
    top_k: int | None = None,
    confidence: float = 0.95,
    target_grade: str = "validated",
    assumed_effect: float | None = None,
    generated_for_report: str | None = None,
) -> dict[str, Any]:
    """Diagnose evidence gaps per card and rank the cheapest grade-moving runs.

    ``assumed_effect`` (positive) overrides the association prior and the
    default minimum effect for cards with no measured signed effect; it never
    overrides a measured one. ``target_grade`` must be one of the claim-ladder
    grades (default ``"validated"``).
    """
    if isinstance(report, dict):
        report = InspectionReport.from_dict(report)
    if target_grade not in _GRADE_RANK:
        known = ", ".join(sorted(_GRADE_RANK))
        raise ValueError(f"unknown target_grade {target_grade!r} (known: {known})")
    if assumed_effect is not None and assumed_effect <= 0.0:
        raise ValueError("assumed_effect must be positive when provided")
    cards = report.cards[:top_k] if top_k is not None else list(report.cards)
    entries = [
        _plan_entry(
            card,
            criterion=report.criterion.text,
            confidence=confidence,
            target_grade=target_grade,
            assumed_effect=assumed_effect,
        )
        for card in cards
    ]
    entries.sort(key=lambda entry: (-entry["priority"], entry["feature_id"]))
    with_gaps = [entry for entry in entries if entry["gaps"]]
    total_prompts = sum(entry["estimated_total_prompts"] for entry in entries)
    return {
        "schema_version": EVIDENCE_PLAN_SCHEMA,
        "created_at": utc_now_iso(),
        "model": report.model,
        "criterion": report.criterion.text,
        "generated_for_report": generated_for_report,
        "confidence": float(confidence),
        "target_grade": target_grade,
        "priority_heuristic": (
            "claim-grade steps to target * (0.5 + 0.5 * importance) / estimated_total_prompts; "
            "0 when no gaps remain"
        ),
        "summary": {
            "cards_assessed": len(entries),
            "cards_with_gaps": len(with_gaps),
            "total_recommended_prompts": total_prompts,
            "highest_priority_feature": with_gaps[0]["feature_id"] if with_gaps else None,
        },
        "plan": entries,
        "agent_next_actions": list(entries[0]["next_actions"]) if entries else [],
    }


def export_evidence_plan(
    report_or_path: InspectionReport | dict | str | Path,
    out: str | Path | None = None,
    markdown_out: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any] | WrittenJsonMarkdown:
    """Build an evidence plan and (optionally) write JSON + Markdown.

    Mirrors ``run_diff``'s writer: with ``out`` set the plan JSON and a sibling
    (or explicit) Markdown summary are written and a ``WrittenJsonMarkdown`` is
    returned; with only ``markdown_out`` the Markdown is still written and the
    plan dict is returned; with neither, the plan dict is returned.
    """
    if isinstance(report_or_path, (str, Path)):
        from interp_lab.reporting import load_inspection_report

        source_path = Path(report_or_path)
        report: InspectionReport | dict = load_inspection_report(source_path)
        kwargs.setdefault("generated_for_report", str(source_path))
    else:
        report = report_or_path
    plan = build_evidence_plan(report, **kwargs)
    if out is None:
        if markdown_out is not None:
            markdown_path = Path(markdown_out)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(render_evidence_plan_markdown(plan), encoding="utf-8")
        return plan
    json_path = Path(out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = Path(markdown_out) if markdown_out is not None else json_path.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_evidence_plan_markdown(plan), encoding="utf-8")
    return WrittenJsonMarkdown(report=plan, json_path=json_path, markdown_path=markdown_path)


def render_evidence_plan_markdown(plan: dict[str, Any]) -> str:
    summary = plan["summary"]
    lines = [
        "# interp-lab Evidence Plan",
        "",
        f"Model: `{plan['model']}`",
        f"Criterion: {plan['criterion']}",
        f"Target grade: `{plan['target_grade']}` at confidence `{plan['confidence']}`",
        f"Cards assessed: `{summary['cards_assessed']}`  |  with gaps: `{summary['cards_with_gaps']}`"
        f"  |  total recommended prompts: `{summary['total_recommended_prompts']}`",
        "",
        "| Feature | Label | Grade est. | Gaps | Rec. n | Prompts | Priority |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for entry in plan["plan"]:
        gaps = ", ".join(gap["gap"] for gap in entry["gaps"]) or "none"
        rec_n = entry.get("recommended_intervention_count")
        lines.append(
            f"| `{entry['feature_id']}` | {entry['label']} | {entry['current_grade_estimate']} "
            f"| {gaps} | {rec_n if rec_n is not None else ''} "
            f"| {entry['estimated_total_prompts']} | {entry['priority']:.4f} |"
        )
    lines.extend(["", "## Notes", ""])
    for entry in plan["plan"]:
        for gap in entry["gaps"]:
            note = f"- `{entry['feature_id']}` [{gap['gap']}]: {gap['detail']}"
            if gap.get("effect_size_source") == "association_prior":
                note += " (effect size is a correlational PRIOR, not a measurement)"
            if gap.get("effect_likely_too_small"):
                note += " (effect likely too small to power at this confidence)"
            lines.append(note)
    lines.extend(["", "## Commands", ""])
    for entry in plan["plan"]:
        if not entry["next_actions"]:
            continue
        lines.append(f"### `{entry['feature_id']}` — {entry['label']}")
        lines.append("")
        for action in entry["next_actions"]:
            if "command" in action:
                lines.append(f"- `{action['id']}`: {action['title']}: `{action['command']}`")
            else:
                lines.append(f"- `{action['id']}`: {action['title']}: {action['instruction']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_plan_evidence_parser() -> argparse.ArgumentParser:
    # Default add_help=True so `plan-evidence --help` works; the cli subparser
    # adopting this as a parent passes add_help=False (the compare-runs pattern).
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="Inspection report.json to plan evidence for.")
    parser.add_argument("--out", help="Output plan JSON path (also writes a sibling .md). Omit to print JSON.")
    parser.add_argument("--markdown-out", help="Optional explicit markdown path.")
    parser.add_argument("--top-k", type=int, help="Plan only the top K report cards.")
    parser.add_argument("--confidence", type=float, default=0.95, help="CI confidence for the power analysis.")
    parser.add_argument(
        "--assumed-effect",
        type=float,
        help="Optional positive prior effect size for cards with no measured signed effect.",
    )
    parser.add_argument("--json", action="store_true", help="Print the plan as JSON.")
    return parser


def run_plan_evidence_from_args(args: argparse.Namespace) -> dict[str, Any] | WrittenJsonMarkdown:
    return export_evidence_plan(
        args.report,
        out=args.out,
        markdown_out=args.markdown_out,
        top_k=args.top_k,
        confidence=args.confidence,
        assumed_effect=args.assumed_effect,
    )


def _plan_entry(
    card: FeatureCard,
    *,
    criterion: str,
    confidence: float,
    target_grade: str,
    assumed_effect: float | None,
) -> dict[str, Any]:
    facts = _card_facts(card)
    effect, effect_source = _effect_estimate(card, facts, assumed_effect=assumed_effect)
    sd, sd_source = _sd_estimate(facts, confidence=confidence)
    rec_n, capped = recommended_intervention_count(effect, sd, confidence=confidence)
    power = {
        "recommended_intervention_count": rec_n,
        "current_intervention_count": facts["intervention_count"],
        "additional_interventions_needed": max(0, rec_n - facts["intervention_count"]),
        "effect_size": round(effect, 6),
        "effect_size_source": effect_source,
        "effect_sd": round(sd, 6),
        "effect_sd_source": sd_source,
        "confidence": float(confidence),
        "effect_likely_too_small": capped,
    }
    gaps = _diagnose_gaps(card, facts, power=power)
    grade = _grade_estimate(gaps)
    sign_known = facts["signed_effect"] is not None and abs(facts["signed_effect"]) >= SIGNED_EFFECT_DIRECTION_MIN
    modes = _intervention_modes(facts["signed_effect"], sign_known)
    intervention_prompts, control_prompts = _prompt_budget(gaps, power=power, modes=modes)
    total_prompts = intervention_prompts + control_prompts
    priority = _priority(grade, target_grade, importance=card.importance, total_prompts=total_prompts)
    actions = _entry_next_actions(
        card,
        criterion=criterion,
        gaps=gaps,
        modes=modes,
        recommended_n=rec_n,
    )
    return {
        "feature_id": card.feature_id,
        "label": card.label,
        "current_grade_estimate": grade,
        "gaps": gaps,
        "priority": priority,
        "recommended_intervention_count": rec_n if any(g["gap"] != "no_controls" for g in gaps) else None,
        "estimated_intervention_prompts": intervention_prompts,
        "estimated_control_prompts": control_prompts,
        "estimated_total_prompts": total_prompts,
        "next_actions": actions,
    }


def _card_facts(card: FeatureCard) -> dict[str, Any]:
    causal = card.causal_effects
    meta = card.metadata.get("interventions")
    meta = meta if isinstance(meta, dict) else {}
    intervention_count = int(
        float(causal.get("intervention_record_count", 0.0) or 0.0)
        or float(meta.get("count", 0) or 0)
    )
    control_count = int(
        float(causal.get("control_record_count", 0.0) or 0.0)
        or float(_dig(meta, "controls", "count") or 0)
    )
    signed, provenance = signed_effect_with_provenance(causal, card.metadata)
    stdev = _optional_float(meta.get("stdev_directed_effect"))
    ci_low = _optional_float(causal.get("criterion_ci_low") if "criterion_ci_low" in causal else meta.get("criterion_ci_low"))
    ci_high = _optional_float(causal.get("criterion_ci_high") if "criterion_ci_high" in causal else meta.get("criterion_ci_high"))
    ci_n = meta.get("criterion_ci_n", causal.get("criterion_ci_n"))
    mean_directed = _optional_float(meta.get("mean_directed_effect"))
    mean_abs_directed = _optional_float(meta.get("mean_abs_directed_effect"))
    return {
        "has_interventions": has_intervention_provenance(causal, card.metadata),
        "has_signed_causal": "signed_causal_effect" in causal,
        "signed_effect": signed,
        "signed_provenance": provenance,
        "intervention_count": intervention_count,
        "control_count": control_count,
        "stdev_directed_effect": stdev,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_n": int(ci_n) if isinstance(ci_n, (int, float)) and ci_n else 0,
        "mean_directed_effect": mean_directed,
        "mean_abs_directed_effect": mean_abs_directed,
    }


def _effect_estimate(
    card: FeatureCard,
    facts: dict[str, Any],
    *,
    assumed_effect: float | None,
) -> tuple[float, str]:
    """Effect-size estimate for the power solve, with honest provenance.

    Precedence: measured intervention effect > caller-supplied assumed effect >
    correlational association prior > documented default minimum effect. The
    association prior is ALWAYS labeled ``association_prior``; it never appears
    as measured.
    """
    if facts["signed_provenance"] == "intervention" and facts["signed_effect"]:
        return abs(facts["signed_effect"]), "measured"
    if assumed_effect is not None:
        return float(assumed_effect), "assumed"
    if facts["signed_provenance"] == "association" and facts["signed_effect"]:
        return abs(facts["signed_effect"]), "association_prior"
    return DEFAULT_MINIMUM_EFFECT, "default_minimum_effect"


def _sd_estimate(facts: dict[str, Any], *, confidence: float) -> tuple[float, str]:
    """Spread estimate: sample stdev > stdev recovered from a reported CI >
    the documented conservative default (:data:`DEFAULT_EFFECT_SD`)."""
    stdev = facts["stdev_directed_effect"]
    if stdev is not None and stdev > 0.0 and facts["intervention_count"] >= 2:
        return float(stdev), "measured"
    ci_low, ci_high, ci_n = facts["ci_low"], facts["ci_high"], facts["ci_n"]
    if ci_low is not None and ci_high is not None and ci_n >= 2:
        # Invert half_width = t * sd / sqrt(n). Card CIs are written at 0.95.
        half_width = (float(ci_high) - float(ci_low)) / 2.0
        if half_width > 0.0:
            sd = half_width * math.sqrt(ci_n) / t_critical(ci_n - 1, 0.95)
            return round(sd, 6), "derived_from_ci"
    return DEFAULT_EFFECT_SD, "conservative_default"


def _diagnose_gaps(
    card: FeatureCard,
    facts: dict[str, Any],
    *,
    power: dict[str, Any],
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    if not facts["has_interventions"]:
        gaps.append(
            {
                "gap": "no_causal_evidence",
                "detail": (
                    "No intervention records back this card; every causal-looking number on it "
                    "is correlational. Run suppress/amplify interventions before any causal claim."
                ),
                **power,
            }
        )
        # Without interventions the downstream gaps (signed effect, power,
        # controls, sign stability) are all subsumed by this one.
        return gaps
    if not facts["has_signed_causal"]:
        gaps.append(
            {
                "gap": "no_signed_effect",
                "detail": (
                    "Intervention provenance is present but no signed_causal_effect was recorded, "
                    "so the direction of the effect is unestablished."
                ),
                **power,
            }
        )
    else:
        sign_alignment = _sign_alignment(facts)
        if sign_alignment is not None and sign_alignment < SIGN_ALIGNMENT_MIN:
            gaps.append(
                {
                    "gap": "sign_inconsistency",
                    "detail": (
                        "Directed effects largely cancel across records "
                        f"(|mean|/mean|x| = {sign_alignment:.3f} < {SIGN_ALIGNMENT_MIN}); the sign of the "
                        "effect is unstable. Rerun both modes with the recommended sample size."
                    ),
                    "sign_alignment": round(sign_alignment, 6),
                    **power,
                }
            )
        if _is_underpowered(facts, power):
            gaps.append(
                {
                    "gap": "insufficient_power",
                    "detail": (
                        f"{facts['intervention_count']} intervention record(s) cannot exclude zero at "
                        f"confidence {power['confidence']}; "
                        f"{power['recommended_intervention_count']} are recommended."
                    ),
                    **power,
                }
            )
    if facts["control_count"] == 0:
        gaps.append(
            {
                "gap": "no_controls",
                "detail": (
                    "No control intervention records; the effect cannot be separated from generic "
                    "disruption. Add " + " and ".join(RECOMMENDED_CONTROL_TYPES) + " control rows "
                    "(metadata.control_type) to the intervention JSONL."
                ),
                "recommended_control_types": list(RECOMMENDED_CONTROL_TYPES),
                "recommended_control_count": _control_budget(power),
            }
        )
    return gaps


def _is_underpowered(facts: dict[str, Any], power: dict[str, Any]) -> bool:
    if facts["intervention_count"] < power["recommended_intervention_count"]:
        return True
    ci_low, ci_high = facts["ci_low"], facts["ci_high"]
    if ci_low is not None and ci_high is not None:
        return ci_low <= 0.0 <= ci_high  # the measured CI itself fails to exclude zero
    return False


def _sign_alignment(facts: dict[str, Any]) -> float | None:
    """|mean| / mean(|x|) over directed effects -- detectable only when the card
    carries both summary moments and at least two records."""
    mean_directed = facts["mean_directed_effect"]
    mean_abs = facts["mean_abs_directed_effect"]
    if mean_directed is None or mean_abs is None or facts["intervention_count"] < 2:
        return None
    if mean_abs <= 1e-12:
        return None
    return abs(mean_directed) / mean_abs


def _grade_estimate(gaps: list[dict[str, Any]]) -> str:
    present = {gap["gap"] for gap in gaps}
    for gap_name in _GAP_ORDER:
        if gap_name in present:
            return _GAP_TO_GRADE[gap_name]
    return "validated"


def _priority(grade: str, target_grade: str, *, importance: float, total_prompts: int) -> float:
    """Expected grade movement per unit cost.

    ``grade_gap`` counts claim-ladder steps from the current estimate to the
    target; cost is the estimated prompt budget. Importance scales the value of
    moving this card's grade (0.5 baseline so unimportant cards still surface),
    so: priority = grade_gap * (0.5 + 0.5 * clamp01(importance)) / max(prompts, 1).
    """
    gap_steps = max(0, _GRADE_RANK[target_grade] - _GRADE_RANK[grade])
    if gap_steps == 0:
        return 0.0
    weight = 0.5 + 0.5 * min(1.0, max(0.0, float(importance)))
    return round(gap_steps * weight / max(total_prompts, 1), 6)


def _prompt_budget(
    gaps: list[dict[str, Any]],
    *,
    power: dict[str, Any],
    modes: list[str],
) -> tuple[int, int]:
    """(intervention_prompts, control_prompts). One record ~= one scored prompt.

    Measurement gaps need the full recommended n per mode when no usable signed
    measurement exists (no_causal_evidence / no_signed_effect / sign_inconsistency),
    or only the shortfall when the measurement is merely underpowered.
    """
    present = {gap["gap"] for gap in gaps}
    rec_n = power["recommended_intervention_count"]
    intervention_prompts = 0
    if present & {"no_causal_evidence", "no_signed_effect", "sign_inconsistency"}:
        intervention_prompts = rec_n * len(modes)
    elif "insufficient_power" in present:
        intervention_prompts = power["additional_interventions_needed"]
    control_prompts = _control_budget(power) if "no_controls" in present else 0
    return intervention_prompts, control_prompts


def _control_budget(power: dict[str, Any]) -> int:
    """Control coverage of the same order as the effect arm, split across the two
    recommended control types, never below :data:`MIN_CONTROL_RECORDS`."""
    return max(MIN_CONTROL_RECORDS, power["recommended_intervention_count"])


def _intervention_modes(signed_effect: float | None, sign_known: bool) -> list[str]:
    if not sign_known:
        return ["suppress", "amplify"]  # direction unknown: probe both ways
    if signed_effect is not None and signed_effect < 0.0:
        return ["amplify"]  # criterion-opposing feature: push it harder
    return ["suppress"]  # criterion-promoting feature: knock it out


def _entry_next_actions(
    card: FeatureCard,
    *,
    criterion: str,
    gaps: list[dict[str, Any]],
    modes: list[str],
    recommended_n: int,
) -> list[dict[str, Any]]:
    if not gaps:
        return [
            next_action(
                action_id="replicate_validated_feature",
                title="No evidence gaps: replicate the measured effect on held-out prompts",
                instruction=(
                    "This card already meets the evidence bar; replicate the intervention effect on a "
                    "held-out prompt set before publishing the claim."
                ),
            )
        ]
    present = {gap["gap"] for gap in gaps}
    actions: list[dict[str, Any]] = []
    is_sae = bool(_SAE_LATENT_PATTERN.match(card.feature_id))
    requires = [f"scored causal prompt JSONL (~{recommended_n} prompts)"]
    if is_sae:
        requires.append("matching interp-lab SAE artifact")
    if present & {"no_causal_evidence", "no_signed_effect", "sign_inconsistency", "insufficient_power"}:
        for mode in modes:
            argv = [
                "interp-lab",
                "intervene",
                "--model",
                card.model,
                "--criterion",
                criterion,
                "--dataset",
                "<causal-prompts.jsonl>",
                "--feature",
                card.feature_id,
            ]
            if is_sae:
                argv.extend(["--sae", "<sae.json>"])
            argv.extend(
                [
                    "--mode",
                    mode,
                    "--target-token",
                    "auto",
                    "--out",
                    "<interventions.jsonl>",
                    "--plan-out",
                    "<intervention-plan.json>",
                    "--dry-run",
                    "--json",
                ]
            )
            actions.append(
                next_action(
                    action_id=f"plan_{mode}_intervention",
                    title=f"Plan a {mode} test for {card.feature_id} (~{recommended_n} prompts recommended)",
                    argv=argv,
                    requires=list(requires),
                )
            )
    if "no_controls" in present:
        # The intervene CLI has no control flags; controls are extra rows in the
        # intervention JSONL tagged via metadata.control_type, so this is prose.
        actions.append(
            next_action(
                action_id="add_control_interventions",
                title="Add control intervention records before claiming specificity",
                instruction=(
                    "Append control rows to the intervention JSONL with metadata.control_type set to "
                    f"{RECOMMENDED_CONTROL_TYPES[0]!r} (a random unrelated feature) and "
                    f"{RECOMMENDED_CONTROL_TYPES[1]!r} (an activation-frequency-matched feature), then "
                    "re-run inspect with --interventions so control separation is graded."
                ),
                requires=["intervention records JSONL"],
            )
        )
    return actions


def _dig(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
