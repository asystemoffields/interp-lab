"""Tests for `plan_evidence` / evidence_planner: gap diagnosis, power math, priorities."""

import json
import re
import shlex
from pathlib import Path

import pytest

from interp_lab.cli import build_parser
from interp_lab.evidence_planner import (
    DEFAULT_EFFECT_SD,
    EVIDENCE_PLAN_SCHEMA,
    MAX_RECOMMENDED_INTERVENTIONS,
    build_evidence_plan,
    build_plan_evidence_parser,
    export_evidence_plan,
    recommended_intervention_count,
    render_evidence_plan_markdown,
)
from interp_lab.schema import Criterion, FeatureCard, FeatureFingerprint, InspectionReport
from interp_lab.stats import t_critical

CRITERION = "the model is aware it is being evaluated"
PLACEHOLDER = re.compile(r"^<.+>$")


def _fingerprint(feature_id: str, model: str = "toy/m", provenance: str = "none") -> FeatureFingerprint:
    return FeatureFingerprint(
        feature_id=feature_id,
        model=model,
        layer=3,
        text="evaluation awareness",
        text_vector=[0.5, 0.5],
        activation_signature=[1.0, 0.0],
        decoder_signature=[0.0, 1.0],
        causal_vector=[],
        causal_provenance=provenance,
    )


def _card(
    feature_id: str,
    *,
    causal_effects: dict | None = None,
    metadata: dict | None = None,
    importance: float = 0.5,
    label: str = "test feature",
    model: str = "toy/m",
) -> FeatureCard:
    return FeatureCard(
        feature_id=feature_id,
        model=model,
        layer=3,
        label=label,
        explanation="",
        importance=importance,
        association=0.4,
        specificity=0.1,
        causal_effect=0.0,
        stability=0.5,
        examples=[],
        source="test",
        fingerprint=_fingerprint(feature_id, model),
        metadata=metadata or {},
        causal_effects=causal_effects or {},
    )


def _report(cards: list[FeatureCard], model: str = "toy/m") -> InspectionReport:
    return InspectionReport(model=model, criterion=Criterion(text=CRITERION), cards=cards)


def _validated_card(feature_id: str = "L3:D1", **kwargs) -> FeatureCard:
    """Measured signed effect, powered (n=12 >> required 3), controls, stable sign."""
    return _card(
        feature_id,
        causal_effects={
            "signed_causal_effect": 0.3,
            "intervention_record_count": 12.0,
            "control_record_count": 4.0,
            "criterion_ci_low": 0.2,
            "criterion_ci_high": 0.4,
        },
        metadata={
            "interventions": {
                "count": 12,
                "mean_directed_effect": 0.3,
                "mean_abs_directed_effect": 0.3,
                "stdev_directed_effect": 0.1,
                "controls": {"count": 4},
            }
        },
        **kwargs,
    )


def _underpowered_card(feature_id: str = "L3:D2", **kwargs) -> FeatureCard:
    """effect=0.2, sd=0.2, n=3: hand-computed required n is 7 (see power test)."""
    return _card(
        feature_id,
        causal_effects={
            "signed_causal_effect": 0.2,
            "intervention_record_count": 3.0,
            "control_record_count": 4.0,
        },
        metadata={
            "interventions": {
                "count": 3,
                "mean_directed_effect": 0.2,
                "mean_abs_directed_effect": 0.2,
                "stdev_directed_effect": 0.2,
                "controls": {"count": 4},
            }
        },
        **kwargs,
    )


def _association_only_card(feature_id: str = "L3:D3", signed: float = 0.4, **kwargs) -> FeatureCard:
    return _card(feature_id, causal_effects={"signed_association": signed}, **kwargs)


# ---------------------------------------------------------------- power math


def test_power_math_matches_hand_computed_t_bound():
    # effect=0.2, sd=0.2 at 0.95: need t(n-1) < sqrt(n).
    # n=6: t(5)=2.571 > 2.449; n=7: t(6)=2.447 < 2.646 -> 7.
    assert recommended_intervention_count(0.2, 0.2) == (7, False)
    # effect=0.5, sd=0.1: need t(n-1) < 5*sqrt(n). n=2: 12.706 > 7.07; n=3: 4.303 < 8.66.
    assert recommended_intervention_count(0.5, 0.1) == (3, False)
    # Zero spread: the t-minimum of two records suffices.
    assert recommended_intervention_count(0.3, 0.0) == (2, False)
    # The returned n satisfies the bound and n-1 does not.
    n, capped = recommended_intervention_count(0.15, 0.3, confidence=0.99)
    assert not capped
    assert t_critical(n - 1, 0.99) * 0.3 / n**0.5 < 0.15
    assert t_critical(n - 2, 0.99) * 0.3 / (n - 1) ** 0.5 >= 0.15


def test_power_math_caps_and_flags_tiny_effects():
    assert recommended_intervention_count(0.001, 0.25) == (MAX_RECOMMENDED_INTERVENTIONS, True)
    with pytest.raises(ValueError):
        recommended_intervention_count(0.0, 0.2)
    with pytest.raises(ValueError):
        recommended_intervention_count(0.1, -0.1)


# ---------------------------------------------------------------- gap diagnosis


def test_no_causal_evidence_gap_uses_association_prior():
    plan = build_evidence_plan(_report([_association_only_card()]))
    assert plan["schema_version"] == EVIDENCE_PLAN_SCHEMA
    (entry,) = plan["plan"]
    assert entry["current_grade_estimate"] == "correlational_only"
    (gap,) = entry["gaps"]
    assert gap["gap"] == "no_causal_evidence"
    # |signed_association|=0.4 is a PRIOR, labeled as such, never "measured".
    assert gap["effect_size_source"] == "association_prior"
    assert gap["effect_size"] == 0.4
    assert gap["effect_sd_source"] == "conservative_default"
    assert gap["effect_sd"] == DEFAULT_EFFECT_SD
    # Hand-computed with sd=0.25: t(n-1) < 1.6*sqrt(n); n=3: 4.303 >= 2.77, n=4: 3.182 < 3.2.
    assert gap["recommended_intervention_count"] == 4
    # Positive association prior -> sign treated as known -> suppress only.
    modes = [action["argv"][action["argv"].index("--mode") + 1] for action in entry["next_actions"] if "argv" in action]
    assert modes == ["suppress"]


def test_no_signed_effect_gap_when_interventions_lack_direction():
    card = _card("L3:D4", metadata={"interventions": {"count": 5}})
    plan = build_evidence_plan(_report([card]))
    (entry,) = plan["plan"]
    gap_names = [gap["gap"] for gap in entry["gaps"]]
    assert "no_signed_effect" in gap_names
    assert "no_controls" in gap_names  # no control rows either
    assert entry["current_grade_estimate"] == "unsigned_causal_evidence"
    # Direction unknown -> both suppress and amplify probes are planned.
    modes = {action["argv"][action["argv"].index("--mode") + 1] for action in entry["next_actions"] if "argv" in action}
    assert modes == {"suppress", "amplify"}


def test_insufficient_power_gap_recommends_hand_computed_n():
    plan = build_evidence_plan(_report([_underpowered_card()]))
    (entry,) = plan["plan"]
    assert entry["current_grade_estimate"] == "underpowered"
    (gap,) = entry["gaps"]
    assert gap["gap"] == "insufficient_power"
    assert gap["effect_size_source"] == "measured"
    assert gap["effect_sd_source"] == "measured"
    assert gap["recommended_intervention_count"] == 7
    assert gap["additional_interventions_needed"] == 4
    assert entry["estimated_total_prompts"] == 4  # only the shortfall; controls exist


def test_no_controls_gap_recommends_both_control_types():
    card = _card(
        "L3:D5",
        causal_effects={
            "signed_causal_effect": 0.3,
            "intervention_record_count": 12.0,
            "criterion_ci_low": 0.2,
            "criterion_ci_high": 0.4,
        },
        metadata={
            "interventions": {
                "count": 12,
                "mean_directed_effect": 0.3,
                "mean_abs_directed_effect": 0.3,
                "stdev_directed_effect": 0.1,
                "controls": {"count": 0},
            }
        },
    )
    plan = build_evidence_plan(_report([card]))
    (entry,) = plan["plan"]
    assert entry["current_grade_estimate"] == "needs_controls"
    (gap,) = entry["gaps"]
    assert gap["gap"] == "no_controls"
    assert gap["recommended_control_types"] == ["random_feature", "matched_frequency"]
    assert gap["recommended_control_count"] >= 4
    # No measurement gap -> no intervention argv, just the controls instruction.
    (action,) = entry["next_actions"]
    assert action["id"] == "add_control_interventions"
    assert "instruction" in action and "argv" not in action
    assert "random_feature" in action["instruction"] and "matched_frequency" in action["instruction"]


def test_sign_inconsistency_detected_from_cancelling_directed_effects():
    card = _card(
        "L3:D6",
        causal_effects={
            "signed_causal_effect": 0.01,
            "intervention_record_count": 8.0,
            "control_record_count": 4.0,
        },
        metadata={
            "interventions": {
                "count": 8,
                "mean_directed_effect": 0.01,
                "mean_abs_directed_effect": 0.2,
                "stdev_directed_effect": 0.25,
                "controls": {"count": 4},
            }
        },
    )
    plan = build_evidence_plan(_report([card]))
    (entry,) = plan["plan"]
    gap_names = [gap["gap"] for gap in entry["gaps"]]
    assert "sign_inconsistency" in gap_names
    assert entry["current_grade_estimate"] == "sign_unstable"
    sign_gap = next(gap for gap in entry["gaps"] if gap["gap"] == "sign_inconsistency")
    assert sign_gap["sign_alignment"] == pytest.approx(0.05)
    # |effect|=0.01 with sd=0.25 cannot be powered within the cap -> flagged.
    assert sign_gap["effect_likely_too_small"] is True
    assert sign_gap["recommended_intervention_count"] == MAX_RECOMMENDED_INTERVENTIONS
    # Sign unknown (|0.01| below the direction floor) -> both modes planned.
    modes = {action["argv"][action["argv"].index("--mode") + 1] for action in entry["next_actions"] if "argv" in action}
    assert modes == {"suppress", "amplify"}


def test_validated_card_has_no_gaps_and_zero_priority():
    plan = build_evidence_plan(_report([_validated_card()]))
    (entry,) = plan["plan"]
    assert entry["gaps"] == []
    assert entry["current_grade_estimate"] == "validated"
    assert entry["priority"] == 0.0
    assert entry["estimated_total_prompts"] == 0
    assert entry["recommended_intervention_count"] is None
    (action,) = entry["next_actions"]
    assert action["id"] == "replicate_validated_feature"
    assert "instruction" in action
    assert plan["summary"]["cards_with_gaps"] == 0
    assert plan["summary"]["highest_priority_feature"] is None


# ---------------------------------------------------------------- prioritization


def test_plan_sorted_by_priority_and_summary_consistent():
    cards = [
        _validated_card("L3:D1"),
        _underpowered_card("L3:D2"),  # 2 grade steps for ~4 prompts
        _association_only_card("L3:D3"),  # 4 grade steps for ~4 prompts -> highest
    ]
    plan = build_evidence_plan(_report(cards))
    priorities = [entry["priority"] for entry in plan["plan"]]
    assert priorities == sorted(priorities, reverse=True)
    assert plan["plan"][0]["feature_id"] == "L3:D3"
    assert plan["plan"][-1]["feature_id"] == "L3:D1"
    assert plan["summary"]["highest_priority_feature"] == "L3:D3"
    assert plan["summary"]["cards_assessed"] == 3
    assert plan["summary"]["cards_with_gaps"] == 2
    assert plan["summary"]["total_recommended_prompts"] == sum(
        entry["estimated_total_prompts"] for entry in plan["plan"]
    )
    # The standard surface convention: top entry's actions are surfaced at top level.
    assert plan["agent_next_actions"] == plan["plan"][0]["next_actions"]


def test_importance_breaks_ties_in_priority():
    cheap_important = _association_only_card("L3:D10", importance=0.9)
    cheap_unimportant = _association_only_card("L3:D11", importance=0.1)
    plan = build_evidence_plan(_report([cheap_unimportant, cheap_important]))
    assert plan["plan"][0]["feature_id"] == "L3:D10"
    assert plan["plan"][0]["priority"] > plan["plan"][1]["priority"]


def test_assumed_effect_overrides_prior_but_never_measured():
    assoc = _association_only_card("L3:D12", signed=0.4)
    measured = _underpowered_card("L3:D13")
    plan = build_evidence_plan(_report([assoc, measured]), assumed_effect=0.5)
    by_id = {entry["feature_id"]: entry for entry in plan["plan"]}
    assoc_gap = by_id["L3:D12"]["gaps"][0]
    assert assoc_gap["effect_size_source"] == "assumed"
    assert assoc_gap["effect_size"] == 0.5
    measured_gap = by_id["L3:D13"]["gaps"][0]
    assert measured_gap["effect_size_source"] == "measured"
    assert measured_gap["effect_size"] == 0.2
    with pytest.raises(ValueError):
        build_evidence_plan(_report([assoc]), assumed_effect=-0.1)
    with pytest.raises(ValueError):
        build_evidence_plan(_report([assoc]), target_grade="amazing")


def test_top_k_limits_assessed_cards():
    cards = [_association_only_card(f"L3:D{i}") for i in range(5)]
    plan = build_evidence_plan(_report(cards), top_k=2)
    assert plan["summary"]["cards_assessed"] == 2


# ---------------------------------------------------------------- runnable actions


def _assert_parses(parser, argv, context):
    filled = ["dummy-path" if PLACEHOLDER.match(str(token)) else str(token) for token in argv]
    if filled and filled[0] == "interp-lab":
        filled = filled[1:]
    try:
        parser.parse_args(filled)
    except SystemExit as exc:  # pragma: no cover - assertion path
        pytest.fail(f"agent action argv does not parse ({context}): {argv} -> exit code {exc.code}")


def test_every_emitted_next_action_is_canonical_and_parses_against_cli():
    cards = [
        _validated_card("L3:D1"),
        _underpowered_card("L3:D2"),
        _association_only_card("L3:D3"),
        _card("L3:D4", metadata={"interventions": {"count": 5}}),
        _card("SAE:L3:F7", causal_effects={"signed_association": 0.3}),
    ]
    plan = build_evidence_plan(_report(cards))
    parser = build_parser()
    seen = 0
    for entry in plan["plan"]:
        for action in entry["next_actions"]:
            seen += 1
            assert action["id"] and action["title"]
            assert ("command" in action) != ("instruction" in action)
            if "command" in action:
                argv = action["argv"]
                assert action["command"] == " ".join(shlex.quote(item) for item in argv)
                assert argv[:2] == ["interp-lab", "intervene"]
                assert "--model" in argv and "--criterion" in argv and "--feature" in argv
                assert argv[argv.index("--model") + 1] == "toy/m"
                assert argv[argv.index("--criterion") + 1] == CRITERION
                assert "--dry-run" in argv and "--json" in argv
                _assert_parses(parser, argv, f"{entry['feature_id']} [{action['id']}]")
    assert seen > 0
    for action in plan["agent_next_actions"]:
        if "command" in action:
            _assert_parses(parser, action["argv"], f"top-level [{action['id']}]")


def test_sae_latents_get_sae_flag_and_negative_effects_get_amplify():
    sae = _card("SAE:L3:F7", causal_effects={"signed_association": 0.3})
    negative = _card(
        "L3:D9",
        causal_effects={
            "signed_causal_effect": -0.3,
            "intervention_record_count": 2.0,
            "control_record_count": 4.0,
        },
        metadata={
            "interventions": {
                "count": 2,
                "mean_directed_effect": -0.3,
                "mean_abs_directed_effect": 0.3,
                "stdev_directed_effect": 0.2,
                "controls": {"count": 4},
            }
        },
    )
    plan = build_evidence_plan(_report([sae, negative]))
    by_id = {entry["feature_id"]: entry for entry in plan["plan"]}
    sae_argvs = [action["argv"] for action in by_id["SAE:L3:F7"]["next_actions"] if "argv" in action]
    assert sae_argvs and all("--sae" in argv for argv in sae_argvs)
    neg_modes = [
        action["argv"][action["argv"].index("--mode") + 1]
        for action in by_id["L3:D9"]["next_actions"]
        if "argv" in action
    ]
    assert neg_modes == ["amplify"]  # criterion-opposing feature: amplify probe


# ---------------------------------------------------------------- writers


def test_export_writes_json_and_markdown_round_trip(tmp_path: Path):
    report = _report([_underpowered_card(), _association_only_card()])
    result = export_evidence_plan(report, out=tmp_path / "plan.json")
    assert result.json_path.exists() and result.markdown_path.exists()
    reloaded = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert reloaded == result.report
    assert reloaded["schema_version"] == EVIDENCE_PLAN_SCHEMA
    text = result.markdown_path.read_text(encoding="utf-8")
    assert "# interp-lab Evidence Plan" in text
    assert "| Feature | Label | Grade est. | Gaps | Rec. n | Prompts | Priority |" in text
    assert "interp-lab intervene" in text
    # Association-derived priors are labeled in prose too.
    assert "PRIOR" in text


def test_markdown_out_alone_writes_markdown_and_returns_dict(tmp_path: Path):
    report = _report([_association_only_card()])
    markdown = tmp_path / "plan.md"
    result = export_evidence_plan(report, markdown_out=markdown)
    assert isinstance(result, dict)
    assert result["schema_version"] == EVIDENCE_PLAN_SCHEMA
    assert "# interp-lab Evidence Plan" in markdown.read_text(encoding="utf-8")


def test_export_accepts_report_path_and_records_provenance(tmp_path: Path):
    report = _report([_underpowered_card()])
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    plan = export_evidence_plan(report_path)
    assert plan["generated_for_report"] == str(report_path)
    assert plan["model"] == "toy/m"
    assert plan["criterion"] == CRITERION
    # In-memory reports have no source path.
    in_memory = export_evidence_plan(report)
    assert in_memory["generated_for_report"] is None
    # Dict payloads are accepted too.
    from_dict = build_evidence_plan(report.to_dict())
    assert from_dict["summary"]["cards_assessed"] == 1


def test_plan_is_json_serializable_and_markdown_renders_empty_plan():
    plan = build_evidence_plan(_report([]))
    json.dumps(plan)
    assert plan["plan"] == []
    assert plan["agent_next_actions"] == []
    assert plan["summary"]["highest_priority_feature"] is None
    assert "# interp-lab Evidence Plan" in render_evidence_plan_markdown(plan)


def test_module_parser_accepts_proposed_flags():
    parser = build_plan_evidence_parser()
    args = parser.parse_args(
        [
            "--report",
            "report.json",
            "--out",
            "plan.json",
            "--markdown-out",
            "plan.md",
            "--top-k",
            "3",
            "--confidence",
            "0.99",
            "--assumed-effect",
            "0.2",
            "--json",
        ]
    )
    assert args.report == "report.json"
    assert args.top_k == 3
    assert args.confidence == 0.99
    assert args.assumed_effect == 0.2
    assert args.json is True
