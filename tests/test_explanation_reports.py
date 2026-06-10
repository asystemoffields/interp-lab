import json
import re
from pathlib import Path

import pytest

from interp_lab.adapters.nla import NlaVerbalizer
from interp_lab.adapters.toy import ToyInterventionRunner, ToyVerbalizer
from interp_lab.cli import build_parser, main
from interp_lab.explanation_reports import (
    build_explanation_consistency_report,
    build_feature_search_report,
    build_model_family_comparison_report,
    build_text_pivot_match_report,
)
from interp_lab.fingerprints import build_fingerprint
from interp_lab.pipeline import inspect_model
from interp_lab.reporting import write_inspection_report
from interp_lab.schema import Criterion, FeatureCard, FeatureEvidence, InspectionReport


def test_nla_verbalizer_uses_external_feature_explanations():
    report = inspect_model(
        model="m",
        criterion_text="successful tool calls",
        feature_provider=_SingleFeatureProvider(),
        verbalizer=NlaVerbalizer(
            {
                "L1:F7": {
                    "explanation": "Fires when the assistant prepares a valid tool-call argument structure.",
                    "confidence": 0.91,
                    "paraphrases": ["tool-call argument planning"],
                }
            }
        ),
        intervention_runner=ToyInterventionRunner(),
        top_k=1,
    )

    card = report.cards[0]
    assert card.explanation.startswith("Fires when the assistant prepares")
    assert card.metadata["verbalizer"]["type"] == "nla"
    assert card.metadata["verbalizer"]["confidence"] == 0.91
    assert report.metadata["verbalizer"]["record_count"] == 1


def test_nla_verbalizer_metadata_reports_actual_fallback_source():
    report = inspect_model(
        model="m",
        criterion_text="successful tool calls",
        feature_provider=_SingleFeatureProvider(),
        verbalizer=NlaVerbalizer(
            {
                "L1:F7": {
                    "explanation": "Low-confidence explanation.",
                    "confidence": 0.1,
                }
            },
            min_confidence=0.8,
            fallback=ToyVerbalizer(),
        ),
        intervention_runner=ToyInterventionRunner(),
        top_k=1,
    )

    metadata = report.cards[0].metadata["verbalizer"]
    assert metadata["source"] == "fallback"
    assert metadata["used_record"] is False
    assert metadata["rejected_record"]["reason"] == "below_min_confidence"


def test_explanation_consistency_report_flags_stable_shared_features(tmp_path: Path):
    left_path = _write_report(
        tmp_path / "left",
        criterion="successful tool calls",
        cards=[_card("L1:F1", "tool call syntax", "Activates on successful structured tool calls.", 0.9)],
    )
    right_path = _write_report(
        tmp_path / "right",
        criterion="the assistant uses tools successfully",
        cards=[_card("L1:F1", "tool call syntax", "Activates on successful structured tool calls.", 0.8)],
    )

    report = build_explanation_consistency_report(reports=[left_path, right_path])

    assert report["schema_version"] == "interp-lab.explanation_consistency.v1"
    assert report["summary"]["consistent_count"] == 1
    assert report["checks"][0]["feature_id"] == "L1:F1"


def test_explanation_consistency_uses_distinct_report_coverage(tmp_path: Path):
    left_path = _write_report(
        tmp_path / "left",
        criterion="successful tool calls",
        cards=[
            _card("L1:F1", "tool call syntax", "Activates on structured tool calls.", 0.9),
            _card("L1:F1", "tool call syntax duplicate", "Activates on structured tool calls.", 0.7),
        ],
    )
    right_path = _write_report(
        tmp_path / "right",
        criterion="successful tool calls",
        cards=[_card("L1:F2", "other feature", "Activates on something else.", 0.6)],
    )

    report = build_explanation_consistency_report(reports=[left_path, right_path])
    check = next(item for item in report["checks"] if item["feature_id"] == "L1:F1")

    assert check["status"] == "missing_in_some_reports"
    assert check["covered_report_count"] == 1
    assert check["duplicate_report_indexes"] == [0]
    assert report["summary"]["duplicate_feature_id_count"] == 1
    assert report["summary"]["shared_feature_count"] == 0


def test_explanation_consistency_does_not_treat_blank_text_as_consistent(tmp_path: Path):
    left_path = _write_report(
        tmp_path / "left",
        criterion="successful tool calls",
        cards=[_card("L1:F1", "", "", 0.9)],
    )
    right_path = _write_report(
        tmp_path / "right",
        criterion="successful tool calls",
        cards=[_card("L1:F1", "", "", 0.8)],
    )

    report = build_explanation_consistency_report(reports=[left_path, right_path])
    check = report["checks"][0]

    assert check["mean_explanation_similarity"] == 0.0
    assert check["status"] == "explanation_drift"
    assert report["summary"]["consistent_count"] == 0


def test_feature_search_ranks_natural_language_query(tmp_path: Path):
    path = _write_report(
        tmp_path / "report",
        criterion="successful tool calls",
        cards=[
            _card("L1:F1", "tool call syntax", "Tracks valid tool calls with arguments.", 0.9),
            _card("L1:F2", "friendly tone", "Tracks warm conversational greetings.", 0.8),
        ],
    )

    report = build_feature_search_report(reports=[path], query="tool calls with arguments", top_k=1)

    assert report["schema_version"] == "interp-lab.feature_search.v1"
    assert report["results"][0]["feature_id"] == "L1:F1"
    assert report["results"][0]["agent_next_actions"][0]["command"].startswith("interp-lab intervene")


def test_feature_search_canonical_action_argv_parses_against_cli(tmp_path: Path):
    path = _write_report(
        tmp_path / "report",
        criterion="successful tool calls",
        cards=[_card("L1:F1", "tool call syntax", "Tracks valid tool calls with arguments.", 0.9)],
    )

    report = build_feature_search_report(reports=[path], query="tool calls", top_k=1)
    result = report["results"][0]

    canonical = result["agent_next_actions"][0]
    assert canonical["id"] == "plan_feature_intervention"
    assert canonical["title"]
    assert canonical["requires"] == ["scored causal prompt JSONL"]
    assert "instruction" not in canonical

    argv = canonical["argv"]
    assert argv[0] == "interp-lab"
    filled = ["dummy-path" if re.fullmatch(r"<.+>", str(token)) else str(token) for token in argv[1:]]
    args = build_parser().parse_args(filled)  # exits 2 if the suggested command is not runnable
    assert args.command == "intervene"
    assert args.model == "m"
    assert args.criterion == "successful tool calls"
    assert args.feature == ["L1:F1"]
    assert args.report == str(path)

    # The legacy flat keys were removed in 3.0.0.
    assert "agent_next_action" not in result
    assert "agent_next_action_argv" not in result
    assert "agent_next_action_requires" not in result

    # Report-level prose actions: instruction only; legacy "description" removed in 3.0.0.
    report_action = report["agent_next_actions"][0]
    assert report_action["id"] and report_action["title"]
    assert report_action["instruction"]
    assert "description" not in report_action


def test_model_family_comparison_report_summarizes_cross_family_matches(tmp_path: Path):
    gemma = _write_report(
        tmp_path / "gemma",
        model="gemma-small",
        criterion="successful tool calls",
        cards=[_card("L1:F1", "tool call syntax", "Tracks valid tool calls with arguments.", 0.9, model="gemma-small")],
    )
    qwen = _write_report(
        tmp_path / "qwen",
        model="qwen-small",
        criterion="successful tool calls",
        cards=[_card("L1:F9", "tool call syntax", "Tracks valid tool calls with arguments.", 0.85, model="qwen-small")],
    )

    report = build_model_family_comparison_report(
        members=[
            {"family": "gemma", "report": str(gemma)},
            {"family": "qwen", "report": str(qwen)},
        ],
        min_score=0.5,
    )

    assert report["schema_version"] == "interp-lab.model_family_comparison.v1"
    assert report["summary"]["family_count"] == 2
    assert report["pairwise"][0]["relation"] == "cross_family"
    assert report["pairwise"][0]["strong_match_count"] >= 1


def test_text_pivot_match_report_uses_explanations_as_bridge(tmp_path: Path):
    left = _write_report(
        tmp_path / "left",
        model="gemma-small",
        criterion="successful tool calls",
        cards=[
            _card("L1:F1", "tool call syntax", "Represents valid tool-call argument construction.", 0.9, model="gemma-small"),
            _card("L1:F2", "friendly tone", "Represents warm social greeting style.", 0.8, model="gemma-small"),
        ],
    )
    right = _write_report(
        tmp_path / "right",
        model="qwen-small",
        criterion="successful tool calls",
        cards=[
            _card("L4:F9", "tool argument schema", "Represents valid tool-call argument construction.", 0.85, model="qwen-small"),
            _card("L4:F3", "poetry style", "Represents rhymed verse completions.", 0.7, model="qwen-small"),
        ],
    )

    report = build_text_pivot_match_report(left_reports=[left], right_reports=[right], top_k=2, per_left=1)

    assert report["schema_version"] == "interp-lab.text_pivot_match.v1"
    assert report["matches"][0]["left_feature_id"] == "L1:F1"
    assert report["matches"][0]["right_feature_id"] == "L4:F9"
    assert report["matches"][0]["components"]["text_pivot"] >= 0.9
    assert report["matches"][0]["evidence_grade"] == "text_pivot_with_causal_support"
    # Canonical per-match action only; legacy agent_next_action* keys removed in 3.0.0.
    match_action = report["matches"][0]["agent_next_actions"][0]
    assert match_action["id"] == "validate_text_pivot_pair"
    assert match_action["argv"][:2] == ["interp-lab", "match"]
    assert match_action["command"].startswith("interp-lab match --left ")
    assert "agent_next_action" not in report["matches"][0]
    assert "agent_next_action_argv" not in report["matches"][0]


def test_text_pivot_does_not_upgrade_label_only_or_association_only_matches(tmp_path: Path):
    left = _write_report(
        tmp_path / "left",
        model="left-model",
        criterion="successful tool calls",
        cards=[
            _card(
                "L1:F1",
                "shared label",
                "Represents valid tool-call argument construction.",
                0.9,
                model="left-model",
                causal_effects={"criterion": 0.9, "signed_association": 0.9},
            )
        ],
    )
    right = _write_report(
        tmp_path / "right",
        model="right-model",
        criterion="successful tool calls",
        cards=[
            _card(
                "L1:F9",
                "shared label",
                "Represents warm social greeting style.",
                0.8,
                model="right-model",
                causal_effects={"criterion": 0.8, "signed_association": 0.8},
            )
        ],
    )

    report = build_text_pivot_match_report(
        left_reports=[left],
        right_reports=[right],
        top_k=1,
        per_left=1,
        min_text_score=0.95,
    )
    match = report["matches"][0]

    assert match["text_pivot_source"] == "label"
    assert match["components"]["causal_evidence"] == 0.0
    assert match["evidence_grade"] == "label_or_example_text_candidate"


def test_text_pivot_requires_measured_intervention_for_causal_support(tmp_path: Path):
    left = _write_report(
        tmp_path / "left",
        criterion="successful tool calls",
        cards=[
            _card(
                "L1:F1",
                "tool call syntax",
                "Represents valid tool-call argument construction.",
                0.9,
                causal_effects={"signed_causal_effect": 0.9, "strong_causal_score": 0.9},
            )
        ],
    )
    right = _write_report(
        tmp_path / "right",
        criterion="successful tool calls",
        cards=[
            _card(
                "L1:F2",
                "tool call syntax",
                "Represents valid tool-call argument construction.",
                0.9,
                causal_effects={"signed_causal_effect": 0.9, "strong_causal_score": 0.9},
            )
        ],
    )

    report = build_text_pivot_match_report(left_reports=[left], right_reports=[right], top_k=1, per_left=1)
    match = report["matches"][0]

    assert match["components"]["causal_evidence"] == 0.0
    assert match["evidence_grade"] != "text_pivot_with_causal_support"


def test_text_pivot_filters_matches_below_min_text_score(tmp_path: Path):
    left = _write_report(
        tmp_path / "left",
        criterion="successful tool calls",
        cards=[_card("L1:F1", "", "", 0.9)],
    )
    right = _write_report(
        tmp_path / "right",
        criterion="successful tool calls",
        cards=[_card("L1:F2", "", "", 0.9)],
    )

    report = build_text_pivot_match_report(
        left_reports=[left],
        right_reports=[right],
        min_text_score=0.1,
    )

    assert report["matches"] == []
    assert report["summary"]["candidate_count"] == 0


def test_text_pivot_rejects_zero_per_left(tmp_path: Path):
    left = _write_report(
        tmp_path / "left",
        criterion="successful tool calls",
        cards=[_card("L1:F1", "tool call syntax", "Tracks tool calls.", 0.9)],
    )
    right = _write_report(
        tmp_path / "right",
        criterion="successful tool calls",
        cards=[_card("L1:F2", "tool call syntax", "Tracks tool calls.", 0.9)],
    )

    with pytest.raises(ValueError, match="per_left must be at least 1"):
        build_text_pivot_match_report(left_reports=[left], right_reports=[right], per_left=0)


def test_text_pivot_rejects_unbounded_pairwise_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    left = _write_report(
        tmp_path / "left",
        criterion="successful tool calls",
        cards=[_card("L1:F1", "tool call syntax", "Tracks tool calls.", 0.9)],
    )
    right = _write_report(
        tmp_path / "right",
        criterion="successful tool calls",
        cards=[_card("L1:F2", "tool call syntax", "Tracks tool calls.", 0.9)],
    )

    monkeypatch.setattr("interp_lab.explanation_reports.DEFAULT_MAX_PAIRWISE_COMPARISONS", 0)
    with pytest.raises(ValueError, match="text-pivot matching would compare"):
        build_text_pivot_match_report(left_reports=[left], right_reports=[right])


def test_malformed_report_errors_are_user_facing(tmp_path: Path):
    bad_report = tmp_path / "bad-report.json"
    bad_report.write_text("[]", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["search-features", "--report", str(bad_report), "--query", "tool calls"])

    assert exc.value.code == 2


def test_new_report_commands_write_json_and_markdown(tmp_path: Path):
    left = _write_report(
        tmp_path / "left",
        model="family-a-model",
        criterion="successful tool calls",
        cards=[_card("L1:F1", "tool call syntax", "Tracks valid tool calls with arguments.", 0.9)],
    )
    right = _write_report(
        tmp_path / "right",
        model="family-b-model",
        criterion="the assistant uses tools successfully",
        cards=[_card("L1:F1", "tool call syntax", "Tracks valid tool calls with arguments.", 0.8)],
    )

    consistency = tmp_path / "consistency.json"
    search = tmp_path / "search.json"
    family = tmp_path / "family.json"
    text_pivot = tmp_path / "text-pivot.json"

    assert (
        main(
            [
                "check-explanation-consistency",
                "--report",
                str(left.parent),
                "--report",
                str(right),
                "--out",
                str(consistency),
                "--html-out",
                str(tmp_path / "consistency.html"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "search-features",
                "--report",
                str(left.parent),
                "--query",
                "tool calls",
                "--out",
                str(search),
                "--html-out",
                str(tmp_path / "search.html"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "compare-model-families",
                "--member",
                f"family-a={left.parent}",
                "--member",
                f"family-b={right}",
                "--out",
                str(family),
                "--html-out",
                str(tmp_path / "family.html"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "match-text-pivot",
                "--left",
                str(left.parent),
                "--right",
                str(right.parent),
                "--out",
                str(text_pivot),
                "--html-out",
                str(tmp_path / "text-pivot.html"),
            ]
        )
        == 0
    )

    assert json.loads(consistency.read_text(encoding="utf-8"))["schema_version"] == "interp-lab.explanation_consistency.v1"
    assert json.loads(text_pivot.read_text(encoding="utf-8"))["schema_version"] == "interp-lab.text_pivot_match.v1"
    assert search.with_suffix(".md").exists()
    assert family.with_suffix(".md").exists()
    assert "Feature Search" in (tmp_path / "search.html").read_text(encoding="utf-8")
    assert "Model-Family Comparison" in (tmp_path / "family.html").read_text(encoding="utf-8")
    assert "Text-Pivot Matches" in (tmp_path / "text-pivot.html").read_text(encoding="utf-8")


def test_inspect_cli_uses_nla_verbalizer_records(tmp_path: Path):
    features = tmp_path / "features.jsonl"
    evidence = FeatureEvidence(
        feature_id="L1:F7",
        model="m",
        layer=1,
        label="tool call arguments",
        examples=["assistant emits a valid tool call"],
        activation_signature=[1.0, 0.0],
        decoder_signature=[0.0, 1.0],
        causal_effects={"criterion": 0.7, "specificity": 0.8},
        source="test",
    )
    features.write_text(json.dumps(evidence.to_dict()) + "\n", encoding="utf-8")
    explanations = tmp_path / "nla.jsonl"
    explanations.write_text(
        json.dumps(
            {
                "feature_id": "L1:F7",
                "explanation": "Represents valid tool-call argument construction.",
                "confidence": 0.93,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report"

    exit_code = main(
        [
            "inspect",
            "--model",
            "m",
            "--criterion",
            "successful tool calls",
            "--backend",
            "jsonl",
            "--features",
            str(features),
            "--verbalizer",
            "nla",
            "--nla-explanations",
            str(explanations),
            "--out",
            str(out),
        ]
    )

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["cards"][0]["explanation"] == "Represents valid tool-call argument construction."
    assert report["cards"][0]["metadata"]["verbalizer"]["confidence"] == 0.93


class _SingleFeatureProvider:
    def features_for(self, model: str, criterion: Criterion) -> list[FeatureEvidence]:
        return [
            FeatureEvidence(
                feature_id="L1:F7",
                model=model,
                layer=1,
                label="tool call arguments",
                examples=["assistant emits a valid tool call"],
                activation_signature=[1.0, 0.0],
                decoder_signature=[0.0, 1.0],
                causal_effects={"criterion": 0.7, "specificity": 0.8},
                source="test",
            )
        ]


def _write_report(
    directory: Path,
    *,
    criterion: str,
    cards: list[FeatureCard],
    model: str = "m",
) -> Path:
    report = InspectionReport(model=model, criterion=Criterion(criterion), cards=cards)
    json_path, _ = write_inspection_report(report, directory)
    return json_path


def _card(
    feature_id: str,
    label: str,
    explanation: str,
    importance: float,
    *,
    model: str = "m",
    causal_effects: dict[str, float] | None = None,
) -> FeatureCard:
    causal = causal_effects or {
        "criterion": importance,
        "signed_causal_effect": importance,
        "intervention_record_count": 2.0,
    }
    evidence = FeatureEvidence(
        feature_id=feature_id,
        model=model,
        layer=1,
        label=label,
        examples=[explanation],
        activation_signature=[1.0, 0.0],
        decoder_signature=[0.0, 1.0],
        causal_effects=causal,
        source="test",
    )
    return FeatureCard(
        feature_id=feature_id,
        model=model,
        layer=1,
        label=label,
        explanation=explanation,
        importance=importance,
        association=importance,
        specificity=importance,
        causal_effect=importance,
        stability=0.8,
        examples=evidence.examples,
        source=evidence.source,
        fingerprint=build_fingerprint(evidence, Criterion("successful tool calls"), explanation),
        causal_effects=evidence.causal_effects,
    )


# =============================================================================
# Provenance gates for text-pivot signed effects (mirrors matching.py): no free
# half-match for unmeasured axes, no measured-vs-association comparisons, and no
# causal-sounding grade from association-vs-association alignment.
# =============================================================================


def _text_pivot_single_match(tmp_path, left_effects, right_effects, **kwargs):
    left = _write_report(
        tmp_path / "left",
        model="left-model",
        criterion="successful tool calls",
        cards=[
            _card(
                "L1:F1",
                "tool call syntax",
                "Represents valid tool-call argument construction.",
                0.9,
                model="left-model",
                causal_effects=left_effects,
            )
        ],
    )
    right = _write_report(
        tmp_path / "right",
        model="right-model",
        criterion="successful tool calls",
        cards=[
            _card(
                "L1:F2",
                "tool call syntax",
                "Represents valid tool-call argument construction.",
                0.9,
                model="right-model",
                causal_effects=right_effects,
            )
        ],
    )
    report = build_text_pivot_match_report(
        left_reports=[left], right_reports=[right], top_k=1, per_left=1, **kwargs
    )
    return report["matches"][0]


def test_text_pivot_missing_signed_effect_is_excluded_not_half_matched(tmp_path: Path):
    # Neither side carries a signed effect: the axis used to earn a free 0.5
    # component; it must now be excluded (0.0, no provenance marker).
    match = _text_pivot_single_match(
        tmp_path,
        {"criterion": 0.9},
        {"criterion": 0.9},
    )

    assert match["components"]["signed_effect"] == 0.0
    assert "signed_effect_provenance_intervention" not in match["components"]
    assert "signed_effect_provenance_association" not in match["components"]
    assert "signed_effect_provenance_mismatch" not in match["components"]


def test_text_pivot_mixed_provenance_signed_effects_are_excluded_not_contradicted(tmp_path: Path):
    # Measured +0.9 vs correlational -0.9 is incomparable: it must neither be
    # compared for similarity nor trigger the opposite-direction 0.49 cap.
    match = _text_pivot_single_match(
        tmp_path,
        {"signed_causal_effect": 0.9},
        {"signed_association": -0.9},
    )

    assert match["components"]["signed_effect_provenance_mismatch"] == 1.0
    assert match["components"]["signed_effect"] == 0.0
    assert match["score"] > 0.49


def test_text_pivot_same_provenance_opposite_signed_effects_still_capped(tmp_path: Path):
    # Two correlational signed effects ARE comparable: opposite directions keep
    # the contradiction cap (parity with matching.py same-provenance handling).
    match = _text_pivot_single_match(
        tmp_path,
        {"signed_association": 0.9},
        {"signed_association": -0.9},
    )

    assert match["components"]["signed_effect_provenance_association"] == 1.0
    assert match["score"] <= 0.49


def test_text_pivot_association_alignment_is_not_causal_support(tmp_path: Path):
    # Both sides have intervention records attached but only correlational signed
    # effects: two perfectly aligned Pearson r's must not produce the
    # causal-sounding "text_pivot_with_causal_support" grade.
    match = _text_pivot_single_match(
        tmp_path,
        {"criterion": 0.9, "signed_association": 0.9, "intervention_record_count": 2.0},
        {"criterion": 0.9, "signed_association": 0.9, "intervention_record_count": 2.0},
    )

    assert match["components"]["causal_evidence"] == 1.0
    assert match["components"]["signed_effect"] >= 0.65
    assert match["components"]["signed_effect_provenance_association"] == 1.0
    assert match["evidence_grade"] != "text_pivot_with_causal_support"


def test_text_pivot_intervention_alignment_keeps_causal_support(tmp_path: Path):
    # Intervention-measured signed effects on both sides keep the causal grade
    # (and the components now record that provenance explicitly).
    match = _text_pivot_single_match(
        tmp_path,
        {"criterion": 0.9, "signed_causal_effect": 0.9, "intervention_record_count": 2.0},
        {"criterion": 0.9, "signed_causal_effect": 0.9, "intervention_record_count": 2.0},
    )

    assert match["components"]["signed_effect_provenance_intervention"] == 1.0
    assert match["evidence_grade"] == "text_pivot_with_causal_support"
