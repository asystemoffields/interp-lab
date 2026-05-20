import json
from pathlib import Path

from oracle_sae.adapters.nla import NlaVerbalizer
from oracle_sae.adapters.toy import ToyInterventionRunner
from oracle_sae.cli import main
from oracle_sae.explanation_reports import (
    build_explanation_consistency_report,
    build_feature_search_report,
    build_model_family_comparison_report,
    build_text_pivot_match_report,
)
from oracle_sae.fingerprints import build_fingerprint
from oracle_sae.pipeline import inspect_model
from oracle_sae.reporting import write_inspection_report
from oracle_sae.schema import Criterion, FeatureCard, FeatureEvidence, InspectionReport


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
    assert report["results"][0]["agent_next_action"].startswith("Run interp-lab intervene")


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
) -> FeatureCard:
    evidence = FeatureEvidence(
        feature_id=feature_id,
        model=model,
        layer=1,
        label=label,
        examples=[explanation],
        activation_signature=[1.0, 0.0],
        decoder_signature=[0.0, 1.0],
        causal_effects={"criterion": importance, "signed_causal_effect": importance},
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
