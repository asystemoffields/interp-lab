from interp_lab.adapters.toy import ToyFeatureProvider, ToyInterventionRunner, ToyVerbalizer
from interp_lab.matching import match_feature_cards
from interp_lab.pipeline import inspect_model, match_reports
from interp_lab.schema import Criterion, FeatureCard, FeatureEvidence, FeatureFingerprint


def test_inspection_report_contains_ranked_feature_cards():
    report = inspect_model(
        model="toy/model-a",
        criterion_text="the model is aware it is being evaluated",
        feature_provider=ToyFeatureProvider(feature_count=6),
        verbalizer=ToyVerbalizer(),
        intervention_runner=ToyInterventionRunner(),
        top_k=4,
    )

    assert len(report.cards) == 4
    assert report.cards[0].importance >= report.cards[-1].importance
    assert report.cards[0].fingerprint.text_vector
    assert " the " not in f" {report.cards[0].label} "


def test_inspection_report_includes_agent_intervention_actions():
    report = inspect_model(
        model="m",
        criterion_text="successful tool calls",
        feature_provider=_HiddenDimensionProvider(),
        verbalizer=ToyVerbalizer(),
        intervention_runner=ToyInterventionRunner(),
        top_k=1,
    )

    assert report.metadata["agent_next_actions"][0]["id"] == "plan_top_feature_interventions"
    actions = report.cards[0].metadata["agent_next_actions"]
    assert actions[0]["id"] == "plan_hidden_suppression"
    assert "--feature" in actions[0]["argv"]
    assert "L6:D12" in actions[0]["command"]


def test_match_reports_returns_candidates():
    left = inspect_model(
        model="toy/model-a",
        criterion_text="Python security bug",
        feature_provider=ToyFeatureProvider(feature_count=4),
        verbalizer=ToyVerbalizer(),
        intervention_runner=ToyInterventionRunner(),
        top_k=4,
    )
    right = inspect_model(
        model="toy/model-b",
        criterion_text="Python security bug",
        feature_provider=ToyFeatureProvider(feature_count=4),
        verbalizer=ToyVerbalizer(),
        intervention_runner=ToyInterventionRunner(),
        top_k=4,
    )

    matches = match_reports(left, right, top_k=3)

    assert len(matches.matches) == 3
    assert matches.matches[0].score >= matches.matches[-1].score
    assert matches.matches[0].left_label


def test_match_ranking_penalizes_opposite_signed_effects():
    left = _card("L", "left", 0.2)
    same_direction = _card("R-same", "same", 0.19)
    opposite_direction = _card("R-opposite", "opposite", -0.2)

    matches = match_feature_cards([left], [opposite_direction, same_direction], top_k=2)

    assert matches[0].right_feature_id == "R-same"
    assert matches[1].right_feature_id == "R-opposite"
    assert matches[1].score <= 0.49


def _card(feature_id: str, label: str, signed: float) -> FeatureCard:
    fingerprint = FeatureFingerprint(
        feature_id=feature_id,
        model="m",
        layer=1,
        text=label,
        text_vector=[1.0, 0.0],
        activation_signature=[1.0, 0.0],
        decoder_signature=[1.0, 0.0],
        causal_vector=[0.2, signed, 0.2, 0.0],
    )
    return FeatureCard(
        feature_id=feature_id,
        model="m",
        layer=1,
        label=label,
        explanation="",
        importance=1.0,
        association=0.5,
        specificity=0.5,
        causal_effect=0.5,
        stability=1.0,
        examples=[],
        source="test",
        fingerprint=fingerprint,
        causal_effects={"signed_causal_effect": signed},
    )


class _HiddenDimensionProvider:
    def features_for(self, model: str, criterion: Criterion) -> list[FeatureEvidence]:
        return [
            FeatureEvidence(
                feature_id="L6:D12",
                model=model,
                layer=6,
                label="tool-call hidden dimension",
                examples=["p1: activation=1.0 | call the tool"],
                activation_signature=[1.0, 0.0],
                decoder_signature=[0.0, 1.0],
                causal_effects={"criterion": 0.2, "specificity": 0.1},
                source="activation-records",
            )
        ]
