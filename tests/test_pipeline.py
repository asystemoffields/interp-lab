from oracle_sae.adapters.toy import ToyFeatureProvider, ToyInterventionRunner, ToyVerbalizer
from oracle_sae.pipeline import inspect_model, match_reports


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
