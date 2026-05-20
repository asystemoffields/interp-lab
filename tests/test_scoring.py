from interp_lab.schema import Criterion, FeatureEvidence
from interp_lab.scoring import score_feature


def test_score_feature_prefers_activation_association_when_available():
    evidence = FeatureEvidence(
        feature_id="f",
        model="m",
        layer=1,
        label="semantically unrelated label",
        activation_signature=[0.1, 0.2],
        causal_effects={
            "criterion": 0.2,
            "signed_association": -0.75,
            "specificity": 0.15,
            "strong_causal_score": 0.18,
        },
    )

    scores = score_feature(evidence, Criterion(text="code-oriented completions"))

    assert scores["association"] == 0.75
    assert scores["importance"] > 0.3
