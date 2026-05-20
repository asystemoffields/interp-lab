from interp_lab.criteria import HeuristicCriterionCompiler
from interp_lab.fingerprints import build_fingerprint
from interp_lab.matching import fingerprint_similarity
from interp_lab.schema import FeatureEvidence


def test_similar_feature_fingerprints_score_highly():
    criterion = HeuristicCriterionCompiler().compile("evaluation awareness")
    left = FeatureEvidence(
        feature_id="L1:F1",
        model="a",
        layer=1,
        label="benchmark test awareness",
        examples=["This appears to be an evaluation."],
        activation_signature=[1.0, 0.0, 0.0],
        decoder_signature=[0.5, 0.5, 0.0],
        causal_effects={"criterion": 0.8, "specificity": 0.7, "side_effect": 0.1},
    )
    right = FeatureEvidence(
        feature_id="L2:F2",
        model="b",
        layer=2,
        label="benchmark test awareness",
        examples=["This appears to be an evaluation."],
        activation_signature=[0.9, 0.1, 0.0],
        decoder_signature=[0.45, 0.5, 0.0],
        causal_effects={"criterion": 0.78, "specificity": 0.72, "side_effect": 0.12},
    )

    left_fp = build_fingerprint(left, criterion, "The model suspects a benchmark.")
    right_fp = build_fingerprint(right, criterion, "The model suspects a benchmark.")
    score, components = fingerprint_similarity(left_fp, right_fp)

    assert score > 0.85
    assert components["causal"] > 0.99


def test_signed_causal_fingerprint_penalizes_opposite_direction():
    criterion = HeuristicCriterionCompiler().compile("evaluation awareness")
    promoter = FeatureEvidence(
        feature_id="L1:F1",
        model="a",
        layer=1,
        label="evaluation awareness",
        activation_signature=[1.0, 0.0],
        decoder_signature=[0.5, 0.5],
        causal_effects={
            "criterion": 0.8,
            "signed_causal_effect": 0.8,
            "specificity": 0.7,
            "side_effect": 0.1,
        },
    )
    suppressor = FeatureEvidence(
        feature_id="L2:F2",
        model="b",
        layer=2,
        label="evaluation awareness",
        activation_signature=[1.0, 0.0],
        decoder_signature=[0.5, 0.5],
        causal_effects={
            "criterion": 0.8,
            "signed_causal_effect": -0.8,
            "specificity": 0.7,
            "side_effect": 0.1,
        },
    )

    same_score, _ = fingerprint_similarity(
        build_fingerprint(promoter, criterion, "promotes evaluation awareness"),
        build_fingerprint(promoter, criterion, "promotes evaluation awareness"),
    )
    opposite_score, components = fingerprint_similarity(
        build_fingerprint(promoter, criterion, "promotes evaluation awareness"),
        build_fingerprint(suppressor, criterion, "suppresses evaluation awareness"),
    )

    assert opposite_score < same_score
    assert components["causal"] < 0.9


def test_weak_causal_vectors_do_not_match_as_strong_causal_evidence():
    criterion = HeuristicCriterionCompiler().compile("evaluation awareness")
    left = FeatureEvidence(
        feature_id="L1:F1",
        model="a",
        layer=1,
        label="evaluation awareness",
        activation_signature=[1.0],
        decoder_signature=[1.0],
        causal_effects={"criterion": 0.001, "signed_causal_effect": 0.001},
    )
    right = FeatureEvidence(
        feature_id="L2:F2",
        model="b",
        layer=2,
        label="evaluation awareness",
        activation_signature=[1.0],
        decoder_signature=[1.0],
        causal_effects={"criterion": 0.001, "signed_causal_effect": 0.001},
    )

    _, components = fingerprint_similarity(
        build_fingerprint(left, criterion, "weak"),
        build_fingerprint(right, criterion, "weak"),
    )

    assert components["causal"] < 0.01
