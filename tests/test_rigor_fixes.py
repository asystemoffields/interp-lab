"""Regression tests for verified correctness/rigor fixes.

Each test pins behavior that, if it regressed, would silently corrupt an evidence
grade or a cross-model ranking -- the failure mode this toolkit exists to prevent.
"""

import pytest

from interp_lab.adapters.interventions import InterventionRecord
from interp_lab.feature_interventions import _select_intervention_strength
from interp_lab.graph_validation import _validation_status
from interp_lab.math_utils import cosine, pearson
from interp_lab.matching import _score_with_signed_effect, fingerprint_similarity
from interp_lab.schema import Criterion, FeatureEvidence, FeatureFingerprint
from interp_lab.scoring import score_feature


# --- bug-7: cosine/pearson must not silently compare mismatched-length vectors ---

def test_cosine_returns_zero_for_mismatched_lengths():
    # Trimming to the shared prefix would report 1.0 here, hiding a real mismatch.
    assert cosine([1.0, 0.0, 0.0, 5.0, 5.0], [1.0, 0.0, 0.0]) == 0.0
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_pearson_returns_zero_for_mismatched_lengths():
    assert pearson([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0]) == 0.0


# --- bug-1: importance must not double-count causal/association via defaults ---

def test_absent_strong_and_specificity_do_not_inflate_importance():
    # Only a causal "criterion" and a (zeroed) association are supplied. With the
    # fix, strong_causal_score and specificity default to 0.0 instead of borrowing
    # causal_effect / association, so importance = 0.20*0.8 + 0.10*stability(0.25).
    evidence = FeatureEvidence(
        feature_id="f",
        model="m",
        layer=0,
        label="x",
        causal_effects={"criterion": 0.8, "signed_association": 0.0},
    )
    scores = score_feature(evidence, Criterion(text="c"))
    assert scores["importance"] == pytest.approx(0.185, abs=1e-6)
    assert scores["specificity"] == 0.0


# --- bug-8 / bug-9: matching must not reward absent or provenance-mismatched causal ---

def _fp(causal_vector, provenance):
    return FeatureFingerprint(
        feature_id="f",
        model="m",
        layer=0,
        text="t",
        text_vector=[1.0, 0.0],
        activation_signature=[1.0, 0.0],
        decoder_signature=[1.0, 0.0],
        causal_vector=causal_vector,
        text_embedder="hash-v1",
        causal_provenance=provenance,
    )


def test_absent_causal_vector_is_excluded_not_scored_half():
    left = _fp([], "none")
    right = _fp([1.0, 0.0], "intervention")
    score, components = fingerprint_similarity(left, right)
    assert "causal" not in components
    assert components["causal_absent"] == 1.0
    # All comparable components are perfect, so excluding causal yields 1.0 --
    # not a depressed score from a phantom 0.5 "half match".
    assert score == pytest.approx(1.0)


def test_mismatched_causal_provenance_is_not_compared():
    left = _fp([1.0, 0.0], "intervention")
    right = _fp([1.0, 0.0], "association")
    _, components = fingerprint_similarity(left, right)
    assert "causal" not in components
    assert components["causal_absent"] == 1.0


def test_matching_causal_compared_when_provenance_matches():
    left = _fp([1.0, 0.0], "association")
    right = _fp([1.0, 0.0], "association")
    _, components = fingerprint_similarity(left, right)
    assert "causal" in components


# --- bug-3: matching and validation agree on what is a real signed effect ---

def test_opposite_signed_effects_capped_at_shared_threshold():
    # Magnitudes at/above 0.02 -> genuine opposite-direction conflict -> capped.
    assert _score_with_signed_effect(0.9, 0.5, 0.03, -0.03) <= 0.49
    # Magnitudes below 0.02 -> too small to be a real conflict -> not capped.
    assert _score_with_signed_effect(0.9, 0.5, 0.01, -0.01) > 0.49


# --- bug-5: strength selection must use the raw float key, not a rounded display value ---

def test_select_intervention_strength_uses_raw_key():
    raw = 10.0 / 3.0  # 3.3333333333333335 -- differs from its 8dp rounding
    rows_by_strength = {raw: [{"baseline_score": 0.5, "intervention_score": 0.8, "metadata": {}}]}
    side_effects = {raw: [0.0]}
    selected, summary = _select_intervention_strength(rows_by_strength, side_effects, mode="amplify")
    # The returned key must index back into the raw-keyed dict without KeyError.
    assert rows_by_strength[selected]
    assert summary[0]["mean_directed_effect"] == pytest.approx(0.3)


# --- bug-14: unknown intervention names have undefined effect direction -> reject ---

def test_unknown_intervention_name_is_rejected():
    with pytest.raises(ValueError):
        InterventionRecord.from_dict(
            {
                "model": "m",
                "feature_id": "f",
                "intervention": "frobnicate",
                "baseline_score": 0.1,
                "intervention_score": 0.2,
            },
            line_label="x:1",
        )


def test_known_intervention_name_is_accepted():
    record = InterventionRecord.from_dict(
        {
            "model": "m",
            "feature_id": "f",
            "intervention": "Ablate",
            "baseline_score": 0.9,
            "intervention_score": 0.3,
        },
        line_label="x:1",
    )
    assert record.intervention == "ablate"


# --- bug-10: a perfectly split sign must not clear the suggestive gate ---

def _thresholds(min_sign_consistency=0.75):
    return {
        "min_effect": 0.05,
        "min_specificity": 0.02,
        "min_effect_control_ratio": 1.5,
        "min_prompt_count": 1,
        "min_sign_consistency": min_sign_consistency,
        "require_controls": False,
    }


def test_perfectly_split_sign_is_weak_not_suggestive():
    status = _validation_status(
        effect_count=2,
        control_count=0,
        prompt_count=2,
        mean_abs_effect=0.3,
        specificity=0.0,
        ratio=None,
        sign_consistency=0.5,  # 50/50 split: no consistent direction
        thresholds=_thresholds(),
    )
    assert status == "weak"


def test_strict_majority_sign_is_suggestive():
    status = _validation_status(
        effect_count=3,
        control_count=0,
        prompt_count=3,
        mean_abs_effect=0.3,
        specificity=0.0,
        ratio=None,
        sign_consistency=0.67,
        thresholds=_thresholds(),
    )
    assert status == "suggestive"
