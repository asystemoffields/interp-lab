"""Regression tests for verified correctness/rigor fixes.

Each test pins behavior that, if it regressed, would silently corrupt an evidence
grade or a cross-model ranking -- the failure mode this toolkit exists to prevent.
"""

import pytest

from interp_lab.adapters.interventions import InterventionRecord
from interp_lab.feature_interventions import _select_intervention_strength
from interp_lab.graph_validation import _validation_status, build_graph_validation_report
from interp_lab.graphs import _top_tokens, build_attribution_graph
from interp_lab.match_validation import build_match_validation_report
from interp_lab.math_utils import cosine, pearson
from interp_lab.matching import (
    _score_with_signed_effect,
    fingerprint_similarity,
    has_intervention_provenance,
    match_feature_cards,
    signed_effect_with_provenance,
)
from interp_lab.schema import (
    CandidateMatch,
    Criterion,
    FeatureCard,
    FeatureEvidence,
    FeatureFingerprint,
    InspectionReport,
    MatchReport,
)
from interp_lab.scoring import score_feature
from interp_lab import stats


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
    # signed_causal_effect marks intervention provenance so the "criterion" value
    # legitimately counts on the causal axis (see the provenance gates below).
    evidence = FeatureEvidence(
        feature_id="f",
        model="m",
        layer=0,
        label="x",
        causal_effects={"criterion": 0.8, "signed_causal_effect": 0.8, "signed_association": 0.0},
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


# --- activation/decoder must be gated like text/causal (no free 0.5 "half match") ---

def _sig_fp(activation, decoder):
    return FeatureFingerprint(
        feature_id="f",
        model="m",
        layer=0,
        text="t",
        text_vector=[1.0, 0.0],
        activation_signature=activation,
        decoder_signature=decoder,
        causal_vector=[1.0, 0.0],
        text_embedder="hash-v1",
        causal_provenance="association",
    )


def test_absent_activation_signature_is_excluded_not_scored_half():
    # One side has no activation signature (a common feature-dump / cross-model case).
    left = _sig_fp([], [1.0, 0.0])
    right = _sig_fp([1.0, 0.0], [1.0, 0.0])
    score, components = fingerprint_similarity(left, right)
    assert "activation" not in components
    assert components["activation_absent"] == 1.0
    # Every comparable axis is perfect, so excluding activation must yield 1.0 -- not a
    # depressed score from a phantom cosine([],x)=0 -> 0.5 contribution at full weight.
    assert score == pytest.approx(1.0)


def test_mismatched_length_decoder_is_excluded():
    # Two models with different hidden sizes: the decoder cosine is undefined, so the
    # component must drop out rather than inject a misaligned 0.5.
    left = _sig_fp([1.0, 0.0], [1.0, 0.0, 0.0])
    right = _sig_fp([1.0, 0.0], [1.0, 0.0])
    score, components = fingerprint_similarity(left, right)
    assert "decoder" not in components
    assert components["decoder_absent"] == 1.0
    assert score == pytest.approx(1.0)


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


# =============================================================================
# Provenance gates: correlational values must never leak onto causal-labeled axes
# (shared accessor: matching.signed_effect_with_provenance / has_intervention_provenance)
# =============================================================================


def test_signed_effect_accessor_reports_provenance():
    assert signed_effect_with_provenance({"signed_causal_effect": 0.3, "signed_association": -0.9}) == (
        0.3,
        "intervention",
    )
    assert signed_effect_with_provenance({"signed_association": -0.9}) == (-0.9, "association")
    assert signed_effect_with_provenance({}, {"signed_association": 0.2}) == (0.2, "association")
    assert signed_effect_with_provenance({}) == (None, "none")
    assert has_intervention_provenance({"signed_causal_effect": 0.0})
    assert has_intervention_provenance({"intervention_record_count": 3.0})
    assert has_intervention_provenance({}, {"interventions": {"count": 2}})
    assert not has_intervention_provenance({"criterion": 0.9, "signed_association": 0.9})


def _provenance_card(feature_id: str, causal_effects: dict[str, float]) -> FeatureCard:
    provenance = "intervention" if "signed_causal_effect" in causal_effects else "association"
    fingerprint = FeatureFingerprint(
        feature_id=feature_id,
        model="m",
        layer=1,
        text="label",
        text_vector=[1.0, 0.0],
        activation_signature=[1.0, 0.0],
        decoder_signature=[1.0, 0.0],
        causal_vector=[0.2, 0.2],
        text_embedder="hash-v1",
        causal_provenance=provenance,
    )
    return FeatureCard(
        feature_id=feature_id,
        model="m",
        layer=1,
        label="label",
        explanation="",
        importance=1.0,
        association=0.5,
        specificity=0.5,
        causal_effect=0.5,
        stability=1.0,
        examples=[],
        source="test",
        fingerprint=fingerprint,
        causal_effects=causal_effects,
    )


# --- HIGH: matching records signed-effect provenance; mixed provenance is excluded ---

def test_matching_tags_signed_effect_provenance():
    intervention = match_feature_cards(
        [_provenance_card("L", {"signed_causal_effect": 0.3})],
        [_provenance_card("R", {"signed_causal_effect": 0.31})],
        top_k=1,
    )[0]
    assert intervention.components["signed_effect_provenance_intervention"] == 1.0
    assert "signed_effect" in intervention.components

    association = match_feature_cards(
        [_provenance_card("L", {"signed_association": 0.3})],
        [_provenance_card("R", {"signed_association": 0.31})],
        top_k=1,
    )[0]
    assert association.components["signed_effect_provenance_association"] == 1.0
    assert "signed_effect" in association.components


def test_mixed_signed_effect_provenance_is_excluded_not_contradicted():
    # Left measured +0.3 vs right correlational -0.3: incomparable, so the pair must
    # neither earn the signed-effect axis nor get the opposite-direction 0.49 cap.
    match = match_feature_cards(
        [_provenance_card("L", {"signed_causal_effect": 0.3})],
        [_provenance_card("R", {"signed_association": -0.3})],
        top_k=1,
    )[0]
    assert match.components["signed_effect_provenance_mismatch"] == 1.0
    assert "signed_effect" not in match.components
    assert match.left_signed_effect is None
    assert match.right_signed_effect is None
    assert match.score > 0.49  # no phantom contradiction cap

    report = build_match_validation_report(
        MatchReport(left_model="lm", right_model="rm", matches=[match])
    )
    row = report["validations"][0]
    assert row["status"] != "contradicted"
    assert row["claim_grade"] != "contradicted_effect"
    assert "signed_effect_provenance_mismatch" in row["reason_codes"]
    assert row["signed_effect_provenance"] == "mismatch"


# --- HIGH: "validated" requires intervention-backed signed effects on both sides ---

def _aligned_match(marker: str | None) -> CandidateMatch:
    components = {
        "text": 0.9,
        "activation": 0.85,
        "decoder": 0.8,
        "causal": 0.8,
        "signed_effect": 0.99,
    }
    if marker is not None:
        components[marker] = 1.0
    return CandidateMatch(
        left_feature_id="L",
        right_feature_id="R",
        left_model="lm",
        right_model="rm",
        score=0.9,
        components=components,
        left_signed_effect=0.11,
        right_signed_effect=0.12,
    )


def test_intervention_backed_aligned_match_is_validated():
    report = build_match_validation_report(
        MatchReport(
            left_model="lm",
            right_model="rm",
            matches=[_aligned_match("signed_effect_provenance_intervention")],
        )
    )
    row = report["validations"][0]
    assert row["status"] == "validated"
    assert row["claim_grade"] == "validated_equivalent"
    assert row["signed_effect_provenance"] == "intervention"


@pytest.mark.parametrize("marker", ["signed_effect_provenance_association", None])
def test_association_only_aligned_match_caps_at_needs_causal_evidence(marker):
    # Two perfectly aligned Pearson r's are still zero interventions: never "validated".
    # marker=None covers legacy match reports written before provenance tagging.
    report = build_match_validation_report(
        MatchReport(left_model="lm", right_model="rm", matches=[_aligned_match(marker)])
    )
    row = report["validations"][0]
    assert row["status"] == "needs_causal_evidence"
    assert row["claim_grade"] == "needs_more_evidence"
    assert "signed_effects_lack_intervention_provenance" in row["reason_codes"]
    assert "interventions" in row["next_action"].lower()
    assert report["summary"]["validated_count"] == 0


# --- MED: scoring must not count the association-proxy "criterion" r as causal ---

def test_association_proxy_criterion_is_not_scored_as_causal():
    # Records backend without interventions: {"criterion": abs(r), "signed_association": r}.
    proxy = FeatureEvidence(
        feature_id="f",
        model="m",
        layer=0,
        label="x",
        causal_effects={"criterion": 0.5, "signed_association": 0.5},
    )
    scores = score_feature(proxy, Criterion(text="c"))
    assert scores["causal_effect"] == 0.0  # the r is already counted as association
    assert scores["association"] == 0.5

    measured = FeatureEvidence(
        feature_id="f",
        model="m",
        layer=0,
        label="x",
        causal_effects={"criterion": 0.5, "signed_causal_effect": 0.5},
    )
    assert score_feature(measured, Criterion(text="c"))["causal_effect"] == 0.5


# --- MED: unmeasured association fallback must not earn a ~0.5 free baseline ---

def test_unmeasured_association_fallback_scores_near_zero_for_unrelated_text():
    unmeasured = FeatureEvidence(
        feature_id="f",
        model="m",
        layer=0,
        label="zzz qqq unrelated words",
    )
    measured_weak = FeatureEvidence(
        feature_id="g",
        model="m",
        layer=0,
        label="zzz qqq unrelated words",
        causal_effects={"signed_association": 0.10},
    )
    criterion = Criterion(text="alpha beta gamma")
    fallback = score_feature(unmeasured, criterion)["association"]
    weak = score_feature(measured_weak, criterion)["association"]
    assert weak == pytest.approx(0.10)
    assert fallback < weak  # a measured weak association outranks no evidence


# --- MED/LOW: graph edges must not publish correlational signed effects as causal ---

def _graph_card(
    feature_id: str,
    causal_effects: dict[str, float],
    *,
    examples: list[str] | None = None,
    layer: int = 12,
) -> FeatureCard:
    fingerprint = FeatureFingerprint(
        feature_id=feature_id,
        model="m",
        layer=layer,
        text=feature_id,
        text_vector=[],
        activation_signature=[0.0, 1.0],
        decoder_signature=[],
        causal_vector=[],
    )
    return FeatureCard(
        feature_id=feature_id,
        model="m",
        layer=layer,
        label=feature_id,
        explanation="",
        importance=0.5,
        association=0.3,
        specificity=0.1,
        causal_effect=0.1,
        stability=1.0,
        examples=examples or [],
        source="test",
        fingerprint=fingerprint,
        causal_effects=causal_effects,
    )


def test_measured_criterion_edge_never_falls_back_to_signed_association():
    # intervention_record_count alone marks the edge measured, but the only signed
    # value is correlational: the causal edge must publish signed_effect=None.
    report = InspectionReport(
        model="m",
        criterion=Criterion(text="criterion"),
        cards=[
            _graph_card(
                "F1",
                {"criterion": 0.4, "signed_association": 0.4, "intervention_record_count": 3.0},
            )
        ],
    )
    graph = build_attribution_graph(report, include_supernodes=False, include_coactivation_edges=False)
    edge = [e for e in graph["edges"] if e["target"] == "criterion"][0]
    assert edge["type"] == "causal_effect"
    assert edge["evidence"] == "measured_intervention"
    assert edge["signed_effect"] is None


def test_supernode_causal_aggregate_averages_only_intervention_backed_members():
    # Two theme-grouped members: one measured (+0.4), one correlational (-0.4).
    # Averaging both would report 0.0 as a causal aggregate; only the measured
    # member may contribute, and the mixed membership must be labeled.
    examples = ["p1: activation=3.0 | Write Python | token[3]='Python'"]
    report = InspectionReport(
        model="m",
        criterion=Criterion(text="criterion"),
        cards=[
            _graph_card(
                "F1",
                {"signed_causal_effect": 0.4, "strong_causal_score": 0.06},
                examples=examples,
                layer=12,
            ),
            _graph_card(
                "F2",
                {"criterion": 0.4, "signed_association": -0.4},
                examples=examples,
                layer=24,
            ),
        ],
    )
    graph = build_attribution_graph(report, include_coactivation_edges=False)
    theme_edges = [
        e
        for e in graph["edges"]
        if e.get("type") == "aggregate_causal_effect" and str(e.get("source", "")).startswith("supernode:theme:")
    ]
    assert theme_edges
    edge = theme_edges[0]
    assert edge["signed_effect"] == pytest.approx(0.4)
    assert edge["measured_member_count"] == 1
    assert edge["member_count"] == 2
    assert edge["evidence"] == "mixed_intervention_and_association"


def test_association_only_supernode_aggregate_stays_correlational():
    examples = ["p1: activation=3.0 | Write Python | token[3]='Python'"]
    report = InspectionReport(
        model="m",
        criterion=Criterion(text="criterion"),
        cards=[
            _graph_card("F1", {"criterion": 0.3, "signed_association": 0.3}, examples=examples, layer=12),
            _graph_card("F2", {"criterion": 0.5, "signed_association": 0.5}, examples=examples, layer=24),
        ],
    )
    graph = build_attribution_graph(report, include_coactivation_edges=False)
    theme_edges = [
        e
        for e in graph["edges"]
        if str(e.get("source", "")).startswith("supernode:theme:") and e.get("target") == "criterion"
    ]
    assert theme_edges
    assert theme_edges[0]["type"] == "aggregate_criterion_association"
    assert theme_edges[0]["signed_effect"] == pytest.approx(0.4)
    assert theme_edges[0]["measured_member_count"] == 0


# --- LOW: robust-without-controls must not claim it "beat controls" ---

def test_robust_without_controls_gets_honest_reason_and_interpretation():
    graph = {
        "schema_version": "interp-lab.attribution_graph.v1",
        "model": "m",
        "criterion": {"text": "criterion"},
        "edges": [],
        "mechanism_summary": {
            "candidate_paths": [
                {
                    "source_feature_id": "SAE:L1:F1",
                    "target_feature_id": "SAE:L2:F8",
                    "evidence": "path_patch",
                }
            ]
        },
    }
    records = [
        {
            "source_feature_id": "SAE:L1:F1",
            "target_feature_id": "SAE:L2:F8",
            "prompt_id": prompt,
            "target_activation_delta": 0.2,
            "strength": 2.0,
        }
        for prompt in ("p1", "p2", "p3")
    ]
    report = build_graph_validation_report(graph, path_records=records, require_controls=False)
    validation = report["path_validations"][0]
    assert validation["status"] == "robust"
    assert validation["reason_codes"] == ["passed_effect_and_sign_thresholds_no_controls"]
    assert "beat controls" not in validation["interpretation"]
    assert "no control" in validation["interpretation"]


# --- LOW: t_critical must use exact small-df tables for 0.90 / 0.99 ---

def test_t_critical_exact_tables_for_common_confidences():
    assert stats.t_critical(1, 0.99) == 63.657  # Cornish-Fisher gave 28.47 here
    assert stats.t_critical(5, 0.99) == 4.032
    assert stats.t_critical(1, 0.90) == 6.314
    assert stats.t_critical(10, 0.90) == 1.812
    assert stats.t_critical(1, 0.95) == 12.706  # 0.95 behavior unchanged


def test_t_critical_uncovered_confidence_is_conservative_at_small_df():
    # 0.97 at df=3 rounds UP to the 0.99 table (wider == conservative), instead of
    # the anti-conservative expansion.
    assert stats.t_critical(3, 0.97) == 5.841
    # > 0.99 clamps to the 0.99 table at small df (documented limitation).
    assert stats.t_critical(2, 0.995) == 9.925
    # At moderate df the expansion is accurate and still used for uncovered levels.
    assert abs(stats.t_critical(50, 0.99) - 2.678) < 0.02


# --- LOW: token displays must escape real newlines (the old replace was a no-op) ---

def test_top_tokens_escapes_real_newlines():
    card = _graph_card(
        "F1",
        {},
        examples=["p1: activation=3.0 | context | token[3]='Py\nthon'"],
    )
    tokens = _top_tokens(card)
    assert tokens == ["Py\\nthon"]
    assert all("\n" not in token for token in tokens)
