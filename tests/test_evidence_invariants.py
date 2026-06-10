"""Generative (property-based) invariant suite: correlational evidence must never
surface on a causal-labeled axis, anywhere in the pipeline.

Philosophy
----------
A recent audit closed four separate doors where the records backend's correlational
``signed_association`` leaked onto causal-labeled outputs (scoring, matching,
match_validation, graphs, reporting, explanation_reports). Each door got a point
regression test in ``tests/test_rigor_fixes.py``. This file is the STRUCTURAL fix:
instead of pinning specific inputs, Hypothesis generates arbitrary association-only
evidence (mirroring exactly what ``adapters/records.py`` emits for its
``association_proxy`` convention, plus adversarial extras), runs the REAL pipeline
functions, and asserts over ALL outputs that no causal-flavored field, edge type,
grade, marker, or phrase appears. Positive-control tests generate
intervention-backed evidence and assert the causal surfaces ARE reachable, so the
gates are proven non-vacuous.

The evidence contract being enforced
------------------------------------
Association-only evidence (the records backend without interventions) carries:
``causal_effects`` drawn from {criterion: abs(r), signed_association: r,
specificity, side_effect} and metadata ``causal_evidence="association_proxy"`` --
and NEVER ``signed_causal_effect``, NEVER ``intervention_record_count > 0``, NEVER
``metadata["interventions"]`` counts. (``strong_causal_score`` is deliberately NOT
generated for association-only cards: it is an intervention-derived quantity and no
adapter emits it without interventions; feeding it without provenance is
contradictory input, not a population this suite warrants.)

For such evidence, everywhere in the pipeline:
  1. scoring        -> the causal axis is 0.0; association reflects |r|; importance
                       is monotone non-decreasing in |r| (all else fixed).
  2. matching       -> no ``signed_effect_provenance_intervention`` marker; mixed
                       (intervention-vs-association) pairs get NO signed_effect
                       component, NO published signed effects, and NO 0.49
                       opposite-direction cap (score == pure fingerprint score).
  3. match_validation -> never grades validated / validated_equivalent AND never
                       grades contradicted / contradicted_effect (both are causal
                       claims requiring intervention provenance); reason codes stay
                       inside the documented non-causal set; intervention-backed
                       aligned pairs CAN validate and intervention-backed opposite
                       pairs CAN contradict.
  4. graphs         -> no edge typed causal_effect / aggregate_causal_effect, no
                       evidence label measured_intervention /
                       mixed_intervention_and_association; in mixed populations,
                       causal-typed edges publish ONLY intervention-measured
                       signed effects.
  5. reporting      -> rendered markdown/HTML never contains the causal-verb
                       phrases ("Causal direction", "promoted the criterion",
                       "suppressed the criterion", "changes the behavior score",
                       "Causal readout:", "Evidence: causal intervention records");
                       intervention-backed cards DO produce them.
  6. explanation_reports -> text-pivot grading never returns
                       ``text_pivot_with_causal_support``; the signed-alignment
                       component never compares across provenance.
  7. end-to-end     -> records JSONL -> inspect_model -> serialized report carries
                       no measured-intervention claim at the field level.

How to extend when adding a new output surface
----------------------------------------------
When a new exporter/renderer consumes FeatureCard/FeatureEvidence causal fields:
  1. Add one test here that feeds it ``association_only_cards()`` (or evidence /
     reports built from them) and asserts the new surface's causal-labeled fields,
     types, or phrases are absent. Inspect the new module first and enumerate its
     EXACT causal-claim strings/field names (as FORBIDDEN_CAUSAL_PHRASES does for
     reporting.py) -- prefer field-level assertions over substring grep.
  2. Add a positive control: the same surface over ``intervention_evidence()``
     must be able to produce the causal output (gate is not vacuously closed).
  3. Keep ``settings(max_examples=60, deadline=None, derandomize=True)`` so CI is
     deterministic and the whole file stays under ~60s.
If a generated case exposes a real leak, minimize it, keep it as an
``xfail(strict=False)`` documenting the door, and fix it in src in a separate
change. (The contradicted_effect door found this way is now closed and its test
is a hard invariant below.)

Requires the dev-only dependency ``hypothesis``.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st

from interp_lab.adapters.records import ActivationRecordFeatureProvider
from interp_lab.adapters.toy import ToyInterventionRunner, ToyVerbalizer
from interp_lab.explanation_reports import _text_pivot_match
from interp_lab.fingerprints import build_fingerprint
from interp_lab.graphs import build_attribution_graph
from interp_lab.match_validation import build_match_validation_report
from interp_lab.matching import (
    fingerprint_similarity,
    has_intervention_provenance,
    match_feature_cards,
)
from interp_lab.pipeline import inspect_model
from interp_lab.reporting import (
    render_inspection_html,
    render_inspection_markdown,
    write_inspection_report,
)
from interp_lab.schema import (
    Criterion,
    FeatureCard,
    FeatureEvidence,
    InspectionReport,
    MatchReport,
)
from interp_lab.scoring import score_feature


# CI determinism + speed: every property uses the same settings profile.
INVARIANT_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much, HealthCheck.data_too_large],
)

CRITERION_TEXT = "tracks the target behavior"
CRITERION = Criterion(text=CRITERION_TEXT)

# Causal-typed graph surfaces (graphs.py _criterion_edge / _supernodes).
CAUSAL_EDGE_TYPES = {"causal_effect", "aggregate_causal_effect"}
CAUSAL_EVIDENCE_LABELS = {"measured_intervention", "mixed_intervention_and_association"}

# EXACT current causal phrasings, extracted from reporting.py:
#   _direction_line          -> "Causal direction" (signed_causal_effect only)
#   _card_interpretation_lines -> "... this feature promoted the criterion" /
#                                 "... suppressed the criterion", "Causal readout:"
#   _mechanism_sketch_lines  -> "changes the behavior score" (intervention provenance only)
#   _evidence_line           -> "Evidence: causal intervention records"
# NOTE: "Activation association: promotes criterion (...)" is the honest correlational
# phrasing and is intentionally ALLOWED; the forbidden set is past-tense / measured-claim
# phrasing only. Update this set whenever reporting.py changes its causal wording.
FORBIDDEN_CAUSAL_PHRASES = (
    "Causal direction",
    "promoted the criterion",
    "suppressed the criterion",
    "changes the behavior score",
    "Causal readout:",
    "Evidence: causal intervention records",
)

# The documented non-causal reason codes from match_validation._match_reason_codes,
# plus the two non-validated success codes. The validated success code
# "passed_score_structural_causal_and_signed_effect_thresholds" is deliberately
# EXCLUDED: it must be unreachable for association-only evidence. So is
# "signed_effect_direction_conflict": it is now emitted only for
# intervention-measured opposite effects (the contradicted gate); association-only
# opposite pairs get "opposite_associations_lack_intervention_provenance" instead.
DOCUMENTED_NON_CAUSAL_REASON_CODES = {
    "score_below_threshold",
    "structural_components_below_threshold",
    "missing_causal_component",
    "causal_component_neutral",
    "causal_component_below_threshold",
    "signed_effect_provenance_mismatch",
    "missing_signed_effects",
    "signed_effects_below_threshold",
    "signed_effect_delta_above_threshold",
    "signed_effects_lack_intervention_provenance",
    "opposite_associations_lack_intervention_provenance",
    "passed_structural_thresholds_but_needs_causal_validation",
    "passed_score_threshold_with_limited_component_support",
}


# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------

# Signed associations: arbitrary finite r in [-1, 1] with the interesting boundary
# cases (zero, ties at the 0.02 direction threshold, exact +/-1) over-weighted.
# No NaN/inf: adapters/records.py rejects non-finite numbers at ingestion.
_r_values = st.one_of(
    st.sampled_from([0.0, 1.0, -1.0, 0.5, -0.5, 0.02, -0.02, 0.019, -0.019, 1e-06]),
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
_unit_floats = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
# Generated free text is drawn from a restricted alphabet so it can never collide
# with a FORBIDDEN_CAUSAL_PHRASES entry (keeps the phrase assertions sound).
_safe_text = st.text(alphabet="abcdef ", max_size=12)
_labels = st.one_of(
    st.sampled_from(["", "alpha detector", "beta latent", "feature 7", "zzz qqq"]),
    _safe_text,
)
_layers = st.one_of(st.none(), st.integers(min_value=0, max_value=30))
_signatures = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=0,
    max_size=6,
)
# token[...] examples drive graph theme supernodes and reporting token readouts.
_examples = st.lists(
    st.sampled_from(
        [
            "p1: activation=1.500, criterion_score=1.000 | alpha beta | token[3]='alpha'",
            "p2: activation=0.200, criterion_score=0.000 | beta beta | token[1]='beta'",
            "plain example with no token marker",
            "",
        ]
    ),
    min_size=0,
    max_size=3,
)


@st.composite
def association_only_evidence(draw, *, require_signed: bool = False) -> FeatureEvidence:
    """FeatureEvidence mirroring adapters/records.py's association_proxy convention.

    causal_effects: any subset of {criterion: abs(r), signed_association: r,
    specificity, side_effect}; NEVER signed_causal_effect, NEVER
    intervention_record_count, NEVER metadata["interventions"]. Adversarial extras:
    signed_association sometimes only in metadata, empty labels, r in {0, +/-1},
    ties at the 0.02 threshold, absent signatures.
    """
    r = draw(_r_values)
    causal_effects: dict[str, float] = {}
    if require_signed or draw(st.booleans()):
        causal_effects["signed_association"] = r
    if draw(st.booleans()):
        causal_effects["criterion"] = abs(r)
    if draw(st.booleans()):
        causal_effects["specificity"] = draw(_unit_floats)
    if draw(st.booleans()):
        causal_effects["side_effect"] = draw(_unit_floats)
    metadata: dict[str, object] = {
        "causal_evidence": "association_proxy",
        "record_count": draw(st.integers(min_value=2, max_value=50)),
    }
    if draw(st.booleans()):
        metadata["signed_association"] = r  # the metadata-side convention
    return FeatureEvidence(
        feature_id="F",
        model="m",
        layer=draw(_layers),
        label=draw(_labels),
        examples=draw(_examples),
        activation_signature=draw(_signatures),
        decoder_signature=draw(_signatures),
        causal_effects=causal_effects,
        source="activation-records",
        metadata=metadata,
    )


@st.composite
def intervention_evidence(draw) -> FeatureEvidence:
    """FeatureEvidence WITH measured interventions (signed_causal_effect + counts).

    May adversarially ALSO carry a correlational signed_association (a measured
    card can still have the proxy r in its history); causal-labeled outputs must
    publish only the measured value.
    """
    signed = draw(
        st.one_of(
            st.sampled_from([0.06, -0.06, 0.4, -0.4, 0.9, -0.9, 0.0]),
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        )
    )
    causal_effects: dict[str, float] = {
        "signed_causal_effect": signed,
        "criterion": abs(signed),
    }
    record_count = draw(st.integers(min_value=1, max_value=12))
    if draw(st.booleans()):
        causal_effects["intervention_record_count"] = float(record_count)
    if draw(st.booleans()):
        causal_effects["strong_causal_score"] = draw(_unit_floats)
    if draw(st.booleans()):
        causal_effects["specificity"] = draw(_unit_floats)
    if draw(st.booleans()):
        causal_effects["signed_association"] = draw(_r_values)  # adversarial extra
    metadata: dict[str, object] = {}
    if draw(st.booleans()):
        # Optional control/CI metadata, in the shape feature_interventions attaches.
        metadata["interventions"] = {
            "count": record_count,
            "mean_directed_effect": signed,
            "criterion_ci_low": signed - 0.05,
            "criterion_ci_high": signed + 0.05,
            "controls": {"count": 2, "mean_abs_directed_effect": 0.01},
        }
    return FeatureEvidence(
        feature_id="F",
        model="m",
        layer=draw(_layers),
        label=draw(_labels),
        examples=draw(_examples),
        activation_signature=draw(_signatures),
        decoder_signature=draw(_signatures),
        causal_effects=causal_effects,
        source="test-interventions",
        metadata=metadata,
    )


def _with_unique_ids(items: list[FeatureEvidence], prefix: str) -> list[FeatureEvidence]:
    return [dataclasses.replace(item, feature_id=f"{prefix}{index}") for index, item in enumerate(items)]


def _card(evidence: FeatureEvidence) -> FeatureCard:
    """Build a FeatureCard exactly the way pipeline.inspect_model does (real
    score_feature + build_fingerprint), so card-level invariants test the real path."""
    scores = score_feature(evidence, CRITERION)
    explanation = f"Activation summary: {evidence.label}." if evidence.label else ""
    fingerprint = build_fingerprint(evidence, CRITERION, explanation)
    return FeatureCard(
        feature_id=evidence.feature_id,
        model=evidence.model,
        layer=evidence.layer,
        label=evidence.label,
        explanation=explanation,
        importance=scores["importance"],
        association=scores["association"],
        specificity=scores["specificity"],
        causal_effect=scores["causal_effect"],
        stability=scores["stability"],
        examples=evidence.examples,
        source=evidence.source,
        fingerprint=fingerprint,
        metadata=dict(evidence.metadata),
        causal_effects=dict(evidence.causal_effects),
    )


def association_only_cards(min_size: int = 1, max_size: int = 3):
    return st.lists(association_only_evidence(), min_size=min_size, max_size=max_size).map(
        lambda items: [_card(item) for item in _with_unique_ids(items, "A")]
    )


def intervention_cards(min_size: int = 1, max_size: int = 3):
    return st.lists(intervention_evidence(), min_size=min_size, max_size=max_size).map(
        lambda items: [_card(item) for item in _with_unique_ids(items, "I")]
    )


@st.composite
def mixed_population(draw) -> InspectionReport:
    """An inspection report containing both association-only and measured cards."""
    cards = draw(association_only_cards(1, 3)) + draw(intervention_cards(1, 3))
    return InspectionReport(model="m", criterion=CRITERION, cards=cards)


class _StaticProvider:
    def __init__(self, items: list[FeatureEvidence]):
        self._items = items

    def features_for(self, model: str, criterion: Criterion) -> list[FeatureEvidence]:
        return list(self._items)


def _assert_no_causal_graph_surfaces(graph: dict) -> None:
    for edge in graph["edges"]:
        assert edge.get("type") not in CAUSAL_EDGE_TYPES, edge
        assert edge.get("evidence") not in CAUSAL_EVIDENCE_LABELS, edge


# ---------------------------------------------------------------------------
# 1. scoring: association-only evidence never earns the causal axis
# ---------------------------------------------------------------------------


@INVARIANT_SETTINGS
@given(evidence=association_only_evidence())
def test_scoring_association_only_causal_axis_is_zero(evidence: FeatureEvidence):
    scores = score_feature(evidence, CRITERION)
    assert scores["causal_effect"] == 0.0
    signed = evidence.causal_effects.get("signed_association")
    if signed is not None:
        # The association axis reflects |r| (clamped to [0, 1], rounded to 6dp).
        assert scores["association"] == pytest.approx(min(1.0, abs(signed)), abs=1e-6)


@INVARIANT_SETTINGS
@given(items=st.lists(association_only_evidence(), min_size=1, max_size=3))
def test_pipeline_cards_from_association_only_evidence_have_zero_causal_effect(items):
    report = inspect_model(
        model="m",
        criterion_text=CRITERION_TEXT,
        feature_provider=_StaticProvider(_with_unique_ids(items, "A")),
        verbalizer=ToyVerbalizer(),
        intervention_runner=ToyInterventionRunner(),  # correlational mode: echoes effects
        top_k=8,
    )
    assert report.cards
    for card in report.cards:
        assert card.causal_effect == 0.0
        assert card.fingerprint.causal_provenance != "intervention"
        assert "signed_causal_effect" not in card.causal_effects
        assert "intervention_record_count" not in card.causal_effects
        assert "interventions" not in card.metadata


@INVARIANT_SETTINGS
@given(evidence=association_only_evidence(require_signed=True), r1=_r_values, r2=_r_values)
def test_scoring_importance_monotone_in_association_strength(evidence, r1, r2):
    # Holds by construction for association-only evidence: with the causal axis gated
    # to 0.0, importance = clamp(0.25*|r| + fixed terms), and clamp/round are monotone.
    if abs(r1) > abs(r2):
        r1, r2 = r2, r1
    low = dataclasses.replace(evidence, causal_effects={**evidence.causal_effects, "signed_association": r1})
    high = dataclasses.replace(evidence, causal_effects={**evidence.causal_effects, "signed_association": r2})
    assert score_feature(low, CRITERION)["importance"] <= score_feature(high, CRITERION)["importance"]


# ---------------------------------------------------------------------------
# 2. matching: provenance markers and the mixed-provenance exclusion
# ---------------------------------------------------------------------------


@INVARIANT_SETTINGS
@given(left=association_only_cards(1, 3), right=association_only_cards(1, 3))
def test_matching_association_only_pairs_never_marked_intervention(left, right):
    matches = match_feature_cards(left, right, top_k=len(left) * len(right))
    assert matches  # min_score=0.0: every pair must surface for inspection
    for match in matches:
        components = match.components
        assert "signed_effect_provenance_intervention" not in components
        # Same-provenance population: the mismatch marker must not appear either.
        assert "signed_effect_provenance_mismatch" not in components
        if "signed_effect" in components:
            assert components.get("signed_effect_provenance_association") == 1.0


@INVARIANT_SETTINGS
@given(
    assoc=association_only_evidence(require_signed=True),
    interv=intervention_evidence(),
    assoc_left=st.booleans(),
)
def test_matching_mixed_provenance_excludes_signed_axis_and_cap(assoc, interv, assoc_left):
    assoc_card = _card(dataclasses.replace(assoc, feature_id="A0"))
    interv_card = _card(dataclasses.replace(interv, feature_id="I0"))
    left, right = ([assoc_card], [interv_card]) if assoc_left else ([interv_card], [assoc_card])
    match = match_feature_cards(left, right, top_k=1)[0]
    assert match.components["signed_effect_provenance_mismatch"] == 1.0
    assert "signed_effect" not in match.components
    assert match.left_signed_effect is None
    assert match.right_signed_effect is None
    # Neither blended into the score nor capped at 0.49 as a phantom contradiction:
    # the score must equal the pure fingerprint similarity.
    pure_score, _ = fingerprint_similarity(left[0].fingerprint, right[0].fingerprint)
    assert match.score == pure_score


# ---------------------------------------------------------------------------
# 3. match_validation: association-only evidence can never validate
# ---------------------------------------------------------------------------


@INVARIANT_SETTINGS
@given(left=association_only_cards(1, 3), right=association_only_cards(1, 3))
def test_match_validation_association_only_never_grades_validated(left, right):
    matches = match_feature_cards(left, right, top_k=len(left) * len(right))
    report = build_match_validation_report(MatchReport(left_model="lm", right_model="rm", matches=matches))
    assert report["summary"]["validated_count"] == 0
    for row in report["validations"]:
        assert row["status"] != "validated"
        assert row["claim_grade"] != "validated_equivalent"
        # "contradicted" is a causal claim too: unreachable without interventions.
        assert row["status"] != "contradicted"
        assert row["claim_grade"] != "contradicted_effect"
        assert row["signed_effect_provenance"] != "intervention"
        assert set(row["reason_codes"]) <= DOCUMENTED_NON_CAUSAL_REASON_CODES, row["reason_codes"]


def _aligned_intervention_evidence(signed: float) -> FeatureEvidence:
    return FeatureEvidence(
        feature_id="F",
        model="m",
        layer=3,
        label="alpha detector",
        examples=["p1: activation=1.500, criterion_score=1.000 | alpha beta | token[3]='alpha'"],
        activation_signature=[0.6, 0.2, 0.1],
        decoder_signature=[0.3, 0.4],
        causal_effects={
            "criterion": 0.8,
            "signed_causal_effect": signed,
            "specificity": 0.5,
            "intervention_record_count": 4.0,
        },
        source="test-interventions",
        metadata={"interventions": {"count": 4, "mean_directed_effect": signed}},
    )


@INVARIANT_SETTINGS
@given(
    signed=st.floats(min_value=0.05, max_value=0.85, allow_nan=False, allow_infinity=False),
    delta=st.floats(min_value=0.0, max_value=0.14, allow_nan=False, allow_infinity=False),
)
def test_match_validation_intervention_backed_aligned_pairs_can_validate(signed, delta):
    # Positive control: the validated gate must not be vacuously closed. Two
    # near-identical intervention-backed cards with aligned signed effects
    # (same direction, delta <= 0.15) must reach "validated".
    left = _card(dataclasses.replace(_aligned_intervention_evidence(signed), feature_id="L"))
    right = _card(
        dataclasses.replace(_aligned_intervention_evidence(min(0.95, signed + delta)), feature_id="R")
    )
    matches = match_feature_cards([left], [right], top_k=1)
    report = build_match_validation_report(MatchReport(left_model="lm", right_model="rm", matches=matches))
    row = report["validations"][0]
    assert row["status"] == "validated"
    assert row["claim_grade"] == "validated_equivalent"
    assert row["signed_effect_provenance"] == "intervention"


# HARD INVARIANT (formerly an xfail documenting a real finding): "contradicted" /
# "contradicted_effect" is a causal claim and now requires intervention provenance
# on the compared signed effects, exactly like "validated". Two association-only
# cards with opposite signed associations (both |r| >= 0.02) are evidence AGAINST
# equivalence, but with zero interventions the causal claim is untested: they grade
# needs_causal_evidence / needs_more_evidence with the dedicated reason code
# opposite_associations_lack_intervention_provenance. matching's 0.49
# same-provenance opposite-direction score cap is unchanged (a similarity penalty,
# not a claim).
@INVARIANT_SETTINGS
@given(magnitude=st.floats(min_value=0.02, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_match_validation_association_only_never_grades_contradicted_effect(magnitude):
    left = _card(
        dataclasses.replace(
            _aligned_intervention_evidence(0.0),  # reuse the shape, replace the effects
            feature_id="L",
            causal_effects={"criterion": magnitude, "signed_association": magnitude},
            metadata={"causal_evidence": "association_proxy"},
        )
    )
    right = _card(
        dataclasses.replace(
            _aligned_intervention_evidence(0.0),
            feature_id="R",
            causal_effects={"criterion": magnitude, "signed_association": -magnitude},
            metadata={"causal_evidence": "association_proxy"},
        )
    )
    matches = match_feature_cards([left], [right], top_k=1)
    report = build_match_validation_report(MatchReport(left_model="lm", right_model="rm", matches=matches))
    row = report["validations"][0]
    assert row["status"] != "contradicted"
    assert row["claim_grade"] != "contradicted_effect"
    assert row["status"] == "needs_causal_evidence"
    assert row["claim_grade"] == "needs_more_evidence"
    assert "opposite_associations_lack_intervention_provenance" in row["reason_codes"]
    assert "signed_effect_direction_conflict" not in row["reason_codes"]


@INVARIANT_SETTINGS
@given(magnitude=st.floats(min_value=0.02, max_value=0.9, allow_nan=False, allow_infinity=False))
def test_match_validation_intervention_backed_opposite_pairs_still_grade_contradicted(magnitude):
    # Positive control: the contradicted gate must not be vacuously closed.
    # Intervention-measured opposite signed effects DO refute causal equivalence.
    left = _card(dataclasses.replace(_aligned_intervention_evidence(magnitude), feature_id="L"))
    right = _card(dataclasses.replace(_aligned_intervention_evidence(-magnitude), feature_id="R"))
    matches = match_feature_cards([left], [right], top_k=1)
    report = build_match_validation_report(MatchReport(left_model="lm", right_model="rm", matches=matches))
    row = report["validations"][0]
    assert row["status"] == "contradicted"
    assert row["claim_grade"] == "contradicted_effect"
    assert row["signed_effect_provenance"] == "intervention"
    assert "signed_effect_direction_conflict" in row["reason_codes"]


# ---------------------------------------------------------------------------
# 4. graphs: no causal-typed edges without interventions
# ---------------------------------------------------------------------------


@INVARIANT_SETTINGS
@given(cards=association_only_cards(1, 4))
def test_graphs_association_only_reports_have_no_causal_edges(cards):
    report = InspectionReport(model="m", criterion=CRITERION, cards=cards)
    graph = build_attribution_graph(report)  # defaults: supernodes + coactivation on
    _assert_no_causal_graph_surfaces(graph)
    for edge in graph["edges"]:
        if edge.get("target") == "criterion" and edge.get("type") not in {
            "aggregate_criterion_association",
        }:
            assert edge["type"] == "criterion_association"
            assert edge["evidence"] == "activation_criterion_association"
    # Feature nodes echo the card's (scored) causal axis, which must be 0.0.
    for node in graph["nodes"]:
        if node.get("type") == "feature":
            assert node["causal_effect"] == 0.0


@INVARIANT_SETTINGS
@given(report=mixed_population())
def test_graphs_mixed_population_causal_edges_only_publish_measured_effects(report):
    graph = build_attribution_graph(report)
    cards_by_node_id = {f"feature:{card.model}:{card.feature_id}": card for card in report.cards}
    cards_by_feature_id = {card.feature_id: card for card in report.cards}
    members_by_supernode = {
        node["id"]: node["member_feature_ids"] for node in graph["nodes"] if node.get("type") == "supernode"
    }
    for edge in graph["edges"]:
        if edge.get("type") == "causal_effect":
            card = cards_by_node_id[edge["source"]]
            assert has_intervention_provenance(card.causal_effects, card.metadata)
            assert edge.get("evidence") == "measured_intervention"
            measured = card.causal_effects.get("signed_causal_effect")
            if measured is None:
                # Never falls back to a correlational signed_association.
                assert edge["signed_effect"] is None
            else:
                assert edge["signed_effect"] == pytest.approx(measured)
        elif edge.get("type") == "aggregate_causal_effect":
            members = [cards_by_feature_id[fid] for fid in members_by_supernode[edge["source"]]]
            measured_signed = [
                member.causal_effects["signed_causal_effect"]
                for member in members
                if "signed_causal_effect" in member.causal_effects
                and has_intervention_provenance(member.causal_effects, member.metadata)
            ]
            if measured_signed:
                assert edge["signed_effect"] == pytest.approx(
                    sum(measured_signed) / len(measured_signed), abs=1e-6
                )
            else:
                assert edge["signed_effect"] is None
        elif edge.get("type") == "criterion_association":
            card = cards_by_node_id[edge["source"]]
            # An intervention-backed card must never be downgraded onto (or leak
            # through) the association edge type.
            assert not has_intervention_provenance(card.causal_effects, card.metadata)


# ---------------------------------------------------------------------------
# 5. reporting: causal-verb phrases require intervention provenance
# ---------------------------------------------------------------------------


@INVARIANT_SETTINGS
@given(cards=association_only_cards(1, 4))
def test_reporting_association_only_markdown_and_html_have_no_causal_phrases(cards):
    report = InspectionReport(model="m", criterion=CRITERION, cards=cards)
    markdown = render_inspection_markdown(report)
    html = render_inspection_html(report)
    for phrase in FORBIDDEN_CAUSAL_PHRASES:
        assert phrase not in markdown, phrase
        assert phrase not in html, phrase


@INVARIANT_SETTINGS
@given(
    magnitude=st.floats(min_value=0.06, max_value=0.95, allow_nan=False, allow_infinity=False),
    positive=st.booleans(),
)
def test_reporting_intervention_backed_cards_do_render_causal_phrases(magnitude, positive):
    # Positive control: with real intervention provenance the causal phrasing is
    # allowed and must actually appear (the suppression is provenance-gated, not blanket).
    signed = magnitude if positive else -magnitude
    evidence = dataclasses.replace(
        _aligned_intervention_evidence(signed),
        causal_effects={
            "criterion": abs(signed),
            "signed_causal_effect": signed,
            "strong_causal_score": 0.2,
            "specificity": 0.5,
            "intervention_record_count": 4.0,
        },
    )
    report = InspectionReport(model="m", criterion=CRITERION, cards=[_card(evidence)])
    markdown = render_inspection_markdown(report)
    assert "Causal direction" in markdown
    assert ("promoted the criterion" if signed > 0 else "suppressed the criterion") in markdown
    assert "changes the behavior score" in markdown  # mechanism sketch, intervention-gated


# ---------------------------------------------------------------------------
# 6. explanation_reports: text-pivot grading is provenance-gated
# ---------------------------------------------------------------------------


def _pivot_item(card: FeatureCard, path: str) -> dict:
    return {"card": card, "rank": 1, "total": 1, "path": path}


@INVARIANT_SETTINGS
@given(left=association_only_cards(1, 1), right=association_only_cards(1, 1))
def test_text_pivot_association_only_never_grades_causal_support(left, right):
    match = _text_pivot_match(_pivot_item(left[0], "l.json"), _pivot_item(right[0], "r.json"), min_text_score=0.55)
    assert match["evidence_grade"] != "text_pivot_with_causal_support"
    assert "signed_effect_provenance_intervention" not in match["components"]
    assert match["components"]["causal_evidence"] == 0.0


@INVARIANT_SETTINGS
@given(
    assoc=association_only_evidence(require_signed=True),
    interv=intervention_evidence(),
    assoc_left=st.booleans(),
)
def test_text_pivot_alignment_never_compares_across_provenance(assoc, interv, assoc_left):
    assoc_card = _card(dataclasses.replace(assoc, feature_id="A0"))
    interv_card = _card(dataclasses.replace(interv, feature_id="I0"))
    left, right = (assoc_card, interv_card) if assoc_left else (interv_card, assoc_card)
    match = _text_pivot_match(_pivot_item(left, "l.json"), _pivot_item(right, "r.json"), min_text_score=0.55)
    components = match["components"]
    # Mixed provenance: the signed axis is excluded (0.0 placeholder), tagged as a
    # mismatch, and can never count as intervention-backed causal support.
    assert components["signed_effect"] == 0.0
    assert components.get("signed_effect_provenance_mismatch") == 1.0
    assert "signed_effect_provenance_intervention" not in components
    assert match["evidence_grade"] != "text_pivot_with_causal_support"


@INVARIANT_SETTINGS
@given(signed=st.floats(min_value=0.05, max_value=0.85, allow_nan=False, allow_infinity=False))
def test_text_pivot_intervention_backed_pairs_can_grade_causal_support(signed):
    # Positive control: two matching intervention-backed cards must be able to reach
    # the causal-support grade, so the gate is not vacuously closed.
    left = _card(dataclasses.replace(_aligned_intervention_evidence(signed), feature_id="L"))
    right = _card(dataclasses.replace(_aligned_intervention_evidence(signed), feature_id="R"))
    match = _text_pivot_match(_pivot_item(left, "l.json"), _pivot_item(right, "r.json"), min_text_score=0.55)
    assert match["evidence_grade"] == "text_pivot_with_causal_support"
    assert match["components"]["signed_effect_provenance_intervention"] == 1.0


# ---------------------------------------------------------------------------
# 7. End-to-end: records JSONL -> inspect_model -> serialized report
# ---------------------------------------------------------------------------


@st.composite
def activation_record_rows(draw) -> list[dict]:
    feature_count = draw(st.integers(min_value=1, max_value=3))
    record_count = draw(st.integers(min_value=2, max_value=5))
    feature_ids = [f"L{index + 1}:F{index}" for index in range(feature_count)]
    rows = []
    for record_index in range(record_count):
        rows.append(
            {
                "model": "m",
                "prompt_id": f"p{record_index}",
                "text": draw(st.sampled_from(["alpha beta", "beta", "cafe dab", "fee fad"])),
                "criterion_score": draw(
                    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
                ),
                "features": {
                    feature_id: draw(
                        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
                    )
                    for feature_id in feature_ids
                },
            }
        )
    return rows


@INVARIANT_SETTINGS
@given(rows=activation_record_rows())
def test_end_to_end_association_only_records_never_serialize_measured_claims(rows):
    with tempfile.TemporaryDirectory() as tmp:
        records_path = Path(tmp) / "records.jsonl"
        records_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        report = inspect_model(
            model="m",
            criterion_text=CRITERION_TEXT,
            feature_provider=ActivationRecordFeatureProvider(records_path),
            verbalizer=ToyVerbalizer(),
            intervention_runner=ToyInterventionRunner(),  # correlational mode
            top_k=8,
        )
        json_path, markdown_path = write_inspection_report(report, Path(tmp) / "out")
        raw = json_path.read_text(encoding="utf-8")
        data = json.loads(raw)

        assert data["cards"]
        for card in data["cards"]:
            causal_effects = card["causal_effects"]
            # Field-level: no key that marks a measured intervention may exist.
            assert "signed_causal_effect" not in causal_effects
            assert "intervention_record_count" not in causal_effects
            assert "strong_causal_score" not in causal_effects
            assert "interventions" not in card["metadata"]
            assert card["metadata"]["causal_evidence"] == "association_proxy"
            assert card["fingerprint"]["causal_provenance"] == "association"
            assert card["causal_effect"] == 0.0
            if "signed_association" in causal_effects:
                assert card["association"] == pytest.approx(
                    min(1.0, abs(causal_effects["signed_association"])), abs=1e-6
                )
        # Backstop on the serialized form: the measured-intervention evidence label
        # must not appear anywhere (all generated text uses a safe alphabet).
        assert "measured_intervention" not in raw

        markdown = markdown_path.read_text(encoding="utf-8")
        for phrase in FORBIDDEN_CAUSAL_PHRASES:
            assert phrase not in markdown, phrase

        # And the graph built from the round-tripped report stays correlational.
        graph = build_attribution_graph(InspectionReport.from_dict(data))
        _assert_no_causal_graph_surfaces(graph)
