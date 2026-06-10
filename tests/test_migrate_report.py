"""Tests for migrate_report.py: re-scoring old reports under current semantics."""

import json
from pathlib import Path

from interp_lab import __version__
from interp_lab.migrate_report import (
    _evidence_from_card,
    build_migrate_report_parser,
    migrate_inspection_report,
    run_migrate_report_from_args,
)
from interp_lab.reporting import load_inspection_report
from interp_lab.schema import (
    INSPECTION_REPORT_SCHEMA,
    Criterion,
    FeatureCard,
    FeatureFingerprint,
    InspectionReport,
)
from interp_lab.scoring import score_feature

CRITERION = "the model is aware it is being evaluated"


def _fingerprint(feature_id: str, *, activation_signature: list[float]) -> FeatureFingerprint:
    return FeatureFingerprint(
        feature_id=feature_id,
        model="toy/m",
        layer=3,
        text="evaluation awareness",
        text_vector=[0.5, 0.5],
        activation_signature=activation_signature,
        decoder_signature=[],
        causal_vector=[],
    )


def _card(
    feature_id: str,
    *,
    causal_effects: dict,
    metadata: dict | None = None,
    importance: float,
    causal_effect: float,
    association: float = 0.4,
    specificity: float = 0.0,
    stability: float = 0.5,
    activation_signature: list[float] | None = None,
) -> FeatureCard:
    return FeatureCard(
        feature_id=feature_id,
        model="toy/m",
        layer=3,
        label=f"feature {feature_id}",
        explanation="",
        importance=importance,
        association=association,
        specificity=specificity,
        causal_effect=causal_effect,
        stability=stability,
        examples=["the assistant suspects a test"],
        source="activation-records",
        fingerprint=_fingerprint(
            feature_id,
            activation_signature=[1.0, 0.0] if activation_signature is None else activation_signature,
        ),
        metadata=metadata or {},
        causal_effects=causal_effects,
    )


def _pre23_report() -> InspectionReport:
    """A pre-2.3-style report: the correlational 'criterion' key was counted as
    causal for card A (no intervention provenance), inflating its old scores."""
    inflated = _card(
        "L3:D1",
        causal_effects={"criterion": 0.9, "signed_association": 0.5},
        importance=0.6,
        causal_effect=0.9,
        association=0.5,
    )
    validated = _card(
        "L3:D2",
        causal_effects={
            "criterion": 0.4,
            "signed_causal_effect": 0.4,
            "strong_causal_score": 0.3,
            "specificity": 0.35,
            "signed_association": 0.3,
            "intervention_record_count": 2.0,
        },
        metadata={"interventions": {"count": 2, "mean_directed_effect": 0.4}},
        importance=0.4,
        causal_effect=0.4,
        association=0.3,
        specificity=0.35,
    )
    return InspectionReport(model="toy/m", criterion=Criterion(text=CRITERION), cards=[inflated, validated])


def test_migrate_rescores_unvalidated_causal_to_zero_and_reorders():
    migrated = migrate_inspection_report(_pre23_report().to_dict())

    assert migrated["schema_version"] == INSPECTION_REPORT_SCHEMA
    cards = {card["feature_id"]: card for card in migrated["cards"]}
    # The correlational card loses its borrowed causal credit...
    assert cards["L3:D1"]["causal_effect"] == 0.0
    # ...and the intervention-backed card keeps its measured one and now ranks first.
    assert cards["L3:D2"]["causal_effect"] == 0.4
    assert [card["feature_id"] for card in migrated["cards"]] == ["L3:D2", "L3:D1"]

    migration = migrated["metadata"]["migration"]
    assert migration["from_tool_version"] == "unknown"
    assert migration["to_tool_version"] == __version__
    assert migration["changes"]["features_reordered"] is True
    assert migration["changes"]["score_deltas"]["L3:D1"]["causal_effect"] == [0.9, 0.0]
    # importance changed too, and the deltas record old -> new.
    old, new = migration["changes"]["score_deltas"]["L3:D1"]["importance"]
    assert old == 0.6 and new == cards["L3:D1"]["importance"] and new < old


def test_migrate_records_deltas_only_for_changed_fields():
    # Build a card whose stored scores already match current semantics exactly.
    causal_effects = {
        "criterion": 0.4,
        "signed_causal_effect": 0.4,
        "strong_causal_score": 0.3,
        "specificity": 0.35,
        "signed_association": 0.3,
        "intervention_record_count": 2.0,
    }
    probe = _card(
        "L3:D2",
        causal_effects=causal_effects,
        importance=0.0,
        causal_effect=0.0,
    )
    report = InspectionReport(model="toy/m", criterion=Criterion(text=CRITERION), cards=[probe])
    expected = score_feature(_evidence_from_card(probe), report.criterion)
    current = _card(
        "L3:D2",
        causal_effects=causal_effects,
        importance=expected["importance"],
        causal_effect=expected["causal_effect"],
        association=expected["association"],
        specificity=expected["specificity"],
        stability=expected["stability"],
    )
    aligned = InspectionReport(model="toy/m", criterion=Criterion(text=CRITERION), cards=[current])

    migrated = migrate_inspection_report(aligned.to_dict())

    assert migrated["metadata"]["migration"]["changes"]["score_deltas"] == {}
    assert migrated["metadata"]["migration"]["changes"]["features_reordered"] is False


def test_migrate_is_idempotent():
    first = migrate_inspection_report(_pre23_report().to_dict())
    assert first["metadata"]["migration"]["changes"]["score_deltas"] != {}

    second = migrate_inspection_report(first)

    assert second["metadata"]["migration"]["changes"]["score_deltas"] == {}
    assert second["metadata"]["migration"]["changes"]["features_reordered"] is False
    # The stamp is refreshed, not accumulated.
    assert second["metadata"]["migration"]["to_tool_version"] == __version__
    assert [card["feature_id"] for card in second["cards"]] == [
        card["feature_id"] for card in first["cards"]
    ]
    # Scores are stable across the second pass.
    assert [card["importance"] for card in second["cards"]] == [
        card["importance"] for card in first["cards"]
    ]


def test_migrate_file_round_trip(tmp_path: Path):
    old = _pre23_report().to_dict()
    old["metadata"] = {"tool": {"name": "interp-lab", "version": "2.1.0"}}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(old), encoding="utf-8")
    out_dir = tmp_path / "migrated"

    migrated = migrate_inspection_report(report_path, out=out_dir)

    assert migrated["metadata"]["migration"]["from_tool_version"] == "2.1.0"
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.md").exists()
    written = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert written["metadata"]["migration"]["from_tool_version"] == "2.1.0"
    # write_inspection_report never overwrites an existing tool stamp.
    assert written["metadata"]["tool"]["version"] == "2.1.0"
    # The written report stays a loadable inspection_report.v1 and re-migrates as a no-op.
    loaded = load_inspection_report(out_dir / "report.json")
    assert [card.feature_id for card in loaded.cards] == ["L3:D2", "L3:D1"]
    second = migrate_inspection_report(out_dir / "report.json")
    assert second["metadata"]["migration"]["changes"]["score_deltas"] == {}
    assert second["metadata"]["migration"]["changes"]["features_reordered"] is False


def test_migration_notes_flag_partial_evidence():
    partial = _card(
        "L3:D9",
        causal_effects={"criterion": 0.7},
        importance=0.5,
        causal_effect=0.7,
        activation_signature=[],
    )
    report = InspectionReport(model="toy/m", criterion=Criterion(text=CRITERION), cards=[partial])

    migrated = migrate_inspection_report(report)

    notes = migrated["cards"][0]["metadata"]["migration_notes"]
    assert any("no intervention provenance" in note for note in notes)
    assert any("signed_association" in note for note in notes)
    assert any("activation_signature" in note for note in notes)
    assert migrated["cards"][0]["causal_effect"] == 0.0
    # The validated fixture card, by contrast, carries no provenance note.
    validated = migrate_inspection_report(_pre23_report())
    by_id = {card["feature_id"]: card for card in validated["cards"]}
    assert all(
        "no intervention provenance" not in note
        for note in by_id["L3:D2"]["metadata"].get("migration_notes", [])
    )


def test_migrate_report_parser_runs_end_to_end(tmp_path: Path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_pre23_report().to_dict()), encoding="utf-8")
    out_dir = tmp_path / "out"
    args = build_migrate_report_parser().parse_args(
        ["--report", str(report_path), "--out", str(out_dir), "--json"]
    )

    migrated = run_migrate_report_from_args(args)

    assert args.json is True
    assert (out_dir / "report.json").exists()
    assert migrated["metadata"]["migration"]["to_tool_version"] == __version__
