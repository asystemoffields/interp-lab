"""Tests for criterion dossiers: cumulative evidence across runs."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from interp_lab.cli import build_parser
from interp_lab.dossier import (
    DOSSIER_SCHEMA,
    dossier_summary,
    load_dossier,
    update_dossier,
    write_dossier_markdown,
)
from interp_lab.reporting import write_inspection_report, write_match_report
from interp_lab.schema import (
    CandidateMatch,
    Criterion,
    FeatureCard,
    FeatureFingerprint,
    InspectionReport,
    MatchReport,
)

MODEL = "toy/model-a"
CRITERION = "the model is aware it is being evaluated"

T1 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
T3 = datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc)


def _fingerprint(feature_id: str, model: str) -> FeatureFingerprint:
    return FeatureFingerprint(
        feature_id=feature_id,
        model=model,
        layer=3,
        text="probe text",
        text_vector=[0.1, 0.2],
        activation_signature=[0.3, 0.4],
        decoder_signature=[0.5, 0.6],
        causal_vector=[0.7],
    )


def _card(
    feature_id: str,
    *,
    model: str = MODEL,
    signed: float | None = None,
    provenance: str | None = None,
    importance: float = 0.5,
) -> FeatureCard:
    causal_effects: dict[str, float] = {}
    if provenance == "intervention":
        causal_effects["signed_causal_effect"] = float(signed or 0.0)
        causal_effects["intervention_record_count"] = 4.0
    elif provenance == "association":
        causal_effects["signed_association"] = float(signed or 0.0)
    return FeatureCard(
        feature_id=feature_id,
        model=model,
        layer=3,
        label=f"label for {feature_id}",
        explanation="",
        importance=importance,
        association=0.4,
        specificity=0.3,
        causal_effect=0.2,
        stability=0.6,
        examples=["token[3]='probe'"],
        source="activation-records",
        fingerprint=_fingerprint(feature_id, model),
        causal_effects=causal_effects,
    )


def _report(cards: list[FeatureCard], *, model: str = MODEL, criterion: str = CRITERION) -> InspectionReport:
    return InspectionReport(model=model, criterion=Criterion(text=criterion), cards=cards)


def _write_report(cards: list[FeatureCard], out_dir: Path, **kwargs) -> Path:
    json_path, _md = write_inspection_report(_report(cards, **kwargs), out_dir)
    return json_path


def _write_match_validation(path: Path, *, grades: dict[str, str], model: str = MODEL) -> Path:
    """Minimal artifact matching match_validation.build_match_validation_report's shape."""
    validations = [
        {
            "left_feature_id": feature_id,
            "right_feature_id": f"R:{feature_id}",
            "left_model": model,
            "right_model": "toy/model-b",
            "claim_grade": grade,
            "status": "validated" if grade == "validated_equivalent" else "weak",
            "score": 0.9,
        }
        for feature_id, grade in grades.items()
    ]
    validated = sum(1 for grade in grades.values() if grade == "validated_equivalent")
    data = {
        "schema_version": "interp-lab.match_validation.v1",
        "left_model": model,
        "right_model": "toy/model-b",
        "summary": {
            "match_count": len(validations),
            "validated_count": validated,
            "contradicted_count": 0,
            "needs_causal_evidence_count": 0,
            "overall_claim_grade": "validated_matches_present" if validated else "weak_matches_only",
        },
        "validations": validations,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _write_matches(path: Path, *, model: str = MODEL) -> Path:
    report = MatchReport(
        left_model=model,
        right_model="toy/model-b",
        matches=[
            CandidateMatch(
                left_feature_id="L3:D7",
                right_feature_id="L3:D9",
                left_model=model,
                right_model="toy/model-b",
                score=0.91,
                components={"text": 0.9},
            )
        ],
    )
    return write_match_report(report, path)


def _write_graph_validation(path: Path) -> Path:
    data = {
        "schema_version": "interp-lab.graph_validation.v1",
        "model": MODEL,
        "criterion": CRITERION,
        "summary": {
            "candidate_count": 3,
            "path_record_count": 12,
            "status_counts": {"validated": 1, "needs_controls": 2},
            "overall_claim_grade": "validated_paths_present",
        },
        "path_validations": [],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# --- create -> update -> update flow ----------------------------------------


def test_create_then_update_flow_with_injected_clock(tmp_path: Path):
    dossier_path = tmp_path / "dossier.json"
    update_dossier(dossier_path, _report([_card("L3:D7")]), now=T1)
    update_dossier(dossier_path, _report([_card("L3:D7"), _card("L3:D8")]), now=T2)
    dossier = update_dossier(dossier_path, _report([_card("L3:D7")]), note="third pass", now=T3)

    assert dossier["schema_version"] == DOSSIER_SCHEMA
    assert dossier["model"] == MODEL
    assert dossier["criterion"] == CRITERION
    assert dossier["created_at"] == T1.isoformat()
    assert dossier["updated_at"] == T3.isoformat()
    assert [run["timestamp"] for run in dossier["runs"]] == [
        T1.isoformat(),
        T2.isoformat(),
        T3.isoformat(),
    ]
    assert dossier["runs"][-1]["note"] == "third pass"
    assert dossier["summary"]["run_count"] == 3
    assert dossier["summary"]["span"] == {"first": T1.isoformat(), "last": T3.isoformat()}
    features = dossier["rollup"]["features"]
    assert features["L3:D7"]["run_count"] == 3
    assert features["L3:D7"]["first_seen"] == T1.isoformat()
    assert features["L3:D7"]["last_seen"] == T3.isoformat()
    # D8 only appeared in the middle run and was never re-ranked.
    assert features["L3:D8"]["run_count"] == 1
    assert features["L3:D8"]["last_seen"] == T2.isoformat()
    assert len(features["L3:D7"]["score_trajectory"]) == 3


def test_update_returns_dict_equal_to_file(tmp_path: Path):
    dossier_path = tmp_path / "dossier.json"
    returned = update_dossier(dossier_path, _report([_card("L3:D7")]), now=T1)
    assert load_dossier(dossier_path) == returned
    assert not dossier_path.with_suffix(".tmp").exists()


# --- identity guard ----------------------------------------------------------


def test_model_mismatch_rejected_with_both_identities(tmp_path: Path):
    dossier_path = tmp_path / "dossier.json"
    update_dossier(dossier_path, _report([_card("L3:D7")]), now=T1)
    other = _report([_card("L3:D7", model="toy/model-b")], model="toy/model-b")
    with pytest.raises(ValueError) as excinfo:
        update_dossier(dossier_path, other, now=T2)
    message = str(excinfo.value)
    assert MODEL in message
    assert "toy/model-b" in message
    # The failed append must not mutate the dossier.
    assert load_dossier(dossier_path)["summary"]["run_count"] == 1


def test_criterion_mismatch_rejected_with_both_identities(tmp_path: Path):
    dossier_path = tmp_path / "dossier.json"
    update_dossier(dossier_path, _report([_card("L3:D7")]), now=T1)
    other = _report([_card("L3:D7")], criterion="the model refuses harmful requests")
    with pytest.raises(ValueError) as excinfo:
        update_dossier(dossier_path, other, now=T2)
    message = str(excinfo.value)
    assert CRITERION in message
    assert "refuses harmful requests" in message


# --- grade transitions -------------------------------------------------------


def test_grade_upgrade_via_match_validation(tmp_path: Path):
    dossier_path = tmp_path / "dossier.json"
    update_dossier(dossier_path, _report([_card("L3:D7", signed=0.2, provenance="association")]), now=T1)
    validation = _write_match_validation(tmp_path / "mv.json", grades={"L3:D7": "validated_equivalent"})
    dossier = update_dossier(
        dossier_path,
        _report([_card("L3:D7", signed=0.2, provenance="association")]),
        match_validation=validation,
        now=T2,
    )
    feature = dossier["rollup"]["features"]["L3:D7"]
    assert feature["grade_history"] == [
        [T1.isoformat(), "associational"],
        [T2.isoformat(), "validated_equivalent"],
    ]
    assert feature["grade_transition"] == "upgraded"
    assert dossier["summary"]["currently_validated"] == 1
    assert dossier["summary"]["regressions"] == 0
    # The run entry records the claim grade for the matching model side.
    assert dossier["runs"][-1]["features"]["L3:D7"]["claim_grade"] == "validated_equivalent"


def test_grade_downgrade_counts_regression(tmp_path: Path):
    dossier_path = tmp_path / "dossier.json"
    good = _write_match_validation(tmp_path / "mv1.json", grades={"L3:D7": "validated_equivalent"})
    bad = _write_match_validation(tmp_path / "mv2.json", grades={"L3:D7": "weak_match"})
    update_dossier(dossier_path, _report([_card("L3:D7")]), match_validation=good, now=T1)
    dossier = update_dossier(dossier_path, _report([_card("L3:D7")]), match_validation=bad, now=T2)
    feature = dossier["rollup"]["features"]["L3:D7"]
    assert feature["grade_transition"] == "downgraded"
    assert dossier["summary"]["currently_validated"] == 0
    assert dossier["summary"]["regressions"] == 1


def test_new_feature_transition_and_stable(tmp_path: Path):
    dossier_path = tmp_path / "dossier.json"
    first = update_dossier(dossier_path, _report([_card("L3:D7")]), now=T1)
    assert first["rollup"]["features"]["L3:D7"]["grade_transition"] == "new"
    second = update_dossier(dossier_path, _report([_card("L3:D7")]), now=T2)
    assert second["rollup"]["features"]["L3:D7"]["grade_transition"] == "stable"


# --- sign flips and provenance ------------------------------------------------


def test_intervention_sign_flip_is_contradiction(tmp_path: Path):
    dossier_path = tmp_path / "dossier.json"
    update_dossier(dossier_path, _report([_card("L3:D7", signed=0.2, provenance="intervention")]), now=T1)
    dossier = update_dossier(
        dossier_path, _report([_card("L3:D7", signed=-0.2, provenance="intervention")]), now=T2
    )
    feature = dossier["rollup"]["features"]["L3:D7"]
    assert feature["sign_flips"] == 1
    assert feature["provenance_changes"] == 0
    assert feature["contradiction"] is True
    assert dossier["summary"]["contradictions"] == 1


def test_association_sign_flip_counts_but_is_not_contradiction(tmp_path: Path):
    dossier_path = tmp_path / "dossier.json"
    update_dossier(dossier_path, _report([_card("L3:D7", signed=0.2, provenance="association")]), now=T1)
    dossier = update_dossier(
        dossier_path, _report([_card("L3:D7", signed=-0.2, provenance="association")]), now=T2
    )
    feature = dossier["rollup"]["features"]["L3:D7"]
    assert feature["sign_flips"] == 1
    assert feature["contradiction"] is False
    assert dossier["summary"]["contradictions"] == 0


def test_provenance_change_is_not_a_contradiction(tmp_path: Path):
    dossier_path = tmp_path / "dossier.json"
    update_dossier(dossier_path, _report([_card("L3:D7", signed=0.2, provenance="association")]), now=T1)
    dossier = update_dossier(
        dossier_path, _report([_card("L3:D7", signed=-0.2, provenance="intervention")]), now=T2
    )
    feature = dossier["rollup"]["features"]["L3:D7"]
    assert feature["sign_flips"] == 0
    assert feature["provenance_changes"] == 1
    assert feature["contradiction"] is False


# --- hashes and attached artifacts --------------------------------------------


def test_report_hash_recorded_for_path_input(tmp_path: Path):
    report_path = _write_report([_card("L3:D7")], tmp_path / "run1")
    dossier = update_dossier(tmp_path / "dossier.json", report_path, now=T1)
    run = dossier["runs"][0]
    assert run["report_path"] == str(report_path)
    assert run["report_hash"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    # Object input carries no path, so no hash is invented.
    dossier = update_dossier(tmp_path / "dossier.json", _report([_card("L3:D7")]), now=T2)
    assert dossier["runs"][-1]["report_path"] is None
    assert dossier["runs"][-1]["report_hash"] is None


def test_attached_artifacts_record_paths_hashes_and_headline_stats(tmp_path: Path):
    matches_path = _write_matches(tmp_path / "matches.json")
    validation_path = _write_match_validation(
        tmp_path / "mv.json", grades={"L3:D7": "needs_more_evidence"}
    )
    graph_path = _write_graph_validation(tmp_path / "gv.json")
    dossier = update_dossier(
        tmp_path / "dossier.json",
        _report([_card("L3:D7")]),
        matches=matches_path,
        match_validation=validation_path,
        graph_validation=graph_path,
        now=T1,
    )
    attached = dossier["runs"][0]["attached"]
    assert attached["matches"]["sha256"] == hashlib.sha256(matches_path.read_bytes()).hexdigest()
    assert attached["matches"]["match_count"] == 1
    assert attached["matches"]["left_model"] == MODEL
    assert attached["match_validation"]["validated_count"] == 0
    assert attached["match_validation"]["match_count"] == 1
    assert attached["match_validation"]["sha256"] == hashlib.sha256(validation_path.read_bytes()).hexdigest()
    assert attached["graph_validation"]["candidate_count"] == 3
    assert attached["graph_validation"]["path_record_count"] == 12
    assert attached["graph_validation"]["status_counts"] == {"validated": 1, "needs_controls": 2}
    # needs_more_evidence keeps the feature in the open-questions count.
    assert dossier["summary"]["open_questions"] == 1


# --- atomic write --------------------------------------------------------------


def test_failed_write_leaves_existing_dossier_intact(tmp_path: Path, monkeypatch):
    dossier_path = tmp_path / "dossier.json"
    update_dossier(dossier_path, _report([_card("L3:D7")]), now=T1)
    before = dossier_path.read_bytes()

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    # Fail after the rollup, during serialization for the tmp file: the
    # tmp+rename pattern must leave the published dossier untouched.
    monkeypatch.setattr("interp_lab.dossier.json.dumps", _boom)
    with pytest.raises(OSError):
        update_dossier(dossier_path, _report([_card("L3:D7")]), now=T2)
    monkeypatch.undo()
    assert dossier_path.read_bytes() == before
    assert load_dossier(dossier_path)["summary"]["run_count"] == 1


# --- summary, markdown, loading -------------------------------------------------


def test_dossier_summary_correctness(tmp_path: Path):
    dossier_path = tmp_path / "dossier.json"
    validation = _write_match_validation(tmp_path / "mv.json", grades={"L3:D7": "validated_equivalent"})
    update_dossier(dossier_path, _report([_card("L3:D7"), _card("L3:D8")]), now=T1)
    dossier = update_dossier(
        dossier_path,
        _report([_card("L3:D7"), _card("L3:D8")]),
        match_validation=validation,
        now=T2,
    )
    summary = dossier_summary(dossier)
    assert summary["model"] == MODEL
    assert summary["criterion"] == CRITERION
    assert summary["run_count"] == 2
    assert summary["features_total"] == 2
    assert summary["currently_validated"] == 1
    assert summary["open_questions"] == 1  # D8 is still associational-only
    assert summary["features"]["L3:D7"]["current_grade"] == "validated_equivalent"
    assert summary["features"]["L3:D7"]["grade_transition"] == "upgraded"
    assert summary["features"]["L3:D8"]["current_grade"] == "associational"


def test_markdown_render(tmp_path: Path):
    dossier_path = tmp_path / "dossier.json"
    update_dossier(dossier_path, _report([_card("L3:D7", signed=0.2, provenance="intervention")]), now=T1)
    dossier = update_dossier(
        dossier_path,
        _report([_card("L3:D7", signed=-0.2, provenance="intervention")]),
        note="checking the flip",
        now=T2,
    )
    md_path = write_dossier_markdown(dossier, tmp_path / "dossier.md")
    text = md_path.read_text(encoding="utf-8")
    assert MODEL in text
    assert CRITERION in text
    assert "L3:D7" in text
    assert "causal_evidence" in text
    assert "YES" in text  # contradiction column
    assert "checking the flip" in text
    assert "## Agent Next Actions" in text


def test_load_dossier_rejects_bad_inputs(tmp_path: Path):
    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text(
        json.dumps({"schema_version": "interp-lab.dossier.v999", "model": "m", "criterion": "c"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_dossier(bad_schema)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid dossier JSON"):
        load_dossier(bad_json)
    missing_identity = tmp_path / "missing.json"
    missing_identity.write_text(json.dumps({"schema_version": DOSSIER_SCHEMA}), encoding="utf-8")
    with pytest.raises(ValueError, match="model and criterion"):
        load_dossier(missing_identity)


# --- agent next actions ----------------------------------------------------------


def test_next_actions_parse_against_cli_where_commands_exist(tmp_path: Path):
    matches_path = _write_matches(tmp_path / "matches.json")
    report_path = _write_report([_card("L3:D7")], tmp_path / "run1")
    dossier = update_dossier(
        tmp_path / "dossier.json", report_path, matches=matches_path, now=T1
    )
    actions = dossier["agent_next_actions"]
    action_ids = [action["id"] for action in actions]
    assert "plan_evidence_latest_report" in action_ids
    assert "reinspect_criterion" in action_ids
    # Matches attached but unvalidated -> the dossier suggests validating them.
    assert "validate_attached_matches" in action_ids

    parser = build_parser()
    existing_commands = {"inspect", "validate-matches"}
    for action in actions:
        assert action["id"] and action["title"]
        assert isinstance(action["argv"], list) and action["argv"][0] == "interp-lab"
        assert action["command"]
        if action["argv"][1] in existing_commands:
            # Must parse against the real CLI today.
            parser.parse_args(action["argv"][1:])
        else:
            # plan-evidence ships with the integration pass; assert shape only.
            assert action["argv"][1] == "plan-evidence"
            assert "--report" in action["argv"]
            assert str(report_path) in action["argv"]


def test_validate_matches_action_absent_when_validation_already_attached(tmp_path: Path):
    matches_path = _write_matches(tmp_path / "matches.json")
    validation_path = _write_match_validation(tmp_path / "mv.json", grades={"L3:D7": "weak_match"})
    dossier = update_dossier(
        tmp_path / "dossier.json",
        _report([_card("L3:D7")]),
        matches=matches_path,
        match_validation=validation_path,
        now=T1,
    )
    action_ids = [action["id"] for action in dossier["agent_next_actions"]]
    assert "validate_attached_matches" not in action_ids
