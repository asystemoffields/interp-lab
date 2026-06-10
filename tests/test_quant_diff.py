"""Tests for quant-diff: which validated features does quantization break?"""

import json
from pathlib import Path

import pytest

from interp_lab.cli import main as cli_main
from interp_lab.match_validation import build_match_validation_report
from interp_lab.pipeline import match_reports
from interp_lab.quant_diff import (
    QUANT_DIFF_SCHEMA,
    build_quant_diff,
    build_quant_diff_parser,
    export_quant_diff,
    render_quant_diff_markdown,
    run_quant_diff_from_args,
)
from interp_lab.reporting import write_inspection_report
from interp_lab.runs import RunOptions, run_config_file
from interp_lab.schema import Criterion, FeatureCard, FeatureFingerprint, InspectionReport
from interp_lab.workflows import quant_diff_workflow

CRITERION = "the model is aware it is being evaluated"
DIMS = 8


def _vec(index: int) -> list[float]:
    values = [0.0] * DIMS
    values[index] = 1.0
    return values


def _card(
    feature_id: str,
    model: str,
    label: str,
    vec: list[float],
    *,
    importance: float,
    association: float = 0.5,
    causal_effect: float = 0.5,
    signed: float | None = None,
    provenance: str = "none",
    layer: int = 4,
) -> FeatureCard:
    causal_effects: dict[str, float] = {}
    causal_vector: list[float] = []
    if provenance == "intervention":
        causal_effects = {
            "signed_causal_effect": signed,
            "intervention_record_count": 2.0,
            "criterion": abs(signed),
            "strong_causal_score": 0.8,
        }
        causal_vector = [0.9, signed, 0.8, 0.1]
    elif provenance == "association":
        causal_effects = {"signed_association": signed, "criterion": abs(signed)}
        causal_vector = [0.9, signed, 0.8, 0.1]
    fingerprint = FeatureFingerprint(
        feature_id=feature_id,
        model=model,
        layer=layer,
        text=label,
        text_vector=vec,
        activation_signature=vec,
        decoder_signature=vec,
        causal_vector=causal_vector,
        text_embedder="hash-v1",
        causal_provenance=provenance,
    )
    return FeatureCard(
        feature_id=feature_id,
        model=model,
        layer=layer,
        label=label,
        explanation=label,
        importance=importance,
        association=association,
        specificity=0.5,
        causal_effect=causal_effect,
        stability=0.5,
        examples=[label],
        source="records",
        fingerprint=fingerprint,
        causal_effects=causal_effects,
    )


def _report(model: str, cards: list[FeatureCard], criterion: str = CRITERION) -> InspectionReport:
    return InspectionReport(model=model, criterion=Criterion(text=criterion), cards=cards)


def _planted_reports() -> tuple[InspectionReport, InspectionReport]:
    """Reports with one planted outcome per verdict.

    Baseline (f16)                              Variant (q4_k_m)
    L:P1  validated, effect +0.50           ->  R:P1  validated, effect +0.50 (preserved)
    L:D1  validated, effect +0.60           ->  R:D1  effect FLIPPED to -0.60 (degraded)
    L:D2  validated, effect +0.50           ->  R:D2  intervention evidence LOST (degraded)
    L:E1  association-only, +0.40           ->  R:E1  association +0.42 (preserved_correlational)
    L:F1  association-only, +0.50           ->  R:F1  association FLIPPED (changed_correlational)
    L:LO  validated, effect +0.50           ->  (no counterpart: lost)
                                                R:EM  association-only (emerged)
    """
    left = _report(
        "llama-3-8b-f16",
        [
            _card("L:P1", "llama-3-8b-f16", "benchmark detector", _vec(0), importance=0.80, association=0.70, causal_effect=0.60, signed=0.50, provenance="intervention"),
            _card("L:D1", "llama-3-8b-f16", "trap detector", _vec(1), importance=0.70, signed=0.60, provenance="intervention"),
            _card("L:D2", "llama-3-8b-f16", "eval framing", _vec(2), importance=0.65, signed=0.50, provenance="intervention"),
            _card("L:E1", "llama-3-8b-f16", "test vocabulary", _vec(3), importance=0.50, signed=0.40, provenance="association"),
            _card("L:F1", "llama-3-8b-f16", "graded answer style", _vec(4), importance=0.45, signed=0.50, provenance="association"),
            _card("L:LO", "llama-3-8b-f16", "scoring rubric", _vec(5), importance=0.90, signed=0.50, provenance="intervention"),
        ],
    )
    right = _report(
        "llama-3-8b-q4_k_m",
        [
            _card("R:P1", "llama-3-8b-q4_k_m", "benchmark detector", _vec(0), importance=0.75, association=0.65, causal_effect=0.55, signed=0.50, provenance="intervention"),
            _card("R:D1", "llama-3-8b-q4_k_m", "trap detector", _vec(1), importance=0.68, signed=-0.60, provenance="intervention"),
            _card("R:D2", "llama-3-8b-q4_k_m", "eval framing", _vec(2), importance=0.60, signed=0.50, provenance="association"),
            _card("R:E1", "llama-3-8b-q4_k_m", "test vocabulary", _vec(3), importance=0.48, signed=0.42, provenance="association"),
            _card("R:F1", "llama-3-8b-q4_k_m", "graded answer style", _vec(4), importance=0.44, signed=-0.50, provenance="association"),
            _card("R:EM", "llama-3-8b-q4_k_m", "quantization noise", _vec(6), importance=0.30, signed=0.30, provenance="association"),
        ],
    )
    return left, right


def _by_left_id(report: dict) -> dict[str, dict]:
    return {entry["left_feature_id"]: entry for entry in report["features"]}


def test_planted_verdicts():
    left, right = _planted_reports()
    diff = build_quant_diff(left, right)

    assert diff["schema_version"] == QUANT_DIFF_SCHEMA
    entries = _by_left_id(diff)
    assert entries["L:P1"]["verdict"] == "preserved"
    assert entries["L:P1"]["validation_status"] == "validated"
    assert entries["L:D1"]["verdict"] == "degraded"
    assert "signed_effect_direction_flipped" in entries["L:D1"]["reasons"]
    assert entries["L:D2"]["verdict"] == "degraded"
    assert "right_lost_intervention_evidence" in entries["L:D2"]["reasons"]
    assert entries["L:E1"]["verdict"] == "preserved_correlational"
    assert entries["L:F1"]["verdict"] == "changed_correlational"
    assert "association_direction_flipped" in entries["L:F1"]["reasons"]
    assert [item["feature_id"] for item in diff["lost_features"]] == ["L:LO"]
    assert diff["lost_features"][0]["intervention_validated"] is True
    assert [item["feature_id"] for item in diff["emerged_features"]] == ["R:EM"]


def test_headline_counts_and_degraded_validated():
    left, right = _planted_reports()
    diff = build_quant_diff(left, right)

    summary = diff["summary"]
    assert summary["features_compared"] == 5
    assert summary["preserved_count"] == 1
    assert summary["degraded_count"] == 2
    assert summary["preserved_correlational_count"] == 1
    assert summary["changed_correlational_count"] == 1
    assert summary["lost_count"] == 1
    assert summary["emerged_count"] == 1
    assert summary["validated_lost_count"] == 1
    assert sorted(item["left_feature_id"] for item in summary["degraded_validated"]) == ["L:D1", "L:D2"]
    # Every degraded entry is intervention-validated on the baseline side by construction.
    assert all(item["left_intervention_validated"] for item in summary["degraded_validated"])
    assert "broke" in diff["interpretation"] or "degraded" in diff["interpretation"]


def test_verdict_thresholds_are_documented():
    left, right = _planted_reports()
    diff = build_quant_diff(left, right)

    thresholds = diff["summary"]["verdict_thresholds"]
    assert set(thresholds) == {
        "min_match_score",
        "min_structural_component",
        "min_structural_components",
        "max_importance_drop",
        "max_signed_effect_drop",
        "min_abs_signed_effect",
    }
    # min_match_score must stay below matching's 0.49 opposite-direction cap so a
    # sign-flipped feature classifies as degraded rather than lost.
    assert thresholds["min_match_score"] < 0.49
    markdown = render_quant_diff_markdown(diff)
    assert "## Verdict thresholds" in markdown
    assert "min_match_score" in markdown


def test_deltas_are_right_minus_left():
    left, right = _planted_reports()
    diff = build_quant_diff(left, right)

    deltas = _by_left_id(diff)["L:P1"]["deltas"]
    assert deltas["importance_delta"] == pytest.approx(-0.05)
    assert deltas["association_delta"] == pytest.approx(-0.05)
    assert deltas["causal_effect_delta"] == pytest.approx(-0.05)


def test_signed_effect_provenance_discipline():
    left, right = _planted_reports()
    diff = build_quant_diff(left, right)

    entries = _by_left_id(diff)
    # Mixed provenance (intervention vs association) is labeled, never numerically compared.
    mixed = entries["L:D2"]["signed_effect_comparison"]
    assert mixed["compared"] is False
    assert mixed["note"] == "mixed_provenance_not_compared"
    assert mixed["delta"] is None
    # Same-provenance intervention pairs ARE compared.
    flipped = entries["L:D1"]["signed_effect_comparison"]
    assert flipped["compared"] is True
    assert flipped["provenance"] == "intervention"
    assert flipped["direction_flip"] is True
    # Association-only pairs never reach degraded_validated, even when they move.
    degraded_ids = {item["left_feature_id"] for item in diff["summary"]["degraded_validated"]}
    assert "L:F1" not in degraded_ids
    assert "L:E1" not in degraded_ids
    assert entries["L:F1"]["verdict"] == "changed_correlational"


def test_in_process_matching_equals_precomputed_artifacts():
    left, right = _planted_reports()
    match_report = match_reports(left, right, top_k=len(left.cards) * len(right.cards), min_score=0.0)
    validation = build_match_validation_report(match_report)

    in_process = build_quant_diff(left, right)
    precomputed = build_quant_diff(left, right, matches=match_report, match_validation=validation)

    in_process.pop("created_at")
    precomputed.pop("created_at")
    assert in_process == precomputed


def test_requires_same_criterion():
    left, _ = _planted_reports()
    other = _report("llama-3-8b-q4_k_m", [], criterion="the text discusses cooking recipes")
    with pytest.raises(ValueError, match="SAME criterion"):
        build_quant_diff(left, other)


def test_model_ids_may_differ_and_labels_flow_through():
    left, right = _planted_reports()
    diff = build_quant_diff(left, right, left_label="f16", right_label="q4_k_m")
    assert diff["left"]["model"] == "llama-3-8b-f16"
    assert diff["right"]["model"] == "llama-3-8b-q4_k_m"
    assert diff["left"]["label"] == "f16"
    assert diff["right"]["label"] == "q4_k_m"


def test_raising_min_match_score_turns_flip_into_lost():
    left, right = _planted_reports()
    diff = build_quant_diff(left, right, min_match_score=0.95)
    # The sign-flipped pair (capped at 0.49 by matching) no longer pairs, so the
    # baseline feature reads as lost -- exactly why the default stays below 0.49.
    lost_ids = {item["feature_id"] for item in diff["lost_features"]}
    assert "L:D1" in lost_ids
    assert diff["summary"]["degraded_count"] < build_quant_diff(left, right)["summary"]["degraded_count"]


def test_no_degraded_features_reads_clean():
    left, right = _planted_reports()
    keep_left = [card for card in left.cards if card.feature_id in {"L:P1", "L:E1"}]
    keep_right = [card for card in right.cards if card.feature_id in {"R:P1", "R:E1"}]
    diff = build_quant_diff(_report(left.model, keep_left), _report(right.model, keep_right))
    assert diff["summary"]["degraded_count"] == 0
    assert diff["summary"]["degraded_validated"] == []
    markdown = render_quant_diff_markdown(diff)
    assert "None -- no intervention-validated baseline feature" in markdown


def test_agent_next_actions_use_canonical_shape():
    left, right = _planted_reports()
    diff = build_quant_diff(left, right)

    actions = {action["id"]: action for action in diff["agent_next_actions"]}
    assert "instruction" in actions["review_quant_diff"]
    assert "command" not in actions["review_quant_diff"]
    plan = actions["plan_evidence_for_degraded"]
    assert plan["argv"][:2] == ["interp-lab", "plan-evidence"]
    assert plan["command"].startswith("interp-lab plan-evidence")
    reexport = actions["reexport_records_at_higher_precision"]
    assert reexport["argv"][:2] == ["interp-lab", "export-gguf-records"]
    for action in actions.values():
        assert action["id"] and action["title"]
        assert ("argv" in action and "command" in action) or "instruction" in action


def test_export_writes_json_and_markdown_with_broken_table_first(tmp_path: Path):
    left, right = _planted_reports()
    left_path, _ = write_inspection_report(left, tmp_path / "left")
    right_path, _ = write_inspection_report(right, tmp_path / "right")

    report = export_quant_diff(
        left_path,
        right_path,
        tmp_path / "quant-diff.json",
        left_label="f16",
        right_label="q4_k_m",
    )

    assert report["schema_version"] == QUANT_DIFF_SCHEMA
    loaded = json.loads((tmp_path / "quant-diff.json").read_text(encoding="utf-8"))
    assert loaded["summary"]["degraded_count"] == 2
    markdown = (tmp_path / "quant-diff.md").read_text(encoding="utf-8")
    assert "# Quantization Feature Diff" in markdown
    # The broken-features table is front and center: before the full verdict table.
    assert markdown.index("## Features broken by quantization") < markdown.index("## All matched features")
    assert "`L:D1`" in markdown
    assert "`L:LO`" in markdown  # validated-but-lost features surface in the broken table too
    assert "no_acceptable_match_in_variant" in markdown


def test_cli_shaped_args_accept_precomputed_artifacts(tmp_path: Path):
    left, right = _planted_reports()
    left_path, _ = write_inspection_report(left, tmp_path / "left")
    right_path, _ = write_inspection_report(right, tmp_path / "right")
    match_report = match_reports(left, right, top_k=36, min_score=0.0)
    matches_path = tmp_path / "matches.json"
    matches_path.write_text(json.dumps(match_report.to_dict()), encoding="utf-8")
    validation_path = tmp_path / "match-validation.json"
    validation_path.write_text(
        json.dumps(build_match_validation_report(match_report)), encoding="utf-8"
    )

    args = build_quant_diff_parser().parse_args(
        [
            "--left-report", str(left_path),
            "--right-report", str(right_path),
            "--matches", str(matches_path),
            "--match-validation", str(validation_path),
            "--out", str(tmp_path / "qd.json"),
            "--left-label", "f16",
            "--right-label", "q4_k_m",
        ]
    )
    report = run_quant_diff_from_args(args)

    assert report["summary"]["degraded_count"] == 2
    assert (tmp_path / "qd.json").exists()
    assert (tmp_path / "qd.md").exists()


def _write_records(path: Path, model: str, scale: float) -> None:
    rows = [
        {
            "model": model,
            "prompt_id": "p1",
            "text": "benchmark prompt",
            "criterion_score": 1,
            "features": {"L1:F1": 0.9 * scale, "L1:F2": 0.4},
            "feature_metadata": {
                "L1:F1": {"label": "benchmark awareness", "layer": 1},
                "L1:F2": {"label": "ordinary answer", "layer": 1},
            },
        },
        {
            "model": model,
            "prompt_id": "p2",
            "text": "ordinary prompt",
            "criterion_score": 0,
            "features": {"L1:F1": 0.1 * scale, "L1:F2": 0.5},
            "feature_metadata": {
                "L1:F1": {"label": "benchmark awareness", "layer": 1},
                "L1:F2": {"label": "ordinary answer", "layer": 1},
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_quant_diff_workflow_config_shape(tmp_path: Path):
    config = quant_diff_workflow(
        tmp_path / "left.jsonl",
        tmp_path / "right.jsonl",
        CRITERION,
        tmp_path / "run",
        model_left="m-f16",
        model_right="m-q4",
        top_k=4,
    )

    assert config["out"] == str(tmp_path / "run")
    names = [step["name"] for step in config["steps"]]
    assert names == ["inspect-baseline", "inspect-variant", "match", "validate-matches", "quant-diff"]
    commands = [step["command"] for step in config["steps"]]
    assert commands == ["inspect", "inspect", "match", "validate-matches", "quant-diff"]
    inspect_left = config["steps"][0]["args"]
    assert inspect_left["backend"] == "records"
    assert inspect_left["model"] == "m-f16"
    assert inspect_left["criterion"] == CRITERION
    # The match step keeps every candidate pair so quant-diff's per-feature
    # best-match selection cannot lose survivors to a top-k cut.
    assert config["steps"][2]["args"]["top_k"] == 16
    final = config["steps"][4]["args"]
    assert final["left_report"] == "{run_dir}/baseline-report/report.json"
    assert final["matches"] == "{run_dir}/matches.json"
    assert final["match_validation"] == "{run_dir}/match-validation.json"
    assert final["left_label"] == "m-f16"

    with pytest.raises(ValueError, match="top_k"):
        quant_diff_workflow("a", "b", CRITERION, "out", top_k=0)


def test_quant_diff_workflow_executes_end_to_end(tmp_path: Path):
    left_records = tmp_path / "left.jsonl"
    right_records = tmp_path / "right.jsonl"
    _write_records(left_records, "m-f16", scale=1.0)
    _write_records(right_records, "m-q4", scale=0.6)
    run_dir = tmp_path / "run"
    config = quant_diff_workflow(
        left_records,
        right_records,
        "benchmark awareness",
        run_dir,
        model_left="m-f16",
        model_right="m-q4",
        top_k=4,
    )
    config_path = tmp_path / "quant-diff-run.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    captured: list[list[str]] = []

    def runner(argv: list[str]) -> int:
        if argv[0] == "quant-diff":
            # The CLI subcommand lands in the integration pass; execute the step
            # through the module's own CLI-shaped entry point in the meantime.
            captured.append(argv)
            run_quant_diff_from_args(build_quant_diff_parser().parse_args(argv[1:]))
            return 0
        return cli_main(argv)

    exit_code = run_config_file(RunOptions(config_path=config_path), command_runner=runner)

    assert exit_code == 0
    assert (run_dir / "baseline-report" / "report.json").exists()
    assert (run_dir / "variant-report" / "report.json").exists()
    assert (run_dir / "matches.json").exists()
    assert (run_dir / "match-validation.json").exists()
    assert (run_dir / "quant-diff.json").exists()
    assert (run_dir / "quant-diff.md").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert [step["status"] for step in manifest["steps"]] == ["succeeded"] * 5

    # Final-step argv shape: this is the wiring contract for the CLI subcommand.
    assert len(captured) == 1
    argv = captured[0]
    assert argv[0] == "quant-diff"
    flags = {argv[index] for index in range(1, len(argv), 2)}
    assert flags == {
        "--left-report",
        "--right-report",
        "--matches",
        "--match-validation",
        "--left-label",
        "--right-label",
        "--out",
        "--markdown-out",
    }
    diff = json.loads((run_dir / "quant-diff.json").read_text(encoding="utf-8"))
    assert diff["schema_version"] == QUANT_DIFF_SCHEMA
    assert diff["left"]["label"] == "m-f16"
    assert diff["summary"]["features_compared"] >= 1


def test_preset_parses_matches_builder_and_templates(tmp_path: Path):
    from interp_lab.runs import _render_value, load_run_config

    preset_path = Path(__file__).resolve().parents[1] / "examples" / "presets" / "quant-diff-run.json"
    config = load_run_config(preset_path)

    # The preset is exactly the workflow builder applied to the documented variables.
    assert config == quant_diff_workflow(
        "${left_records}",
        "${right_records}",
        "${criterion}",
        "reports/quant-diff/${name}",
    )

    rendered = _render_value(
        config,
        {
            "config_dir": str(preset_path.parent),
            "run_dir": "reports/quant-diff/smoke",
            "left_records": str(tmp_path / "left.jsonl"),
            "right_records": str(tmp_path / "right.jsonl"),
            "criterion": "benchmark awareness",
            "name": "smoke",
        },
    )
    text = json.dumps(rendered)
    assert "${" not in text
    assert "{run_dir}" not in text
    assert rendered["out"] == "reports/quant-diff/smoke"
    assert rendered["steps"][0]["args"]["records"] == str(tmp_path / "left.jsonl")
    assert rendered["steps"][4]["args"]["out"] == "reports/quant-diff/smoke/quant-diff.json"
