"""Tests for the calibration harness (planted ground truth vs the real pipeline)."""

from __future__ import annotations

import json

import pytest

from interp_lab.adapters.interventions import InterventionRecordRunner, summarize_intervention_file
from interp_lab.adapters.records import ActivationRecordFeatureProvider
from interp_lab.calibration import (
    CALIBRATION_SCHEMA,
    KIND_CAUSAL,
    KIND_DECOY,
    KIND_NOISE,
    TIER_CORRELATIONAL_ONLY,
    TIER_INTERVENTION_NULL,
    TIER_MEASURED_CAUSAL,
    VERDICT_NO_CAUSAL_TRUTH,
    VERDICT_WELL_CALIBRATED,
    export_calibration_report,
    generate_planted_world,
    render_calibration_markdown,
    run_calibration,
    wilson_interval,
)
from interp_lab.schema import Criterion

# Small, fast world settings shared by most tests (full suite stays well under a
# minute; the default-config run is exercised once in test_default_settings_runtime).
SMALL_WORLD = {
    "n_features": 12,
    "n_causal": 4,
    "n_decoys": 4,
    "n_prompts": 32,
    "noise": 0.2,
}


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("calibration-small")
    return run_calibration(out, seeds=[0, 1], **SMALL_WORLD)


def test_schema_constant_and_report_schema(small_run):
    assert CALIBRATION_SCHEMA == "interp-lab.calibration_report.v1"
    assert small_run["schema_version"] == CALIBRATION_SCHEMA


def test_planted_world_artifacts_parse_through_real_loaders(tmp_path):
    world = generate_planted_world(7, **SMALL_WORLD)
    activation_path, interventions_path = world.write(tmp_path)

    evidence = ActivationRecordFeatureProvider(activation_path).features_for(
        world.model, Criterion(text=world.criterion)
    )
    assert len(evidence) == SMALL_WORLD["n_features"]
    assert {item.feature_id for item in evidence} == {f.feature_id for f in world.features}

    counts = summarize_intervention_file(interventions_path)
    assert sum(counts.values()) == len(world.intervention_records)
    assert set(counts) <= {"ablate", "amplify"}

    records = InterventionRecordRunner(interventions_path, require_criterion_match=False)._load_records()
    tested_ids = world.causal_ids | world.decoy_ids
    assert {record.feature_id for record in records} == tested_ids
    # Control records (random_feature / matched_frequency / placebo) are present.
    control_types = {record.control_type for record in records if record.is_control}
    assert control_types == {"random_feature", "matched_frequency", "placebo"}
    # Noise features have no intervention records at all.
    assert not any(record.feature_id in world.noise_ids for record in records)


def test_world_generation_is_deterministic():
    left = generate_planted_world(3, **SMALL_WORLD)
    right = generate_planted_world(3, **SMALL_WORLD)
    assert left.features == right.features
    assert left.activation_records == right.activation_records
    assert left.intervention_records == right.intervention_records
    assert generate_planted_world(4, **SMALL_WORLD).activation_records != left.activation_records


def test_run_calibration_payload_is_deterministic(tmp_path):
    first = run_calibration(tmp_path / "a", seeds=[5], **SMALL_WORLD)
    second = run_calibration(tmp_path / "b", seeds=[5], **SMALL_WORLD)
    # Identical payloads even across different out dirs: no timestamps, no abs paths.
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_high_signal_world_precision_and_decoy_resistance(small_run):
    pooled = small_run["pooled"]
    assert pooled["discovery"]["precision_at_k"] >= 0.9
    assert pooled["discovery"]["recall_at_k"] >= 0.9
    assert pooled["decoy_resistance"]["decoy_resistance"] == 1.0
    assert pooled["decoy_resistance"]["false_causal_count"] == 0
    tier_table = pooled["grade_calibration"]["by_evidence_tier"]
    assert tier_table[TIER_MEASURED_CAUSAL]["p_truly_causal"] == 1.0
    assert small_run["assessment"]["verdict"] == VERDICT_WELL_CALIBRATED


def test_evidence_tiers_track_planted_kinds(small_run):
    rows = [row for seed_result in small_run["per_seed"] for row in seed_result["features"]]
    decoy_tiers = {row["evidence_tier"] for row in rows if row["kind"] == KIND_DECOY}
    noise_tiers = {row["evidence_tier"] for row in rows if row["kind"] == KIND_NOISE}
    causal_tiers = {row["evidence_tier"] for row in rows if row["kind"] == KIND_CAUSAL}
    # Decoys carry intervention records, so they are never mistaken for
    # correlational-only -- but they must never earn measured_causal either.
    assert decoy_tiers == {TIER_INTERVENTION_NULL}
    assert noise_tiers == {TIER_CORRELATIONAL_ONLY}
    assert causal_tiers == {TIER_MEASURED_CAUSAL}


def test_self_match_validated_grade_has_perfect_precision(small_run):
    grade_table = small_run["pooled"]["grade_calibration"]["by_self_match_claim_grade"]
    validated = grade_table.get("validated_equivalent")
    assert validated is not None
    assert validated["p_truly_causal"] == 1.0
    # And no decoy/noise feature graded validated: precision holds row-by-row too.
    rows = [row for seed_result in small_run["per_seed"] for row in seed_result["features"]]
    for row in rows:
        if row["self_match_claim_grade"] == "validated_equivalent":
            assert row["kind"] == KIND_CAUSAL


def test_effect_rank_correlation_is_high(small_run):
    correlation = small_run["pooled"]["effect_rank_correlation"]
    assert correlation["n"] == 2 * SMALL_WORLD["n_causal"]
    assert correlation["spearman"] >= 0.8


def test_calibration_table_counts_sum(small_run):
    seeds = len(small_run["per_seed"])
    expected_total = SMALL_WORLD["n_features"] * seeds
    expected_causal = SMALL_WORLD["n_causal"] * seeds
    for table_name in ("by_evidence_tier", "by_self_match_claim_grade"):
        table = small_run["pooled"]["grade_calibration"][table_name]
        assert sum(cell["count"] for cell in table.values()) == expected_total
        assert sum(cell["truly_causal_count"] for cell in table.values()) == expected_causal
        for cell in table.values():
            assert 0 <= cell["truly_causal_count"] <= cell["count"]
            assert cell["ci_low"] <= cell["p_truly_causal"] <= cell["ci_high"]


def test_pure_noise_world_is_reported_honestly(tmp_path):
    report = run_calibration(
        tmp_path, seeds=[0], n_features=8, n_causal=0, n_decoys=0, n_prompts=32
    )
    pooled = report["pooled"]
    tier_table = pooled["grade_calibration"]["by_evidence_tier"]
    assert TIER_MEASURED_CAUSAL not in tier_table
    assert pooled["discovery"]["precision_at_k"] is None
    assert pooled["decoy_resistance"]["decoy_resistance"] is None
    assessment = report["assessment"]
    assert assessment["verdict"] == VERDICT_NO_CAUSAL_TRUTH
    assert "honestly reported no measured-causal evidence" in assessment["summary"]
    # The markdown renders the honest verdict instead of crashing on None metrics.
    markdown = render_calibration_markdown(report)
    assert VERDICT_NO_CAUSAL_TRUTH in markdown
    assert "n/a" in markdown


def test_markdown_render_contains_key_sections(small_run):
    markdown = render_calibration_markdown(small_run)
    assert "# interp-lab Calibration Report" in markdown
    assert CALIBRATION_SCHEMA in markdown
    assert "## Headline" in markdown
    assert "Decoy resistance" in markdown
    assert TIER_MEASURED_CAUSAL in markdown
    assert "## Per-seed metrics" in markdown
    assert "## Caveats" in markdown
    assert "Synthetic worlds are not real models" in markdown


def test_export_writes_json_and_markdown(tmp_path, small_run):
    created_at = "2026-06-10T00:00:00+00:00"
    result = export_calibration_report(
        tmp_path / "calibration.json",
        tmp_path / "calibration.md",
        report=small_run,
        created_at=created_at,
    )
    assert result.json_path.exists()
    assert result.markdown_path.exists()
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CALIBRATION_SCHEMA
    assert payload["created_at"] == created_at
    assert payload["pooled"] == small_run["pooled"]
    assert result.markdown_path.read_text(encoding="utf-8").startswith(
        "# interp-lab Calibration Report"
    )


def test_export_defaults_markdown_path_and_runs_fresh(tmp_path):
    result = export_calibration_report(
        tmp_path / "report.json",
        work_dir=tmp_path / "worlds",
        seeds=[0],
        **SMALL_WORLD,
    )
    assert result.markdown_path == tmp_path / "report.md"
    assert result.markdown_path.exists()
    assert (tmp_path / "worlds" / "seed-0" / "activation_records.jsonl").exists()
    assert "created_at" in result.report


def test_seeds_accepts_int_count(tmp_path):
    report = run_calibration(
        tmp_path, seeds=2, n_features=6, n_causal=2, n_decoys=2, n_prompts=16
    )
    assert [seed_result["seed"] for seed_result in report["per_seed"]] == [0, 1]
    assert report["config"]["seeds"] == [0, 1]


def test_world_argument_validation():
    with pytest.raises(ValueError):
        generate_planted_world(0, n_features=4, n_causal=5)
    with pytest.raises(ValueError):
        generate_planted_world(0, effect_range=(0.8, 0.2))
    with pytest.raises(ValueError):
        generate_planted_world(0, intervention_repeats=1)
    with pytest.raises(ValueError):
        generate_planted_world(0, n_features=8, n_causal=4, n_decoys=6)
    with pytest.raises(ValueError):
        run_calibration("unused", seeds=[])


def test_wilson_interval_sanity():
    interval = wilson_interval(30, 30)
    assert interval["method"] == "wilson"
    assert 0.85 <= interval["low"] < 1.0
    assert interval["high"] == 1.0
    interval = wilson_interval(0, 60)
    assert interval["low"] == 0.0
    assert 0.0 < interval["high"] <= 0.1
    empty = wilson_interval(0, 0)
    assert empty["method"] == "no_data"
    assert empty["low"] is None and empty["high"] is None
    with pytest.raises(ValueError):
        wilson_interval(5, 3)


def test_artifact_paths_in_payload_are_relative(small_run):
    for seed_result in small_run["per_seed"]:
        artifacts = seed_result["world"]["artifacts"]
        for value in artifacts.values():
            assert not value.startswith("/")
            assert value.startswith(f"seed-{seed_result['seed']}/")


def test_default_settings_runtime_and_headline(tmp_path):
    import time

    start = time.monotonic()
    report = run_calibration(tmp_path)  # full default config: 5 seeds, 24 features
    elapsed = time.monotonic() - start
    assert elapsed < 60.0
    headline = report["assessment"]["headline"]
    assert headline["decoy_resistance"] == 1.0
    assert headline["p_truly_causal_given_measured_causal"] == 1.0
    assert headline["discovery_precision_at_k"] >= 0.9
    assert report["assessment"]["verdict"] == VERDICT_WELL_CALIBRATED
