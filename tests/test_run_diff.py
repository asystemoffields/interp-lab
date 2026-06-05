"""Tests for `compare-runs` / compare_runs: report-to-report diffing."""

from pathlib import Path

from interp_lab import compare_runs, inspect
from interp_lab.adapters.toy import ToyFeatureProvider, ToyInterventionRunner, ToyVerbalizer
from interp_lab.pipeline import inspect_model
from interp_lab.reporting import write_inspection_report
from interp_lab.run_diff import RUN_DIFF_SCHEMA, build_run_diff_report

CRITERION = "the model is aware it is being evaluated"


def _write(model: str, *, measured: bool, out: Path) -> Path:
    report = inspect_model(
        model=model,
        criterion_text=CRITERION,
        feature_provider=ToyFeatureProvider(),
        verbalizer=ToyVerbalizer(),
        intervention_runner=ToyInterventionRunner(measured=measured),
        top_k=8,
    )
    json_path, _ = write_inspection_report(report, out)
    return json_path


def test_identical_runs_report_no_drift(tmp_path: Path):
    a = _write("toy/m", measured=False, out=tmp_path / "a")
    b = _write("toy/m", measured=False, out=tmp_path / "b")
    diff = compare_runs(a, b)
    assert diff["schema_version"] == RUN_DIFF_SCHEMA
    assert diff["summary"]["rank_stability"] == 1.0
    assert diff["summary"]["added_count"] == 0
    assert diff["summary"]["dropped_count"] == 0
    assert diff["summary"]["max_abs_importance_delta"] == 0.0
    assert "identical" in diff["interpretation"].lower()


def test_measured_vs_correlational_shows_movers(tmp_path: Path):
    base = _write("toy/model-a", measured=False, out=tmp_path / "plain")
    cand = _write("toy/model-a", measured=True, out=tmp_path / "measured")
    diff = compare_runs(base, cand)
    # Same model+criterion -> same feature ids, so everything is "changed", not added/dropped.
    assert diff["summary"]["common_count"] == 8
    assert diff["summary"]["added_count"] == 0
    assert diff["changed_features"]
    top = diff["changed_features"][0]
    # Measured mode adds causal evidence, lifting importance.
    assert top["importance_delta"] > 0
    assert "strong_causal_score" in top["score_deltas"]
    # Sorted by absolute importance delta.
    deltas = [abs(item["importance_delta"]) for item in diff["changed_features"]]
    assert deltas == sorted(deltas, reverse=True)


def test_different_criterion_is_all_added_and_dropped(tmp_path: Path):
    a = inspect("toy/m", CRITERION, backend="toy", out=tmp_path / "a", top_k=6)
    b = inspect("toy/m", "the text discusses cooking recipes", backend="toy", out=tmp_path / "b", top_k=6)
    diff = compare_runs(a.json_path, b.json_path)
    assert diff["summary"]["criterion_match"] is False
    assert diff["summary"]["common_count"] == 0
    assert diff["summary"]["added_count"] == 6
    assert diff["summary"]["dropped_count"] == 6


def test_compare_runs_writes_json_and_markdown(tmp_path: Path):
    a = _write("toy/m", measured=False, out=tmp_path / "a")
    b = _write("toy/m", measured=True, out=tmp_path / "b")
    result = compare_runs(a, b, out=tmp_path / "diff.json")
    assert result.json_path.exists()
    assert result.markdown_path.exists()
    text = result.markdown_path.read_text(encoding="utf-8")
    assert "# interp-lab Run Diff" in text
    assert "Biggest movers" in text


def test_build_run_diff_is_pure_function():
    left = inspect_model(
        model="toy/m", criterion_text=CRITERION, feature_provider=ToyFeatureProvider(),
        verbalizer=ToyVerbalizer(), intervention_runner=ToyInterventionRunner(), top_k=4,
    )
    right = inspect_model(
        model="toy/m", criterion_text=CRITERION, feature_provider=ToyFeatureProvider(),
        verbalizer=ToyVerbalizer(), intervention_runner=ToyInterventionRunner(measured=True), top_k=4,
    )
    diff = build_run_diff_report(left, right)
    import json
    assert json.loads(json.dumps(diff)) == diff  # JSON round-trippable
