"""Notebook ergonomics: chainable results, compact reprs, table accessors, loaders."""

from pathlib import Path

import interp_lab
from interp_lab import (
    compare,
    inspect,
    load_inspection_report,
    load_match_report,
    validate_matches,
)
from interp_lab.schema import InspectionReport, MatchReport

CRITERION = "the model is aware it is being evaluated"


def test_written_results_chain_through_the_api(tmp_path: Path):
    # The README workflow must work passing the Written* wrappers directly.
    a = inspect("toy/a", CRITERION, backend="toy", out=tmp_path / "a", top_k=4)
    b = inspect("toy/b", CRITERION, backend="toy", out=tmp_path / "b", top_k=4)
    matches = compare(a, b, out=tmp_path / "m.json")          # WrittenInspection -> compare
    validation = validate_matches(matches, out=tmp_path / "v.json")  # WrittenMatch -> validate
    assert validation.report["summary"]["match_count"] > 0
    assert validation.json_path.exists()


def test_compare_accepts_in_memory_reports():
    a = inspect("toy/a", CRITERION, backend="toy", top_k=3)
    b = inspect("toy/b", CRITERION, backend="toy", top_k=3)
    assert isinstance(a, InspectionReport)
    result = compare(a, b)
    assert isinstance(result, MatchReport)
    assert result.matches


def test_reprs_are_compact_and_hide_vectors():
    report = inspect("toy/a", CRITERION, backend="toy", top_k=2)
    # A notebook cell must not flood the output with float vectors.
    for text in (repr(report), repr(report.cards[0]), repr(report.cards[0].fingerprint)):
        assert len(text) < 200
        assert "0.0," not in text  # no raw vector contents
    assert "cards=2" in repr(report)


def test_cards_table_is_stdlib_friendly():
    report = inspect("toy/a", CRITERION, backend="toy", top_k=3)
    rows = report.cards_table()
    assert len(rows) == 3
    assert rows[0]["rank"] == 1
    assert {"feature_id", "importance", "causal_provenance", "strong_causal_score"} <= set(rows[0])
    # Vector-free: every value is a JSON scalar.
    for value in rows[0].values():
        assert value is None or isinstance(value, (str, int, float))


def test_matches_table_is_stdlib_friendly():
    a = inspect("toy/a", CRITERION, backend="toy", top_k=3)
    b = inspect("toy/b", CRITERION, backend="toy", top_k=3)
    rows = compare(a, b).matches_table()
    assert rows and rows[0]["rank"] == 1
    assert {"left_feature_id", "right_feature_id", "score"} <= set(rows[0])


def test_loaders_round_trip_from_top_level(tmp_path: Path):
    written = inspect("toy/a", CRITERION, backend="toy", out=tmp_path / "a", top_k=2)
    loaded = load_inspection_report(written.json_path)
    assert isinstance(loaded, InspectionReport)
    assert [c.feature_id for c in loaded.cards] == [c.feature_id for c in written.report.cards]
    matches = compare(written.report, written.report, out=tmp_path / "m.json")
    assert isinstance(load_match_report(matches.json_path), MatchReport)


def test_package_ships_py_typed_marker():
    marker = Path(interp_lab.__file__).with_name("py.typed")
    assert marker.exists()


def test_writing_a_report_does_not_mutate_the_in_memory_report(tmp_path: Path):
    # The version/platform stamp must land in the file, never in the caller's report.
    report = inspect("toy/a", CRITERION, backend="toy", top_k=2)
    before = report.to_dict()
    written = inspect("toy/a", CRITERION, backend="toy", out=tmp_path / "a", top_k=2)
    # to_dict() stays idempotent across a write; no 'tool' key leaks into the live object.
    assert report.to_dict() == before
    assert "tool" not in written.report.metadata
    # ...but the file on disk IS stamped.
    import json
    assert "tool" in json.loads(written.json_path.read_text(encoding="utf-8"))["metadata"]


def test_analysis_subcommands_help_works():
    # These used add_help=False builders, so `<cmd> --help` errored (exit 2). Now it works.
    import pytest
    from interp_lab.cli import main
    for cmd in ("compare-runs", "search-features", "compare-model-families",
                "match-text-pivot", "check-explanation-consistency"):
        with pytest.raises(SystemExit) as exc:
            main([cmd, "--help"])
        assert int(exc.value.code or 0) == 0, cmd
