"""Exposed matcher knobs (--min-score / --weights) and the inline-SVG layer profile."""

from pathlib import Path

from interp_lab import compare, inspect
from interp_lab.cli import _parse_match_weights
from interp_lab.pipeline import inspect_model, match_reports
from interp_lab.adapters.toy import ToyFeatureProvider, ToyInterventionRunner, ToyVerbalizer
from interp_lab.reporting import render_inspection_html

CRITERION = "the model is aware it is being evaluated"


def _report(model: str):
    return inspect_model(
        model=model, criterion_text=CRITERION, feature_provider=ToyFeatureProvider(),
        verbalizer=ToyVerbalizer(), intervention_runner=ToyInterventionRunner(measured=True), top_k=8,
    )


def test_min_score_filters_weak_matches():
    left, right = _report("toy/a"), _report("toy/b")
    unfiltered = match_reports(left, right, top_k=20)
    threshold = max(m.score for m in unfiltered.matches)
    filtered = match_reports(left, right, top_k=20, min_score=threshold)
    assert filtered.matches  # at least the best survives
    assert all(m.score >= threshold for m in filtered.matches)
    assert len(filtered.matches) <= len(unfiltered.matches)


def test_weights_override_changes_score():
    left, right = _report("toy/a"), _report("toy/b")
    default_top = match_reports(left, right, top_k=1).matches[0].score
    text_only_top = match_reports(left, right, top_k=1, weights={"text": 1.0}).matches[0].score
    assert default_top != text_only_top


def test_parse_match_weights_validates():
    assert _parse_match_weights("text=0.4,causal=0.6") == {"text": 0.4, "causal": 0.6}
    import pytest
    with pytest.raises(SystemExit):
        _parse_match_weights("bogus=1.0")
    with pytest.raises(SystemExit):
        _parse_match_weights("text")  # not KEY=VALUE


def test_compare_api_accepts_min_score_and_weights(tmp_path: Path):
    a = inspect("toy/a", CRITERION, backend="toy", top_k=5)
    b = inspect("toy/b", CRITERION, backend="toy", top_k=5)
    result = compare(a, b, min_score=0.0, weights={"text": 0.5, "causal": 0.5})
    assert result.matches


def test_html_report_has_layer_profile_svg():
    html = render_inspection_html(_report("toy/a"))
    assert "Importance by layer" in html
    assert "<svg" in html and "</svg>" in html
    assert "measured causal" in html  # the legend


def test_degenerate_weights_warn_on_cli(tmp_path, capsys):
    from interp_lab.cli import main
    import json
    # Build two reports, then strip causal so --weights causal=1.0 has nothing comparable.
    for side in ("a", "b"):
        rep = _report(f"toy/{side}")
        from interp_lab.reporting import write_inspection_report
        json_path, _ = write_inspection_report(rep, tmp_path / side)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for card in data["cards"]:
            card["fingerprint"]["causal_vector"] = []
            card["fingerprint"]["causal_provenance"] = "none"
        json_path.write_text(json.dumps(data), encoding="utf-8")
    code = main([
        "match", "--left", str(tmp_path / "a" / "report.json"),
        "--right", str(tmp_path / "b" / "report.json"),
        "--weights", "causal=1.0", "--out", str(tmp_path / "m.json"),
    ])
    err = capsys.readouterr().err
    assert code == 0
    assert "none of the --weights component(s)" in err
    assert "causal" in err


def test_legit_weights_do_not_warn(tmp_path, capsys):
    from interp_lab.cli import main
    a, b = _report("toy/a"), _report("toy/b")
    from interp_lab.reporting import write_inspection_report
    ap, _ = write_inspection_report(a, tmp_path / "a")
    bp, _ = write_inspection_report(b, tmp_path / "b")
    main(["match", "--left", str(ap), "--right", str(bp), "--weights", "text=0.6,causal=0.4",
          "--out", str(tmp_path / "m.json")])
    assert "warning" not in capsys.readouterr().err


def test_layer_profile_absent_with_one_feature():
    report = inspect_model(
        model="toy/a", criterion_text=CRITERION, feature_provider=ToyFeatureProvider(feature_count=1),
        verbalizer=ToyVerbalizer(), intervention_runner=ToyInterventionRunner(), top_k=1,
    )
    html = render_inspection_html(report)
    assert "Importance by layer" not in html  # nothing to localize with <2 points
