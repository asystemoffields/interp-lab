import json
from pathlib import Path

from interp_lab.cli import main
from interp_lab.match_validation import (
    build_match_validation_report,
    export_match_validation_report,
    render_match_validation_html,
    render_match_validation_markdown,
)
from interp_lab.schema import CandidateMatch, MatchReport


def test_match_validation_grades_cross_model_equivalence_claims():
    report = _match_report()

    validation = build_match_validation_report(report)

    assert validation["schema_version"] == "interp-lab.match_validation.v1"
    assert validation["summary"]["status_counts"] == {
        "contradicted": 1,
        "needs_causal_evidence": 1,
        "validated": 1,
        "weak": 1,
    }
    assert validation["summary"]["validated_count"] == 1
    assert validation["summary"]["needs_causal_evidence_count"] == 1
    assert validation["summary"]["contradicted_count"] == 1
    assert validation["summary"]["overall_claim_grade"] == "validated_matches_present"
    assert any(action["id"] == "replicate_validated_matches" for action in validation["agent_next_actions"])

    by_pair = {
        (item["left_feature_id"], item["right_feature_id"]): item
        for item in validation["validations"]
    }
    assert by_pair[("L1:F1", "R3:F8")]["status"] == "validated"
    assert by_pair[("L1:F1", "R3:F8")]["claim_grade"] == "validated_equivalent"
    assert by_pair[("L1:F1", "R3:F8")]["reason_codes"] == [
        "passed_score_structural_causal_and_signed_effect_thresholds"
    ]
    assert by_pair[("L1:F2", "R3:F9")]["status"] == "needs_causal_evidence"
    assert "missing_signed_effects" in by_pair[("L1:F2", "R3:F9")]["reason_codes"]
    assert by_pair[("L1:F3", "R3:F10")]["status"] == "contradicted"
    assert by_pair[("L1:F3", "R3:F10")]["claim_grade"] == "contradicted_effect"
    assert "signed_effect_direction_conflict" in by_pair[("L1:F3", "R3:F10")]["reason_codes"]
    assert by_pair[("L1:F4", "R3:F11")]["status"] == "weak"
    assert "score_below_threshold" in by_pair[("L1:F4", "R3:F11")]["reason_codes"]


def test_match_validation_markdown_and_export(tmp_path: Path):
    matches_path = tmp_path / "matches.json"
    matches_path.write_text(json.dumps(_match_report().to_dict()), encoding="utf-8")

    result = export_match_validation_report(
        matches_path=matches_path,
        out_path=tmp_path / "validation.json",
        html_out_path=tmp_path / "validation.html",
        top_k=2,
    )

    assert result.json_path == tmp_path / "validation.json"
    assert result.markdown_path == tmp_path / "validation.md"
    assert result.html_path == tmp_path / "validation.html"
    loaded = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["match_count"] == 2
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "Cross-Model Match Validation" in markdown
    assert "validated_equivalent" in markdown
    assert "`L1:F1 -> R3:F8`" in markdown
    assert "## Agent Next Actions" in markdown

    rendered = render_match_validation_markdown(loaded)
    assert "needs_more_evidence" in rendered
    html = result.html_path.read_text(encoding="utf-8")
    assert "Cross-Model Match Validation" in html
    assert "match-search" in html
    assert "status-filter" in html
    assert "validated_equivalent" in html
    assert "L1:F1" in html
    assert "visibleCount" in html
    assert "signed_effect_direction_conflict" not in render_match_validation_html(loaded)


def test_validate_matches_cli_writes_json_and_markdown(tmp_path: Path):
    matches_path = tmp_path / "matches.json"
    matches_path.write_text(json.dumps(_match_report().to_dict()), encoding="utf-8")

    exit_code = main(
        [
            "validate-matches",
            "--matches",
            str(matches_path),
            "--out",
            str(tmp_path / "validation.json"),
            "--html-out",
            str(tmp_path / "validation.html"),
            "--top-k",
            "1",
        ]
    )

    assert exit_code == 0
    validation = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))
    assert validation["summary"]["match_count"] == 1
    assert validation["validations"][0]["status"] == "validated"
    assert (tmp_path / "validation.md").exists()
    assert (tmp_path / "validation.html").exists()


def _match_report() -> MatchReport:
    return MatchReport(
        left_model="left/model",
        right_model="right/model",
        matches=[
            CandidateMatch(
                left_feature_id="L1:F1",
                right_feature_id="R3:F8",
                left_model="left/model",
                right_model="right/model",
                score=0.92,
                components={
                    "text": 0.91,
                    "activation": 0.83,
                    "decoder": 0.79,
                    "causal": 0.78,
                    "signed_effect": 0.99,
                    # "validated" requires intervention-backed signed effects.
                    "signed_effect_provenance_intervention": 1.0,
                },
                left_label="evaluation awareness",
                right_label="evaluation awareness",
                left_signed_effect=0.11,
                right_signed_effect=0.12,
            ),
            CandidateMatch(
                left_feature_id="L1:F2",
                right_feature_id="R3:F9",
                left_model="left/model",
                right_model="right/model",
                score=0.86,
                components={
                    "text": 0.88,
                    "activation": 0.82,
                    "decoder": 0.74,
                    "causal": 0.5,
                },
                left_label="math phrasing",
                right_label="math phrasing",
            ),
            CandidateMatch(
                left_feature_id="L1:F3",
                right_feature_id="R3:F10",
                left_model="left/model",
                right_model="right/model",
                score=0.89,
                components={
                    "text": 0.87,
                    "activation": 0.8,
                    "decoder": 0.7,
                    "causal": 0.76,
                    "signed_effect": 0.75,
                    "signed_effect_provenance_intervention": 1.0,
                },
                left_label="promotes criterion",
                right_label="suppresses criterion",
                left_signed_effect=0.14,
                right_signed_effect=-0.11,
            ),
            CandidateMatch(
                left_feature_id="L1:F4",
                right_feature_id="R3:F11",
                left_model="left/model",
                right_model="right/model",
                score=0.41,
                components={
                    "text": 0.55,
                    "activation": 0.51,
                    "decoder": 0.48,
                    "causal": 0.5,
                },
                left_label="weak left",
                right_label="weak right",
            ),
        ],
    )
