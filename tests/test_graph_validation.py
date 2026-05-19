import json
from pathlib import Path

from oracle_sae.graph_validation import (
    annotate_graph_with_validation,
    build_graph_validation_report,
    export_graph_validation_report,
    select_graph_path_pairs,
)


def test_graph_validation_classifies_robust_path_against_controls():
    graph = _graph()
    records = [
        _path_row("p1", 0.21),
        _path_row("p2", 0.19),
        _path_row("p3", 0.23),
        _path_row("p1", 0.03, control=True),
        _path_row("p2", 0.02, control=True),
        _path_row("p3", 0.01, control=True),
    ]

    report = build_graph_validation_report(graph, path_records=records)

    validation = report["path_validations"][0]
    assert report["schema_version"] == "interp-lab.graph_validation.v1"
    assert validation["status"] == "robust"
    assert validation["reason_codes"] == ["passed_effect_control_and_sign_thresholds"]
    assert validation["record_count"] == 3
    assert validation["control_record_count"] == 3
    assert validation["path_specificity_score"] == 0.19
    assert validation["effect_control_ratio"] > 6
    assert validation["sign_consistency"] == 1.0
    assert validation["target_activation_delta_ci"]["low"] < validation["target_activation_delta_ci"]["high"]


def test_graph_validation_marks_control_matched_path_as_failed_control():
    graph = _graph()
    records = [
        _path_row("p1", 0.21),
        _path_row("p2", 0.19),
        _path_row("p3", 0.23),
        _path_row("p1", 0.20, control=True),
        _path_row("p2", 0.18, control=True),
        _path_row("p3", 0.22, control=True),
    ]

    report = build_graph_validation_report(graph, path_records=records)

    validation = report["path_validations"][0]
    assert validation["status"] == "failed_control"
    assert "control_specificity_below_threshold" in validation["reason_codes"]
    assert "comparable target-latent deltas" in validation["interpretation"]


def test_export_graph_validation_report_writes_json_and_markdown(tmp_path: Path):
    graph_path = tmp_path / "graph.json"
    graph = _graph()
    graph["edges"] = [
        {
            "source": "SAE:L1:F1",
            "target": "SAE:L2:F8",
            "type": "path_patch",
        }
    ]
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    records_path = tmp_path / "paths.jsonl"
    records_path.write_text(
        "\n".join(json.dumps(row) for row in [_path_row("p1", 0.2), _path_row("p1", 0.02, control=True)]),
        encoding="utf-8",
    )

    result = export_graph_validation_report(
        graph_path=graph_path,
        path_records_path=records_path,
        out_path=tmp_path / "validation.json",
        graph_out_path=tmp_path / "validated-graph.json",
        min_prompt_count=1,
    )

    assert result.json_path.exists()
    assert result.markdown_path.exists()
    assert result.annotated_graph_path is not None
    assert result.annotated_graph_path.exists()
    assert result.annotated_graph_markdown_path is not None
    assert result.annotated_graph_markdown_path.exists()
    annotated = json.loads(result.annotated_graph_path.read_text(encoding="utf-8"))
    assert annotated["edges"][0]["validation"]["status"] == "robust"
    assert annotated["edges"][0]["validation"]["reason_codes"] == ["passed_effect_control_and_sign_thresholds"]
    assert annotated["mechanism_summary"]["candidate_paths"][0]["validation"]["status"] == "robust"
    assert "Attribution Graph Validation" in result.markdown_path.read_text(encoding="utf-8")
    annotated_markdown = result.annotated_graph_markdown_path.read_text(encoding="utf-8")
    assert "Path validation: `robust=1`" in annotated_markdown
    assert "`SAE:L1:F1 -> SAE:L2:F8`" in annotated_markdown


def test_annotate_graph_with_validation_preserves_input_graph():
    graph = _graph()
    graph["edges"] = [{"source": "SAE:L1:F1", "target": "SAE:L2:F8", "type": "path_patch"}]
    report = build_graph_validation_report(
        graph,
        path_records=[_path_row("p1", 0.2), _path_row("p1", 0.02, control=True)],
        min_prompt_count=1,
    )

    annotated = annotate_graph_with_validation(graph, report)

    assert "validation" not in graph["edges"][0]
    assert annotated["edges"][0]["validation"]["status"] == "robust"
    assert annotated["metadata"]["graph_validation"]["summary"]["validated_path_count"] == 1


def test_select_graph_path_pairs_deduplicates_candidate_paths():
    graph = _graph()
    graph["mechanism_summary"]["candidate_paths"].append(
        {
            "source_feature_id": "SAE:L1:F1",
            "target_feature_id": "SAE:L2:F8",
            "evidence": "path_patch",
        }
    )
    graph["mechanism_summary"]["candidate_paths"].append(
        {
            "source_feature_id": "SAE:L1:F2",
            "target_feature_id": "SAE:L2:F9",
            "evidence": "path_patch",
        }
    )

    assert select_graph_path_pairs(graph, top_k=2) == [
        ("SAE:L1:F1", "SAE:L2:F8"),
        ("SAE:L1:F2", "SAE:L2:F9"),
    ]


def _graph() -> dict:
    return {
        "schema_version": "interp-lab.attribution_graph.v1",
        "model": "m",
        "criterion": {"text": "criterion"},
        "edges": [],
        "mechanism_summary": {
            "candidate_paths": [
                {
                    "source_feature_id": "SAE:L1:F1",
                    "target_feature_id": "SAE:L2:F8",
                    "source_label": "source",
                    "target_label": "target",
                    "evidence": "path_patch",
                }
            ]
        },
    }


def _path_row(prompt_id: str, delta: float, *, control: bool = False) -> dict:
    row = {
        "source_feature_id": "SAE:L1:F1",
        "target_feature_id": "SAE:L2:F8",
        "prompt_id": prompt_id,
        "target_activation_delta": delta,
        "score_delta": delta / 10,
        "strength": 2.0,
    }
    if control:
        row["metadata"] = {
            "control_type": "random_source",
            "control_source_feature_id": "SAE:L1:F7",
        }
    return row
