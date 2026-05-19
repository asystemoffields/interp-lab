import json
from pathlib import Path

from oracle_sae.graph_validation import GraphValidationWriteResult
from oracle_sae.hf_sae_validation import export_hf_sae_path_validation


def test_hf_sae_path_validation_reruns_selected_graph_pairs(tmp_path: Path, monkeypatch):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "model": "m",
                "criterion": {"text": "criterion from graph"},
                "mechanism_summary": {
                    "candidate_paths": [
                        {
                            "source_feature_id": "SAE:L1:F1",
                            "target_feature_id": "SAE:L2:F8",
                            "evidence": "path_patch",
                        },
                        {
                            "source_feature_id": "SAE:L1:F3",
                            "target_feature_id": "SAE:L2:F9",
                            "evidence": "path_patch",
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_export_paths(**kwargs):
        captured["path_kwargs"] = kwargs
        path = Path(kwargs["out_path"])
        path.write_text("", encoding="utf-8")
        return path

    def fake_export_validation(**kwargs):
        captured["validation_kwargs"] = kwargs
        return GraphValidationWriteResult(
            report={"ok": True},
            json_path=Path(kwargs["out_path"]),
            markdown_path=Path(kwargs["markdown_out_path"] or Path(kwargs["out_path"]).with_suffix(".md")),
            annotated_graph_path=Path(kwargs["graph_out_path"]) if kwargs.get("graph_out_path") else None,
        )

    monkeypatch.setattr("oracle_sae.hf_sae_validation.export_hf_sae_path_records", fake_export_paths)
    monkeypatch.setattr("oracle_sae.hf_sae_validation.export_graph_validation_report", fake_export_validation)

    result = export_hf_sae_path_validation(
        graph_path=graph_path,
        model_name="m",
        dataset_path="heldout.jsonl",
        source_artifact_path="source-sae.json",
        target_artifact_path="target-sae.json",
        path_records_out_path=tmp_path / "paths.jsonl",
        validation_out_path=tmp_path / "validation.json",
        graph_out_path=tmp_path / "validated-graph.json",
        top_k=1,
        random_source_controls=3,
    )

    assert result.selected_path_pairs == [("SAE:L1:F1", "SAE:L2:F8")]
    assert captured["path_kwargs"]["criterion"] == "criterion from graph"
    assert captured["path_kwargs"]["path_pairs"] == [("SAE:L1:F1", "SAE:L2:F8")]
    assert captured["path_kwargs"]["random_source_controls"] == 3
    assert captured["validation_kwargs"]["path_records_path"] == tmp_path / "paths.jsonl"
    assert captured["validation_kwargs"]["graph_out_path"] == tmp_path / "validated-graph.json"
    assert result.validation.annotated_graph_path == tmp_path / "validated-graph.json"
