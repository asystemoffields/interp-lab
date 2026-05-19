import json
from pathlib import Path

from interp_lab import (
    attribution_graph,
    attribution_graph_summary,
    build_prompts,
    compare,
    doctor,
    inspect,
    profile_environment,
    publish_hf_artifact,
    run,
    scale_plan,
    scaffold_run,
    train_sae,
    validate_attribution_graph,
    validate_hf_sae_paths,
    validate_matches,
)
from oracle_sae.graph_validation import GraphValidationWriteResult
from oracle_sae.hf_sae_validation import HfSaePathValidationResult
from interp_lab.artifacts import InspectionReport, load_inspection_report


def test_inspect_api_returns_report():
    report = inspect(
        "toy/a",
        "the model is aware it is being evaluated",
        backend="toy",
        top_k=2,
    )

    assert isinstance(report, InspectionReport)
    assert report.cards
    assert len(report.cards) == 2


def test_inspect_api_can_write_report(tmp_path: Path):
    result = inspect(
        "toy/a",
        "the model is aware it is being evaluated",
        backend="toy",
        out=tmp_path,
        top_k=1,
    )

    assert result.json_path == tmp_path / "report.json"
    assert result.markdown_path == tmp_path / "report.md"
    assert result.report.cards
    assert load_inspection_report(result.json_path).model == "toy/a"


def test_compare_api_accepts_reports_and_paths(tmp_path: Path):
    left = inspect("toy/a", "benchmark awareness", backend="toy", out=tmp_path / "left")
    right = inspect("toy/b", "benchmark awareness", backend="toy", top_k=3)

    result = compare(left.json_path, right, out=tmp_path / "matches.json")

    assert result.json_path == tmp_path / "matches.json"
    assert result.markdown_path == tmp_path / "matches.md"
    assert result.report.matches


def test_validate_matches_public_api_accepts_report_and_path(tmp_path: Path):
    matches = compare(
        inspect("toy/a", "benchmark awareness", backend="toy", top_k=2),
        inspect("toy/b", "benchmark awareness", backend="toy", top_k=2),
        top_k=2,
        out=tmp_path / "matches.json",
    )

    in_memory = validate_matches(matches.report, top_k=1)
    assert in_memory["summary"]["match_count"] == 1
    assert in_memory["validations"][0]["claim_grade"]

    written = validate_matches(
        matches.report,
        out=tmp_path / "match-validation.json",
        html_out=tmp_path / "match-validation.html",
    )
    assert written.json_path == tmp_path / "match-validation.json"
    assert written.markdown_path == tmp_path / "match-validation.md"
    assert written.html_path == tmp_path / "match-validation.html"
    assert written.html_path.exists()
    assert written.report["summary"]["match_count"] == len(matches.report.matches)

    loaded = validate_matches(matches.json_path, out=tmp_path / "match-validation-2.json")
    assert loaded.json_path == tmp_path / "match-validation-2.json"


def test_build_prompts_public_api_accepts_files_and_inline_prompts(tmp_path: Path):
    positive = tmp_path / "positive.txt"
    positive.write_text("The answer is measured in meters.\n\nA speed is given in miles per hour.", encoding="utf-8")

    result = build_prompts(
        positive=positive,
        negative_prompt="The answer is a proper name.",
        split="paragraphs",
        id_prefix="unit",
        out=tmp_path / "prompts.jsonl",
    )

    rows = [
        json.loads(line)
        for line in result.path.read_text(encoding="utf-8").splitlines()
    ]
    assert result.record_count == 3
    assert result.positive_count == 2
    assert result.negative_count == 1
    assert rows[0]["prompt_id"] == "unit-positive-001"
    assert rows[-1]["criterion_score"] == 0.0


def test_scaffold_run_public_api_writes_records_workflow(tmp_path: Path):
    result = scaffold_run(
        out=tmp_path / "run.json",
        workflow="records",
        model="m",
        criterion="benchmark awareness",
        run_dir=tmp_path / "run",
        records=tmp_path / "records.jsonl",
        top_k=3,
    )

    assert result.path == tmp_path / "run.json"
    assert result.config["steps"][0]["command"] == "inspect"
    assert result.config["steps"][0]["args"]["records"] == str(tmp_path / "records.jsonl")
    assert json.loads(result.path.read_text(encoding="utf-8"))["steps"][1]["command"] == "export-attribution-graph"


def test_scaffold_run_public_api_writes_sae_path_workflow(tmp_path: Path):
    result = scaffold_run(
        out=tmp_path / "run.json",
        workflow="sae-paths",
        model="distilgpt2",
        criterion="unit prediction",
        run_dir=tmp_path / "run",
        dataset=tmp_path / "prompts.jsonl",
        validation_dataset=tmp_path / "heldout.jsonl",
        source_layer=2,
        target_layer=4,
        include_causal=True,
        target_token="auto",
        validate_paths=True,
        model_class="auto-image-text-to-text",
        torch_dtype="auto",
        device_map="auto",
    )

    commands = [step["command"] for step in result.config["steps"]]
    assert commands == [
        "train-sae",
        "train-sae",
        "inspect",
        "inspect",
        "export-hf-sae-paths",
        "export-attribution-graph",
        "summarize-attribution-graph",
        "validate-hf-sae-paths",
        "summarize-attribution-graph",
    ]
    assert result.config["steps"][0]["args"]["layer"] == 2
    assert result.config["steps"][1]["args"]["layer"] == 4
    assert result.config["steps"][0]["args"]["model_class"] == "auto-image-text-to-text"
    assert result.config["steps"][0]["args"]["torch_dtype"] == "auto"
    assert result.config["steps"][0]["args"]["device_map"] == "auto"
    assert result.config["steps"][2]["args"]["require_interventions"] is True
    assert result.config["steps"][5]["args"]["path_records"] == "{run_dir}/paths.jsonl"
    assert result.config["steps"][6]["args"]["out"] == "{run_dir}/graph-summary.json"
    assert result.config["steps"][7]["args"]["dataset"] == str(tmp_path / "heldout.jsonl")
    assert result.config["steps"][7]["args"]["model_class"] == "auto-image-text-to-text"
    saved = json.loads(result.path.read_text(encoding="utf-8"))
    assert saved["steps"][-2]["args"]["graph_out"] == "{run_dir}/validated-graph.json"
    assert saved["steps"][-2]["args"]["graph_html_out"] == "{run_dir}/validated-graph.html"
    assert saved["steps"][-1]["args"]["out"] == "{run_dir}/validated-graph-summary.json"


def test_train_sae_api_from_records(tmp_path: Path):
    records = tmp_path / "records.jsonl"
    rows = [
        _row("m", "p1", 1.0, {"raw-a": 2.0, "raw-b": 0.0}),
        _row("m", "p2", 0.0, {"raw-a": 0.0, "raw-b": 2.0}),
    ]
    records.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = train_sae(
        records=records,
        model="m",
        out=tmp_path / "sae.json",
        records_out=tmp_path / "sae-records.jsonl",
        method="fallback",
        latent_dim=2,
    )

    assert result.artifact_path.exists()
    assert result.records_path is not None
    assert result.records_path.exists()
    artifact = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    assert artifact["format"] == "interp-lab.sae.v1"


def test_run_and_doctor_api(tmp_path: Path):
    config = tmp_path / "run.json"
    run_dir = tmp_path / "run"
    config.write_text(
        json.dumps(
            {
                "out": str(run_dir),
                "model": "toy/a",
                "criterion": "benchmark awareness",
                "backend": "toy",
            }
        ),
        encoding="utf-8",
    )

    assert run(config) == 0
    assert (run_dir / "manifest.json").exists()
    assert doctor()["tool"] == "interp-lab"
    assert profile_environment(tmp_path)["schema_version"] == "interp-lab.env_profile.v1"


def test_graph_publish_and_scale_public_apis(tmp_path: Path):
    result = inspect("toy/a", "benchmark awareness", backend="toy", out=tmp_path / "inspect", top_k=2)
    graph = attribution_graph(result.report)
    paths = tmp_path / "paths.jsonl"
    paths.write_text(
        json.dumps(
            {
                "source_feature_id": result.report.cards[0].feature_id,
                "target_feature_id": result.report.cards[1].feature_id,
                "target_activation_delta": 0.1,
                "strength": 2.0,
                "prompt_id": "p",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    graph_with_paths = attribution_graph(result.report, path_records=paths)
    written_graph = attribution_graph(
        result.json_path,
        out=tmp_path / "graph.json",
        markdown_out=tmp_path / "graph.md",
        html_out=tmp_path / "graph.html",
    )
    graph_summary = attribution_graph_summary(written_graph.graph, out=tmp_path / "graph-summary.json")
    dry_run = publish_hf_artifact(
        repo_id="user/interp-lab-demo",
        paths=[result.json_path],
        dry_run=True,
    )
    plan = scale_plan(model_params=1e12, tokens=1000, d_model=1024)

    assert graph["schema_version"] == "interp-lab.attribution_graph.v1"
    assert any(edge["type"] == "path_patch" for edge in graph_with_paths["edges"])
    assert written_graph.json_path == tmp_path / "graph.json"
    assert written_graph.markdown_path == tmp_path / "graph.md"
    assert written_graph.markdown_path.exists()
    assert written_graph.html_path == tmp_path / "graph.html"
    assert written_graph.html_path.exists()
    assert graph_summary.json_path == tmp_path / "graph-summary.json"
    assert graph_summary.summary["schema_version"] == "interp-lab.attribution_graph_summary.v1"
    assert dry_run.uploaded == ["report.json"]
    assert plan["schema_version"] == "interp-lab.scale_plan.v2"
    assert any("1T+" in item for item in plan["recommendations"])


def test_validate_attribution_graph_public_api(tmp_path: Path):
    graph = {
        "schema_version": "interp-lab.attribution_graph.v1",
        "model": "m",
        "criterion": {"text": "criterion"},
        "mechanism_summary": {
            "candidate_paths": [
                {
                    "source_feature_id": "SAE:L1:F1",
                    "target_feature_id": "SAE:L2:F8",
                    "evidence": "path_patch",
                }
            ]
        },
    }
    paths = tmp_path / "paths.jsonl"
    rows = [
        {
            "source_feature_id": "SAE:L1:F1",
            "target_feature_id": "SAE:L2:F8",
            "target_activation_delta": 0.2,
            "prompt_id": "p1",
        },
        {
            "source_feature_id": "SAE:L1:F1",
            "target_feature_id": "SAE:L2:F8",
            "target_activation_delta": 0.02,
            "prompt_id": "p1",
            "metadata": {"control_type": "random_source"},
        },
    ]
    paths.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = validate_attribution_graph(graph, path_records=paths, min_prompt_count=1)
    written = validate_attribution_graph(
        graph,
        path_records=paths,
        min_prompt_count=1,
        out=tmp_path / "validation.json",
        graph_out=tmp_path / "validated-graph.json",
    )

    assert report["path_validations"][0]["status"] == "robust"
    assert written.json_path.exists()
    assert written.markdown_path.exists()
    assert written.annotated_graph_path is not None
    assert written.annotated_graph_path.exists()
    assert written.annotated_graph_markdown_path == tmp_path / "validated-graph.md"
    assert written.annotated_graph_markdown_path.exists()
    assert written.annotated_graph_html_path == tmp_path / "validated-graph.html"
    assert written.annotated_graph_html_path.exists()


def test_validate_hf_sae_paths_public_api(tmp_path: Path, monkeypatch):
    def fake_export_hf_sae_path_validation(**_kwargs):
        return HfSaePathValidationResult(
            selected_path_pairs=[("SAE:L1:F1", "SAE:L2:F8")],
            path_records_path=tmp_path / "paths.jsonl",
            validation=GraphValidationWriteResult(
                report={"ok": True},
                json_path=tmp_path / "validation.json",
                markdown_path=tmp_path / "validation.md",
                annotated_graph_path=tmp_path / "validated-graph.json",
                annotated_graph_html_path=tmp_path / "validated-graph.html",
            ),
        )

    monkeypatch.setattr(
        "interp_lab.api.export_hf_sae_path_validation",
        fake_export_hf_sae_path_validation,
    )

    result = validate_hf_sae_paths(
        graph=tmp_path / "graph.json",
        model="m",
        dataset=tmp_path / "heldout.jsonl",
        source_sae=tmp_path / "source.json",
        target_sae=tmp_path / "target.json",
        path_records_out=tmp_path / "paths.jsonl",
        out=tmp_path / "validation.json",
        graph_out=tmp_path / "validated-graph.json",
    )

    assert result.selected_path_pairs == [("SAE:L1:F1", "SAE:L2:F8")]
    assert result.validation_report == {"ok": True}
    assert result.annotated_graph_path == tmp_path / "validated-graph.json"
    assert result.annotated_graph_html_path == tmp_path / "validated-graph.html"


def _row(model: str, prompt_id: str, score: float, features: dict[str, float]):
    return {
        "model": model,
        "prompt_id": prompt_id,
        "text": prompt_id,
        "criterion_score": score,
        "features": features,
    }
