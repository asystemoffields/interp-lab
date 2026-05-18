import json
from pathlib import Path

from interp_lab import (
    attribution_graph,
    compare,
    doctor,
    inspect,
    profile_environment,
    publish_hf_artifact,
    run,
    scale_plan,
    train_sae,
)
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
    written_graph = attribution_graph(result.json_path, out=tmp_path / "graph.json")
    dry_run = publish_hf_artifact(
        repo_id="user/interp-lab-demo",
        paths=[result.json_path],
        dry_run=True,
    )
    plan = scale_plan(model_params=1e12, tokens=1000, d_model=1024)

    assert graph["schema_version"] == "interp-lab.attribution_graph.v1"
    assert written_graph.json_path == tmp_path / "graph.json"
    assert dry_run.uploaded == ["report.json"]
    assert plan["schema_version"] == "interp-lab.scale_plan.v2"
    assert any("1T+" in item for item in plan["recommendations"])


def _row(model: str, prompt_id: str, score: float, features: dict[str, float]):
    return {
        "model": model,
        "prompt_id": prompt_id,
        "text": prompt_id,
        "criterion_score": score,
        "features": features,
    }
