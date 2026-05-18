from pathlib import Path
import json

from oracle_sae.cli import main


def test_demo_command_writes_reports(tmp_path: Path):
    exit_code = main(["demo", "--out", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "model-a" / "report.json").exists()
    assert (tmp_path / "model-b" / "report.json").exists()
    assert (tmp_path / "matches.json").exists()
    assert (tmp_path / "matches.md").exists()


def test_records_backend_writes_report(tmp_path: Path):
    records = tmp_path / "records.jsonl"
    rows = [
        {
            "model": "m",
            "prompt_id": "p1",
            "text": "benchmark-like prompt",
            "criterion_score": 1,
            "features": {"L1:F1": 0.9},
            "feature_metadata": {"L1:F1": {"label": "benchmark awareness"}},
        },
        {
            "model": "m",
            "prompt_id": "p2",
            "text": "ordinary prompt",
            "criterion_score": 0,
            "features": {"L1:F1": 0.1},
            "feature_metadata": {"L1:F1": {"label": "benchmark awareness"}},
        },
    ]
    records.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    out = tmp_path / "report"

    exit_code = main(
        [
            "inspect",
            "--model",
            "m",
            "--criterion",
            "benchmark awareness",
            "--backend",
            "records",
            "--records",
            str(records),
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    assert (out / "report.json").exists()


def test_records_backend_accepts_intervention_records(tmp_path: Path):
    records = tmp_path / "records.jsonl"
    rows = [
        {
            "model": "m",
            "prompt_id": "p1",
            "text": "benchmark-like prompt",
            "criterion_score": 1,
            "features": {"L1:F1": 0.9},
            "feature_metadata": {"L1:F1": {"label": "benchmark awareness"}},
        },
        {
            "model": "m",
            "prompt_id": "p2",
            "text": "ordinary prompt",
            "criterion_score": 0,
            "features": {"L1:F1": 0.1},
            "feature_metadata": {"L1:F1": {"label": "benchmark awareness"}},
        },
    ]
    records.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    interventions = tmp_path / "interventions.jsonl"
    intervention_rows = [
        {
            "model": "m",
            "feature_id": "L1:F1",
            "criterion": "benchmark awareness",
            "intervention": "ablate",
            "prompt_id": "p1",
            "baseline_score": 0.8,
            "intervention_score": 0.3,
            "side_effect_score": 0.05,
        }
    ]
    interventions.write_text(
        "\n".join(json.dumps(row) for row in intervention_rows),
        encoding="utf-8",
    )
    out = tmp_path / "report"

    exit_code = main(
        [
            "inspect",
            "--model",
            "m",
            "--criterion",
            "benchmark awareness",
            "--backend",
            "records",
            "--records",
            str(records),
            "--interventions",
            str(interventions),
            "--out",
            str(out),
        ]
    )

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    card = report["cards"][0]
    assert exit_code == 0
    assert card["causal_effect"] == 0.5
    assert card["metadata"]["interventions"]["count"] == 1


def test_doctor_command_reports_environment():
    exit_code = main(["doctor", "--json"])

    assert exit_code == 0


def test_run_config_writes_manifest_and_report(tmp_path: Path):
    records = tmp_path / "records.jsonl"
    rows = [
        {
            "model": "m",
            "prompt_id": "p1",
            "text": "benchmark-like prompt",
            "criterion_score": 1,
            "features": {"L1:F1": 0.9},
            "feature_metadata": {"L1:F1": {"label": "benchmark awareness"}},
        },
        {
            "model": "m",
            "prompt_id": "p2",
            "text": "ordinary prompt",
            "criterion_score": 0,
            "features": {"L1:F1": 0.1},
            "feature_metadata": {"L1:F1": {"label": "benchmark awareness"}},
        },
    ]
    records.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    run_dir = tmp_path / "run"
    config = tmp_path / "run.json"
    config.write_text(
        json.dumps(
            {
                "out": str(run_dir),
                "model": "m",
                "criterion": "benchmark awareness",
                "backend": "records",
                "records": str(records),
                "top_k": 3,
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["run", str(config)])

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert (run_dir / "inspect" / "report.json").exists()
    assert manifest["status"] == "succeeded"
    assert manifest["steps"][0]["command"] == "inspect"
    assert manifest["steps"][0]["exit_code"] == 0
    assert manifest["inputs"][0]["sha256"]


def test_run_config_steps_support_template_variables(tmp_path: Path):
    config = tmp_path / "run.json"
    run_dir = tmp_path / "templated"
    config.write_text(
        json.dumps(
            {
                "out": str(run_dir),
                "steps": [
                    {
                        "name": "demo",
                        "command": "demo",
                        "args": {"out": "{run_dir}/demo"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["run", str(config)])

    assert exit_code == 0
    assert (run_dir / "demo" / "matches.json").exists()
