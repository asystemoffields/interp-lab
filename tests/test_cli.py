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


def test_doctor_command_reports_environment(capsys):
    exit_code = main(["doctor", "--json"])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["tool"] == "interp-lab"


def test_build_prompts_command_writes_prompt_jsonl(tmp_path: Path, capsys):
    positive = tmp_path / "positive.txt"
    positive.write_text("Prompt one\nPrompt two\n", encoding="utf-8")
    out = tmp_path / "prompts.jsonl"

    exit_code = main(
        [
            "build-prompts",
            "--positive",
            str(positive),
            "--negative-prompt",
            "Ordinary control prompt",
            "--split",
            "lines",
            "--id-prefix",
            "custom",
            "--out",
            str(out),
        ]
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Prompts: 3 total" in output
    assert rows[0]["prompt_id"] == "custom-positive-001"
    assert rows[-1]["criterion_score"] == 0.0


def test_publish_hf_artifact_dry_run_command(tmp_path: Path, capsys):
    artifact = tmp_path / "report.json"
    artifact.write_text("{}", encoding="utf-8")

    exit_code = main(
        [
            "publish-hf-artifact",
            "--repo-id",
            "user/interp-lab-demo",
            "--path",
            str(artifact),
            "--path-in-repo",
            "reports/report.json",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Would upload 1 artifact" in output
    assert "reports/report.json" in output


def test_plan_scale_command_outputs_estimate(capsys):
    exit_code = main(
        [
            "plan-scale",
            "--model-params",
            "1T",
            "--tokens",
            "1K",
            "--d-model",
            "1024",
            "--selected-layers",
            "2",
            "--latent-dim",
            "4096",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "interp-lab scale plan" in output
    assert "Agent next actions" in output
    assert "1T+" in output


def test_plan_scale_command_writes_agent_json(tmp_path: Path):
    plan_path = tmp_path / "scale-plan.json"
    exit_code = main(
        [
            "plan-scale",
            "--model-params",
            "70B",
            "--tokens",
            "10M",
            "--d-model",
            "8192",
            "--selected-layers",
            "4",
            "--target-shard-size",
            "4GB",
            "--out",
            str(plan_path),
            "--json",
        ]
    )

    data = json.loads(plan_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert data["schema_version"] == "interp-lab.scale_plan.v2"
    assert data["shard_plan"]["target_shard_size_human"] == "4.00 GB"
    assert data["agent_next_actions"]


def test_export_attribution_graph_command(tmp_path: Path):
    exit_code = main(["demo", "--out", str(tmp_path / "demo")])
    assert exit_code == 0

    graph = tmp_path / "graph.json"
    markdown = tmp_path / "graph.md"
    exit_code = main(
        [
            "export-attribution-graph",
            "--report",
            str(tmp_path / "demo" / "model-a" / "report.json"),
            "--out",
            str(graph),
            "--markdown-out",
            str(markdown),
        ]
    )

    data = json.loads(graph.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert data["schema_version"] == "interp-lab.attribution_graph.v1"
    assert "# Attribution Graph" in markdown.read_text(encoding="utf-8")


def test_export_attribution_graph_command_accepts_path_records(tmp_path: Path):
    exit_code = main(["demo", "--out", str(tmp_path / "demo")])
    assert exit_code == 0
    path_records = tmp_path / "paths.jsonl"
    path_records.write_text(
        json.dumps(
            {
                "source_feature_id": "L8:F1",
                "target_feature_id": "L9:F2",
                "target_activation_delta": 0.25,
                "strength": 2.0,
                "prompt_id": "p",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    graph = tmp_path / "graph.json"
    exit_code = main(
        [
            "export-attribution-graph",
            "--report",
            str(tmp_path / "demo" / "model-a" / "report.json"),
            "--path-records",
            str(path_records),
            "--out",
            str(graph),
        ]
    )

    data = json.loads(graph.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert any(edge["type"] == "path_patch" for edge in data["edges"])


def test_validate_attribution_graph_command(tmp_path: Path):
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    path_records = tmp_path / "paths.jsonl"
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
    path_records.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    out = tmp_path / "validation.json"

    exit_code = main(
        [
            "validate-attribution-graph",
            "--graph",
            str(graph),
            "--path-records",
            str(path_records),
            "--min-prompt-count",
            "1",
            "--out",
            str(out),
            "--graph-out",
            str(tmp_path / "validated-graph.json"),
        ]
    )

    data = json.loads(out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert data["path_validations"][0]["status"] == "robust"
    assert out.with_suffix(".md").exists()
    assert (tmp_path / "validated-graph.json").exists()
    assert (tmp_path / "validated-graph.md").exists()
    graph_markdown = (tmp_path / "validated-graph.md").read_text(encoding="utf-8")
    assert "Path validation: `robust=1`" in graph_markdown


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
    assert manifest["tool"] == "interp-lab"
    assert manifest["schema_version"] == "interp-lab.run.v1"
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
