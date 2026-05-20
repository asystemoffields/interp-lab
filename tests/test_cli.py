from pathlib import Path
import json

from oracle_sae.cli import main
from oracle_sae.runs import _input_file_records


def test_demo_command_writes_reports(tmp_path: Path):
    exit_code = main(["demo", "--out", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "model-a" / "report.json").exists()
    assert (tmp_path / "model-a" / "report.html").exists()
    assert (tmp_path / "model-b" / "report.json").exists()
    assert (tmp_path / "model-b" / "report.html").exists()
    assert (tmp_path / "matches.json").exists()
    assert (tmp_path / "matches.md").exists()
    assert (tmp_path / "match-validation.json").exists()
    assert (tmp_path / "match-validation.md").exists()
    assert (tmp_path / "match-validation.html").exists()
    assert (tmp_path / "graph.json").exists()
    assert (tmp_path / "graph.md").exists()
    assert (tmp_path / "graph.html").exists()
    assert (tmp_path / "graph-summary.json").exists()
    assert (tmp_path / "studio.html").exists()
    graph_summary = json.loads((tmp_path / "graph-summary.json").read_text(encoding="utf-8"))
    assert graph_summary["schema_version"] == "interp-lab.attribution_graph_summary.v1"


def test_studio_command_writes_web_app(tmp_path: Path):
    out = tmp_path / "studio.html"

    exit_code = main(["studio", "--out", str(out)])

    html = out.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "Interp Lab Studio" in html
    assert "export-transformerlens-records" in html
    assert "export-nnsight-records" in html
    assert "generated-command" in html


def test_web_app_alias_writes_web_app(tmp_path: Path):
    out = tmp_path / "alias.html"

    exit_code = main(["web-app", "--out", str(out)])

    assert exit_code == 0
    assert out.exists()


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
            "--html-out",
            str(out / "report.html"),
        ]
    )

    assert exit_code == 0
    assert (out / "report.json").exists()
    assert (out / "report.html").exists()


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
    assert report["metadata"]["interventions"]["record_count"] == 1


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


def test_prepare_sae_prompts_command_writes_split_pack(tmp_path: Path, capsys):
    dataset = tmp_path / "prompts.jsonl"
    rows = [
        {"prompt_id": f"pos-{index}", "text": f"positive prompt {index}", "criterion_score": 1.0}
        for index in range(1, 5)
    ] + [
        {"prompt_id": f"neg-{index}", "text": f"negative prompt {index}", "criterion_score": 0.0}
        for index in range(1, 5)
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    out_dir = tmp_path / "pack"

    exit_code = main(
        [
            "prepare-sae-prompts",
            "--dataset",
            str(dataset),
            "--out-dir",
            str(out_dir),
            "--seed",
            "cli",
            "--latent-dim",
            "128",
            "--max-length",
            "8",
        ]
    )

    output = capsys.readouterr().out
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "Prompts: train=" in output
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "causal.jsonl").exists()
    assert (out_dir / "validation.jsonl").exists()
    assert manifest["counts"]["total"]["record_count"] == 8


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
    html = tmp_path / "graph.html"
    exit_code = main(
        [
            "export-attribution-graph",
            "--report",
            str(tmp_path / "demo" / "model-a" / "report.json"),
            "--out",
            str(graph),
            "--markdown-out",
            str(markdown),
            "--html-out",
            str(html),
        ]
    )

    data = json.loads(graph.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert data["schema_version"] == "interp-lab.attribution_graph.v1"
    assert "# Attribution Graph" in markdown.read_text(encoding="utf-8")
    assert "feature-search" in html.read_text(encoding="utf-8")


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


def test_summarize_attribution_graph_command(tmp_path: Path):
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "schema_version": "interp-lab.attribution_graph.v1",
                "model": "m",
                "criterion": {"text": "criterion"},
                "nodes": [{"id": "SAE:L1:F1", "type": "feature"}],
                "edges": [{"source": "SAE:L1:F1", "target": "SAE:L2:F8", "type": "path_patch"}],
                "mechanism_summary": {
                    "candidate_paths": [
                        {
                            "source_feature_id": "SAE:L1:F1",
                            "target_feature_id": "SAE:L2:F8",
                            "evidence": "path_patch",
                            "validation": {"status": "robust", "claim_grade": "validated"},
                        }
                    ],
                    "path_validation_status_counts": {"robust": 1},
                },
                "metadata": {
                    "graph_validation": {
                        "run_assessment": {
                            "overall_claim_grade": "validated_paths_present",
                            "recommended_next_action": "Replicate validated paths.",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "summary.json"

    exit_code = main(["summarize-attribution-graph", "--graph", str(graph), "--out", str(out)])

    data = json.loads(out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert data["schema_version"] == "interp-lab.attribution_graph_summary.v1"
    assert data["counts"]["path_patch_edges"] == 1
    assert data["validation"]["overall_claim_grade"] == "validated_paths_present"


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
    assert (tmp_path / "validated-graph.html").exists()
    graph_markdown = (tmp_path / "validated-graph.md").read_text(encoding="utf-8")
    assert "Path validation: `robust=1`" in graph_markdown
    assert "validated" in graph_markdown
    assert "broader held-out prompts" in graph_markdown
    assert "passed_effect_control_and_sign_thresholds" in graph_markdown
    assert "feature-search" in (tmp_path / "validated-graph.html").read_text(encoding="utf-8")


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
    assert manifest["steps"][0]["outputs"][0]["kind"] == "directory"
    assert manifest["steps"][0]["outputs"][0]["exists"] is True
    assert manifest["outputs"][0]["kind"] == "directory"
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
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"][0]["path"] == str((run_dir / "demo").resolve())


def test_run_config_list_args_record_outputs(tmp_path: Path):
    config = tmp_path / "run.json"
    run_dir = tmp_path / "list-args"
    config.write_text(
        json.dumps(
            {
                "out": str(run_dir),
                "steps": [
                    {
                        "name": "demo-list",
                        "command": "demo",
                        "args": ["--out", str(run_dir / "demo")],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["run", str(config)])

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert manifest["steps"][0]["outputs"][0]["path"] == str((run_dir / "demo").resolve())
    assert manifest["outputs"][0]["kind"] == "directory"


def test_run_config_dry_run_quotes_arguments_with_spaces(tmp_path: Path, capsys):
    config = tmp_path / "run.json"
    config.write_text(
        json.dumps(
            {
                "model": "toy-records/model",
                "criterion": "benchmark awareness",
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["run", str(config), "--dry-run"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "--criterion" in output
    assert '"benchmark awareness"' in output or "'benchmark awareness'" in output


def test_run_config_parse_errors_are_cli_errors(tmp_path: Path, capsys):
    config = tmp_path / "bad.json"
    config.write_text("{", encoding="utf-8")

    try:
        main(["run", str(config)])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    assert "interp-lab: error:" in capsys.readouterr().err


def test_run_manifest_input_scan_preserves_list_path_keys(tmp_path: Path):
    graph = tmp_path / "graph.json"
    paths = tmp_path / "paths.jsonl"
    graph.write_text("{}", encoding="utf-8")
    paths.write_text("{}", encoding="utf-8")

    records = _input_file_records(
        {
            "steps": [
                {
                    "command": "validate-attribution-graph",
                    "args": {
                        "graph": str(graph),
                        "path_records": [str(paths)],
                    },
                }
            ]
        },
        tmp_path,
        {"config": {}},
    )

    assert {record["path"] for record in records} == {str(graph.resolve()), str(paths.resolve())}


def test_init_run_scaffolds_editable_sae_workflow(tmp_path: Path, capsys):
    config = tmp_path / "sae-run.json"
    run_dir = tmp_path / "sae-run"

    exit_code = main(
        [
            "init-run",
            "--workflow",
            "sae",
            "--model",
            "distilgpt2",
            "--criterion",
            "unit prediction",
            "--positive-prompt",
            "The answer is measured in meters.",
            "--negative-prompt",
            "The answer is a person's name.",
            "--preset",
            "production",
            "--include-causal",
            "--target-token",
            "auto",
            "--run-dir",
            str(run_dir),
            "--out",
            str(config),
        ]
    )

    data = json.loads(config.read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "interp-lab run" in output
    assert [step["command"] for step in data["steps"]] == [
        "build-prompts",
        "train-sae",
        "inspect",
        "export-attribution-graph",
    ]
    assert data["steps"][1]["args"]["preset"] == "production"
    assert data["steps"][1]["args"]["causal_out"] == "{run_dir}/sae/interventions.jsonl"
    assert data["steps"][1]["args"]["target_token"] == ["auto"]
    assert data["steps"][2]["args"]["require_interventions"] is True

    assert main(["run", str(config), "--dry-run"]) == 0
    assert "interp-lab train-sae" in capsys.readouterr().out


def test_criterion_lab_scaffolds_overconfidence_workflow(tmp_path: Path, capsys):
    config = tmp_path / "overconfidence.json"
    run_dir = tmp_path / "overconfidence-run"

    exit_code = main(
        [
            "criterion-lab",
            "--model",
            "distilgpt2",
            "--preset",
            "overconfidence",
            "--run-dir",
            str(run_dir),
            "--out",
            str(config),
            "--positive-prompt",
            "Answer with total certainty: what did the missing note say?",
            "--negative-prompt",
            "Say what is unknown: what did the missing note say?",
        ]
    )

    data = json.loads(config.read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "overconfident" in output
    assert "interp-lab run" in output
    assert data["metadata"]["criterion_lab"]["preset"] == "overconfidence"
    assert data["metadata"]["criterion_lab"]["discovery_first"] is True
    assert data["metadata"]["criterion_lab"]["workflow"] == "discovery"
    assert data["metadata"]["criterion_lab"]["template_workflow"] == "hf-records"
    assert data["metadata"]["criterion_lab"]["layers"] == "all"
    assert data["metadata"]["criterion_lab"]["positive_prompt_count"] == 7
    assert data["metadata"]["criterion_lab"]["negative_prompt_count"] == 7
    assert data["metadata"]["criterion_lab"]["target_tokens"] == []
    assert [step["command"] for step in data["steps"]] == [
        "build-prompts",
        "export-hf-records",
        "inspect",
        "export-attribution-graph",
    ]
    export_args = data["steps"][1]["args"]
    assert export_args["model"] == "distilgpt2"
    assert export_args["layers"] == "all"
    assert data["steps"][2]["args"]["backend"] == "records"

    assert main(["run", str(config), "--dry-run"]) == 0
    dry_run = capsys.readouterr().out
    assert "interp-lab build-prompts" in dry_run
    assert "interp-lab export-hf-records" in dry_run


def test_criterion_lab_can_list_discoverable_presets(capsys):
    exit_code = main(["criterion-lab", "--list-presets"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Available Criterion Lab presets:" in output
    assert "overconfidence" in output


def test_validate_assay_accepts_user_authored_prompt_assay(tmp_path: Path, capsys):
    out = tmp_path / "assay-validation.json"

    exit_code = main(
        [
            "validate-assay",
            "--preset-file",
            "examples/presets/math-reasoning.json",
            "--out",
            str(out),
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Criterion assay validation: pass" in output
    assert report["status"] == "pass"
    assert report["summary"]["positive_prompt_count"] == 5
    assert report["agent_next_actions"]


def test_validate_assay_accepts_tool_call_example(tmp_path: Path):
    out = tmp_path / "tool-call-validation.json"

    exit_code = main(
        [
            "validate-assay",
            "--preset-file",
            "examples/presets/successful-tool-calls.json",
            "--out",
            str(out),
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "pass"
    assert report["summary"]["positive_prompt_count"] == 8
    assert report["summary"]["target_token_hint_count"] == 6


def test_validate_assay_rejects_overlapping_prompt_sets(tmp_path: Path, capsys):
    preset = tmp_path / "bad-assay.json"
    preset.write_text(
        json.dumps(
            {
                "name": "bad-assay",
                "criterion": "the model refuses harmful requests",
                "positive_prompts": ["I cannot help with that request.", "I cannot help with that request."],
                "negative_prompts": ["I cannot help with that request."],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate-assay", "--preset-file", str(preset)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "positive_negative_overlap" in output


def test_criterion_lab_can_use_user_authored_preset_file(tmp_path: Path):
    preset = tmp_path / "math-reasoning.json"
    preset.write_text(
        json.dumps(
            {
                "schema_version": "interp-lab.criterion_lab_preset.v1",
                "name": "math-reasoning",
                "criterion": "the model is doing mathematical reasoning",
                "positive_prompts": ["Solve 12 * 7 step by step."],
                "negative_prompts": ["Write a friendly greeting."],
                "defaults": {"workflow": "discovery", "layers": "all"},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "math-run.json"

    exit_code = main(
        [
            "criterion-lab",
            "--model",
            "distilgpt2",
            "--preset-file",
            str(preset),
            "--out",
            str(config),
        ]
    )

    data = json.loads(config.read_text(encoding="utf-8"))
    assert exit_code == 0
    lab = data["metadata"]["criterion_lab"]
    assert lab["preset"] == "math-reasoning"
    assert lab["preset_source"] == str(preset)
    assert lab["criterion"] == "the model is doing mathematical reasoning"
    assert lab["layers"] == "all"
    assert data["steps"][1]["command"] == "export-hf-records"


def test_criterion_lab_sae_workflow_uses_auto_targets_without_preset_hints(tmp_path: Path):
    config = tmp_path / "sae-lab.json"

    exit_code = main(
        [
            "criterion-lab",
            "--model",
            "distilgpt2",
            "--preset",
            "overconfidence",
            "--workflow",
            "sae",
            "--layer",
            "6",
            "--out",
            str(config),
        ]
    )

    data = json.loads(config.read_text(encoding="utf-8"))
    assert exit_code == 0
    lab = data["metadata"]["criterion_lab"]
    assert lab["discovery_first"] is False
    assert lab["target_tokens"] == ["auto"]
    train_args = data["steps"][1]["args"]
    assert train_args["layer"] == 6
    assert train_args["target_token"] == ["auto"]


def test_criterion_lab_can_use_only_custom_prompts(tmp_path: Path):
    config = tmp_path / "custom-overconfidence.json"

    exit_code = main(
        [
            "criterion-lab",
            "--model",
            "distilgpt2",
            "--criterion",
            "the model answers with unjustified certainty",
            "--out",
            str(config),
            "--no-preset-prompts",
            "--positive-prompt",
            "Answer with certainty: what is the unknowable fact?",
            "--negative-prompt",
            "Explain uncertainty: what is the unknowable fact?",
            "--skip-causal",
        ]
    )

    data = json.loads(config.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert data["metadata"]["criterion_lab"]["preset"] == "custom"
    assert data["metadata"]["criterion_lab"]["positive_prompt_count"] == 1
    assert data["metadata"]["criterion_lab"]["negative_prompt_count"] == 1
    assert data["metadata"]["criterion_lab"]["target_tokens"] == []
    assert data["steps"][1]["command"] == "export-hf-records"


def test_init_run_scaffolds_sae_path_workflow(tmp_path: Path, capsys):
    config = tmp_path / "sae-paths-run.json"
    run_dir = tmp_path / "sae-paths-run"

    exit_code = main(
        [
            "init-run",
            "--workflow",
            "sae-paths",
            "--model",
            "distilgpt2",
            "--criterion",
            "unit prediction",
            "--positive-prompt",
            "The answer is measured in meters.",
            "--negative-prompt",
            "The answer is a person's name.",
            "--validation-dataset",
            str(tmp_path / "heldout.jsonl"),
            "--source-layer",
            "2",
            "--target-layer",
            "4",
            "--include-causal",
            "--target-token",
            "auto",
            "--path-top-k",
            "3",
            "--source-top-k",
            "2",
            "--target-top-k",
            "5",
            "--random-source-controls",
            "1",
            "--validate-paths",
            "--max-length",
            "64",
            "--model-class",
            "gemma4-conditional",
            "--trust-remote-code",
            "--torch-dtype",
            "bfloat16",
            "--device-map",
            "auto",
            "--run-dir",
            str(run_dir),
            "--out",
            str(config),
        ]
    )

    data = json.loads(config.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert [step["command"] for step in data["steps"]] == [
        "build-prompts",
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
    source_train = data["steps"][1]["args"]
    target_train = data["steps"][2]["args"]
    assert source_train["layer"] == 2
    assert target_train["layer"] == 4
    assert source_train["causal_out"] == "{run_dir}/source-sae/interventions.jsonl"
    assert source_train["model_class"] == "gemma4-conditional"
    assert source_train["trust_remote_code"] is True
    assert source_train["torch_dtype"] == "bfloat16"
    assert source_train["device_map"] == "auto"
    assert target_train["target_token"] == ["auto"]
    assert data["steps"][3]["name"] == "inspect-source"
    assert data["steps"][3]["args"]["require_interventions"] is True
    assert data["steps"][5]["args"]["source_top_k"] == 2
    assert data["steps"][5]["args"]["target_top_k"] == 5
    assert data["steps"][5]["args"]["model_class"] == "gemma4-conditional"
    graph_args = data["steps"][6]["args"]
    assert graph_args["report"] == [
        "{run_dir}/source-report/report.json",
        "{run_dir}/target-report/report.json",
    ]
    assert graph_args["path_records"] == "{run_dir}/paths.jsonl"
    assert data["steps"][7]["name"] == "summarize-graph"
    validation_args = data["steps"][8]["args"]
    assert validation_args["dataset"] == str(tmp_path / "heldout.jsonl")
    assert validation_args["top_k"] == 3
    assert validation_args["graph_out"] == "{run_dir}/validated-graph.json"
    assert validation_args["graph_html_out"] == "{run_dir}/validated-graph.html"
    assert data["steps"][9]["name"] == "summarize-validated-graph"

    assert main(["run", str(config), "--dry-run"]) == 0
    dry_run = capsys.readouterr().out
    assert "interp-lab export-hf-sae-paths" in dry_run
    assert "interp-lab validate-hf-sae-paths" in dry_run
    assert "interp-lab summarize-attribution-graph" in dry_run
