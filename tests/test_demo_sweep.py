import argparse
import json
from pathlib import Path

import pytest

import interp_lab
from interp_lab.cli import build_parser, main
from interp_lab.demo_sweep import _planned_command, build_demo_sweep_report, render_demo_sweep_text


def test_demo_sweep_reports_missing_artifacts(tmp_path: Path):
    _write_demo_manifest(tmp_path)
    out = tmp_path / "sweep.json"

    exit_code = main(
        [
            "demo-sweep",
            "--repo-root",
            str(tmp_path),
            "--manifest-dir",
            "examples/real_model_demos",
            "--out",
            str(out),
            "--strict",
        ]
    )
    report = json.loads(out.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert report["schema_version"] == "interp-lab.real_model_demo_sweep.v1"
    assert report["status"] == "incomplete"
    assert report["demos"][0]["artifact_summary"]["missing"] == 3
    assert "INCOMPLETE" in render_demo_sweep_text(report)

    # Next actions are canonical objects (not the pre-2.3 plain strings): id+title
    # always, plus exactly one of command/instruction; manifest-authored strings
    # are coerced.
    demo_actions = {action["id"]: action for action in report["demos"][0]["agent_next_actions"]}
    assert demo_actions["run_demo_sweep"]["argv"] == ["interp-lab", "demo-sweep", "--run"]
    assert demo_actions["generate_missing_artifacts"]["instruction"]
    assert demo_actions["manifest_note_1"]["title"] == "Archive outputs."
    for action in [*report["agent_next_actions"], *report["demos"][0]["agent_next_actions"]]:
        assert action["id"] and action["title"]
        assert ("command" in action) != ("instruction" in action)
    # The text view renders the first action's title, not a raw dict.
    assert "{'" not in render_demo_sweep_text(report)


def test_demo_sweep_passes_when_expected_artifacts_exist(tmp_path: Path):
    _write_demo_manifest(tmp_path)
    for relative in ["reports/demo/a.json", "reports/demo/b.html", "reports/demo/c.jsonl"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")

    report = build_demo_sweep_report(
        repo_root=tmp_path,
        manifest_dir="examples/real_model_demos",
    )

    assert report["status"] == "passed"
    assert report["summary"]["passed"] == 1
    assert report["repo_root"] == "."
    assert report["manifest_dir"] == "examples/real_model_demos"
    assert str(tmp_path) not in json.dumps(report)
    assert report["demos"][0]["artifacts"][0]["sha256"]
    assert "absolute_path" not in report["demos"][0]["artifacts"][0]


def test_demo_sweep_runs_internal_commands_and_skips_external_by_default(tmp_path: Path):
    _write_demo_manifest(
        tmp_path,
        commands=[
            ["interp-lab", "demo", "--out", "reports/toy"],
            ["modal", "run", "examples/remote.py"],
            ["inspect", "--model", "toy/model", "--criterion", "demo"],
        ],
    )
    for relative in ["reports/demo/a.json", "reports/demo/b.html", "reports/demo/c.jsonl"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")
    seen = []

    def runner(argv: list[str]) -> int:
        seen.append(argv)
        return 0

    report = build_demo_sweep_report(
        repo_root=tmp_path,
        manifest_dir="examples/real_model_demos",
        run=True,
        command_runner=runner,
    )

    assert seen == [["demo", "--out", "reports/toy"]]
    assert report["status"] == "incomplete"
    assert report["demos"][0]["command_summary"]["skipped"] == 1
    assert report["demos"][0]["command_summary"]["blocked"] == 1


def test_every_cli_command_is_classified_internal():
    # Regression: the old hand-maintained allowlist drifted from the CLI, so
    # commands like demo-sweep or search-features were skipped as "external" (or
    # exec'd as nonexistent binaries under --allow-external).
    choices: set[str] = set()
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            choices.update(action.choices)
    assert choices

    for name in sorted(choices):
        record = _planned_command({"name": name, "argv": [name, "--help"]})
        assert record["kind"] == "internal", name

    prefixed = _planned_command({"name": "sweep", "argv": ["interp-lab", "demo-sweep", "--run"]})
    assert prefixed["kind"] == "internal"
    external = _planned_command({"name": "modal", "argv": ["modal", "run", "examples/remote.py"]})
    assert external["kind"] == "external"


def test_verify_only_sweep_does_not_clobber_archived_default_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    monkeypatch.chdir(tmp_path)
    _write_demo_manifest(tmp_path)
    archived = tmp_path / "reports" / "real-model-demo-sweep.json"
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_text('{"archived": true}\n', encoding="utf-8")

    exit_code = main(["demo-sweep", "--repo-root", str(tmp_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "demo sweep" in out  # results are still printed
    assert "Wrote" not in out  # but nothing is written without --out
    assert json.loads(archived.read_text(encoding="utf-8")) == {"archived": True}


def test_run_sweep_writes_default_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    _write_demo_manifest(tmp_path)

    exit_code = main(["demo-sweep", "--repo-root", str(tmp_path), "--run"])

    default_out = tmp_path / "reports" / "real-model-demo-sweep.json"
    report = json.loads(default_out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["run_commands"] is True


def test_demo_sweep_preflights_required_inputs(tmp_path: Path):
    _write_demo_manifest(tmp_path, required_inputs=["examples/missing.jsonl"])
    report = build_demo_sweep_report(
        repo_root=tmp_path,
        manifest_dir="examples/real_model_demos",
        run=True,
        command_runner=lambda argv: 0,
    )

    demo = report["demos"][0]
    assert report["status"] == "incomplete"
    assert demo["input_summary"]["missing"] == 1
    assert demo["command_summary"]["blocked"] == 3
    assert "required input" in demo["detail"]


def test_public_api_demo_sweep_writes_report(tmp_path: Path):
    _write_demo_manifest(tmp_path)
    out = tmp_path / "api-sweep.json"

    report = interp_lab.demo_sweep(
        repo_root=tmp_path,
        manifest_dir="examples/real_model_demos",
        out=out,
    )

    assert report["schema_version"] == "interp-lab.real_model_demo_sweep.v1"
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "incomplete"


def _write_demo_manifest(
    root: Path,
    *,
    commands: list[list[str]] | None = None,
    required_inputs: list[str] | None = None,
) -> Path:
    doc = root / "docs" / "DEMO.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("# Demo\n", encoding="utf-8")
    manifest_dir = root / "examples" / "real_model_demos"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "interp-lab.real_model_demo.v1",
        "id": "sample-demo",
        "title": "Sample Demo",
        "model": "toy/model",
        "criterion": "sample behavior",
        "workflow": "sample-workflow",
        "doc": "docs/DEMO.md",
        "estimated_runtime": "quick",
        "required_inputs": required_inputs or [],
        "commands": [
            {"name": f"step {index}", "argv": argv}
            for index, argv in enumerate(
                commands
                or [
                    ["demo", "--out", "reports/toy"],
                    ["demo", "--out", "reports/toy-2"],
                    ["demo", "--out", "reports/toy-3"],
                ],
                start=1,
            )
        ],
        "expected_artifacts": [
            {
                "path": "reports/demo/a.json",
                "kind": "json",
                "why_it_matters": "Confirms an artifact exists.",
                "interpretation_notes": "Review before claiming evidence.",
            },
            {
                "path": "reports/demo/b.html",
                "kind": "html",
                "why_it_matters": "Confirms a readable report exists.",
                "interpretation_notes": "Open before sharing.",
            },
            {
                "path": "reports/demo/c.jsonl",
                "kind": "records",
                "why_it_matters": "Confirms records exist.",
                "interpretation_notes": "Check records before causal claims.",
            },
        ],
        "evidence_checks": ["Check artifacts.", "Check limitations."],
        "agent_next_actions": ["Archive outputs.", "Repeat on held-out prompts."],
    }
    path = manifest_dir / "sample-demo.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
