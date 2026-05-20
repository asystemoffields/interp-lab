import json
from pathlib import Path

import interp_lab
from oracle_sae.cli import main
from oracle_sae.demo_sweep import build_demo_sweep_report, render_demo_sweep_text


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
    assert report["demos"][0]["artifacts"][0]["sha256"]


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

    assert seen == [["demo", "--out", "reports/toy"], ["inspect", "--model", "toy/model", "--criterion", "demo"]]
    assert report["status"] == "incomplete"
    assert report["demos"][0]["command_summary"]["skipped"] == 1


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
