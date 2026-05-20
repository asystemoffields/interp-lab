import json
from pathlib import Path

from interp_lab.cli import main
from interp_lab.release_check import build_release_readiness_report, render_release_check_text


def test_release_check_reports_stable_gate_status():
    report = build_release_readiness_report(Path.cwd())

    assert report["schema_version"] == "interp-lab.release_check.v1"
    assert "ready_for_stable_release" in report
    classifier_check = next(check for check in report["checks"] if check["id"] == "development_classifier")
    assert classifier_check["status"] == "pass"
    blockers_check = next(check for check in report["checks"] if check["id"] == "known_stable_blockers")
    assert blockers_check["status"] == "pass"
    demo_check = next(check for check in report["checks"] if check["id"] == "real_model_demo_coverage")
    assert demo_check["status"] == "pass"
    assert "valid real-model demo manifest" in demo_check["detail"]
    assert "interp-lab stable release check:" in render_release_check_text(report)


def test_release_check_cli_writes_json_and_strict_reflects_report(tmp_path: Path, capsys):
    out = tmp_path / "release-check.json"

    exit_code = main(["release-check", "--json", "--out", str(out)])
    printed = json.loads(capsys.readouterr().out.split("Wrote", maxsplit=1)[0])
    written = json.loads(out.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert printed["schema_version"] == "interp-lab.release_check.v1"
    assert written["schema_version"] == "interp-lab.release_check.v1"

    strict_exit_code = main(["release-check", "--strict", "--out", str(tmp_path / "strict.json")])
    capsys.readouterr()

    assert strict_exit_code == (0 if written["ready_for_stable_release"] else 1)


def test_release_check_strict_fails_for_incomplete_repo(tmp_path: Path, capsys):
    exit_code = main(["release-check", "--repo-root", str(tmp_path), "--strict"])
    capsys.readouterr()

    assert exit_code == 1
