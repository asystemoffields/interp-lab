import json
from pathlib import Path

from oracle_sae.cli import main
from oracle_sae.release_check import build_release_readiness_report, render_release_check_text


def test_release_check_reports_current_stable_blockers():
    report = build_release_readiness_report(Path.cwd())

    assert report["schema_version"] == "interp-lab.release_check.v1"
    assert report["ready_for_stable_release"] is False
    assert report["summary"]["blocker"] >= 1
    assert any(check["id"] == "development_classifier" for check in report["checks"])
    assert any(check["id"] == "known_stable_blockers" for check in report["checks"])
    assert any(action["id"] == "development_classifier" for action in report["agent_next_actions"])
    assert "NOT READY" in render_release_check_text(report)


def test_release_check_cli_writes_json_and_strict_fails(tmp_path: Path, capsys):
    out = tmp_path / "release-check.json"

    exit_code = main(["release-check", "--json", "--out", str(out)])
    printed = json.loads(capsys.readouterr().out.split("Wrote", maxsplit=1)[0])
    written = json.loads(out.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert printed["schema_version"] == "interp-lab.release_check.v1"
    assert written["schema_version"] == "interp-lab.release_check.v1"
    assert written["ready_for_stable_release"] is False

    strict_exit_code = main(["release-check", "--strict", "--out", str(tmp_path / "strict.json")])
    capsys.readouterr()

    assert strict_exit_code == 1
