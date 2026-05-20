import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from oracle_sae.cli import build_parser
from oracle_sae.web_server import build_studio_server
from oracle_sae.web_app import command_specs_from_parser, render_web_app_html, write_web_app


def test_command_specs_cover_cli_subcommands():
    specs = command_specs_from_parser(build_parser())
    ids = {spec["id"] for spec in specs}

    assert {
        "inspect",
        "train-sae",
        "export-transformerlens-records",
        "export-nnsight-records",
        "validate-matches",
        "export-attribution-graph",
        "validate-attribution-graph",
        "criterion-lab",
        "studio",
        "demo",
    } <= ids
    assert "web-app" not in ids


def test_render_web_app_contains_required_surfaces():
    specs = command_specs_from_parser(build_parser())

    html = render_web_app_html(command_specs=specs)

    assert "Interp Lab Studio" in html
    assert "command-specs" in html
    assert "generated-command" in html
    assert "run-config-output" in html
    assert "validate-hf-sae-paths" in html
    assert "criterion-lab" in html
    assert "Discovery-first Criterion Lab" in html
    assert "--preset-file" in html
    assert "--list-presets" in html
    assert "--trust-remote-code" in html
    assert "Choose..." in html
    assert "splitExtraFlags" in html
    assert "buildArgv(spec, data, false).slice(1)" in html
    assert "args: stepArgs" in html
    assert "server-status" in html
    assert "start-job" in html
    assert "artifact-list" in html
    assert "/api/jobs" in html
    assert "renderGraphOverview" in html


def test_optional_select_fields_do_not_receive_parser_choice_defaults():
    specs = command_specs_from_parser(build_parser())
    inspect = next(spec for spec in specs if spec["id"] == "inspect")
    scope_source = next(field for field in inspect["fields"] if field["key"] == "scope_source")

    assert scope_source["type"] == "select"
    assert "default" not in scope_source


def test_write_web_app(tmp_path: Path):
    out = tmp_path / "studio.html"

    path = write_web_app(out, command_specs=command_specs_from_parser(build_parser()))

    assert path == out
    assert "Interp Lab Studio" in out.read_text(encoding="utf-8")


def test_studio_server_health_jobs_and_artifacts(tmp_path: Path):
    reports = tmp_path / "reports"
    reports.mkdir()
    graph = reports / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "schema_version": "interp-lab.attribution_graph.v1",
                "nodes": [{"id": "criterion", "type": "criterion"}],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    (reports / "report.html").write_text("<h1>Report</h1>", encoding="utf-8")
    ran: list[list[str]] = []

    def runner(argv: list[str], workspace: Path):
        ran.append(argv)
        out = workspace / "reports" / "job-output.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('{"ok": true}', encoding="utf-8")
        return 0, "done", ""

    specs = [{"id": "demo", "label": "Demo", "group": "Utility", "fields": []}]
    server = build_studio_server(
        host="127.0.0.1",
        port=0,
        reports_dir=reports,
        command_specs=specs,
        command_runner=runner,
        workspace=tmp_path,
    )
    server.start()
    try:
        health = _get_json(server.url + "api/health")
        assert health["ok"] is True
        assert health["command_count"] == 1

        artifacts = _get_json(server.url + "api/artifacts")["artifacts"]
        assert {item["relative_path"] for item in artifacts} >= {"graph.json", "report.html"}

        artifact = _get_json(server.url + "api/artifact?path=" + _quote(str(graph)))
        assert artifact["kind"] == "graph"
        assert "attribution_graph" in artifact["text"]

        created = _post_json(server.url + "api/jobs", {"argv": ["demo", "--out", "reports/demo"]})["job"]
        job = _wait_for_job(server.url, created["id"])
        assert job["status"] == "succeeded"
        assert job["stdout"] == "done"
        assert ran == [["demo", "--out", "reports/demo"]]
    finally:
        server.stop()


def test_studio_server_rejects_unknown_commands(tmp_path: Path):
    specs = [{"id": "demo", "label": "Demo", "group": "Utility", "fields": []}]
    server = build_studio_server(
        host="127.0.0.1",
        port=0,
        reports_dir=tmp_path / "reports",
        command_specs=specs,
        command_runner=lambda argv, workspace: (0, "", ""),
        workspace=tmp_path,
    )
    server.start()
    try:
        try:
            _post_json(server.url + "api/jobs", {"argv": ["not-real"]})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            body = json.loads(exc.read().decode("utf-8"))
            assert "unknown interp-lab command" in body["error"]
        else:  # pragma: no cover - failure branch.
            raise AssertionError("unknown command was accepted")
    finally:
        server.stop()


def _get_json(url: str):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_job(base_url: str, job_id: str):
    for _ in range(50):
        job = _get_json(base_url + "api/jobs/" + job_id)["job"]
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")
