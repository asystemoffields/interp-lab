import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from interp_lab.cli import build_parser
from interp_lab.web_server import build_studio_server
from interp_lab.web_app import COMMAND_SPECS, command_specs_from_parser, render_web_app_html, write_web_app


def test_command_specs_cover_cli_subcommands():
    specs = command_specs_from_parser(build_parser())
    ids = {spec["id"] for spec in specs}

    assert {
        "inspect",
        "train-sae",
        "export-transformerlens-records",
        "export-nnsight-records",
        "validate-matches",
        "check-explanation-consistency",
        "search-features",
        "compare-model-families",
        "match-text-pivot",
        "export-attribution-graph",
        "validate-attribution-graph",
        "validate-assay",
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
    assert "validate-assay" in html
    assert "criterion-lab" in html
    assert "check-explanation-consistency" in html
    assert "search-features" in html
    assert "compare-model-families" in html
    assert "match-text-pivot" in html
    assert "--nla-explanations" in html
    assert "Discovery-first Criterion Lab" in html
    assert "Validate an assay" in html
    assert "--preset-file" in html
    assert "--list-presets" in html
    assert "--trust-remote-code" in html
    assert "Choose..." in html
    assert "splitExtraFlags" in html
    assert "buildArgv(spec, data, false).slice(1)" in html
    assert "args: stepArgs" in html
    assert "server-status" in html
    assert "start-job" in html
    assert "run-config-import" in html
    assert "start-imported-config" in html
    assert "artifact-list" in html
    assert "/api/jobs" in html
    assert "renderGraphOverview" in html


def test_optional_select_fields_do_not_receive_parser_choice_defaults():
    specs = command_specs_from_parser(build_parser())
    inspect = next(spec for spec in specs if spec["id"] == "inspect")
    scope_source = next(field for field in inspect["fields"] if field["key"] == "scope_source")

    assert scope_source["type"] == "select"
    assert "default" not in scope_source


def test_explanation_studio_specs_round_trip_through_parser():
    parser = build_parser()
    specs = command_specs_from_parser(parser)
    ids = {spec["id"]: spec for spec in specs}
    samples = {
        "check-explanation-consistency": [
            "--report",
            "reports/a/report.json",
            "--report",
            "reports/b/report.json",
            "--max-rank-span",
            "3",
        ],
        "search-features": ["--report", "reports/a/report.json", "--query", "tool calls"],
        "compare-model-families": [
            "--member",
            "gemma=reports/gemma/report.json",
            "--member",
            "qwen=reports/qwen/report.json",
        ],
        "match-text-pivot": ["--left", "reports/a/report.json", "--right", "reports/b/report.json"],
    }

    for command, args in samples.items():
        assert command in ids
        namespace = parser.parse_args([command, *args])
        assert namespace.command == command


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
        if argv[0] == "run":
            assert Path(argv[1]).exists()
        out = workspace / "reports" / "job-output.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('{"ok": true}', encoding="utf-8")
        return 0, "done", ""

    specs = [
        {"id": "demo", "label": "Demo", "group": "Utility", "fields": []},
        {"id": "run", "label": "Run", "group": "Utility", "fields": []},
    ]
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
        health = _get_json(server.url + "api/health", token=server.token)
        assert health["ok"] is True
        assert health["command_count"] == 2
        assert health["history_schema_version"] == "interp-lab.studio_history.v1"
        assert health["history_path"].endswith("jobs.json")

        artifacts = _get_json(server.url + "api/artifacts", token=server.token)["artifacts"]
        assert {item["relative_path"] for item in artifacts} >= {"graph.json", "report.html"}

        artifact = _get_json(server.url + "api/artifact?path=" + _quote(str(graph)), token=server.token)
        assert artifact["kind"] == "graph"
        assert "attribution_graph" in artifact["text"]

        created = _post_json(server.url + "api/jobs", {"argv": ["demo", "--out", "reports/demo"]}, token=server.token)["job"]
        job = _wait_for_job(server.url, created["id"], token=server.token)
        assert job["status"] == "succeeded"
        assert job["stdout"] == "done"
        assert ran == [["demo", "--out", "reports/demo"]]

        config = {
            "schema_version": "interp-lab.run.v1",
            "steps": [{"name": "demo", "command": "demo", "args": {"out": "reports/demo"}}],
        }
        created_config = _post_json(server.url + "api/jobs", {"run_config": config}, token=server.token)["job"]
        config_job = _wait_for_job(server.url, created_config["id"], token=server.token)
        assert config_job["status"] == "succeeded"
        assert config_job["source"] == "run_config"
        assert config_job["run_config_path"]
        assert json.loads(Path(config_job["run_config_path"]).read_text(encoding="utf-8")) == config
        assert ran[1][0] == "run"

        job_listing = _get_json(server.url + "api/jobs", token=server.token)
        assert job_listing["schema_version"] == "interp-lab.studio_history.v1"
        assert len(job_listing["jobs"]) == 2
    finally:
        server.stop()


def test_studio_server_persists_job_history(tmp_path: Path):
    reports = tmp_path / "reports"
    specs = [{"id": "demo", "label": "Demo", "group": "Utility", "fields": []}]

    def runner(argv: list[str], workspace: Path):
        return 0, "persisted", ""

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
        created = _post_json(server.url + "api/jobs", {"argv": ["demo"]}, token=server.token)["job"]
        completed = _wait_for_job(server.url, created["id"], token=server.token)
        assert completed["status"] == "succeeded"
    finally:
        server.stop()

    history_path = reports / ".studio" / "jobs.json"
    assert history_path.exists()
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert history["schema_version"] == "interp-lab.studio_history.v1"

    restarted = build_studio_server(
        host="127.0.0.1",
        port=0,
        reports_dir=reports,
        command_specs=specs,
        command_runner=runner,
        workspace=tmp_path,
    )
    restarted.start()
    try:
        jobs = _get_json(restarted.url + "api/jobs", token=restarted.token)["jobs"]
        assert jobs[0]["id"] == created["id"]
        assert jobs[0]["stdout"] == "persisted"
        assert jobs[0]["status"] == "succeeded"
    finally:
        restarted.stop()


def test_studio_server_marks_stale_running_jobs_interrupted(tmp_path: Path):
    reports = tmp_path / "reports"
    history_path = reports / ".studio" / "jobs.json"
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps(
            {
                "schema_version": "interp-lab.studio_history.v1",
                "jobs": [
                    {
                        "id": "stale-job",
                        "argv": ["demo"],
                        "command": "demo",
                        "status": "running",
                        "created_at": "2026-05-20T00:00:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    server = build_studio_server(
        host="127.0.0.1",
        port=0,
        reports_dir=reports,
        command_specs=[{"id": "demo", "label": "Demo", "group": "Utility", "fields": []}],
        command_runner=lambda argv, workspace: (0, "", ""),
        workspace=tmp_path,
    )
    server.start()
    try:
        jobs = _get_json(server.url + "api/jobs", token=server.token)["jobs"]
        assert jobs[0]["id"] == "stale-job"
        assert jobs[0]["status"] == "interrupted"
        assert "stopped before this job completed" in jobs[0]["stderr"]
    finally:
        server.stop()

    persisted = json.loads(history_path.read_text(encoding="utf-8"))["jobs"][0]
    assert persisted["status"] == "interrupted"


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
            _post_json(server.url + "api/jobs", {"argv": ["not-real"]}, token=server.token)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            body = json.loads(exc.read().decode("utf-8"))
            assert "unknown interp-lab command" in body["error"]
        else:  # pragma: no cover - failure branch.
            raise AssertionError("unknown command was accepted")
    finally:
        server.stop()


def test_studio_server_requires_token_for_api(tmp_path: Path):
    server = build_studio_server(
        host="127.0.0.1",
        port=0,
        reports_dir=tmp_path / "reports",
        command_specs=[{"id": "demo", "label": "Demo", "group": "Utility", "fields": []}],
        command_runner=lambda argv, workspace: (0, "", ""),
        workspace=tmp_path,
    )
    server.start()
    try:
        try:
            _post_json(server.url + "api/jobs", {"argv": ["demo"]})
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
            body = json.loads(exc.read().decode("utf-8"))
            assert "token" in body["error"]
        else:  # pragma: no cover - failure branch.
            raise AssertionError("missing token was accepted")
    finally:
        server.stop()


def test_studio_server_rejects_cross_origin_api_requests(tmp_path: Path):
    server = build_studio_server(
        host="127.0.0.1",
        port=0,
        reports_dir=tmp_path / "reports",
        command_specs=[{"id": "demo", "label": "Demo", "group": "Utility", "fields": []}],
        command_runner=lambda argv, workspace: (0, "", ""),
        workspace=tmp_path,
    )
    server.start()
    try:
        try:
            _post_json(
                server.url + "api/jobs",
                {"argv": ["demo"]},
                token=server.token,
                headers={"Origin": "https://example.com"},
            )
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
            body = json.loads(exc.read().decode("utf-8"))
            assert "origin" in body["error"]
        else:  # pragma: no cover - failure branch.
            raise AssertionError("cross-origin request was accepted")
    finally:
        server.stop()


def test_studio_server_rejects_imported_run_config_nested_server(tmp_path: Path):
    server = build_studio_server(
        host="127.0.0.1",
        port=0,
        reports_dir=tmp_path / "reports",
        command_specs=[
            {"id": "run", "label": "Run", "group": "Utility", "fields": []},
            {"id": "studio", "label": "Studio", "group": "Utility", "fields": []},
        ],
        command_runner=lambda argv, workspace: (0, "", ""),
        workspace=tmp_path,
    )
    server.start()
    try:
        try:
            _post_json(
                server.url + "api/jobs",
                {
                    "run_config": {
                        "schema_version": "interp-lab.run.v1",
                        "steps": [{"name": "serve", "command": "studio", "args": {"serve": True}}],
                    }
                },
                token=server.token,
            )
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            body = json.loads(exc.read().decode("utf-8"))
            assert "nested server" in body["error"]
        else:  # pragma: no cover - failure branch.
            raise AssertionError("nested Studio run config was accepted")
    finally:
        server.stop()


def test_studio_artifacts_are_limited_to_reports_dir(tmp_path: Path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "report.json").write_text('{"ok": true}', encoding="utf-8")
    secret = tmp_path / "pyproject.toml"
    secret.write_text("[project]\nname = 'private'\n", encoding="utf-8")
    server = build_studio_server(
        host="127.0.0.1",
        port=0,
        reports_dir=reports,
        command_specs=[{"id": "demo", "label": "Demo", "group": "Utility", "fields": []}],
        command_runner=lambda argv, workspace: (0, "", ""),
        workspace=tmp_path,
    )
    server.start()
    try:
        artifact = _get_json(server.url + "api/artifact?path=report.json", token=server.token)
        assert artifact["path"] == "report.json"

        try:
            _get_json(server.url + "api/artifact?path=" + _quote(str(secret)), token=server.token)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            body = json.loads(exc.read().decode("utf-8"))
            assert "reports directory" in body["error"]
        else:  # pragma: no cover - failure branch.
            raise AssertionError("workspace file outside reports was accepted")
    finally:
        server.stop()


def _get_json(url: str, *, token: str = "", headers: dict[str, str] | None = None):
    request_headers = dict(headers or {})
    if token:
        request_headers["X-Interp-Lab-Studio-Token"] = token
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict, *, token: str = "", headers: dict[str, str] | None = None):
    data = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    if token:
        request_headers["X-Interp-Lab-Studio-Token"] = token
    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_job(base_url: str, job_id: str, *, token: str):
    for _ in range(50):
        job = _get_json(base_url + "api/jobs/" + job_id, token=token)["job"]
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def test_demo_sweep_out_field_does_not_prefill_archived_evidence_path():
    # Verify-only sweeps deliberately write nothing unless --out is set, so the
    # Studio form must not prefill the archived-evidence default path (a
    # run_commands=false sweep would clobber reports/real-model-demo-sweep.json).
    for specs in (COMMAND_SPECS, command_specs_from_parser(build_parser())):
        spec = next(item for item in specs if item["id"] == "demo-sweep")
        out_field = next(field for field in spec["fields"] if field["key"] == "out")
        assert not out_field.get("default")
        assert "real-model-demo-sweep" in out_field.get("placeholder", "")
