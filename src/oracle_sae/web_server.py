from __future__ import annotations

import json
import mimetypes
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import parse_qs, unquote, urlparse

from oracle_sae.web_app import COMMAND_SPECS, render_web_app_html

CommandRunner = Callable[[list[str], Path], tuple[int, str, str]]


@dataclass
class StudioServer:
    server: ThreadingHTTPServer
    thread: threading.Thread | None
    url: str

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)


def build_studio_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    reports_dir: str | Path = "reports",
    command_specs: Sequence[dict[str, Any]] | None = None,
    command_runner: CommandRunner | None = None,
    workspace: str | Path | None = None,
) -> StudioServer:
    workspace_path = Path.cwd() if workspace is None else Path(workspace)
    workspace_path = workspace_path.resolve()
    reports_path = _resolve_under_workspace(reports_dir, workspace_path)
    specs = list(COMMAND_SPECS if command_specs is None else command_specs)
    allowed_commands = {str(spec.get("id")) for spec in specs if spec.get("id")}
    html = render_web_app_html(command_specs=specs)
    state = _ServerState(
        workspace=workspace_path,
        reports_dir=reports_path,
        command_specs=specs,
        allowed_commands=allowed_commands,
        command_runner=command_runner or _subprocess_runner,
    )

    class Handler(_StudioRequestHandler):
        server_state = state
        index_html = html

    httpd = ThreadingHTTPServer((host, int(port)), Handler)
    actual_host, actual_port = httpd.server_address[:2]
    display_host = host if host not in {"", "0.0.0.0"} else str(actual_host)
    return StudioServer(
        server=httpd,
        thread=None,
        url=f"http://{display_host}:{actual_port}/",
    )


def serve_web_app(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    reports_dir: str | Path = "reports",
    command_specs: Sequence[dict[str, Any]] | None = None,
    open_browser: bool = False,
) -> None:
    runtime = build_studio_server(
        host=host,
        port=port,
        reports_dir=reports_dir,
        command_specs=command_specs,
    )
    print(f"Serving interp-lab Studio at {runtime.url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(runtime.url)
    try:
        runtime.server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping interp-lab Studio.")
    finally:
        runtime.server.server_close()


class _ServerState:
    def __init__(
        self,
        *,
        workspace: Path,
        reports_dir: Path,
        command_specs: list[dict[str, Any]],
        allowed_commands: set[str],
        command_runner: CommandRunner,
    ) -> None:
        self.workspace = workspace
        self.reports_dir = reports_dir
        self.command_specs = command_specs
        self.allowed_commands = allowed_commands
        self.command_runner = command_runner
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def snapshot_jobs(self) -> list[dict[str, Any]]:
        with self.lock:
            jobs = [dict(job) for job in self.jobs.values()]
        return sorted(jobs, key=lambda item: str(item.get("created_at", "")), reverse=True)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job is not None else None

    def create_job(self, argv: list[str], *, job_id: str | None = None) -> dict[str, Any]:
        self._validate_argv(argv)
        job_id = job_id or uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "argv": argv,
            "command": argv[0],
            "status": "queued",
            "created_at": _utc_timestamp(),
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "workspace": str(self.workspace),
        }
        with self.lock:
            self.jobs[job_id] = job
        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()
        return dict(job)

    def create_config_job(self, config: dict[str, Any]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        run_dir = self.reports_dir / "studio-runs" / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "run-config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return self.create_job(["run", str(config_path)], job_id=job_id)

    def _validate_argv(self, argv: list[str]) -> None:
        if not argv:
            raise ValueError("request needs a non-empty argv list")
        if not all(isinstance(item, str) and item for item in argv):
            raise ValueError("argv must contain non-empty strings")
        command = argv[0]
        if self.allowed_commands and command not in self.allowed_commands:
            raise ValueError(f"unknown interp-lab command: {command}")
        if command in {"studio", "web-app"} and "--serve" in argv[1:]:
            raise ValueError("Studio cannot launch a nested server job")

    def _run_job(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job["status"] = "running"
            job["started_at"] = _utc_timestamp()
            argv = list(job["argv"])
        try:
            exit_code, stdout, stderr = self.command_runner(argv, self.workspace)
        except Exception as exc:  # pragma: no cover - defensive boundary.
            exit_code = 1
            stdout = ""
            stderr = f"{type(exc).__name__}: {exc}"
        with self.lock:
            job = self.jobs[job_id]
            job["status"] = "succeeded" if exit_code == 0 else "failed"
            job["finished_at"] = _utc_timestamp()
            job["exit_code"] = exit_code
            job["stdout"] = stdout
            job["stderr"] = stderr


class _StudioRequestHandler(BaseHTTPRequestHandler):
    server_state: _ServerState
    index_html: str

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
        parsed = urlparse(self.path)
        if parsed.path in {"", "/"}:
            self._send_text(self.index_html, content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "workspace": str(self.server_state.workspace),
                    "reports_dir": str(self.server_state.reports_dir),
                    "command_count": len(self.server_state.command_specs),
                }
            )
            return
        if parsed.path == "/api/specs":
            self._send_json({"commands": self.server_state.command_specs})
            return
        if parsed.path == "/api/jobs":
            self._send_json({"jobs": self.server_state.snapshot_jobs()})
            return
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            job = self.server_state.get_job(job_id)
            if job is None:
                self._send_error_json(HTTPStatus.NOT_FOUND, "job not found")
                return
            self._send_json({"job": job})
            return
        if parsed.path == "/api/artifacts":
            self._send_json({"artifacts": _artifact_records(self.server_state.reports_dir)})
            return
        if parsed.path == "/api/artifact":
            self._send_artifact(parsed.query, raw=False)
            return
        if parsed.path == "/api/raw":
            self._send_artifact(parsed.query, raw=True)
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
        parsed = urlparse(self.path)
        if parsed.path != "/api/jobs":
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            payload = self._read_json()
            if "run_config" in payload:
                config = payload["run_config"]
                if not isinstance(config, dict):
                    raise ValueError("run_config must be an object")
                job = self.server_state.create_config_job(config)
            else:
                argv = payload.get("argv")
                if not isinstance(argv, list):
                    raise ValueError("argv must be a list")
                job = self.server_state.create_job([str(item) for item in argv])
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_json({"job": job}, status=HTTPStatus.ACCEPTED)

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler API.
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_common_headers()
        self.end_headers()

    def log_message(self, _format: str, *args: Any) -> None:
        return

    def _send_artifact(self, query: str, *, raw: bool) -> None:
        params = parse_qs(query)
        path_values = params.get("path", [])
        if not path_values:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "path is required")
            return
        try:
            path = _safe_workspace_path(path_values[0], self.server_state.workspace)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if not path.exists() or not path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "artifact not found")
            return
        if raw:
            content_type = mimetypes.guess_type(path.name)[0] or "text/plain"
            self._send_bytes(path.read_bytes(), content_type=content_type)
            return
        text = path.read_text(encoding="utf-8", errors="replace")
        self._send_json({"path": str(path), "kind": _artifact_kind(path), "text": text})

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        data = self.rfile.read(length)
        payload = json.loads(data.decode("utf-8") if data else "{}")
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self._send_bytes(body, status=status, content_type="application/json; charset=utf-8")

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"ok": False, "error": message}, status=status)

    def _send_text(
        self,
        text: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        self._send_bytes(text.encode("utf-8"), status=status, content_type=content_type)

    def _send_bytes(
        self,
        body: bytes,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        content_type: str,
    ) -> None:
        self.send_response(status)
        self._send_common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_common_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def _subprocess_runner(argv: list[str], workspace: Path) -> tuple[int, str, str]:
    process = subprocess.run(
        [sys.executable, "-m", "oracle_sae", *argv],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    return process.returncode, process.stdout, process.stderr


def _artifact_records(reports_dir: Path, *, limit: int = 300) -> list[dict[str, Any]]:
    if not reports_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    suffixes = {".html", ".json", ".jsonl", ".md"}
    for path in sorted(reports_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        stat = path.stat()
        records.append(
            {
                "path": str(path),
                "name": path.name,
                "relative_path": str(path.relative_to(reports_dir)),
                "kind": _artifact_kind(path),
                "size_bytes": stat.st_size,
                "modified_at": _utc_timestamp(stat.st_mtime),
            }
        )
    return sorted(records, key=lambda item: str(item["modified_at"]), reverse=True)[:limit]


def _artifact_kind(path: Path) -> str:
    name = path.name.lower()
    if path.suffix.lower() == ".html":
        return "html"
    if name in {"graph.json", "annotated-graph.json"} or "graph" in name:
        return "graph"
    if "match" in name:
        return "match"
    if name == "report.json":
        return "inspection_report"
    if name == "manifest.json":
        return "manifest"
    if path.suffix.lower() == ".md":
        return "markdown"
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    return "json"


def _safe_workspace_path(path_text: str, workspace: Path) -> Path:
    text = unquote(path_text)
    path = Path(text)
    candidate = path if path.is_absolute() else workspace / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("artifact path must stay inside the workspace") from exc
    return resolved


def _resolve_under_workspace(path: str | Path, workspace: Path) -> Path:
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("reports directory must stay inside the workspace") from exc
    return resolved


def _utc_timestamp(seconds: float | None = None) -> str:
    if seconds is None:
        seconds = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))
