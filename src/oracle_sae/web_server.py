from __future__ import annotations

import json
import mimetypes
import secrets
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

from oracle_sae.web_app import COMMAND_SPECS, command_specs_from_parser, render_web_app_html

CommandRunner = Callable[[list[str], Path], tuple[int, str, str]]
STUDIO_HISTORY_SCHEMA = "interp-lab.studio_history.v1"
MAX_CAPTURED_OUTPUT_CHARS = 20000
MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
STUDIO_TOKEN_HEADER = "X-Interp-Lab-Studio-Token"


@dataclass
class StudioServer:
    server: ThreadingHTTPServer
    thread: threading.Thread | None
    url: str
    token: str

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
    specs = list(_default_command_specs() if command_specs is None else command_specs)
    allowed_commands = {str(spec.get("id")) for spec in specs if spec.get("id")}
    token = secrets.token_urlsafe(32)
    html = render_web_app_html(command_specs=specs, studio_token=token)
    state = _ServerState(
        workspace=workspace_path,
        reports_dir=reports_path,
        command_specs=specs,
        allowed_commands=allowed_commands,
        command_runner=command_runner or _subprocess_runner,
        token=token,
    )

    class Handler(_StudioRequestHandler):
        server_state = state
        index_html = html

    httpd = ThreadingHTTPServer((host, int(port)), Handler)
    actual_host, actual_port = httpd.server_address[:2]
    state.allowed_hosts = _allowed_hosts(host, str(actual_host), int(actual_port))
    display_host = host if host not in {"", "0.0.0.0"} else str(actual_host)
    return StudioServer(
        server=httpd,
        thread=None,
        url=f"http://{display_host}:{actual_port}/",
        token=token,
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
        token: str,
    ) -> None:
        self.workspace = workspace
        self.reports_dir = reports_dir
        self.command_specs = command_specs
        self.allowed_commands = allowed_commands
        self.command_runner = command_runner
        self.token = token
        self.allowed_hosts: set[str] = set()
        self.history_path = self.reports_dir / ".studio" / "jobs.json"
        self.lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = self._load_jobs()
        if self.jobs:
            with self.lock:
                self._persist_jobs_locked()

    def snapshot_jobs(self) -> list[dict[str, Any]]:
        with self.lock:
            return self.snapshot_jobs_locked()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job is not None else None

    def create_job(
        self,
        argv: list[str],
        *,
        job_id: str | None = None,
        source: str = "argv",
        run_config_path: Path | None = None,
    ) -> dict[str, Any]:
        self._validate_argv(argv)
        job_id = job_id or uuid.uuid4().hex[:12]
        job = {
            "schema_version": STUDIO_HISTORY_SCHEMA,
            "id": job_id,
            "argv": argv,
            "command": argv[0],
            "source": source,
            "status": "queued",
            "created_at": _utc_timestamp(),
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "captured_output_truncated": False,
            "workspace": str(self.workspace),
            "run_config_path": str(run_config_path) if run_config_path else None,
        }
        with self.lock:
            self.jobs[job_id] = job
            self._persist_jobs_locked()
        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()
        return dict(job)

    def create_config_job(self, config: dict[str, Any]) -> dict[str, Any]:
        self._validate_run_config(config)
        job_id = uuid.uuid4().hex[:12]
        run_dir = self.reports_dir / "studio-runs" / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "run-config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return self.create_job(["run", str(config_path)], job_id=job_id, source="run_config", run_config_path=config_path)

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

    def _validate_run_config(self, config: dict[str, Any]) -> None:
        if "steps" not in config:
            if self.allowed_commands and "inspect" not in self.allowed_commands:
                raise ValueError("imported run config resolves to inspect, which is not an allowed Studio command")
            return
        steps = config.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("run_config steps must be a non-empty list")
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                raise ValueError(f"run_config step {index} must be an object")
            command = step.get("command")
            if not isinstance(command, str) or not command:
                raise ValueError(f"run_config step {index} is missing command")
            if command == "run":
                raise ValueError("Studio run configs cannot recursively call interp-lab run")
            if self.allowed_commands and command not in self.allowed_commands:
                raise ValueError(f"unknown interp-lab command in run_config step {index}: {command}")
            if command in {"studio", "web-app"} and _step_requests_server(step.get("args", {})):
                raise ValueError("Studio cannot launch a nested server job from an imported run_config")

    def _run_job(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job["status"] = "running"
            job["started_at"] = _utc_timestamp()
            argv = list(job["argv"])
            self._persist_jobs_locked()
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
            job["stdout"] = _truncate_output(stdout)
            job["stderr"] = _truncate_output(stderr)
            job["captured_output_truncated"] = stdout != job["stdout"] or stderr != job["stderr"]
            self._persist_jobs_locked()

    def _load_jobs(self) -> dict[str, dict[str, Any]]:
        if not self.history_path.exists():
            return {}
        try:
            payload = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        jobs = payload.get("jobs", [])
        if not isinstance(jobs, list):
            return {}
        loaded: dict[str, dict[str, Any]] = {}
        for item in jobs:
            if not isinstance(item, dict):
                continue
            job_id = item.get("id")
            if not isinstance(job_id, str) or not job_id:
                continue
            job = _normalize_job_record(item, self.workspace)
            loaded[job_id] = job
        return loaded

    def _persist_jobs_locked(self) -> None:
        payload = {
            "schema_version": STUDIO_HISTORY_SCHEMA,
            "updated_at": _utc_timestamp(),
            "workspace": str(self.workspace),
            "reports_dir": str(self.reports_dir),
            "jobs": self.snapshot_jobs_locked(limit=200),
        }
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.history_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temp_path.replace(self.history_path)
        except OSError:
            return

    def snapshot_jobs_locked(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        jobs = [dict(job) for job in self.jobs.values()]
        ordered = sorted(jobs, key=lambda item: str(item.get("created_at", "")), reverse=True)
        return ordered if limit is None else ordered[:limit]


class _StudioRequestHandler(BaseHTTPRequestHandler):
    server_state: _ServerState
    index_html: str

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
        parsed = urlparse(self.path)
        if parsed.path in {"", "/"}:
            self._send_text(self.index_html, content_type="text/html; charset=utf-8")
            return
        if parsed.path.startswith("/api/") and not self._authorize_api_request(parsed.query):
            return
        if parsed.path == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "workspace": str(self.server_state.workspace),
                    "reports_dir": str(self.server_state.reports_dir),
                    "history_path": str(self.server_state.history_path),
                    "history_schema_version": STUDIO_HISTORY_SCHEMA,
                    "command_count": len(self.server_state.command_specs),
                }
            )
            return
        if parsed.path == "/api/specs":
            self._send_json({"commands": self.server_state.command_specs})
            return
        if parsed.path == "/api/jobs":
            self._send_json({"schema_version": STUDIO_HISTORY_SCHEMA, "jobs": self.server_state.snapshot_jobs()})
            return
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            job = self.server_state.get_job(job_id)
            if job is None:
                self._send_error_json(HTTPStatus.NOT_FOUND, "job not found")
                return
            self._send_json({"schema_version": STUDIO_HISTORY_SCHEMA, "job": job})
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
        if parsed.path.startswith("/api/") and not self._authorize_api_request(parsed.query):
            return
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
        if not self._origin_allowed():
            self._send_error_json(HTTPStatus.FORBIDDEN, "origin is not allowed")
            return
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
            path = _safe_artifact_path(path_values[0], self.server_state.reports_dir)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if not path.exists() or not path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "artifact not found")
            return
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            self._send_error_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "artifact is too large to preview")
            return
        if raw:
            content_type = mimetypes.guess_type(path.name)[0] or "text/plain"
            self._send_bytes(path.read_bytes(), content_type=content_type)
            return
        text = path.read_text(encoding="utf-8", errors="replace")
        self._send_json({"path": _public_artifact_path(path, self.server_state.reports_dir), "kind": _artifact_kind(path), "text": text})

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
        origin = self.headers.get("Origin")
        if origin and self._origin_allowed():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", f"Content-Type, {STUDIO_TOKEN_HEADER}, Authorization")

    def _authorize_api_request(self, query: str) -> bool:
        if not self._host_allowed():
            self._send_error_json(HTTPStatus.FORBIDDEN, "host is not allowed")
            return False
        if not self._origin_allowed():
            self._send_error_json(HTTPStatus.FORBIDDEN, "origin is not allowed")
            return False
        if not self._token_allowed(query):
            self._send_error_json(HTTPStatus.FORBIDDEN, "Studio token is required")
            return False
        return True

    def _host_allowed(self) -> bool:
        host = self.headers.get("Host")
        if not host or not self.server_state.allowed_hosts:
            return True
        return host.lower() in self.server_state.allowed_hosts

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlparse(origin)
        if parsed.scheme not in {"http", "https"}:
            return False
        netloc = parsed.netloc.lower()
        return not self.server_state.allowed_hosts or netloc in self.server_state.allowed_hosts

    def _token_allowed(self, query: str) -> bool:
        token = self.headers.get(STUDIO_TOKEN_HEADER, "")
        if not token:
            authorization = self.headers.get("Authorization", "")
            if authorization.startswith("Bearer "):
                token = authorization.removeprefix("Bearer ").strip()
        if not token:
            token = parse_qs(query).get("token", [""])[0]
        return secrets.compare_digest(token, self.server_state.token)


def _subprocess_runner(argv: list[str], workspace: Path) -> tuple[int, str, str]:
    process = subprocess.run(
        [sys.executable, "-m", "oracle_sae", *argv],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    return process.returncode, process.stdout, process.stderr


def _default_command_specs() -> list[dict[str, Any]]:
    try:
        from oracle_sae.cli import build_parser
    except ImportError:  # pragma: no cover - import-cycle fallback.
        return list(COMMAND_SPECS)
    return command_specs_from_parser(build_parser())


def _normalize_job_record(item: dict[str, Any], workspace: Path) -> dict[str, Any]:
    argv = item.get("argv")
    if not isinstance(argv, list):
        argv = []
    argv = [str(value) for value in argv if value is not None]
    command = str(item.get("command") or (argv[0] if argv else "unknown"))
    status = str(item.get("status") or "unknown")
    stderr = str(item.get("stderr") or "")
    if status in {"queued", "running"}:
        status = "interrupted"
        note = "Studio server stopped before this job completed."
        stderr = f"{stderr}\n{note}".strip()
    return {
        "schema_version": STUDIO_HISTORY_SCHEMA,
        "id": str(item.get("id")),
        "argv": argv,
        "command": command,
        "source": str(item.get("source") or "history"),
        "status": status,
        "created_at": item.get("created_at") or _utc_timestamp(),
        "started_at": item.get("started_at"),
        "finished_at": item.get("finished_at") if status != "interrupted" else item.get("finished_at") or _utc_timestamp(),
        "exit_code": item.get("exit_code"),
        "stdout": _truncate_output(str(item.get("stdout") or "")),
        "stderr": _truncate_output(stderr),
        "captured_output_truncated": bool(item.get("captured_output_truncated", False)),
        "workspace": str(item.get("workspace") or workspace),
        "run_config_path": item.get("run_config_path"),
    }


def _truncate_output(text: str) -> str:
    if len(text) <= MAX_CAPTURED_OUTPUT_CHARS:
        return text
    return text[-MAX_CAPTURED_OUTPUT_CHARS:]


def _artifact_records(reports_dir: Path, *, limit: int = 300) -> list[dict[str, Any]]:
    if not reports_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    suffixes = {".html", ".json", ".jsonl", ".md"}
    for path in sorted(reports_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if _contains_private_path_part(path.relative_to(reports_dir)):
            continue
        stat = path.stat()
        records.append(
            {
                "path": _public_artifact_path(path, reports_dir),
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


def _safe_artifact_path(path_text: str, reports_dir: Path) -> Path:
    text = unquote(path_text)
    path = Path(text)
    candidate = path if path.is_absolute() else reports_dir / path
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(reports_dir)
    except ValueError as exc:
        raise ValueError("artifact path must stay inside the reports directory") from exc
    if _contains_private_path_part(relative):
        raise ValueError("artifact path cannot reference private Studio files")
    return resolved


def _public_artifact_path(path: Path, reports_dir: Path) -> str:
    return str(path.relative_to(reports_dir))


def _contains_private_path_part(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _resolve_under_workspace(path: str | Path, workspace: Path) -> Path:
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("reports directory must stay inside the workspace") from exc
    return resolved


def _allowed_hosts(configured_host: str, actual_host: str, port: int) -> set[str]:
    host_candidates = {configured_host, actual_host, "127.0.0.1", "localhost", "::1"}
    values = set()
    for host in host_candidates:
        if not host or host == "0.0.0.0":
            continue
        display = f"[{host}]" if ":" in host and not host.startswith("[") else host
        values.add(f"{display}:{port}".lower())
    return values


def _step_requests_server(args: Any) -> bool:
    if isinstance(args, list):
        return "--serve" in {str(item) for item in args}
    if isinstance(args, dict):
        value = args.get("serve")
        if isinstance(value, str):
            return value.lower() not in {"", "0", "false", "no"}
        return bool(value)
    return False


def _utc_timestamp(seconds: float | None = None) -> str:
    if seconds is None:
        seconds = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))
