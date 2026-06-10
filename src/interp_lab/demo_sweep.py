from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from interp_lab.agent_actions import next_action
from interp_lab.release_check import REAL_MODEL_DEMO_SCHEMA, _validate_real_model_demo_manifest

REAL_MODEL_DEMO_SWEEP_SCHEMA = "interp-lab.real_model_demo_sweep.v1"

DEFAULT_SWEEP_OUT = "reports/real-model-demo-sweep.json"

_INTERNAL_COMMANDS_CACHE: frozenset[str] | None = None


def _internal_commands() -> frozenset[str]:
    """Internal subcommand names derived from the live CLI parser.

    Deriving the set from build_parser() (instead of a hand-maintained allowlist)
    means a new CLI command can never be misclassified as an external binary.
    """
    global _INTERNAL_COMMANDS_CACHE
    if _INTERNAL_COMMANDS_CACHE is None:
        from interp_lab.cli import build_parser  # local import: cli imports this module

        names: set[str] = set()
        for action in build_parser()._actions:
            if isinstance(action, argparse._SubParsersAction):
                names.update(action.choices)
        _INTERNAL_COMMANDS_CACHE = frozenset(names)
    return _INTERNAL_COMMANDS_CACHE


def _command_kind(raw_argv: list[str], normalized: list[str]) -> str:
    # Anything invoked through the interp-lab entry point is internal by definition;
    # the prefix is stripped before exec, so it must never reach subprocess.run.
    if raw_argv and raw_argv[0] == "interp-lab":
        return "internal"
    if normalized and normalized[0] in _internal_commands():
        return "internal"
    return "external"

CommandRunner = Callable[[list[str]], int]


@dataclass(frozen=True)
class DemoSweepResult:
    report: dict[str, Any]
    path: Path | None = None


def build_demo_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify or execute the real-model demo suite and write an archival sweep report."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root for docs, commands, and artifacts.")
    parser.add_argument(
        "--manifest-dir",
        default="examples/real_model_demos",
        help="Directory containing real-model demo manifest JSON files. Relative paths are resolved from --repo-root.",
    )
    parser.add_argument(
        "--demo",
        action="append",
        default=[],
        help="Demo id, manifest stem, or manifest filename to include. Repeat to select multiple demos.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run manifest commands before verifying expected artifacts.",
    )
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="Allow non-interp-lab commands such as modal to execute during --run.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop executing additional commands after the first command failure.",
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=float,
        default=0.0,
        help="Timeout per external command. Use 0 for no timeout.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            f"Output JSON sweep report path. Defaults to {DEFAULT_SWEEP_OUT} with --run; "
            "verify-only sweeps print results but only write when --out is set, so they "
            "never clobber archived release evidence."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print the machine-readable report.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero unless every selected demo passes artifact verification.",
    )
    return parser


def run_demo_sweep_from_args(
    args: argparse.Namespace,
    *,
    command_runner: CommandRunner | None = None,
) -> DemoSweepResult:
    report = build_demo_sweep_report(
        repo_root=args.repo_root,
        manifest_dir=args.manifest_dir,
        demos=args.demo,
        run=args.run,
        allow_external=args.allow_external,
        stop_on_failure=args.stop_on_failure,
        command_timeout_seconds=args.command_timeout_seconds,
        command_runner=command_runner,
    )
    out = args.out
    if out is None and args.run:
        # Only --run sweeps fall back to the archival default path: a verify-only
        # sweep with run_commands=false would overwrite release evidence that
        # release-check then flags as a blocker.
        out = DEFAULT_SWEEP_OUT
    path = Path(out) if out else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return DemoSweepResult(report=report, path=path)


def build_demo_sweep_report(
    *,
    repo_root: str | Path = ".",
    manifest_dir: str | Path = "examples/real_model_demos",
    demos: Sequence[str] | None = None,
    run: bool = False,
    allow_external: bool = False,
    stop_on_failure: bool = False,
    command_timeout_seconds: float = 0.0,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    demo_dir = _resolve_under_root(root, manifest_dir)
    selected = set(demos or [])
    manifest_paths = _selected_manifest_paths(demo_dir, selected)
    demo_reports: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        demo_report = _build_single_demo_report(
            manifest_path=manifest_path,
            repo_root=root,
            run=run,
            allow_external=allow_external,
            stop_on_failure=stop_on_failure,
            command_timeout_seconds=command_timeout_seconds,
            command_runner=command_runner,
        )
        demo_reports.append(demo_report)
    if selected and not manifest_paths:
        demo_reports.append(
            {
                "status": "invalid_selection",
                "manifest_path": None,
                "detail": f"No demo manifests matched: {', '.join(sorted(selected))}",
                "commands": [],
                "artifacts": [],
                "agent_next_actions": [
                    next_action(
                        action_id="fix_demo_selection",
                        title="Fix the --demo selection",
                        instruction="Check --demo values against manifest id, stem, or filename.",
                    )
                ],
            }
        )
    counts = _status_counts(demo_reports)
    overall_status = _overall_status(demo_reports)
    return {
        "schema_version": REAL_MODEL_DEMO_SWEEP_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": _public_path(root, root),
        "manifest_dir": _public_path(demo_dir, root),
        "selected_demo_count": len(demo_reports),
        "run_commands": bool(run),
        "allow_external": bool(allow_external),
        "status": overall_status,
        "summary": counts,
        "demos": demo_reports,
        "agent_next_actions": _sweep_next_actions(overall_status, demo_reports, run),
    }


def render_demo_sweep_text(report: dict[str, Any]) -> str:
    status = str(report.get("status", "unknown")).upper()
    summary = report.get("summary", {})
    lines = [
        f"interp-lab real-model demo sweep: {status}",
        (
            f"demos={report.get('selected_demo_count', 0)} "
            f"passed={summary.get('passed', 0)} "
            f"incomplete={summary.get('incomplete', 0)} "
            f"failed={summary.get('failed', 0)}"
        ),
        "",
    ]
    for demo in report.get("demos", []):
        demo_status = str(demo.get("status", "unknown")).upper()
        title = demo.get("title") or demo.get("id") or demo.get("manifest_path") or "demo"
        lines.append(f"[{demo_status}] {title}")
        if demo.get("model"):
            lines.append(f"  model: {demo['model']}")
        if demo.get("workflow"):
            lines.append(f"  workflow: {demo['workflow']}")
        input_summary = demo.get("input_summary", {})
        if input_summary:
            lines.append(
                "  inputs: "
                f"{input_summary.get('present', 0)}/{input_summary.get('total', 0)} present"
            )
        artifact_summary = demo.get("artifact_summary", {})
        if artifact_summary:
            lines.append(
                "  artifacts: "
                f"{artifact_summary.get('present', 0)}/{artifact_summary.get('total', 0)} present"
            )
        command_summary = demo.get("command_summary", {})
        if command_summary:
            lines.append(
                "  commands: "
                f"passed={command_summary.get('passed', 0)} "
                f"planned={command_summary.get('planned', 0)} "
                f"skipped={command_summary.get('skipped', 0)} "
                f"blocked={command_summary.get('blocked', 0)} "
                f"failed={command_summary.get('failed', 0)}"
            )
        detail = demo.get("detail")
        if detail:
            lines.append(f"  {detail}")
        next_actions = demo.get("agent_next_actions") or []
        if next_actions:
            lines.append(f"  Next: {_action_line(next_actions[0])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _action_line(action: Any) -> str:
    """Render one agent next action for the text view.

    Pre-2.3 sweep reports stored plain strings; current reports store canonical
    {id, title, command|instruction} objects. Both render cleanly.
    """
    if isinstance(action, dict):
        label = str(action.get("title") or action.get("id") or "")
        command = action.get("command")
        instruction = action.get("instruction")
        if command:
            return f"{label} ({command})"
        if instruction and instruction != label:
            return f"{label}. {instruction}"
        return label
    return str(action)


def _build_single_demo_report(
    *,
    manifest_path: Path,
    repo_root: Path,
    run: bool,
    allow_external: bool,
    stop_on_failure: bool,
    command_timeout_seconds: float,
    command_runner: CommandRunner | None,
) -> dict[str, Any]:
    ok, detail = _validate_real_model_demo_manifest(manifest_path, repo_root)
    if not ok:
        return {
            "status": "failed",
            "manifest_path": _public_path(manifest_path, repo_root),
            "detail": detail,
            "commands": [],
            "artifacts": [],
            "agent_next_actions": [
                next_action(
                    action_id="fix_manifest_schema",
                    title="Fix the manifest schema before running the demo",
                    instruction=f"Fix the manifest schema before running the demo: {detail}",
                )
            ],
        }
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    commands = []
    command_failed = False
    command_blocked = False
    inputs = [_input_record(item, repo_root=repo_root) for item in payload.get("required_inputs", [])]
    input_summary = {
        "total": len(inputs),
        "present": sum(1 for item in inputs if item["exists"]),
        "missing": sum(1 for item in inputs if not item["exists"]),
    }
    if run:
        if input_summary["missing"]:
            commands = [_blocked_command(command, detail="Command was not run because required inputs are missing.") for command in payload.get("commands", [])]
            command_blocked = True
        runnable_commands = [] if input_summary["missing"] else payload.get("commands", [])
        for command in runnable_commands:
            if command_blocked:
                commands.append(_blocked_command(command))
                continue
            command_report = _run_or_skip_command(
                command,
                repo_root=repo_root,
                allow_external=allow_external,
                command_timeout_seconds=command_timeout_seconds,
                command_runner=command_runner,
            )
            commands.append(command_report)
            if command_report["status"] == "failed":
                command_failed = True
                if stop_on_failure:
                    break
            if command_report["status"] == "skipped":
                command_blocked = True
    else:
        commands = [_planned_command(command) for command in payload.get("commands", [])]

    artifacts = [_artifact_record(artifact, repo_root=repo_root) for artifact in payload.get("expected_artifacts", [])]
    artifact_summary = {
        "total": len(artifacts),
        "present": sum(1 for artifact in artifacts if artifact["exists"]),
        "missing": sum(1 for artifact in artifacts if not artifact["exists"]),
    }
    command_summary = _command_summary(commands)
    incomplete = (
        input_summary["missing"] > 0
        or artifact_summary["missing"] > 0
        or command_summary["skipped"] > 0
        or command_summary["blocked"] > 0
    )
    status = "passed"
    if command_failed:
        status = "failed"
    elif incomplete:
        status = "incomplete"
    return {
        "schema_version": REAL_MODEL_DEMO_SCHEMA,
        "id": payload["id"],
        "title": payload["title"],
        "model": payload["model"],
        "criterion": payload["criterion"],
        "workflow": payload["workflow"],
        "doc": payload["doc"],
        "manifest_path": _public_path(manifest_path, repo_root),
        "status": status,
        "detail": _demo_detail(status, input_summary, artifact_summary, command_summary, run),
        "inputs": inputs,
        "input_summary": input_summary,
        "commands": commands,
        "command_summary": command_summary,
        "artifacts": artifacts,
        "artifact_summary": artifact_summary,
        "evidence_checks": payload.get("evidence_checks", []),
        "agent_next_actions": _demo_next_actions(payload, status, input_summary, artifact_summary, command_summary, run),
    }


def _run_or_skip_command(
    command: dict[str, Any],
    *,
    repo_root: Path,
    allow_external: bool,
    command_timeout_seconds: float,
    command_runner: CommandRunner | None,
) -> dict[str, Any]:
    raw_argv = list(command["argv"])
    normalized = _normalize_argv(raw_argv)
    kind = _command_kind(raw_argv, normalized)
    base = {
        "name": command["name"],
        "argv": raw_argv,
        "normalized_argv": normalized,
        "kind": kind,
    }
    if kind == "external" and not allow_external:
        return {
            **base,
            "status": "skipped",
            "detail": "External command skipped. Re-run with --allow-external to execute it.",
        }
    if kind == "internal":
        if command_runner is None:
            return {
                **base,
                "status": "skipped",
                "detail": "No internal command runner was provided.",
            }
        return {**base, **_run_internal_command(normalized, repo_root=repo_root, command_runner=command_runner)}
    return {
        **base,
        **_run_external_command(
            normalized,
            repo_root=repo_root,
            command_timeout_seconds=command_timeout_seconds,
        ),
    }


def _run_internal_command(
    argv: list[str],
    *,
    repo_root: Path,
    command_runner: CommandRunner,
) -> dict[str, Any]:
    started = time.monotonic()
    stdout = io.StringIO()
    stderr = io.StringIO()
    previous_cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = int(command_runner(argv))
    except SystemExit as exc:
        exit_code = int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:  # pragma: no cover - surfaced in report detail.
        exit_code = 1
        stderr.write(str(exc))
    finally:
        os.chdir(previous_cwd)
    duration = time.monotonic() - started
    return {
        "status": "passed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "duration_seconds": round(duration, 3),
        "stdout_tail": _tail(stdout.getvalue()),
        "stderr_tail": _tail(stderr.getvalue()),
    }


def _run_external_command(
    argv: list[str],
    *,
    repo_root: Path,
    command_timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    timeout = command_timeout_seconds if command_timeout_seconds > 0 else None
    try:
        completed = subprocess.run(
            argv,
            cwd=repo_root,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "exit_code": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "stdout_tail": _tail(exc.stdout or ""),
            "stderr_tail": _tail(exc.stderr or ""),
            "detail": f"Command timed out after {command_timeout_seconds:g} seconds.",
        }
    except OSError as exc:
        return {
            "status": "failed",
            "exit_code": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "detail": str(exc),
        }
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def _planned_command(command: dict[str, Any]) -> dict[str, Any]:
    raw_argv = list(command["argv"])
    normalized = _normalize_argv(raw_argv)
    return {
        "name": command["name"],
        "argv": raw_argv,
        "normalized_argv": normalized,
        "kind": _command_kind(raw_argv, normalized),
        "status": "planned",
        "detail": "Command execution was not requested.",
    }


def _blocked_command(
    command: dict[str, Any],
    *,
    detail: str = "Command was not run because an earlier command was skipped.",
) -> dict[str, Any]:
    raw_argv = list(command["argv"])
    normalized = _normalize_argv(raw_argv)
    return {
        "name": command["name"],
        "argv": raw_argv,
        "normalized_argv": normalized,
        "kind": _command_kind(raw_argv, normalized),
        "status": "blocked",
        "detail": detail,
    }


def _input_record(item: str | dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    if isinstance(item, str):
        relative = item
        kind = "file"
        description = ""
    else:
        relative = str(item.get("path", ""))
        kind = str(item.get("kind", "file"))
        description = str(item.get("description", ""))
    path = _resolve_under_root(repo_root, relative)
    return {
        "path": relative,
        "kind": kind,
        "description": description,
        "exists": path.exists(),
    }


def _artifact_record(artifact: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    path = _resolve_under_root(repo_root, artifact["path"])
    exists = path.exists()
    record: dict[str, Any] = {
        "path": artifact["path"],
        "kind": artifact["kind"],
        "exists": exists,
        "why_it_matters": artifact["why_it_matters"],
        "interpretation_notes": artifact["interpretation_notes"],
    }
    if not exists:
        return record
    record["is_dir"] = path.is_dir()
    if path.is_file():
        stat = path.stat()
        record["size_bytes"] = stat.st_size
        if stat.st_size <= 50 * 1024 * 1024:
            record["sha256"] = _sha256(path)
    return record


def _selected_manifest_paths(demo_dir: Path, selected: set[str]) -> list[Path]:
    if not demo_dir.exists():
        return []
    paths = sorted(demo_dir.glob("*.json"))
    if not selected:
        return paths
    chosen = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        keys = {path.name, path.stem, str(payload.get("id", ""))}
        if keys & selected:
            chosen.append(path)
    return chosen


def _normalize_argv(argv: list[str]) -> list[str]:
    if argv and argv[0] == "interp-lab":
        return argv[1:]
    return argv


def _resolve_under_root(root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    return resolved.resolve()


def _public_path(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return path.name
    if str(relative) == ".":
        return "."
    return relative.as_posix()


def _status_counts(demos: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "passed": sum(1 for demo in demos if demo.get("status") == "passed"),
        "incomplete": sum(1 for demo in demos if demo.get("status") == "incomplete"),
        "failed": sum(1 for demo in demos if demo.get("status") in {"failed", "invalid_selection"}),
    }


def _overall_status(demos: list[dict[str, Any]]) -> str:
    if not demos:
        return "failed"
    if any(demo.get("status") in {"failed", "invalid_selection"} for demo in demos):
        return "failed"
    if all(demo.get("status") == "passed" for demo in demos):
        return "passed"
    return "incomplete"


def _command_summary(commands: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(commands),
        "passed": sum(1 for command in commands if command.get("status") == "passed"),
        "planned": sum(1 for command in commands if command.get("status") == "planned"),
        "skipped": sum(1 for command in commands if command.get("status") == "skipped"),
        "blocked": sum(1 for command in commands if command.get("status") == "blocked"),
        "failed": sum(1 for command in commands if command.get("status") == "failed"),
    }


def _demo_detail(
    status: str,
    input_summary: dict[str, int],
    artifact_summary: dict[str, int],
    command_summary: dict[str, int],
    run: bool,
) -> str:
    if status == "passed":
        return "All expected artifacts are present."
    if status == "failed":
        return "One or more commands failed during the sweep."
    if input_summary["missing"]:
        return f"{input_summary['missing']} required input(s) are missing."
    if artifact_summary["missing"]:
        return f"{artifact_summary['missing']} expected artifact(s) are missing."
    if run and command_summary["skipped"]:
        return f"{command_summary['skipped']} command(s) were skipped."
    if run and command_summary["blocked"]:
        return f"{command_summary['blocked']} command(s) were blocked by an earlier skipped command."
    return "Demo verification is incomplete."


def _demo_next_actions(
    payload: dict[str, Any],
    status: str,
    input_summary: dict[str, int],
    artifact_summary: dict[str, int],
    command_summary: dict[str, int],
    run: bool,
) -> list[dict[str, Any]]:
    if status == "passed":
        return [
            next_action(
                action_id="archive_demo_evidence",
                title="Archive this demo's evidence",
                instruction="Archive this manifest, the sweep report, and the listed artifacts together.",
            ),
            *_manifest_actions(payload),
        ]
    actions = []
    if input_summary["missing"]:
        actions.append(
            next_action(
                action_id="add_missing_inputs",
                title="Add or correct the missing required inputs",
                instruction="Add or correct the missing required inputs before running this demo.",
            )
        )
    if not run:
        actions.append(
            next_action(
                action_id="run_demo_sweep",
                title="Re-run this sweep with --run when dependencies and credentials are ready",
                argv=["interp-lab", "demo-sweep", "--run"],
            )
        )
    if command_summary["skipped"]:
        actions.append(
            next_action(
                action_id="allow_external_commands",
                title="Use --allow-external for trusted external launchers such as Modal",
                argv=["interp-lab", "demo-sweep", "--run", "--allow-external"],
            )
        )
    if command_summary["blocked"]:
        actions.append(
            next_action(
                action_id="run_blocked_steps",
                title="Run the blocked steps after the skipped launcher finishes",
                instruction="Run blocked steps after the skipped launcher has produced its expected artifacts.",
            )
        )
    if artifact_summary["missing"]:
        actions.append(
            next_action(
                action_id="generate_missing_artifacts",
                title="Generate the missing expected artifacts",
                instruction="Generate the missing expected artifacts, then repeat the sweep.",
            )
        )
    actions.extend(_manifest_actions(payload))
    return actions


def _manifest_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Coerce manifest-authored next actions (historically plain strings) to the
    canonical {id, title, instruction} shape."""
    actions: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("agent_next_actions", [])[:2], start=1):
        if isinstance(item, dict) and item.get("id") and item.get("title"):
            actions.append(item)
            continue
        text = str(item)
        actions.append(
            next_action(action_id=f"manifest_note_{index}", title=text, instruction=text)
        )
    return actions


def _sweep_next_actions(overall_status: str, demos: list[dict[str, Any]], run: bool) -> list[dict[str, Any]]:
    if overall_status == "passed":
        return [
            next_action(
                action_id="archive_sweep_report",
                title="Archive the sweep report with its evidence",
                instruction=(
                    "Archive reports/real-model-demo-sweep.json with the produced manifests, "
                    "reports, graphs, and notes."
                ),
            ),
            next_action(
                action_id="run_release_check",
                title="Run the strict release check after the stable classifier change is prepared",
                argv=["interp-lab", "release-check", "--strict"],
            ),
        ]
    actions = []
    if not run:
        actions.append(
            next_action(
                action_id="run_demo_sweep",
                title="Run the sweep with command execution after installing the optional dependencies needed by selected demos",
                argv=["interp-lab", "demo-sweep", "--run"],
            )
        )
    missing = [demo.get("id") for demo in demos if demo.get("artifact_summary", {}).get("missing", 0)]
    if missing:
        actions.append(
            next_action(
                action_id="produce_missing_artifacts",
                title="Produce the missing expected artifacts",
                instruction=f"Produce missing artifacts for: {', '.join(str(item) for item in missing if item)}.",
            )
        )
    failed = [demo.get("id") or demo.get("manifest_path") for demo in demos if demo.get("status") == "failed"]
    if failed:
        actions.append(
            next_action(
                action_id="fix_failed_demos",
                title="Fix the failed demos",
                instruction=f"Fix failed demos: {', '.join(str(item) for item in failed if item)}.",
            )
        )
    return actions


def _tail(text: str, *, lines: int = 40) -> str:
    split = text.splitlines()
    return "\n".join(split[-lines:])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
