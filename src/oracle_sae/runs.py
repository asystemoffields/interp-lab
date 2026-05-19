from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - exercised on Python 3.10 in CI.
    import tomli as tomllib

from oracle_sae import __version__

CommandRunner = Callable[[list[str]], int]

DEFAULT_HASH_LIMIT_BYTES = 64 * 1024 * 1024
INPUT_PATH_KEYS = {
    "dataset",
    "features",
    "graph",
    "records",
    "interventions",
    "neuronpedia_features",
    "saelens_feature_metadata",
    "left",
    "right",
    "report",
    "path",
    "path_records",
    "source_sae",
    "target_sae",
    "source_report",
    "target_report",
}
OUTPUT_PATH_KEYS = {
    "out",
    "records_out",
    "causal_out",
    "path_records_out",
    "graph_out",
    "graph_html_out",
    "graph_markdown_out",
    "html_out",
    "markdown_out",
}


@dataclass(frozen=True)
class RunOptions:
    config_path: Path
    dry_run: bool = False
    variables: dict[str, str] | None = None


def run_config_file(options: RunOptions, *, command_runner: CommandRunner) -> int:
    config_path = options.config_path
    config = load_run_config(config_path)
    config_dir = config_path.resolve().parent
    run_dir = Path(str(config.get("out", config.get("run_dir", "reports/run"))))
    variables = {
        "config_dir": str(config_dir),
        "run_dir": str(run_dir),
        **{key: str(value) for key, value in (options.variables or {}).items()},
    }
    rendered_config = _render_value(config, variables)
    steps = _steps_from_config(rendered_config)
    if options.dry_run:
        for step in steps:
            argv = _argv_from_step(step)
            print(_format_command(["interp-lab", *argv]))
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest = _new_manifest(config_path, run_dir, rendered_config)
    manifest["inputs"] = _input_file_records(rendered_config, config_dir, manifest)
    _write_manifest(manifest_path, manifest)
    try:
        for step in steps:
            step_record = _run_step(step, command_runner)
            step_record["outputs"] = _output_file_records(step, config_dir, manifest)
            manifest["steps"].append(step_record)
            manifest["outputs"] = _merge_output_records(manifest["steps"])
            _write_manifest(manifest_path, manifest)
            if step_record["status"] != "succeeded":
                manifest["status"] = "failed"
                manifest["finished_at"] = _utc_now()
                _write_manifest(manifest_path, manifest)
                raise RuntimeError(
                    f"run step {step_record['name']} failed with exit code {step_record['exit_code']}"
                )
    except Exception:
        if manifest.get("status") != "failed":
            manifest["status"] = "failed"
            manifest["finished_at"] = _utc_now()
            _write_manifest(manifest_path, manifest)
        raise
    manifest["status"] = "succeeded"
    manifest["finished_at"] = _utc_now()
    _write_manifest(manifest_path, manifest)
    print(f"Wrote {manifest_path}")
    return 0


def load_run_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix == ".toml":
        data = tomllib.loads(text)
    elif suffix in {".yaml", ".yml"}:
        data = _load_yaml(text)
    else:
        raise ValueError("run config must be JSON, TOML, YAML, or YML")
    if not isinstance(data, dict):
        raise ValueError("run config must contain an object at the top level")
    return data


def _load_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML configs require PyYAML. Install `interp-lab` with production dependencies.") from exc
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML run config must contain an object at the top level")
    return data


def _steps_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    if "steps" in config:
        steps = config["steps"]
        if not isinstance(steps, list) or not steps:
            raise ValueError("run config `steps` must be a non-empty list")
        return [_normalize_step(step, index) for index, step in enumerate(steps, start=1)]
    return [
        {
            "name": "inspect",
            "command": "inspect",
            "args": _top_level_inspect_args(config),
        }
    ]


def _normalize_step(step: Any, index: int) -> dict[str, Any]:
    if not isinstance(step, dict):
        raise ValueError(f"run step {index} must be an object")
    if "command" not in step:
        raise ValueError(f"run step {index} is missing `command`")
    command = str(step["command"])
    if command == "run":
        raise ValueError("run configs cannot recursively call `interp-lab run`")
    args = step.get("args", {})
    if not isinstance(args, (dict, list)):
        raise ValueError(f"run step {index} `args` must be an object or list")
    return {
        "name": str(step.get("name", f"{index}-{command}")),
        "command": command,
        "args": args,
    }


def _top_level_inspect_args(config: dict[str, Any]) -> dict[str, Any]:
    required = ["model", "criterion"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("run config needs `steps` or top-level `model` and `criterion`")
    allowed = {
        "model",
        "criterion",
        "backend",
        "features",
        "records",
        "interventions",
        "allow_intervention_criterion_mismatch",
        "require_interventions",
        "neuronpedia_feature",
        "neuronpedia_features",
        "neuronpedia_base_url",
        "saelens_release",
        "saelens_sae_id",
        "saelens_feature_indexes",
        "saelens_max_features",
        "saelens_device",
        "saelens_force_download",
        "saelens_feature_metadata",
        "goodfire_top_k",
        "goodfire_api_key_env",
        "scope_source",
        "scope_release",
        "scope_sae_id",
        "scope_feature_indexes",
        "scope_max_features",
        "scope_device",
        "scope_force_download",
        "scope_feature_metadata",
        "top_k",
    }
    args = {key: value for key, value in config.items() if key in allowed}
    args.setdefault("backend", "toy")
    args.setdefault("out", str(Path(str(config.get("out", "reports/run"))) / "inspect"))
    return args


def _run_step(step: dict[str, Any], command_runner: CommandRunner) -> dict[str, Any]:
    argv = _argv_from_step(step)
    record = {
        "name": step["name"],
        "command": step["command"],
        "argv": argv,
        "started_at": _utc_now(),
        "finished_at": None,
        "exit_code": None,
        "status": "running",
    }
    try:
        exit_code = int(command_runner(argv))
    except SystemExit as exc:
        exit_code = _system_exit_code(exc)
    record["exit_code"] = exit_code
    record["finished_at"] = _utc_now()
    record["status"] = "succeeded" if exit_code == 0 else "failed"
    return record


def _argv_from_step(step: dict[str, Any]) -> list[str]:
    args = step["args"]
    if isinstance(args, list):
        return [step["command"], *[str(item) for item in args]]
    return [step["command"], *_dict_to_argv(args)]


def _dict_to_argv(args: Mapping[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in args.items():
        if value is None or value is False:
            continue
        flag = f"--{str(key).replace('_', '-')}"
        if value is True:
            argv.append(flag)
        elif isinstance(value, list):
            for item in value:
                argv.extend([flag, str(item)])
        else:
            argv.extend([flag, str(value)])
    return argv


def _format_command(argv: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _new_manifest(config_path: Path, run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "interp-lab.run.v1",
        "tool": "interp-lab",
        "version": __version__,
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "config": config,
        "inputs": [],
        "outputs": [],
        "steps": [],
    }


def _input_file_records(config: Any, config_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    hash_limit = int(
        manifest.get("config", {})
        .get("manifest", {})
        .get("max_hash_bytes", DEFAULT_HASH_LIMIT_BYTES)
    )
    records = []
    seen: set[Path] = set()
    for key, value in _walk_items(config):
        if key not in INPUT_PATH_KEYS or not isinstance(value, str):
            continue
        path = _resolve_existing_path(value, config_dir)
        if path is None or path in seen:
            continue
        seen.add(path)
        records.append(_file_record(path, hash_limit))
    return records


def _output_file_records(step: dict[str, Any], config_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    hash_limit = int(
        manifest.get("config", {})
        .get("manifest", {})
        .get("max_hash_bytes", DEFAULT_HASH_LIMIT_BYTES)
    )
    records = []
    seen: set[Path] = set()
    for _key, value in _output_path_items(step):
        path = _resolve_output_path(value, config_dir)
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        records.append(_artifact_record(resolved, hash_limit))
    return records


def _merge_output_records(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for step in steps:
        for record in step.get("outputs", []):
            path = record.get("path")
            if not path or path in seen:
                continue
            seen.add(path)
            merged.append(record)
    return merged


def _output_path_items(step: dict[str, Any]):
    args = step.get("args", {})
    if isinstance(args, dict):
        for key, value in _walk_items(args):
            if key in OUTPUT_PATH_KEYS and isinstance(value, str):
                yield key, value
        return
    if not isinstance(args, list):
        return
    items = [str(item) for item in args]
    index = 0
    while index < len(items):
        item = items[index]
        if not item.startswith("--"):
            index += 1
            continue
        key_value = item[2:]
        if "=" in key_value:
            key, value = key_value.split("=", 1)
            if key.replace("-", "_") in OUTPUT_PATH_KEYS:
                yield key, value
            index += 1
            continue
        key = key_value.replace("-", "_")
        if key in OUTPUT_PATH_KEYS and index + 1 < len(items):
            value = items[index + 1]
            if not value.startswith("--"):
                yield key, value
                index += 2
                continue
        index += 1


def _walk_items(value: Any, parent_key: str | None = None):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_items(child, str(key))
    elif isinstance(value, list):
        for child in value:
            if parent_key is not None:
                yield parent_key, child
            yield from _walk_items(child, parent_key)


def _resolve_existing_path(value: str, config_dir: Path) -> Path | None:
    if "://" in value:
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, config_dir / path]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _file_record(path: Path, hash_limit: int) -> dict[str, Any]:
    size = path.stat().st_size
    record = {
        "path": str(path),
        "size_bytes": size,
        "sha256": None,
        "hash_skipped": size > hash_limit,
    }
    if size <= hash_limit:
        record["sha256"] = _sha256(path)
    return record


def _resolve_output_path(value: str, config_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    config_path = config_dir / path
    if config_path.exists():
        return config_path
    return cwd_path


def _artifact_record(path: Path, hash_limit: int) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "kind": "missing",
            "size_bytes": 0,
            "sha256": None,
            "hash_skipped": True,
        }
    if path.is_file():
        record = _file_record(path, hash_limit)
        record["exists"] = True
        record["kind"] = "file"
        return record
    if path.is_dir():
        files = [item for item in path.rglob("*") if item.is_file()]
        return {
            "path": str(path),
            "exists": True,
            "kind": "directory",
            "file_count": len(files),
            "size_bytes": sum(item.stat().st_size for item in files),
            "sha256": None,
            "hash_skipped": True,
        }
    return {
        "path": str(path),
        "exists": True,
        "kind": "other",
        "size_bytes": 0,
        "sha256": None,
        "hash_skipped": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_value(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        rendered = value
        for key, replacement in variables.items():
            rendered = rendered.replace("${" + key + "}", replacement)
        return rendered.format_map(_SafeFormatMap(variables))
    if isinstance(value, list):
        return [_render_value(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _render_value(item, variables) for key, item in value.items()}
    return value


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _system_exit_code(exc: SystemExit) -> int:
    if exc.code is None:
        return 0
    if isinstance(exc.code, int):
        return exc.code
    return 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class _SafeFormatMap(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
