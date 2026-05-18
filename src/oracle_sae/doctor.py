from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import sys
from typing import Any

from oracle_sae import __version__


def collect_diagnostics() -> dict[str, Any]:
    checks = [
        _check_python(),
        _check_package("interp-lab", required=True),
        _check_module("yaml", package_name="PyYAML", required=False, purpose="YAML run configs"),
        _check_module("torch", required=False, purpose="HF activation export and SAE training"),
        _check_module("transformers", required=False, purpose="Hugging Face model adapters"),
        _check_module("sae_lens", package_name="sae-lens", required=False, purpose="SAE Lens adapter"),
    ]
    return {
        "tool": "interp-lab",
        "version": __version__,
        "platform": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "checks": checks,
        "ok": all(check["ok"] for check in checks if check["required"]),
    }


def diagnostics_to_text(diagnostics: dict[str, Any]) -> str:
    lines = [
        f"interp-lab {diagnostics['version']}",
        (
            f"Python {diagnostics['platform']['python']} "
            f"on {diagnostics['platform']['system']} {diagnostics['platform']['machine']}"
        ),
        "",
        "Checks:",
    ]
    for check in diagnostics["checks"]:
        status = "ok" if check["ok"] else "missing"
        required = "required" if check["required"] else "optional"
        version = f" ({check['version']})" if check.get("version") else ""
        purpose = f" - {check['purpose']}" if check.get("purpose") else ""
        lines.append(f"- {check['name']}: {status}{version} [{required}]{purpose}")
    lines.append("")
    lines.append("Environment ready." if diagnostics["ok"] else "Required checks failed.")
    return "\n".join(lines)


def diagnostics_to_json(diagnostics: dict[str, Any]) -> str:
    return json.dumps(diagnostics, indent=2, sort_keys=True)


def _check_python() -> dict[str, Any]:
    ok = sys.version_info >= (3, 10)
    return {
        "name": "python>=3.10",
        "ok": ok,
        "required": True,
        "version": sys.version.split()[0],
        "purpose": "package runtime",
    }


def _check_package(name: str, *, required: bool) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = __version__ if name == "interp-lab" else None
    return {
        "name": name,
        "ok": version is not None,
        "required": required,
        "version": version,
        "purpose": "installed package",
    }


def _check_module(
    module_name: str,
    *,
    package_name: str | None = None,
    required: bool,
    purpose: str,
) -> dict[str, Any]:
    package_name = package_name or module_name
    found = importlib.util.find_spec(module_name) is not None
    version = None
    if found:
        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            version = "available"
    return {
        "name": package_name,
        "ok": found,
        "required": required,
        "version": version,
        "purpose": purpose,
    }
