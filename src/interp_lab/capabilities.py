"""Single machine-readable discovery endpoint for agents.

``build_capabilities()`` assembles everything an agent needs to drive interp-lab
in one payload: the full structured CLI surface, the Python API contract
(exports, schemas, signatures), optional-module availability, and the house
conventions (JSON-first outputs, error shape, next-action templates, MCP).
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

from interp_lab import __version__
from interp_lab.doctor import collect_diagnostics

CAPABILITIES_SCHEMA = "interp-lab.capabilities.v1"

_ENVIRONMENT_CHECK_KEYS = ("name", "ok", "version", "purpose")


def build_capabilities() -> dict[str, Any]:
    """Return the interp-lab capabilities payload for agents."""
    # Imported lazily: cli (and web_app behind it) import broadly across the
    # package, while contracts imports this module for the schema id.
    from interp_lab.cli import build_parser
    from interp_lab.contracts import public_api_contract
    from interp_lab.web_app import command_specs_from_parser

    diagnostics = collect_diagnostics()
    return {
        "schema_version": CAPABILITIES_SCHEMA,
        "tool": {
            "name": "interp-lab",
            "version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "commands": command_specs_from_parser(build_parser()),
        "python_api": public_api_contract(),
        "environment": {
            # The stable, fast subset of `doctor`: module availability probes
            # (importlib only, no model loads or hardware sampling) plus the
            # active text embedder. Run the `doctor` command/tool for the rest.
            "optional_modules": [
                {key: check.get(key) for key in _ENVIRONMENT_CHECK_KEYS}
                for check in diagnostics["checks"]
                if not check["required"]
            ],
            "text_embedder": diagnostics["text_embedder"],
        },
        "conventions": {
            "json_first": True,
            "errors": "interp-lab: error: ... on stderr, exit 2",
            "outputs": "JSON artifacts carry schema_version; --json keeps stdout machine-pure",
            "next_actions": {
                "shape": {
                    "id": "stable action identifier",
                    "title": "human-readable summary",
                    "command": "shell-quoted command string (runnable actions; absent on prose actions)",
                    "argv": "the same command as an argv list (always paired with command)",
                    "instruction": "prose guidance (non-runnable actions; mutually exclusive with command/argv)",
                    "requires": "artifacts the action needs first",
                },
                "placeholders": "<angle-bracket> tokens mark run-local artifacts the agent must substitute",
            },
            "mcp": {"command": "interp-lab mcp", "transport": "stdio"},
        },
    }


def write_capabilities(out: str | Path) -> Path:
    """Write the capabilities payload as JSON and return the path."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_capabilities(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
