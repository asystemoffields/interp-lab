"""Model Context Protocol server over stdio, pure stdlib.

Transport: newline-delimited JSON-RPC 2.0 -- one UTF-8 JSON object per line on
stdin/stdout. stdout carries nothing but protocol messages; all logging goes to
stderr. Start it with ``interp-lab mcp``.

Design notes for agents:
- Tools wrap the public Python API (``interp_lab.api``). Tools that produce
  large artifacts require an ``out`` path, write the artifact to disk, and
  return a compact summary plus the written paths -- read the artifact for the
  full payload. Small results (capabilities, doctor, searches without ``out``)
  are returned whole.
- Tool-level failures (bad paths, invalid params at execution time, missing
  optional extras) come back as tool results with ``isError: true``, mirroring
  the CLI error boundary. JSON-RPC errors are reserved for protocol problems:
  -32700 parse, -32600 invalid request, -32601 unknown method, -32602 invalid
  params (unknown tool name or missing required arguments).
- Resources use the ``interp-lab://docs/<name>`` URI scheme and expose the
  project docs (README.md, COMMANDS.md, AGENTS.md) when they exist on disk,
  looked up under the current working directory first, then the package
  checkout root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, TextIO

from interp_lab import __version__

KNOWN_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
DEFAULT_PROTOCOL_VERSION = "2025-06-18"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Mirror cli.main's error boundary; json.JSONDecodeError is a ValueError.
_TOOL_ERRORS = (ValueError, OSError, ImportError, RuntimeError)

_SERVER_INSTRUCTIONS = (
    "interp-lab finds, causally tests, and cross-model matches model features for "
    "a plain-language criterion. Call the 'capabilities' tool first: it returns the "
    "full CLI surface, the Python API contract, artifact schemas, and environment. "
    "Conventions are JSON-first: every artifact carries schema_version, and tools "
    "that write artifacts return compact summaries plus the written paths."
)

_BACKEND_CHOICES = ["toy", "jsonl", "records", "neuronpedia", "saelens", "goodfire", "scope"]

_DOC_RESOURCES = (
    ("README.md", "README.md", "Project overview and quick start."),
    ("COMMANDS.md", "docs/COMMANDS.md", "Complete CLI and data-format reference."),
    ("AGENTS.md", "AGENTS.md", "Agent-facing usage notes."),
)


def _doc_roots() -> list[Path]:
    roots = [Path.cwd()]
    package_root = Path(__file__).resolve().parents[2]
    if package_root not in roots:
        roots.append(package_root)
    return roots


def _resolve_doc(relative: str) -> Path | None:
    for root in _doc_roots():
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def _paths(**named: Any) -> dict[str, str]:
    """Stringify written artifact paths, dropping the ones not produced."""
    return {key: str(value) for key, value in named.items() if value is not None}


def _schema(
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "description": description,
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _path_property(description: str) -> dict[str, Any]:
    return {"type": "string", "description": f"{description} (filesystem path, relative to the server's cwd)"}


def _path_array_property(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string"},
        "description": f"{description} (filesystem paths, relative to the server's cwd)",
    }


# --------------------------------------------------------------------------- tools


def _tool_capabilities(_params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab.capabilities import build_capabilities

    return build_capabilities()


def _tool_doctor(_params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    return api.doctor()


def _tool_inspect(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    written = api.inspect(
        params["model"],
        params["criterion"],
        backend=params.get("backend", "toy"),
        features=params.get("features"),
        records=params.get("records"),
        interventions=params.get("interventions"),
        out=params["out"],
        html_out=params.get("html_out"),
        csv_out=params.get("csv_out"),
        top_k=params.get("top_k", 8),
        require_interventions=params.get("require_interventions", False),
    )
    report = written.report
    return {
        "model": report.model,
        "criterion": report.criterion.text,
        "card_count": len(report.cards),
        "top_features": [
            {
                "feature_id": card.feature_id,
                "label": card.label,
                "importance": card.importance,
                "causal_effect": card.causal_effect,
            }
            for card in report.cards[:5]
        ],
        "paths": _paths(
            report_json=written.json_path,
            report_markdown=written.markdown_path,
            report_html=written.html_path,
            report_csv=written.csv_path,
        ),
    }


def _tool_compare(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    written = api.compare(
        params["left_report"],
        params["right_report"],
        top_k=params.get("top_k", 10),
        min_score=params.get("min_score", 0.0),
        out=params["out"],
    )
    matches = written.report.matches
    return {
        "match_count": len(matches),
        "top_matches": [
            {
                "left_feature_id": match.left_feature_id,
                "right_feature_id": match.right_feature_id,
                "score": match.score,
            }
            for match in matches[:5]
        ],
        "paths": _paths(matches_json=written.json_path, matches_markdown=written.markdown_path),
    }


def _tool_validate_matches(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    written = api.validate_matches(params["match_report"], out=params["out"])
    return {
        "summary": written.report.get("summary", {}),
        "paths": _paths(validation_json=written.json_path, validation_markdown=written.markdown_path),
    }


def _tool_search_features(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    result = api.search_features(
        params["query"],
        list(params["reports"]),
        out=params.get("out"),
        top_k=params.get("top_k", 10),
    )
    if isinstance(result, dict):
        return result
    return {
        "summary": result.report.get("summary", {}),
        "results": result.report.get("results", []),
        "paths": _paths(search_json=result.json_path, search_markdown=result.markdown_path),
    }


def _tool_compare_runs(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    written = api.compare_runs(
        params["left"],
        params["right"],
        out=params["out"],
        markdown_out=params.get("markdown_out"),
    )
    return {
        "summary": written.report.get("summary", {}),
        "paths": _paths(diff_json=written.json_path, diff_markdown=written.markdown_path),
    }


def _tool_check_explanation_consistency(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    result = api.check_explanation_consistency(list(params["reports"]), out=params.get("out"))
    if isinstance(result, dict):
        return result
    return {
        "summary": result.report.get("summary", {}),
        "paths": _paths(consistency_json=result.json_path, consistency_markdown=result.markdown_path),
    }


def _tool_attribution_graph(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    written = api.attribution_graph(
        params["report"],
        out=params["out"],
        markdown_out=params.get("markdown_out"),
        html_out=params.get("html_out"),
    )
    graph = written.graph
    return {
        "node_count": len(graph.get("nodes", [])),
        "edge_count": len(graph.get("edges", [])),
        "candidate_path_count": len(graph.get("mechanism_summary", {}).get("candidate_paths", [])),
        "paths": _paths(
            graph_json=written.json_path,
            graph_markdown=written.markdown_path,
            graph_html=written.html_path,
        ),
    }


def _tool_validate_attribution_graph(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    written = api.validate_attribution_graph(
        params["graph"],
        path_records=params["records"],
        out=params["out"],
        require_controls=not params.get("allow_missing_controls", False),
    )
    return {
        "summary": written.report.get("summary", {}),
        "paths": _paths(validation_json=written.json_path, validation_markdown=written.markdown_path),
    }


_COMPACT_NOTE = "Writes artifacts to disk and returns a compact summary plus the written paths; read the artifacts for the full payload."

TOOLS: list[dict[str, Any]] = [
    {
        "name": "capabilities",
        "description": (
            "Discovery endpoint: the full interp-lab CLI surface, Python API contract, "
            "artifact schemas, optional-module availability, and conventions. Call this first. "
            "Returns the full payload."
        ),
        "inputSchema": _schema("No parameters.", {}, []),
        "handler": _tool_capabilities,
    },
    {
        "name": "doctor",
        "description": "Environment diagnostics: required runtime and optional adapter availability. Returns the full payload.",
        "inputSchema": _schema("No parameters.", {}, []),
        "handler": _tool_doctor,
    },
    {
        "name": "inspect",
        "description": (
            "Rank and explain model features for a natural-language criterion. "
            f"Requires 'out'. {_COMPACT_NOTE}"
        ),
        "inputSchema": _schema(
            "Inspection parameters.",
            {
                "model": {"type": "string", "description": "Model identifier."},
                "criterion": {"type": "string", "description": "Natural-language criterion."},
                "backend": {
                    "type": "string",
                    "enum": _BACKEND_CHOICES,
                    "description": "Feature backend (default: toy).",
                },
                "features": _path_property("JSONL feature dump for backend 'jsonl'"),
                "records": _path_property("Activation records JSONL for backend 'records'"),
                "interventions": _path_property("Optional intervention records JSONL"),
                "out": _path_property("Output directory for report.json/report.md"),
                "html_out": _path_property("Optional self-contained HTML report"),
                "csv_out": _path_property("Optional CSV of ranked features"),
                "top_k": {"type": "integer", "description": "Feature cards to keep (default 8)."},
                "require_interventions": {
                    "type": "boolean",
                    "description": "Give untested features zero criterion effect when interventions are set.",
                },
            },
            ["model", "criterion", "out"],
        ),
        "handler": _tool_inspect,
    },
    {
        "name": "compare",
        "description": (
            "Match candidate equivalent features across two inspection reports. "
            f"{_COMPACT_NOTE}"
        ),
        "inputSchema": _schema(
            "Match parameters.",
            {
                "left_report": _path_property("Left report.json"),
                "right_report": _path_property("Right report.json"),
                "out": _path_property("Output matches.json path"),
                "min_score": {"type": "number", "description": "Drop candidates scoring below this (0..1)."},
                "top_k": {"type": "integer", "description": "Matches to keep (default 10)."},
            },
            ["left_report", "right_report", "out"],
        ),
        "handler": _tool_compare,
    },
    {
        "name": "validate_matches",
        "description": (
            "Grade cross-model candidate matches by evidence (validated / plausible / "
            f"contradicted / weak). {_COMPACT_NOTE}"
        ),
        "inputSchema": _schema(
            "Match validation parameters.",
            {
                "match_report": _path_property("matches.json from the compare tool"),
                "out": _path_property("Output validation JSON path"),
            },
            ["match_report", "out"],
        ),
        "handler": _tool_validate_matches,
    },
    {
        "name": "search_features",
        "description": (
            "Search inspection reports for features matching a natural-language query. "
            "Returns results directly; pass 'out' to also write the search report (then "
            "returns summary, results, and paths)."
        ),
        "inputSchema": _schema(
            "Feature search parameters.",
            {
                "reports": _path_array_property("Inspection report.json files to search"),
                "query": {"type": "string", "description": "Natural-language feature description."},
                "out": _path_property("Optional output search-report JSON path"),
                "top_k": {"type": "integer", "description": "Hits to keep (default 10)."},
            },
            ["reports", "query"],
        ),
        "handler": _tool_search_features,
    },
    {
        "name": "compare_runs",
        "description": (
            "Diff two inspection reports: rank drift, score deltas, added/dropped features. "
            f"{_COMPACT_NOTE}"
        ),
        "inputSchema": _schema(
            "Run diff parameters.",
            {
                "left": _path_property("Baseline report.json"),
                "right": _path_property("Candidate report.json"),
                "out": _path_property("Output diff JSON path"),
                "markdown_out": _path_property("Optional Markdown summary path"),
            },
            ["left", "right", "out"],
        ),
        "handler": _tool_compare_runs,
    },
    {
        "name": "check_explanation_consistency",
        "description": (
            "Check whether feature explanations and ranks stay stable across paraphrased "
            "inspection reports. Returns the report directly; pass 'out' to write it and "
            "get a compact summary plus paths instead."
        ),
        "inputSchema": _schema(
            "Explanation consistency parameters.",
            {
                "reports": _path_array_property("Paraphrased inspection report.json files"),
                "out": _path_property("Optional output JSON path"),
            },
            ["reports"],
        ),
        "handler": _tool_check_explanation_consistency,
    },
    {
        "name": "attribution_graph",
        "description": (
            "Export an inspection report as a candidate mechanism (attribution) graph. "
            f"{_COMPACT_NOTE}"
        ),
        "inputSchema": _schema(
            "Attribution graph parameters.",
            {
                "report": _path_property("Inspection report.json"),
                "out": _path_property("Output graph JSON path"),
                "markdown_out": _path_property("Optional Markdown digest path"),
                "html_out": _path_property("Optional offline HTML viewer path"),
            },
            ["report", "out"],
        ),
        "handler": _tool_attribution_graph,
    },
    {
        "name": "validate_attribution_graph",
        "description": (
            "Validate measured attribution-graph paths against path-patching records. "
            f"{_COMPACT_NOTE}"
        ),
        "inputSchema": _schema(
            "Graph validation parameters.",
            {
                "graph": _path_property("Attribution graph JSON"),
                "records": _path_property("Path-patch records JSONL"),
                "out": _path_property("Output validation JSON path"),
                "allow_missing_controls": {
                    "type": "boolean",
                    "description": "Accept path claims without control records (default false).",
                },
            },
            ["graph", "records", "out"],
        ),
        "handler": _tool_validate_attribution_graph,
    },
]


class McpServer:
    """MCP server core: one ``handle_message`` call per JSON-RPC message."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {tool["name"]: tool for tool in TOOLS}

    # ------------------------------------------------------------- protocol

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle one JSON-RPC message; return the response, or None for notifications."""
        if not isinstance(message, dict):
            return self._error(None, INVALID_REQUEST, "Invalid request: expected a JSON object")
        method = message.get("method")
        message_id = message.get("id")
        if isinstance(method, str) and method.startswith("notifications/"):
            return None
        if message_id is None:
            # A request without an id is a notification; never respond.
            return None
        if not isinstance(method, str):
            return self._error(message_id, INVALID_REQUEST, "Invalid request: missing method")
        params = message.get("params") or {}
        if method == "initialize":
            return self._result(message_id, self._initialize(params))
        if method == "ping":
            return self._result(message_id, {})
        if method == "tools/list":
            return self._result(message_id, {"tools": self._tool_listing()})
        if method == "tools/call":
            return self._tools_call(message_id, params)
        if method == "resources/list":
            return self._result(message_id, {"resources": self._resource_listing()})
        if method == "resources/read":
            return self._resources_read(message_id, params)
        return self._error(message_id, METHOD_NOT_FOUND, f"Method not found: {method}")

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        version = requested if requested in KNOWN_PROTOCOL_VERSIONS else DEFAULT_PROTOCOL_VERSION
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {}, "resources": {}},
            "serverInfo": {"name": "interp-lab", "version": __version__},
            "instructions": _SERVER_INSTRUCTIONS,
        }

    def _tool_listing(self) -> list[dict[str, Any]]:
        return [
            {"name": tool["name"], "description": tool["description"], "inputSchema": tool["inputSchema"]}
            for tool in TOOLS
        ]

    def _tools_call(self, message_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        tool = self._tools.get(name) if isinstance(name, str) else None
        if tool is None:
            return self._error(message_id, INVALID_PARAMS, f"Unknown tool: {name!r}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return self._error(message_id, INVALID_PARAMS, "Tool arguments must be an object")
        missing = [key for key in tool["inputSchema"]["required"] if key not in arguments]
        if missing:
            return self._error(
                message_id,
                INVALID_PARAMS,
                f"Tool {name!r} missing required argument(s): {', '.join(missing)}",
            )
        handler: Callable[[dict[str, Any]], dict[str, Any]] = tool["handler"]
        try:
            result = handler(arguments)
        except _TOOL_ERRORS as exc:
            # Tool-level failure: a result with isError, never a JSON-RPC error.
            return self._result(
                message_id,
                {"content": [{"type": "text", "text": str(exc)}], "isError": True},
            )
        tool_result: dict[str, Any] = {
            "content": [{"type": "text", "text": json.dumps(result, indent=2, sort_keys=True)}],
            "isError": False,
        }
        if isinstance(result, dict):
            tool_result["structuredContent"] = result
        return self._result(message_id, tool_result)

    def _resource_listing(self) -> list[dict[str, Any]]:
        resources = []
        for name, relative, description in _DOC_RESOURCES:
            if _resolve_doc(relative) is not None:
                resources.append(
                    {
                        "uri": f"interp-lab://docs/{name}",
                        "name": name,
                        "description": description,
                        "mimeType": "text/markdown",
                    }
                )
        return resources

    def _resources_read(self, message_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri")
        for name, relative, _description in _DOC_RESOURCES:
            if uri == f"interp-lab://docs/{name}":
                path = _resolve_doc(relative)
                if path is None:
                    break
                return self._result(
                    message_id,
                    {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "text/markdown",
                                "text": path.read_text(encoding="utf-8"),
                            }
                        ]
                    },
                )
        return self._error(message_id, INVALID_PARAMS, f"Unknown resource: {uri!r}")

    # ------------------------------------------------------------ transport

    def serve_stdio(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
        """Serve newline-delimited JSON-RPC until stdin closes. Flushes per message."""
        stdin = sys.stdin if stdin is None else stdin
        stdout = sys.stdout if stdout is None else stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                response: dict[str, Any] | None = self._error(None, PARSE_ERROR, f"Parse error: {exc}")
            else:
                try:
                    response = self.handle_message(message)
                except Exception as exc:  # noqa: BLE001 -- the server must outlive any one message
                    message_id = message.get("id") if isinstance(message, dict) else None
                    print(f"interp-lab mcp: internal error: {exc}", file=sys.stderr)
                    response = self._error(message_id, INTERNAL_ERROR, f"Internal error: {exc}")
            if response is not None:
                stdout.write(json.dumps(response, sort_keys=True) + "\n")
                stdout.flush()
        return 0

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _result(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "result": result}

    @staticmethod
    def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def run_mcp_server() -> int:
    """CLI entry: serve MCP over stdio. Diagnostics go to stderr only."""
    print(
        f"interp-lab {__version__} MCP server ready on stdio (newline-delimited JSON-RPC 2.0)",
        file=sys.stderr,
    )
    return McpServer().serve_stdio()
