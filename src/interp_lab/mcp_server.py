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
- ``apply-steering`` is deliberately not exposed as a tool: generating text from
  arbitrary models is a host-agent decision, not something a server should make
  one tool call away. Use the ``interp-lab apply-steering`` CLI once decided.
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
    "that write artifacts return compact summaries plus the written paths. "
    "The full investigation loop is driveable over MCP: compile_criterion "
    "(operationalize the criterion into a scored, gated prompt dataset; "
    "generator=agent returns a generation request so YOU write the candidate "
    "prompts, then re-call with the candidates path — the intended agent path) "
    "/ score_prompts (re-score any dataset against a criterion) -> inspect -> "
    "plan_evidence "
    "(gaps and costed interventions) -> intervene (dry_run defaults to true: plan "
    "first, then re-call with dry_run false to spend model time) -> inspect with "
    "interventions attached -> dossier_update / dossier_show (cumulative evidence "
    "per model+criterion) -> export_steering (the deliverable; refuses cards without "
    "intervention provenance). calibrate audits the grading itself against planted "
    "ground truth; quant_diff compares precision variants; migrate_report re-scores "
    "old reports. apply-steering is deliberately NOT exposed as an MCP tool: "
    "generating text from arbitrary models is a host-agent decision -- use the "
    "'interp-lab apply-steering' CLI when you have made it."
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


def _tool_plan_evidence(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    result = api.plan_evidence(
        params["report"],
        out=params.get("out"),
        markdown_out=params.get("markdown_out"),
        top_k=params.get("top_k"),
        confidence=params.get("confidence", 0.95),
    )
    if isinstance(result, dict):
        return result
    return {
        "summary": result.report.get("summary", {}),
        "paths": _paths(plan_json=result.json_path, plan_markdown=result.markdown_path),
    }


def _tool_dossier_update(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    dossier = api.update_dossier(
        dossier=params["dossier"],
        report=params["report"],
        matches=params.get("matches"),
        match_validation=params.get("match_validation"),
        graph_validation=params.get("graph_validation"),
        note=params.get("note"),
    )
    return {
        "summary": api.dossier_summary(dossier=dossier),
        "paths": _paths(dossier_json=params["dossier"]),
    }


def _tool_dossier_show(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    summary = api.dossier_summary(
        dossier=params["dossier"],
        markdown_out=params.get("markdown_out"),
    )
    return {
        "summary": summary,
        "paths": _paths(dossier_json=params["dossier"], dossier_markdown=params.get("markdown_out")),
    }


def _tool_quant_diff(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    report = api.quant_diff(
        params["left_report"],
        params["right_report"],
        out=params["out"],
        markdown_out=params.get("markdown_out"),
        left_label=params.get("left_label", "baseline"),
        right_label=params.get("right_label", "variant"),
    )
    summary = report.get("summary", {})
    json_path = Path(params["out"])
    markdown_path = (
        Path(params["markdown_out"]) if params.get("markdown_out") else json_path.with_suffix(".md")
    )
    return {
        "headline": {
            "preserved_count": summary.get("preserved_count"),
            "degraded_count": summary.get("degraded_count"),
            "lost_count": summary.get("lost_count"),
            "emerged_count": summary.get("emerged_count"),
            "validated_lost_count": summary.get("validated_lost_count"),
            "degraded_validated": summary.get("degraded_validated", []),
        },
        "interpretation": report.get("interpretation"),
        "paths": _paths(diff_json=json_path, diff_markdown=markdown_path),
    }


def _tool_calibrate(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    result = api.calibrate(
        params["out"],
        markdown_out=params.get("markdown_out"),
        work_dir=params.get("work_dir"),
        seeds=params.get("seeds", 5),
        features=params.get("features", 24),
        causal=params.get("causal", 6),
        prompts=params.get("prompts", 64),
        noise=params.get("noise", 0.3),
    )
    assessment = result.report.get("assessment", {})
    return {
        "verdict": assessment.get("verdict"),
        "headline": assessment.get("headline", {}),
        "summary": assessment.get("summary"),
        "paths": _paths(calibration_json=result.json_path, calibration_markdown=result.markdown_path),
    }


def _tool_migrate_report(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    migrated = api.migrate_inspection_report(params["report"], out=params.get("out"))
    migration = migrated.get("metadata", {}).get("migration", {})
    changes = migration.get("changes", {})
    score_deltas = changes.get("score_deltas", {})
    out = params.get("out")
    return {
        "from_tool_version": migration.get("from_tool_version"),
        "to_tool_version": migration.get("to_tool_version"),
        "features_reordered": changes.get("features_reordered"),
        "score_delta_feature_count": len(score_deltas),
        "score_deltas": score_deltas,
        "paths": _paths(report_json=Path(out) / "report.json" if out is not None else None),
    }


def _tool_export_steering(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    artifact = api.export_steering_vector(
        params["report"],
        params["feature_id"],
        sae=params.get("sae"),
        out=params["out"],
        strength=params.get("strength"),
        allow_unvalidated=params.get("allow_unvalidated", False),
    )
    summary = {
        "feature_id": artifact["feature_id"],
        "label": artifact["label"],
        "layer": artifact["layer"],
        "provenance": artifact["provenance"],
        "recommended_strength": artifact["recommended_strength"],
        "measured_signed_effect": artifact["measured_signed_effect"],
        "signed_effect_provenance": artifact["signed_effect_provenance"],
        "paths": _paths(steering_json=params["out"]),
    }
    if "unvalidated_warning" in artifact:
        summary["unvalidated_warning"] = artifact["unvalidated_warning"]
    return summary


def _tool_intervene(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    result = api.intervene(
        model=params["model"],
        dataset=params["dataset"],
        criterion=params["criterion"],
        out=params["out"],
        features=params.get("features"),
        report=params.get("report"),
        records=params.get("records"),
        top_k=params.get("top_k", 8),
        sae=params.get("sae"),
        mode=params.get("mode", "suppress"),
        strength_sweep=params.get("strength_sweep"),
        target_tokens=params.get("target_tokens"),
        device=params.get("device", "cpu"),
        max_length=params.get("max_length", 128),
        plan_out=params.get("plan_out"),
        dry_run=params.get("dry_run", True),
    )
    return result.to_dict()


def _tool_train_sae(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    result = api.train_sae(
        out=params["out"],
        records=params.get("records"),
        model=params.get("model"),
        hf_model=params.get("hf_model"),
        dataset=params.get("dataset"),
        records_out=params.get("records_out"),
        preset=params.get("preset", "minimal"),
        layer=params.get("layer"),
        latent_dim=params.get("latent_dim"),
        expansion_factor=params.get("expansion_factor"),
        method=params.get("method"),
        epochs=params.get("epochs"),
        seed=params.get("seed", 0),
        device=params.get("device", "cpu"),
        max_records=params.get("max_records"),
        criterion=params.get("criterion"),
        causal_out=params.get("causal_out"),
    )
    artifact = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
    return {
        "method": artifact.get("method"),
        "input_dim": artifact.get("input_dim"),
        "latent_dim": artifact.get("latent_dim"),
        "metrics": artifact.get("metrics", {}),
        "paths": _paths(
            sae_json=result.artifact_path,
            records_jsonl=result.records_path,
            interventions_jsonl=result.interventions_path,
        ),
    }


def _tool_score_prompts(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api

    return api.score_prompts(
        params["dataset"],
        params["criterion"],
        hypothesis=params.get("hypothesis"),
        scorer=params.get("scorer", "nli"),
        scorer_model=params.get("scorer_model"),
        out=params["out"],
        binarize=params.get("binarize"),
    )


def _tool_compile_criterion(params: dict[str, Any]) -> dict[str, Any]:
    from interp_lab import api
    from interp_lab.criterion_compile import GENERATION_REQUEST_SCHEMA

    result = api.compile_criterion(
        params["criterion"],
        out=params["out"],
        generator=params.get("generator", "heuristic"),
        candidates=params.get("candidates"),
        n=params.get("n", 32),
        hypothesis=params.get("hypothesis"),
        scorer=params.get("scorer", "nli"),
        scorer_model=params.get("scorer_model"),
        pos_threshold=params.get("pos_threshold", 0.7),
        neg_threshold=params.get("neg_threshold", 0.3),
        min_per_side=params.get("min_per_side", 8),
        model_path=params.get("model"),
    )
    if result.get("schema_version") == GENERATION_REQUEST_SCHEMA:
        # Two-phase agent flow: return the generation request whole — the
        # natural MCP shape; the caller writes the candidates and re-calls.
        return result
    return {
        "status": result["status"],
        "criterion": result["criterion"],
        "hypothesis": result["hypothesis"],
        "scorer": result["scorer"],
        "counts": result["counts"],
        "gates": {
            "margins": result["gates"]["margins"],
            "balance": result["gates"]["balance"],
            "min_per_side": result["gates"]["min_per_side"],
            "assay_validation_status": result["gates"]["assay_validation"].get("status"),
        },
        "warnings": result["warnings"],
        "agent_next_actions": result["agent_next_actions"],
        "paths": _paths(
            prompts_jsonl=result["outputs"]["prompts"],
            preset_json=result["outputs"]["preset"],
            report_json=result["outputs"]["report"],
            report_markdown=result["outputs"]["report_markdown"],
        ),
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
    {
        "name": "plan_evidence",
        "description": (
            "Diagnose a report's evidence gaps and rank the cheapest grade-moving "
            "interventions (recommended sample sizes, ready-to-run next actions). "
            "Returns the full plan directly; pass 'out' to write JSON+Markdown and "
            "get a compact summary plus paths instead."
        ),
        "inputSchema": _schema(
            "Evidence planning parameters.",
            {
                "report": _path_property("Inspection report.json"),
                "out": _path_property("Optional output plan JSON path (Markdown sibling is written too)"),
                "markdown_out": _path_property("Optional explicit Markdown summary path"),
                "top_k": {"type": "integer", "description": "Only plan the top-k cards (default: all)."},
                "confidence": {
                    "type": "number",
                    "description": "CI confidence for power analysis (default 0.95).",
                },
            },
            ["report"],
        ),
        "handler": _tool_plan_evidence,
    },
    {
        "name": "dossier_update",
        "description": (
            "Append an inspection run to the cumulative (model, criterion) evidence "
            "dossier -- created when absent, identity-checked, rewritten atomically. "
            "Returns the rollup summary (grade transitions, sign flips, contradictions) "
            "plus the dossier path; read the dossier JSON for full per-run detail."
        ),
        "inputSchema": _schema(
            "Dossier update parameters.",
            {
                "dossier": _path_property("Dossier JSON path (created when absent)"),
                "report": _path_property("Inspection report.json to append"),
                "matches": _path_property("Optional matches.json to attach"),
                "match_validation": _path_property("Optional match-validation JSON to attach"),
                "graph_validation": _path_property("Optional graph-validation JSON to attach"),
                "note": {"type": "string", "description": "Optional free-text note for this run entry."},
            },
            ["dossier", "report"],
        ),
        "handler": _tool_dossier_update,
    },
    {
        "name": "dossier_show",
        "description": (
            "Summarize an evidence dossier: per-feature standing, grade transitions, "
            "contradictions. Pass 'markdown_out' to also write the full Markdown rendering."
        ),
        "inputSchema": _schema(
            "Dossier summary parameters.",
            {
                "dossier": _path_property("Dossier JSON path"),
                "markdown_out": _path_property("Optional Markdown rendering path"),
            },
            ["dossier"],
        ),
        "handler": _tool_dossier_show,
    },
    {
        "name": "quant_diff",
        "description": (
            "Diff a baseline report against a quantized/precision variant of the same "
            "criterion: which intervention-validated features survived, degraded, were "
            f"lost, or emerged. {_COMPACT_NOTE}"
        ),
        "inputSchema": _schema(
            "Quantization diff parameters.",
            {
                "left_report": _path_property("Higher-precision baseline report.json"),
                "right_report": _path_property("Quantized variant report.json (same criterion)"),
                "out": _path_property("Output diff JSON path (Markdown sibling is written too)"),
                "markdown_out": _path_property("Optional explicit Markdown path"),
                "left_label": {"type": "string", "description": "Baseline label (default 'baseline')."},
                "right_label": {"type": "string", "description": "Variant label (default 'variant')."},
            },
            ["left_report", "right_report", "out"],
        ),
        "handler": _tool_quant_diff,
    },
    {
        "name": "calibrate",
        "description": (
            "Audit interp-lab's own claim grading against planted synthetic ground truth "
            "(truly causal features, correlational decoys, noise) and return the headline "
            "trust metrics: discovery precision/recall, decoy resistance, P(truly causal | "
            "tier), effect rank correlation, and an overall verdict. Compute-heavy at the "
            f"defaults; shrink seeds/features/prompts for a quick check. {_COMPACT_NOTE}"
        ),
        "inputSchema": _schema(
            "Calibration parameters.",
            {
                "out": _path_property("Output calibration JSON path (Markdown sibling is written too)"),
                "markdown_out": _path_property("Optional explicit Markdown path"),
                "work_dir": _path_property("Optional directory for planted-world artifacts (default: temp)"),
                "seeds": {"type": "integer", "description": "Number of planted worlds (default 5)."},
                "features": {"type": "integer", "description": "Features per world (default 24)."},
                "causal": {"type": "integer", "description": "Truly causal features per world (default 6)."},
                "prompts": {"type": "integer", "description": "Prompts per world (default 64, min 4)."},
                "noise": {"type": "number", "description": "Activation noise level (default 0.3)."},
            },
            ["out"],
        ),
        "handler": _tool_calibrate,
    },
    {
        "name": "migrate_report",
        "description": (
            "Re-score an older inspection report under current scoring semantics and "
            "re-rank its cards (pre-2.3 reports counted correlational evidence on the "
            "causal axis). Returns the migration summary (reorder flag, per-feature score "
            "deltas); pass 'out' to write the migrated report.json/report.md directory."
        ),
        "inputSchema": _schema(
            "Report migration parameters.",
            {
                "report": _path_property("Inspection report.json to migrate"),
                "out": _path_property("Optional output directory for the migrated report"),
            },
            ["report"],
        ),
        "handler": _tool_migrate_report,
    },
    {
        "name": "export_steering",
        "description": (
            "Export one report feature as a reusable steering-vector artifact. REFUSES "
            "cards without intervention-measured evidence (provenance gate); set "
            "'allow_unvalidated' to export anyway with the artifact stamped "
            f"provenance=unvalidated. {_COMPACT_NOTE}"
        ),
        "inputSchema": _schema(
            "Steering export parameters.",
            {
                "report": _path_property("Inspection report.json"),
                "feature_id": {
                    "type": "string",
                    "description": "Steerable feature id: L<layer>:D<dim> or SAE:L<layer>:F<latent>.",
                },
                "out": _path_property("Output steering-artifact JSON path"),
                "sae": _path_property("interp-lab SAE artifact JSON (required for SAE:* latents)"),
                "strength": {"type": "number", "description": "Override the derived recommended_strength."},
                "allow_unvalidated": {
                    "type": "boolean",
                    "description": "Export association-only cards, stamped provenance=unvalidated (default false).",
                },
            },
            ["report", "feature_id", "out"],
        ),
        "handler": _tool_export_steering,
    },
    {
        "name": "intervene",
        "description": (
            "Amplify, suppress, or ablate selected features on a Hugging Face model and "
            "write intervention records (requires the [hf] extra and a scored prompt "
            "JSONL). dry_run defaults to TRUE: the safe default returns the machine-"
            "readable plan (features, expected forward passes, follow-up commands) "
            "without loading any model. Re-call with dry_run=false to execute; the "
            "records JSONL then feeds inspect's 'interventions' argument."
        ),
        "inputSchema": _schema(
            "Feature intervention parameters.",
            {
                "model": {"type": "string", "description": "Hugging Face model name."},
                "dataset": _path_property("Prompt JSONL with text and criterion_score"),
                "criterion": {"type": "string", "description": "Criterion text stored in intervention rows."},
                "out": _path_property("Output intervention-record JSONL path"),
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature ids to intervene on (e.g. L6:D512, SAE:L6:F30).",
                },
                "report": _path_property("Optional report.json; top features are used when 'features' is omitted"),
                "records": _path_property("Optional activation-record JSONL for runnable follow-up commands"),
                "top_k": {"type": "integer", "description": "Top report features to use (default 8)."},
                "sae": _path_property("SAE artifact JSON required for SAE:* latent interventions"),
                "mode": {
                    "type": "string",
                    "enum": ["amplify", "suppress", "ablate"],
                    "description": "Feature edit to test (default suppress).",
                },
                "strength_sweep": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Steering strengths to sweep; sign is inferred from mode.",
                },
                "target_tokens": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Target tokens or ['auto'] (default: shared behavior token set).",
                },
                "device": {"type": "string", "description": "Torch device (default cpu)."},
                "max_length": {"type": "integer", "description": "Tokenizer max length (default 128)."},
                "plan_out": _path_property("Optional JSON path for the intervention plan/manifest"),
                "dry_run": {
                    "type": "boolean",
                    "description": "Default TRUE: plan only, no model load. Set false to execute.",
                },
            },
            ["model", "dataset", "criterion", "out"],
        ),
        "handler": _tool_intervene,
    },
    {
        "name": "train_sae",
        "description": (
            "Train an SAE from activation records ('records') or Hugging Face hidden "
            "states ('hf_model' + 'dataset'). method=auto uses torch when installed and "
            "falls back to the deterministic stdlib trainer otherwise; the HF path "
            f"requires the [hf] extra. {_COMPACT_NOTE}"
        ),
        "inputSchema": _schema(
            "SAE training parameters.",
            {
                "out": _path_property("Output SAE artifact JSON path"),
                "records": _path_property("Activation-record JSONL to train from"),
                "model": {"type": "string", "description": "Model name stamped into the artifact (records path)."},
                "hf_model": {"type": "string", "description": "Hugging Face model to capture activations from."},
                "dataset": _path_property("Prompt JSONL (required with hf_model)"),
                "records_out": _path_property("Optional SAE-feature activation records output"),
                "preset": {"type": "string", "description": "Training preset (default 'minimal')."},
                "layer": {"type": "integer", "description": "Hidden-state layer to train on (hf_model path)."},
                "latent_dim": {"type": "integer", "description": "Explicit latent dimension."},
                "expansion_factor": {"type": "number", "description": "Latent dim as a multiple of input dim."},
                "method": {
                    "type": "string",
                    "enum": ["auto", "torch", "fallback"],
                    "description": "Trainer backend; auto falls back to stdlib without torch.",
                },
                "epochs": {"type": "integer", "description": "Training epochs."},
                "seed": {"type": "integer", "description": "Random seed (default 0)."},
                "device": {"type": "string", "description": "Torch device (default cpu)."},
                "max_records": {"type": "integer", "description": "Cap on training records."},
                "criterion": {"type": "string", "description": "Criterion for the optional causal sweep (hf_model path)."},
                "causal_out": _path_property("Optional intervention-record output for the causal sweep"),
            },
            ["out"],
        ),
        "handler": _tool_train_sae,
    },
    {
        "name": "score_prompts",
        "description": (
            "Score a prompt dataset against a natural-language criterion via a scoring "
            "hypothesis. scorer='nli' uses a compact zero-shot NLI cross-encoder (needs "
            "the [criteria] extra; any HF zero-shot/NLI model id works via "
            "'scorer_model'); scorer='hash' is the dependency-free lexical fallback, "
            "always labeled weak. Every row carries criterion_score_source provenance. "
            f"Requires 'out'. {_COMPACT_NOTE}"
        ),
        "inputSchema": _schema(
            "Prompt scoring parameters.",
            {
                "dataset": _path_property(
                    "Prompt JSONL (text/prompt fields) or plain text file, one prompt per line"
                ),
                "criterion": {"type": "string", "description": "Natural-language criterion."},
                "hypothesis": {
                    "type": "string",
                    "description": 'Override the scoring hypothesis (default: "This text clearly involves <criterion>.").',
                },
                "scorer": {
                    "type": "string",
                    "enum": ["nli", "hash"],
                    "description": "Scorer backend (default nli; hash is weak/lexical but dependency-free).",
                },
                "scorer_model": {
                    "type": "string",
                    "description": "HF zero-shot/NLI cross-encoder id for scorer=nli.",
                },
                "out": _path_property("Output scored-prompt JSONL path"),
                "binarize": {
                    "type": "number",
                    "description": "Threshold continuous scores to 0/1 (raw kept as criterion_score_raw).",
                },
            },
            ["dataset", "criterion", "out"],
        ),
        "handler": _tool_score_prompts,
    },
    {
        "name": "compile_criterion",
        "description": (
            "Compile a natural-language criterion into a scored, gated prompt dataset "
            "(prompts.jsonl + Criterion Lab preset + compile report). Generators: "
            "'heuristic' (default; zero-dep templates), 'llamacpp' (local GGUF via "
            "'model'), or 'agent' — the intended agent path: NO model is called, the "
            "tool result IS the generation request (criterion, hypothesis, counts, "
            "diversity/confound constraints, candidates format); write the candidates "
            "JSONL yourself, then re-call with 'candidates' to score, gate, and "
            "package. The gate enforces score margins (per-prompt exclusions with "
            "reasons), positive/negative balance, and the real assay validation; with "
            "scorer='hash' margins are advisory only. Gate failure is a tool error "
            f"that names the already-written report. {_COMPACT_NOTE}"
        ),
        "inputSchema": _schema(
            "Criterion compile parameters.",
            {
                "criterion": {"type": "string", "description": "Natural-language criterion."},
                "out": _path_property("Output directory for the compiled artifacts"),
                "generator": {
                    "type": "string",
                    "enum": ["heuristic", "llamacpp", "agent"],
                    "description": "Candidate generator (default heuristic).",
                },
                "candidates": _path_property(
                    'Candidates JSONL ({"label": "positive"|"negative", "text": ...}); skips generation'
                ),
                "n": {"type": "integer", "description": "Candidates per side to generate (default 32)."},
                "hypothesis": {
                    "type": "string",
                    "description": "Override the scoring hypothesis.",
                },
                "scorer": {
                    "type": "string",
                    "enum": ["nli", "hash"],
                    "description": "Scorer backend (default nli; hash is weak/lexical, margins advisory).",
                },
                "scorer_model": {
                    "type": "string",
                    "description": "HF zero-shot/NLI cross-encoder id for scorer=nli.",
                },
                "model": _path_property("GGUF model path for generator=llamacpp"),
                "pos_threshold": {
                    "type": "number",
                    "description": "Minimum positive score; lower-scoring positives are excluded (default 0.7).",
                },
                "neg_threshold": {
                    "type": "number",
                    "description": "Maximum negative score; higher-scoring negatives are excluded (default 0.3).",
                },
                "min_per_side": {
                    "type": "integer",
                    "description": "Minimum surviving prompts per side (default 8).",
                },
            },
            ["criterion", "out"],
        ),
        "handler": _tool_compile_criterion,
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
