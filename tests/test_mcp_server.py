import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

import interp_lab
from interp_lab.mcp_server import (
    DEFAULT_PROTOCOL_VERSION,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    McpServer,
)


def _request(message_id, method, params=None):
    message = {"jsonrpc": "2.0", "id": message_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def _call(server, message_id, name, arguments):
    return server.handle_message(
        _request(message_id, "tools/call", {"name": name, "arguments": arguments})
    )


def _inspect_toy(server, message_id, model, out):
    return _call(
        server,
        message_id,
        "inspect",
        {"model": model, "criterion": "benchmark awareness", "backend": "toy", "out": str(out)},
    )


def test_initialize_echoes_known_protocol_version():
    response = McpServer().handle_message(
        _request(1, "initialize", {"protocolVersion": "2025-03-26"})
    )

    result = response["result"]
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert result["protocolVersion"] == "2025-03-26"
    assert result["capabilities"] == {"tools": {}, "resources": {}}
    assert result["serverInfo"] == {"name": "interp-lab", "version": interp_lab.__version__}
    assert "capabilities" in result["instructions"]


def test_initialize_falls_back_to_default_protocol_version():
    response = McpServer().handle_message(
        _request(1, "initialize", {"protocolVersion": "1999-01-01"})
    )

    assert response["result"]["protocolVersion"] == DEFAULT_PROTOCOL_VERSION


def test_notifications_get_no_response():
    server = McpServer()

    assert server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert server.handle_message({"jsonrpc": "2.0", "method": "notifications/cancelled"}) is None
    # Any message without an id is a notification -- never answered.
    assert server.handle_message({"jsonrpc": "2.0", "method": "ping"}) is None


def test_ping_returns_empty_result():
    response = McpServer().handle_message(_request(7, "ping"))

    assert response == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_tools_list_schema_sanity():
    tools = McpServer().handle_message(_request(1, "tools/list"))["result"]["tools"]

    names = [tool["name"] for tool in tools]
    assert names == [
        "capabilities",
        "doctor",
        "inspect",
        "compare",
        "validate_matches",
        "search_features",
        "compare_runs",
        "check_explanation_consistency",
        "attribution_graph",
        "validate_attribution_graph",
        "plan_evidence",
        "dossier_update",
        "dossier_show",
        "quant_diff",
        "calibrate",
        "migrate_report",
        "export_steering",
        "intervene",
        "train_sae",
        "score_prompts",
        "compile_criterion",
    ]
    assert len(names) == 21
    # apply_steering is deliberately NOT served: generation against arbitrary
    # models is a host-agent decision (see the server instructions).
    assert "apply_steering" not in names
    for tool in tools:
        assert tool["name"]
        assert tool["description"]
        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert isinstance(schema["properties"], dict)
        assert isinstance(schema["required"], list)
        for required in schema["required"]:
            assert required in schema["properties"], (tool["name"], required)


def test_tools_call_capabilities():
    response = _call(McpServer(), 2, "capabilities", {})

    result = response["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["schema_version"] == "interp-lab.capabilities.v1"
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]


def test_tools_call_doctor():
    result = _call(McpServer(), 3, "doctor", {})["result"]

    assert result["isError"] is False
    assert result["structuredContent"]["tool"] == "interp-lab"
    assert result["structuredContent"]["checks"]


def test_toy_workflow_inspect_compare_validate(tmp_path: Path):
    server = McpServer()

    left = _inspect_toy(server, 1, "toy/model-a", tmp_path / "a")["result"]
    right = _inspect_toy(server, 2, "toy/model-b", tmp_path / "b")["result"]
    assert left["isError"] is False and right["isError"] is False
    left_summary = left["structuredContent"]
    assert left_summary["model"] == "toy/model-a"
    assert left_summary["criterion"] == "benchmark awareness"
    assert left_summary["card_count"] > 0
    assert left_summary["top_features"][0]["feature_id"]
    left_report = Path(left_summary["paths"]["report_json"])
    right_report = Path(right["structuredContent"]["paths"]["report_json"])
    assert left_report.exists() and right_report.exists()

    compared = _call(
        server,
        3,
        "compare",
        {
            "left_report": str(left_report),
            "right_report": str(right_report),
            "out": str(tmp_path / "matches.json"),
        },
    )["result"]
    assert compared["isError"] is False
    match_summary = compared["structuredContent"]
    assert match_summary["match_count"] > 0
    assert 0.0 <= match_summary["top_matches"][0]["score"] <= 1.0
    matches_path = Path(match_summary["paths"]["matches_json"])
    assert matches_path.exists()
    assert Path(match_summary["paths"]["matches_markdown"]).exists()

    validated = _call(
        server,
        4,
        "validate_matches",
        {"match_report": str(matches_path), "out": str(tmp_path / "validation.json")},
    )["result"]
    assert validated["isError"] is False
    summary = validated["structuredContent"]["summary"]
    assert summary["match_count"] == match_summary["match_count"]
    assert summary["overall_claim_grade"]
    assert Path(validated["structuredContent"]["paths"]["validation_json"]).exists()


def test_search_features_tool_returns_results_and_writes_with_out(tmp_path: Path):
    server = McpServer()
    report = _inspect_toy(server, 1, "toy/a", tmp_path / "a")["result"]["structuredContent"]
    report_path = report["paths"]["report_json"]

    direct = _call(server, 2, "search_features", {"reports": [report_path], "query": "awareness"})
    written = _call(
        server,
        3,
        "search_features",
        {"reports": [report_path], "query": "awareness", "out": str(tmp_path / "search.json")},
    )

    assert direct["result"]["structuredContent"]["schema_version"] == "interp-lab.feature_search.v1"
    assert Path(written["result"]["structuredContent"]["paths"]["search_json"]).exists()


def test_compare_runs_and_explanation_consistency_tools(tmp_path: Path):
    server = McpServer()
    report = _inspect_toy(server, 1, "toy/a", tmp_path / "a")["result"]["structuredContent"]
    report_path = report["paths"]["report_json"]

    diff = _call(
        server,
        2,
        "compare_runs",
        {"left": report_path, "right": report_path, "out": str(tmp_path / "diff.json")},
    )["result"]["structuredContent"]
    consistency = _call(
        server,
        3,
        "check_explanation_consistency",
        {"reports": [report_path, report_path]},
    )["result"]["structuredContent"]

    assert diff["summary"]["added_count"] == 0
    assert Path(diff["paths"]["diff_json"]).exists()
    assert consistency["schema_version"] == "interp-lab.explanation_consistency.v1"


def test_attribution_graph_and_validation_tools(tmp_path: Path):
    server = McpServer()
    report = _inspect_toy(server, 1, "toy/a", tmp_path / "a")["result"]["structuredContent"]

    graph = _call(
        server,
        2,
        "attribution_graph",
        {"report": report["paths"]["report_json"], "out": str(tmp_path / "graph.json")},
    )["result"]["structuredContent"]
    assert graph["node_count"] > 0
    assert graph["edge_count"] > 0
    graph_path = Path(graph["paths"]["graph_json"])
    assert graph_path.exists()

    synthetic_graph = tmp_path / "synthetic-graph.json"
    synthetic_graph.write_text(
        json.dumps(
            {
                "schema_version": "interp-lab.attribution_graph.v1",
                "model": "m",
                "criterion": {"text": "criterion"},
                "mechanism_summary": {
                    "candidate_paths": [
                        {
                            "source_feature_id": "SAE:L1:F1",
                            "target_feature_id": "SAE:L2:F8",
                            "evidence": "path_patch",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    records = tmp_path / "paths.jsonl"
    records.write_text(
        json.dumps(
            {
                "source_feature_id": "SAE:L1:F1",
                "target_feature_id": "SAE:L2:F8",
                "target_activation_delta": 0.2,
                "prompt_id": "p1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validation = _call(
        server,
        3,
        "validate_attribution_graph",
        {
            "graph": str(synthetic_graph),
            "records": str(records),
            "out": str(tmp_path / "graph-validation.json"),
            "allow_missing_controls": True,
        },
    )["result"]["structuredContent"]
    assert validation["summary"]
    assert Path(validation["paths"]["validation_json"]).exists()


def test_initialize_instructions_cover_the_investigation_loop():
    instructions = McpServer().handle_message(
        _request(1, "initialize", {"protocolVersion": DEFAULT_PROTOCOL_VERSION})
    )["result"]["instructions"]

    assert "plan_evidence" in instructions
    assert "dossier_update" in instructions
    assert "compile_criterion" in instructions
    assert "score_prompts" in instructions
    assert "apply-steering" in instructions  # documented as deliberately not exposed


def test_full_toy_investigation_loop_over_mcp_alone(tmp_path: Path):
    """inspect -> plan_evidence -> dossier_update -> dossier_show -> migrate_report
    (idempotent) -> quant_diff, all through tool calls only."""
    server = McpServer()
    left = _inspect_toy(server, 1, "toy/model-a", tmp_path / "a")["result"]["structuredContent"]
    report_path = left["paths"]["report_json"]

    # plan_evidence without out: the full plan comes back directly.
    plan = _call(server, 2, "plan_evidence", {"report": report_path})["result"]
    assert plan["isError"] is False
    full_plan = plan["structuredContent"]
    assert full_plan["schema_version"] == "interp-lab.evidence_plan.v1"
    assert full_plan["summary"]["cards_assessed"] > 0
    assert full_plan["plan"]

    # plan_evidence with out: compact summary plus written JSON + Markdown.
    written_plan = _call(
        server,
        3,
        "plan_evidence",
        {"report": report_path, "out": str(tmp_path / "plan.json"), "top_k": 3},
    )["result"]["structuredContent"]
    assert "plan" not in written_plan
    assert written_plan["summary"]["cards_assessed"] == 3
    assert Path(written_plan["paths"]["plan_json"]).exists()
    assert Path(written_plan["paths"]["plan_markdown"]).exists()

    # dossier_update twice: cumulative memory across rounds.
    dossier_path = tmp_path / "dossier.json"
    first = _call(
        server,
        4,
        "dossier_update",
        {"dossier": str(dossier_path), "report": report_path, "note": "round 1"},
    )["result"]["structuredContent"]
    assert first["summary"]["run_count"] == 1
    assert dossier_path.exists()
    second = _call(
        server,
        5,
        "dossier_update",
        {"dossier": str(dossier_path), "report": report_path, "note": "round 2"},
    )["result"]["structuredContent"]
    assert second["summary"]["run_count"] == 2
    assert second["summary"]["features"]  # per-feature standing in the summary

    shown = _call(
        server,
        6,
        "dossier_show",
        {"dossier": str(dossier_path), "markdown_out": str(tmp_path / "dossier.md")},
    )["result"]["structuredContent"]
    assert shown["summary"]["run_count"] == 2
    for standing in shown["summary"]["features"].values():
        assert standing["current_grade"]
    assert Path(shown["paths"]["dossier_markdown"]).exists()

    # migrate_report: current-version reports migrate cleanly and idempotently.
    migrated = _call(
        server,
        7,
        "migrate_report",
        {"report": report_path, "out": str(tmp_path / "migrated")},
    )["result"]["structuredContent"]
    migrated_report = Path(migrated["paths"]["report_json"])
    assert migrated_report.exists()
    again = _call(server, 8, "migrate_report", {"report": str(migrated_report)})["result"][
        "structuredContent"
    ]
    assert again["features_reordered"] is False
    assert again["score_delta_feature_count"] == 0
    assert again["score_deltas"] == {}

    # quant_diff: baseline vs variant of the same criterion.
    right = _inspect_toy(server, 9, "toy/model-b", tmp_path / "b")["result"]["structuredContent"]
    diff = _call(
        server,
        10,
        "quant_diff",
        {
            "left_report": report_path,
            "right_report": right["paths"]["report_json"],
            "out": str(tmp_path / "quant-diff.json"),
            "left_label": "f16",
            "right_label": "q4_k_m",
        },
    )["result"]["structuredContent"]
    headline = diff["headline"]
    for key in ("preserved_count", "degraded_count", "lost_count", "emerged_count"):
        assert isinstance(headline[key], int)
    assert isinstance(headline["degraded_validated"], list)
    assert diff["interpretation"]
    assert Path(diff["paths"]["diff_json"]).exists()
    assert Path(diff["paths"]["diff_markdown"]).exists()


def test_calibrate_tool_tiny_world(tmp_path: Path):
    result = _call(
        McpServer(),
        1,
        "calibrate",
        {
            "out": str(tmp_path / "calibration.json"),
            "seeds": 1,
            "features": 6,
            "causal": 2,
            "prompts": 8,
            "noise": 0.2,
            "work_dir": str(tmp_path / "worlds"),
        },
    )["result"]
    assert result["isError"] is False
    summary = result["structuredContent"]
    assert summary["verdict"]
    assert "decoy_resistance" in summary["headline"]
    assert "p_truly_causal_given_measured_causal" in summary["headline"]
    calibration_path = Path(summary["paths"]["calibration_json"])
    assert calibration_path.exists()
    assert Path(summary["paths"]["calibration_markdown"]).exists()
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "interp-lab.calibration_report.v1"
    assert payload["config"]["world"]["n_features"] == 6


def _write_association_only_report(path: Path) -> Path:
    """A hand-built report with a steerable hidden-dim card carrying only
    correlational evidence -- the case the export_steering provenance gate exists for."""
    from interp_lab.schema import Criterion, FeatureCard, FeatureFingerprint, InspectionReport

    card = FeatureCard(
        feature_id="L3:D7",
        model="toy/m",
        layer=3,
        label="eval awareness candidate",
        explanation="",
        importance=0.5,
        association=0.4,
        specificity=0.1,
        causal_effect=0.0,
        stability=0.5,
        examples=["the assistant suspects a test"],
        source="test",
        fingerprint=FeatureFingerprint(
            feature_id="L3:D7",
            model="toy/m",
            layer=3,
            text="evaluation awareness",
            text_vector=[0.5, 0.5],
            activation_signature=[1.0, 0.0],
            decoder_signature=[],
            causal_vector=[],
        ),
        metadata={},
        causal_effects={"criterion": 0.5, "signed_association": 0.4},
    )
    report = InspectionReport(
        model="toy/m", criterion=Criterion(text="benchmark awareness"), cards=[card]
    )
    path.write_text(json.dumps(report.to_dict()), encoding="utf-8")
    return path


def test_export_steering_gate_refusal_and_allow_unvalidated(tmp_path: Path):
    server = McpServer()
    report_path = _write_association_only_report(tmp_path / "report.json")
    out = tmp_path / "steer.json"

    refused = _call(
        server,
        1,
        "export_steering",
        {"report": str(report_path), "feature_id": "L3:D7", "out": str(out)},
    )["result"]
    assert refused["isError"] is True
    assert "--allow-unvalidated" in refused["content"][0]["text"]
    assert not out.exists()

    allowed = _call(
        server,
        2,
        "export_steering",
        {
            "report": str(report_path),
            "feature_id": "L3:D7",
            "out": str(out),
            "allow_unvalidated": True,
        },
    )["result"]
    assert allowed["isError"] is False
    summary = allowed["structuredContent"]
    assert summary["provenance"] == "unvalidated"
    assert summary["signed_effect_provenance"] == "association"
    assert "UNVALIDATED" in summary["unvalidated_warning"]
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["provenance"] == "unvalidated"


def _write_prompt_dataset(path: Path) -> Path:
    rows = [
        {"id": "p1", "text": "the assistant suspects this is a test", "criterion_score": 1.0},
        {"id": "p2", "text": "a plain weather report", "criterion_score": 0.0},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_intervene_defaults_to_dry_run_planning_without_any_model(tmp_path: Path):
    dataset = _write_prompt_dataset(tmp_path / "prompts.jsonl")

    result = _call(
        McpServer(),
        1,
        "intervene",
        {
            "model": "distilgpt2",
            "dataset": str(dataset),
            "criterion": "benchmark awareness",
            "out": str(tmp_path / "interventions.jsonl"),
            "features": ["L3:D7"],
            "plan_out": str(tmp_path / "plan.json"),
        },
    )["result"]

    assert result["isError"] is False
    plan_result = result["structuredContent"]
    assert plan_result["dry_run"] is True
    assert plan_result["records_path"] is None
    assert plan_result["plan"]["schema_version"] == "interp-lab.intervention_plan.v1"
    assert Path(plan_result["plan_path"]).exists()
    assert not (tmp_path / "interventions.jsonl").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is not None,
    reason="verifies the clean install-hint error on a torch-free environment",
)
def test_intervene_execution_without_torch_is_a_clean_tool_error(tmp_path: Path):
    dataset = _write_prompt_dataset(tmp_path / "prompts.jsonl")

    result = _call(
        McpServer(),
        1,
        "intervene",
        {
            "model": "distilgpt2",
            "dataset": str(dataset),
            "criterion": "benchmark awareness",
            "out": str(tmp_path / "interventions.jsonl"),
            "features": ["L3:D7"],
            "dry_run": False,
        },
    )["result"]

    assert result["isError"] is True
    assert "interp-lab[hf]" in result["content"][0]["text"]


def test_train_sae_tool_fallback_method_from_records(tmp_path: Path):
    server = McpServer()
    # Reuse the calibration world generator for a real activation-records file.
    from interp_lab.calibration import generate_planted_world

    world = generate_planted_world(0, n_features=4, n_causal=1, n_prompts=8)
    records_path, _interventions = world.write(tmp_path / "world")

    result = _call(
        server,
        1,
        "train_sae",
        {
            "out": str(tmp_path / "sae.json"),
            "records": str(records_path),
            "method": "fallback",
            "latent_dim": 2,
        },
    )["result"]

    assert result["isError"] is False
    summary = result["structuredContent"]
    assert summary["method"] == "fallback-dictionary"
    assert summary["latent_dim"] == 2
    assert Path(summary["paths"]["sae_json"]).exists()


def test_score_prompts_tool_writes_scored_dataset_with_hash_scorer(tmp_path: Path):
    dataset = _write_prompt_dataset(tmp_path / "prompts.jsonl")

    result = _call(
        McpServer(),
        1,
        "score_prompts",
        {
            "dataset": str(dataset),
            "criterion": "benchmark awareness",
            "scorer": "hash",
            "out": str(tmp_path / "scored.jsonl"),
        },
    )["result"]

    assert result["isError"] is False
    summary = result["structuredContent"]
    assert summary["count"] == 2
    assert summary["scorer"] == "hash_cosine"
    assert summary["hypothesis"] == "This text clearly involves benchmark awareness."
    assert summary["warnings"]  # hash scorer is always labeled weak
    assert "rows" not in summary  # out required over MCP: compact summary only
    rows = [
        json.loads(line)
        for line in Path(summary["out"]).read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["criterion_score_source"] == "hash_cosine" for row in rows)


def test_compile_criterion_tool_agent_generator_returns_request_payload(tmp_path: Path):
    result = _call(
        McpServer(),
        1,
        "compile_criterion",
        {
            "criterion": "benchmark awareness",
            "out": str(tmp_path / "compile"),
            "generator": "agent",
        },
    )["result"]

    assert result["isError"] is False
    request = result["structuredContent"]
    # The generation request IS the tool result — the two-phase agent flow.
    assert request["schema_version"] == "interp-lab.criterion_generation_request.v1"
    assert request["counts"] == {"positive": 32, "negative": 32}
    assert Path(request["request_path"]).exists()
    action_ids = [action["id"] for action in request["agent_next_actions"]]
    assert action_ids == ["write_candidate_prompts", "finish_compile_criterion"]


def test_compile_criterion_tool_heuristic_with_hash_scorer(tmp_path: Path):
    result = _call(
        McpServer(),
        1,
        "compile_criterion",
        {
            "criterion": "benchmark awareness",
            "out": str(tmp_path / "compile"),
            "generator": "heuristic",
            "scorer": "hash",
            "n": 10,
        },
    )["result"]

    assert result["isError"] is False
    summary = result["structuredContent"]
    assert summary["status"] == "pass"
    assert summary["gates"]["margins"]["mode"] == "advisory"
    assert summary["warnings"]
    assert Path(summary["paths"]["prompts_jsonl"]).exists()
    assert Path(summary["paths"]["preset_json"]).exists()
    assert Path(summary["paths"]["report_json"]).exists()


def test_unknown_method_returns_method_not_found():
    response = McpServer().handle_message(_request(9, "definitely/not-a-method"))

    assert response["error"]["code"] == METHOD_NOT_FOUND


def test_unknown_tool_returns_invalid_params():
    response = _call(McpServer(), 10, "not-a-tool", {})

    assert response["error"]["code"] == INVALID_PARAMS
    assert "not-a-tool" in response["error"]["message"]


def test_missing_required_arguments_return_invalid_params():
    response = _call(McpServer(), 11, "inspect", {"model": "toy/a"})

    assert response["error"]["code"] == INVALID_PARAMS
    assert "criterion" in response["error"]["message"]
    assert "out" in response["error"]["message"]


def test_tool_execution_failure_is_a_tool_result_not_a_protocol_error(tmp_path: Path):
    response = _call(
        McpServer(),
        12,
        "compare",
        {
            "left_report": str(tmp_path / "missing.json"),
            "right_report": str(tmp_path / "missing-2.json"),
            "out": str(tmp_path / "matches.json"),
        },
    )

    assert "error" not in response
    result = response["result"]
    assert result["isError"] is True
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"]


def test_resources_list_and_read_project_docs():
    server = McpServer()

    resources = server.handle_message(_request(1, "resources/list"))["result"]["resources"]
    uris = {resource["uri"] for resource in resources}
    assert "interp-lab://docs/README.md" in uris
    assert "interp-lab://docs/COMMANDS.md" in uris
    for resource in resources:
        assert resource["uri"].startswith("interp-lab://docs/")
        assert resource["mimeType"] == "text/markdown"

    read = server.handle_message(
        _request(2, "resources/read", {"uri": "interp-lab://docs/COMMANDS.md"})
    )["result"]["contents"][0]
    assert read["uri"] == "interp-lab://docs/COMMANDS.md"
    assert "interp-lab command reference" in read["text"]

    missing = server.handle_message(
        _request(3, "resources/read", {"uri": "interp-lab://docs/NOPE.md"})
    )
    assert missing["error"]["code"] == INVALID_PARAMS


def test_serve_stdio_loop_handles_messages_and_malformed_json():
    lines = "\n".join(
        [
            json.dumps(_request(1, "initialize", {"protocolVersion": DEFAULT_PROTOCOL_VERSION})),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            "{this is not json",
            json.dumps(_request(2, "ping")),
            "",
        ]
    )
    stdout = io.StringIO()

    exit_code = McpServer().serve_stdio(stdin=io.StringIO(lines), stdout=stdout)

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert exit_code == 0
    assert len(responses) == 3  # the notification is never answered
    assert responses[0]["id"] == 1
    assert responses[0]["result"]["protocolVersion"] == DEFAULT_PROTOCOL_VERSION
    assert responses[1]["id"] is None
    assert responses[1]["error"]["code"] == PARSE_ERROR
    assert responses[2] == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_mcp_subprocess_smoke():
    lines = "\n".join(
        [
            json.dumps(_request(1, "initialize", {"protocolVersion": DEFAULT_PROTOCOL_VERSION})),
            json.dumps(_request(2, "tools/list")),
            "",
        ]
    )

    completed = subprocess.run(
        [sys.executable, "-m", "interp_lab", "mcp"],
        input=lines,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(responses) == 2
    assert responses[0]["result"]["serverInfo"]["name"] == "interp-lab"
    tool_names = {tool["name"] for tool in responses[1]["result"]["tools"]}
    assert "capabilities" in tool_names
    assert "inspect" in tool_names
    # stdout carried nothing but protocol messages; diagnostics went to stderr.
    assert "MCP server ready" in completed.stderr
