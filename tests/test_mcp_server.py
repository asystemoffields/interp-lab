import io
import json
import subprocess
import sys
from pathlib import Path

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
    ]
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
