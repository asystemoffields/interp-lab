import json
from pathlib import Path

import interp_lab
from interp_lab.capabilities import CAPABILITIES_SCHEMA, build_capabilities, write_capabilities
from interp_lab.cli import main
from interp_lab.contracts import SCHEMA_CONTRACTS, public_api_contract


def test_build_capabilities_payload_structure():
    payload = build_capabilities()

    assert payload["schema_version"] == "interp-lab.capabilities.v1"
    assert payload["tool"]["name"] == "interp-lab"
    assert payload["tool"]["version"] == interp_lab.__version__
    assert payload["tool"]["python"]
    assert payload["tool"]["platform"]
    assert json.loads(json.dumps(payload)) == payload


def test_capabilities_schema_is_registered_in_contracts():
    assert SCHEMA_CONTRACTS["capabilities"] == CAPABILITIES_SCHEMA
    assert build_capabilities()["schema_version"] == SCHEMA_CONTRACTS["capabilities"]


def test_capabilities_commands_cover_the_cli_surface():
    payload = build_capabilities()
    ids = {spec["id"] for spec in payload["commands"]}

    assert payload["commands"]
    assert "inspect" in ids
    assert "capabilities" in ids
    assert "mcp" in ids


def test_capabilities_python_api_matches_public_contract():
    assert build_capabilities()["python_api"] == public_api_contract()


def test_capabilities_environment_reports_optional_modules_and_embedder():
    environment = build_capabilities()["environment"]

    assert environment["optional_modules"]
    for module in environment["optional_modules"]:
        assert set(module) == {"name", "ok", "version", "purpose"}
    names = {module["name"] for module in environment["optional_modules"]}
    # Required checks (python, the package itself) stay out of the optional list.
    assert "python>=3.10" not in names
    assert "interp-lab" not in names
    assert environment["text_embedder"]


def test_capabilities_conventions_describe_agent_contract():
    conventions = build_capabilities()["conventions"]

    assert conventions["json_first"] is True
    assert "exit 2" in conventions["errors"]
    assert "schema_version" in conventions["outputs"]
    assert set(conventions["next_actions"]["shape"]) == {
        "id",
        "title",
        "command",
        "argv",
        "instruction",
        "requires",
    }
    assert "<angle-bracket>" in conventions["next_actions"]["placeholders"]
    assert conventions["mcp"] == {"command": "interp-lab mcp", "transport": "stdio"}


def test_write_capabilities_writes_json_file(tmp_path: Path):
    out = tmp_path / "nested" / "capabilities.json"

    path = write_capabilities(out)

    assert path == out
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CAPABILITIES_SCHEMA


def test_capabilities_public_api_export():
    payload = interp_lab.capabilities()

    assert payload["schema_version"] == CAPABILITIES_SCHEMA
    assert "capabilities" in interp_lab.__all__


def test_cli_capabilities_json_keeps_stdout_pure(capsys):
    assert main(["capabilities", "--json"]) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["schema_version"] == CAPABILITIES_SCHEMA


def test_cli_capabilities_json_with_out_confirms_on_stderr(tmp_path: Path, capsys):
    out = tmp_path / "capabilities.json"

    assert main(["capabilities", "--json", "--out", str(out)]) == 0

    captured = capsys.readouterr()
    json.loads(captured.out)  # stdout stays machine-pure
    assert f"Wrote {out}" in captured.err
    assert out.exists()


def test_cli_capabilities_out_without_json_writes_file(tmp_path: Path, capsys):
    out = tmp_path / "capabilities.json"

    assert main(["capabilities", "--out", str(out)]) == 0

    captured = capsys.readouterr()
    assert f"Wrote {out}" in captured.out
    assert json.loads(out.read_text(encoding="utf-8"))["schema_version"] == CAPABILITIES_SCHEMA


def test_cli_capabilities_default_prints_human_summary_with_json_hint(capsys):
    assert main(["capabilities"]) == 0

    out = capsys.readouterr().out
    assert "interp-lab" in out
    assert "capabilities --json" in out
    assert "mcp" in out
