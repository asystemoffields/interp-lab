from pathlib import Path

from oracle_sae.cli import build_parser
from oracle_sae.web_app import command_specs_from_parser, render_web_app_html, write_web_app


def test_command_specs_cover_cli_subcommands():
    specs = command_specs_from_parser(build_parser())
    ids = {spec["id"] for spec in specs}

    assert {
        "inspect",
        "train-sae",
        "export-transformerlens-records",
        "export-nnsight-records",
        "validate-matches",
        "export-attribution-graph",
        "validate-attribution-graph",
        "studio",
        "demo",
    } <= ids
    assert "web-app" not in ids


def test_render_web_app_contains_required_surfaces():
    specs = command_specs_from_parser(build_parser())

    html = render_web_app_html(command_specs=specs)

    assert "Interp Lab Studio" in html
    assert "command-specs" in html
    assert "generated-command" in html
    assert "run-config-output" in html
    assert "validate-hf-sae-paths" in html
    assert "--trust-remote-code" in html
    assert "Choose..." in html


def test_optional_select_fields_do_not_receive_parser_choice_defaults():
    specs = command_specs_from_parser(build_parser())
    inspect = next(spec for spec in specs if spec["id"] == "inspect")
    scope_source = next(field for field in inspect["fields"] if field["key"] == "scope_source")

    assert scope_source["type"] == "select"
    assert "default" not in scope_source


def test_write_web_app(tmp_path: Path):
    out = tmp_path / "studio.html"

    path = write_web_app(out, command_specs=command_specs_from_parser(build_parser()))

    assert path == out
    assert "Interp Lab Studio" in out.read_text(encoding="utf-8")
