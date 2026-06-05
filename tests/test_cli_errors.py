"""Friendly-error and real-world-input robustness tests.

These pin the contract that ordinary user mistakes (a typo'd field, a Windows BOM,
an unknown embedder, a path without an extension) produce a clean one-line
``interp-lab: error: ...`` (exit 2) or a usable result -- never a raw traceback.
They also cover CLI-consistency papercuts (--version, --json stdout purity,
positional assay, the --out alias).
"""

import json
from pathlib import Path

import pytest

from interp_lab import __version__
from interp_lab.cli import main


def _run(argv: list[str]) -> int:
    """Run the CLI, normalizing the parser.exit(2) SystemExit into an int."""
    try:
        return main(argv)
    except SystemExit as exc:  # parser.exit raises SystemExit(code)
        return int(exc.code or 0)


def test_jsonl_missing_field_is_friendly_error(tmp_path: Path, capsys):
    features = tmp_path / "bad.jsonl"
    features.write_text('{"foo": "bar"}\n', encoding="utf-8")
    code = _run(
        ["inspect", "--model", "m", "--criterion", "x", "--backend", "jsonl",
         "--features", str(features), "--out", str(tmp_path / "out")]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "invalid feature record" in err
    assert "feature_id" in err
    assert "Traceback" not in err


def test_jsonl_non_object_row_is_friendly_error(tmp_path: Path, capsys):
    features = tmp_path / "bad.jsonl"
    features.write_text("[1, 2, 3]\n", encoding="utf-8")
    code = _run(
        ["inspect", "--model", "m", "--criterion", "x", "--backend", "jsonl",
         "--features", str(features), "--out", str(tmp_path / "out")]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "must be a JSON object" in err


def test_jsonl_utf8_bom_is_tolerated(tmp_path: Path):
    features = tmp_path / "bom.jsonl"
    features.write_bytes(
        b"\xef\xbb\xbf" + b'{"feature_id": "L1:F1", "model": "m", "label": "x"}\n'
    )
    code = _run(
        ["inspect", "--model", "m", "--criterion", "x", "--backend", "jsonl",
         "--features", str(features), "--out", str(tmp_path / "out")]
    )
    assert code == 0
    assert (tmp_path / "out" / "report.json").exists()


def test_jsonl_model_typo_warns_but_still_writes(tmp_path: Path, capsys):
    features = tmp_path / "feat.jsonl"
    features.write_text('{"feature_id": "L1:F1", "model": "real", "label": "x"}\n', encoding="utf-8")
    code = _run(
        ["inspect", "--model", "typo", "--criterion", "x", "--backend", "jsonl",
         "--features", str(features), "--out", str(tmp_path / "out")]
    )
    err = capsys.readouterr().err
    assert code == 0
    assert "no rows with model='typo'" in err
    assert "real" in err


def test_unknown_text_embedder_is_friendly_error(capsys):
    code = _run(["doctor", "--text-embedder", "does-not-exist"])
    err = capsys.readouterr().err
    assert code == 2
    assert "Unknown text embedder" in err
    assert "Traceback" not in err


def test_missing_features_file_is_friendly_error(tmp_path: Path, capsys):
    code = _run(
        ["inspect", "--model", "m", "--criterion", "x", "--backend", "jsonl",
         "--features", str(tmp_path / "does-not-exist.jsonl"), "--out", str(tmp_path / "out")]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "Traceback" not in err


def test_empty_criterion_is_rejected(tmp_path: Path, capsys):
    code = _run(["inspect", "--model", "toy/m", "--criterion", "   ", "--backend", "toy",
                 "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 2
    assert "criterion must be a non-empty" in err
    assert not (tmp_path / "out").exists()


def test_non_finite_score_is_rejected(tmp_path: Path, capsys):
    records = tmp_path / "rec.jsonl"
    records.write_text('{"model": "m", "criterion_score": NaN, "features": {"L1:F1": 0.5}}\n', encoding="utf-8")
    code = _run(["inspect", "--model", "m", "--criterion", "x", "--backend", "records",
                 "--records", str(records), "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 2
    assert "finite number" in err


def test_criterion_with_literal_braces_does_not_crash(tmp_path: Path):
    config = tmp_path / "run.json"
    config.write_text(
        '{"model": "toy/m", "criterion": "emits JSON like {\\"ok\\": true} and the set {1, 2}",'
        ' "backend": "toy", "out": "' + str(tmp_path / "run").replace("\\", "/") + '"}\n',
        encoding="utf-8",
    )
    code = _run(["run", str(config)])
    assert code == 0
    report = tmp_path / "run" / "inspect" / "report.json"
    assert report.exists()
    assert "{1, 2}" in report.read_text(encoding="utf-8")


def test_match_with_suffixless_out_writes_sibling_markdown(tmp_path: Path):
    from interp_lab.cli import main as cli_main
    assert cli_main(["demo", "--out", str(tmp_path / "demo")]) == 0
    out = tmp_path / "matches"  # no .json suffix
    code = _run(["match", "--left", str(tmp_path / "demo" / "model-a" / "report.json"),
                 "--right", str(tmp_path / "demo" / "model-b" / "report.json"), "--out", str(out)])
    assert code == 0
    assert out.exists()  # the JSON, written at the suffixless path
    assert (tmp_path / "matches.md").exists()  # markdown sibling, not a colliding dir


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert int(exc.value.code or 0) == 0
    assert __version__ in capsys.readouterr().out


def test_validate_assay_accepts_positional_file(capsys):
    # The positional routes to --preset-file: a missing file is reported, not ignored.
    code = _run(["validate-assay", "/no/such/assay.json"])
    assert code != 0
    assert "preset file not found" in capsys.readouterr().out


def test_release_check_json_keeps_stdout_pure(tmp_path: Path, capsys):
    code = _run(["release-check", "--json", "--out", str(tmp_path / "rc.json")])
    captured = capsys.readouterr()
    assert code == 0
    parsed = json.loads(captured.out)  # stdout is valid JSON -- no "Wrote ..." mixed in
    assert "ready_for_stable_release" in parsed
    assert "Wrote" in captured.err  # the confirmation went to stderr instead


def test_prepare_sae_prompts_out_alias(tmp_path: Path):
    scored = tmp_path / "scored.jsonl"
    scored.write_text(
        '{"text": "a", "criterion_score": 1.0}\n{"text": "b", "criterion_score": 0.0}\n',
        encoding="utf-8",
    )
    code = _run(["prepare-sae-prompts", "--dataset", str(scored), "--out", str(tmp_path / "pack")])
    assert code == 0
    assert (tmp_path / "pack").exists()
