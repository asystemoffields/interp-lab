import json
import re
import shlex
from pathlib import Path

import pytest

from interp_lab.cli import build_parser
from interp_lab.criterion_compile import (
    CRITERION_COMPILE_SCHEMA,
    GENERATION_REQUEST_SCHEMA,
    build_compile_criterion_parser,
    compile_criterion,
)
from interp_lab.criterion_lab import (
    build_criterion_assay_validation_report,
    load_criterion_lab_preset,
)
from interp_lab.hf_records import load_prompt_records

CRITERION = "the model is aware it is being evaluated"


class FakeScorer:
    """Deterministic scorer keyed on text markers (the scorer_factory seam)."""

    def __init__(self, scorer_id="nli:fake-model", scores_by_marker=None, default=0.05, weak=False):
        self.id = scorer_id
        self.weak = weak
        self.scores_by_marker = scores_by_marker or {}
        self.default = default

    def score_batch(self, hypothesis, texts):
        return [
            next(
                (score for marker, score in self.scores_by_marker.items() if marker in text),
                self.default,
            )
            for text in texts
        ]


def _write_candidates(path: Path, positives, negatives):
    rows = [{"label": "positive", "text": text} for text in positives]
    rows += [{"label": "negative", "text": text} for text in negatives]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _candidates(tmp_path: Path, *, positives=4, negatives=4):
    return _write_candidates(
        tmp_path / "candidates.jsonl",
        [f"POS prompt number {index} about being tested" for index in range(positives)],
        [f"NEG note number {index} about the garden" for index in range(negatives)],
    )


def _good_scorer():
    # POS texts score high (clear margin), NEG texts low.
    return FakeScorer(scores_by_marker={"POS": 0.92, "NEG": 0.04})


def test_compile_heuristic_end_to_end_with_fake_scorer(tmp_path: Path):
    out = tmp_path / "compile"

    report = compile_criterion(
        CRITERION,
        out=out,
        generator="heuristic",
        n=10,
        min_per_side=8,
        # Heuristic positives mention the criterion topic; negatives never do.
        scorer_factory=lambda scorer, model: FakeScorer(scores_by_marker={"evaluated": 0.9, "aware": 0.9}),
    )

    assert report["schema_version"] == CRITERION_COMPILE_SCHEMA
    assert report["status"] == "pass"
    assert report["generator"] == "heuristic"
    assert report["candidate_source"] == "heuristic"
    assert report["counts"]["positive_survivors"] == 10
    assert report["counts"]["negative_survivors"] == 10
    assert report["gates"]["margins"]["mode"] == "enforced"
    assert report["gates"]["margins"]["pass"] is True
    assert report["gates"]["balance"]["pass"] is True
    # Provenance discipline: hypothesis + scorer id recorded everywhere.
    assert report["hypothesis"] == f"This text clearly involves {CRITERION}."
    assert report["scorer"] == "nli:fake-model"

    # prompts.jsonl loads through the REAL scored-prompt loader.
    records = load_prompt_records(out / "prompts.jsonl")
    assert len(records) == 20
    rows = [json.loads(line) for line in (out / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["criterion_score_source"] == "nli:fake-model" for row in rows)

    # preset.json round-trips through the real Criterion Lab preset loader.
    preset = load_criterion_lab_preset(preset_file=out / "preset.json")
    assert preset["criterion"] == CRITERION
    assert len(preset["positive_prompts"]) == 10
    assert len(preset["negative_prompts"]) == 10
    cached = json.loads((out / "preset.json").read_text(encoding="utf-8"))["criterion_compile"]
    assert cached["hypothesis"] == report["hypothesis"]
    assert cached["scorer"] == "nli:fake-model"

    assert (out / "compile-report.json").exists()
    markdown = (out / "compile-report.md").read_text(encoding="utf-8")
    assert "Criterion Compile Report" in markdown
    assert "pass" in markdown


def test_compile_uses_real_assay_validation(tmp_path: Path):
    # Plant a duplicate positive candidate: the assay validation (the REAL
    # criterion_lab gate) must flag it in the compile report.
    candidates = _write_candidates(
        tmp_path / "candidates.jsonl",
        ["POS the same exact text", "POS the same exact text", "POS a different text"],
        ["NEG one", "NEG two", "NEG three"],
    )

    report = compile_criterion(
        CRITERION,
        out=tmp_path / "compile",
        candidates=candidates,
        min_per_side=3,
        scorer_factory=lambda scorer, model: _good_scorer(),
    )

    assay = report["gates"]["assay_validation"]
    assert assay["schema_version"] == "interp-lab.criterion_assay_validation.v1"
    codes = {issue["code"] for issue in assay["issues"]}
    assert "duplicate_positive_prompts" in codes
    # And it agrees with calling the validator directly on the written preset.
    direct = build_criterion_assay_validation_report(preset_file=tmp_path / "compile" / "preset.json")
    assert direct["status"] == assay["status"]


def test_low_margin_positive_is_excluded_with_reason(tmp_path: Path):
    candidates = _write_candidates(
        tmp_path / "candidates.jsonl",
        ["POS strong one", "POS strong two", "POS strong three", "WEAK positive outlier"],
        ["NEG a", "NEG b", "NEG c"],
    )

    report = compile_criterion(
        CRITERION,
        out=tmp_path / "compile",
        candidates=candidates,
        min_per_side=3,
        scorer_factory=lambda scorer, model: FakeScorer(
            scores_by_marker={"POS": 0.9, "WEAK": 0.4, "NEG": 0.05}
        ),
    )

    assert report["status"] == "pass"
    assert report["counts"]["positive_survivors"] == 3
    exclusions = report["exclusions"]
    assert len(exclusions) == 1
    assert exclusions[0]["side"] == "positive"
    assert exclusions[0]["text"] == "WEAK positive outlier"
    assert "below pos_threshold 0.7" in exclusions[0]["reason"]
    # The excluded prompt never reaches the dataset.
    texts = {record.text for record in load_prompt_records(tmp_path / "compile" / "prompts.jsonl")}
    assert "WEAK positive outlier" not in texts


def test_high_scoring_negative_is_excluded_with_reason(tmp_path: Path):
    candidates = _write_candidates(
        tmp_path / "candidates.jsonl",
        ["POS one", "POS two", "POS three"],
        ["NEG a", "NEG b", "NEG c", "LEAKY negative that matches the criterion"],
    )

    report = compile_criterion(
        CRITERION,
        out=tmp_path / "compile",
        candidates=candidates,
        min_per_side=3,
        scorer_factory=lambda scorer, model: FakeScorer(
            scores_by_marker={"POS": 0.9, "LEAKY": 0.8, "NEG": 0.05}
        ),
    )

    assert report["counts"]["negative_survivors"] == 3
    assert any(
        exclusion["side"] == "negative" and "above neg_threshold 0.3" in exclusion["reason"]
        for exclusion in report["exclusions"]
    )


def test_gate_failure_raises_with_report_already_written(tmp_path: Path):
    out = tmp_path / "compile"
    candidates = _candidates(tmp_path, positives=4, negatives=4)

    with pytest.raises(ValueError) as excinfo:
        compile_criterion(
            CRITERION,
            out=out,
            candidates=candidates,
            min_per_side=8,
            scorer_factory=lambda scorer, model: _good_scorer(),
        )

    message = str(excinfo.value)
    assert "need 8 per side" in message
    report_path = out / "compile-report.json"
    assert str(report_path) in message
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert report["gates"]["min_per_side"]["pass"] is False
    # The partial dataset is still on disk for inspection.
    assert (out / "prompts.jsonl").exists()
    assert (out / "compile-report.md").exists()


def test_balance_trims_over_represented_side_lowest_margin_first(tmp_path: Path):
    candidates = _write_candidates(
        tmp_path / "candidates.jsonl",
        ["POS high alpha", "POS high beta", "POS high gamma", "MID positive borderline"],
        ["NEG a", "NEG b", "NEG c"],
    )

    report = compile_criterion(
        CRITERION,
        out=tmp_path / "compile",
        candidates=candidates,
        min_per_side=3,
        scorer_factory=lambda scorer, model: FakeScorer(
            scores_by_marker={"POS": 0.95, "MID": 0.75, "NEG": 0.05}
        ),
    )

    # 4 positives all pass the margin, 3 negatives: the lowest-margin positive
    # (0.75) is trimmed for balance, with the reason recorded.
    assert report["counts"]["positive_survivors"] == 3
    assert report["counts"]["negative_survivors"] == 3
    assert report["counts"]["balance_trimmed"] == 1
    trims = [exclusion for exclusion in report["exclusions"] if "balance trim" in exclusion["reason"]]
    assert len(trims) == 1
    assert trims[0]["text"] == "MID positive borderline"
    assert report["gates"]["balance"]["pass"] is True


def test_hash_scorer_degrades_margin_gate_to_advisory(tmp_path: Path):
    # With the lexical scorer nothing is excluded on score and the report says
    # so loudly, even though hash scores would fail the enforced thresholds.
    report = compile_criterion(
        CRITERION,
        out=tmp_path / "compile",
        generator="heuristic",
        scorer="hash",
        n=8,
        min_per_side=8,
    )

    assert report["status"] == "pass"
    assert report["scorer"] == "hash_cosine"
    assert report["scorer_weak"] is True
    assert report["gates"]["margins"]["mode"] == "advisory"
    assert not any("threshold" in exclusion["reason"] for exclusion in report["exclusions"])
    assert any("ADVISORY" in warning for warning in report["warnings"])
    markdown = (tmp_path / "compile" / "compile-report.md").read_text(encoding="utf-8")
    assert "WEAK/lexical" in markdown
    assert "ADVISORY" in markdown


def test_compile_report_next_actions_are_canonical_and_parse(tmp_path: Path):
    report = compile_criterion(
        CRITERION,
        out=tmp_path / "compile",
        candidates=_candidates(tmp_path),
        min_per_side=4,
        scorer_factory=lambda scorer, model: _good_scorer(),
    )

    actions = report["agent_next_actions"]
    assert [action["id"] for action in actions] == [
        "inspect_compiled_dataset",
        "rescore_compiled_prompts",
    ]
    parser = build_parser()
    for action in actions:
        assert action["command"] == " ".join(shlex.quote(token) for token in action["argv"])
        argv = [
            "dummy-path" if re.match(r"^<.+>$", token) else token for token in action["argv"][1:]
        ]
        parser.parse_args(argv)
    # The rescore action carries the exact hypothesis + scorer provenance.
    rescore = actions[1]
    assert report["hypothesis"] in rescore["argv"]
    assert str(tmp_path / "compile" / "prompts.jsonl") in rescore["argv"]


# ---------------------------------------------------------------- llamacpp


class FakeLlama:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def create_completion(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"choices": [{"text": self._text}]}


def test_llamacpp_generator_with_stubbed_factory(tmp_path: Path):
    lines = []
    for index in range(4):
        lines.append(json.dumps({"label": "positive", "text": f"POS generated {index}"}))
        lines.append(json.dumps({"label": "negative", "text": f"NEG generated {index}"}))
    fake_llama = FakeLlama("\n".join(lines))

    report = compile_criterion(
        CRITERION,
        out=tmp_path / "compile",
        generator="llamacpp",
        model_path=tmp_path / "tiny-instruct.Q4_K_M.gguf",
        n=4,
        min_per_side=4,
        llama_factory=lambda model_path: fake_llama,
        scorer_factory=lambda scorer, model: _good_scorer(),
    )

    assert report["status"] == "pass"
    assert report["candidate_source"] == "llamacpp:tiny-instruct.Q4_K_M.gguf"
    assert report["counts"]["candidates"] == 8
    # The generation prompt carries the criterion, hypothesis, and constraints.
    prompt = fake_llama.calls[0][0]
    assert CRITERION in prompt
    assert "No shared template strings" in prompt


def test_llamacpp_generator_malformed_output_diagnostics(tmp_path: Path):
    fake_llama = FakeLlama('{"label": "positive", "text": "ok"}\n{broken json\nplain prose line\n')

    with pytest.raises(ValueError, match="0 negative candidate") as excinfo:
        compile_criterion(
            CRITERION,
            out=tmp_path / "compile",
            generator="llamacpp",
            model_path=tmp_path / "m.gguf",
            llama_factory=lambda model_path: fake_llama,
            scorer_factory=lambda scorer, model: _good_scorer(),
        )
    message = str(excinfo.value)
    assert "m.gguf output:2" in message  # line-numbered diagnostics
    assert "invalid JSON" in message
    assert "m.gguf output:3" in message


def test_llamacpp_generator_requires_model_path(tmp_path: Path):
    with pytest.raises(ValueError, match="model_path"):
        compile_criterion(CRITERION, out=tmp_path / "compile", generator="llamacpp")


# ---------------------------------------------------------- agent two-phase


def test_agent_generator_writes_request_and_round_trips(tmp_path: Path):
    out = tmp_path / "compile"

    # Phase one: no model call, the generation request is the result.
    request = compile_criterion(CRITERION, out=out, generator="agent", n=4, min_per_side=4)

    assert request["schema_version"] == GENERATION_REQUEST_SCHEMA
    assert request["criterion"] == CRITERION
    assert request["hypothesis"] == f"This text clearly involves {CRITERION}."
    assert request["counts"] == {"positive": 4, "negative": 4}
    assert request["constraints"]["diversity"]
    assert any("No shared template strings" in item for item in request["constraints"]["confounds"])
    assert request["candidates_format"]["line"]["label"] == "positive | negative"
    assert request["scoring"]["min_per_side"] == 4
    on_disk = json.loads((out / "generation-request.json").read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == GENERATION_REQUEST_SCHEMA

    # The two next actions: an instruction to write candidates, then the finish command.
    instruction, command = request["agent_next_actions"]
    assert instruction["id"] == "write_candidate_prompts"
    assert "instruction" in instruction and "argv" not in instruction
    assert str(out / "candidates.jsonl") in instruction["instruction"]
    assert command["id"] == "finish_compile_criterion"
    assert command["command"] == " ".join(shlex.quote(token) for token in command["argv"])
    build_parser().parse_args(command["argv"][1:])  # parses against the real CLI

    # Phase two: the agent writes candidates per the request, then finishes.
    candidates_path = Path(request["candidates_format"]["path"])
    _write_candidates(
        candidates_path,
        [f"POS agent-authored {index}" for index in range(4)],
        [f"NEG agent-authored {index}" for index in range(4)],
    )
    report = compile_criterion(
        CRITERION,
        out=out,
        candidates=candidates_path,
        min_per_side=4,
        scorer_factory=lambda scorer, model: _good_scorer(),
    )
    assert report["schema_version"] == CRITERION_COMPILE_SCHEMA
    assert report["status"] == "pass"
    assert report["candidate_source"] == str(candidates_path)
    assert load_prompt_records(out / "prompts.jsonl")


def test_candidates_loader_diagnostics(tmp_path: Path):
    path = tmp_path / "candidates.jsonl"
    path.write_text('{"label": "positive", "text": "ok"}\n{bad\n', encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"{path}:2") + ".*invalid JSON"):
        compile_criterion(CRITERION, out=tmp_path / "c", candidates=path)

    path.write_text('{"label": "maybe", "text": "ok"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"{path}:1") + ".*label must be"):
        compile_criterion(CRITERION, out=tmp_path / "c", candidates=path)

    path.write_text('{"label": "positive"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"{path}:1") + ".*missing text"):
        compile_criterion(CRITERION, out=tmp_path / "c", candidates=path)


def test_compile_parser_defaults_match_design_thresholds():
    parser = build_compile_criterion_parser()
    args = parser.parse_args(["--criterion", "c", "--out", "dir"])

    assert args.generator == "heuristic"
    assert args.scorer == "nli"
    assert args.n == 32
    assert args.pos_threshold == 0.7
    assert args.neg_threshold == 0.3
    assert args.min_per_side == 8
    assert args.json is False


def test_compile_input_validation(tmp_path: Path):
    with pytest.raises(ValueError, match="criterion is required"):
        compile_criterion("   ", out=tmp_path / "c")
    with pytest.raises(ValueError, match="generator must be one of"):
        compile_criterion(CRITERION, out=tmp_path / "c", generator="gpt5")
    with pytest.raises(ValueError, match="n must be at least 1"):
        compile_criterion(CRITERION, out=tmp_path / "c", n=0)
    with pytest.raises(ValueError, match="min_per_side must be at least 1"):
        compile_criterion(CRITERION, out=tmp_path / "c", min_per_side=0)
