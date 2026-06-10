import json
import re
import sys
from pathlib import Path

import pytest

from interp_lab.criterion_scoring import (
    CRITERIA_INSTALL_MESSAGE,
    DEFAULT_NLI_MODEL,
    HASH_SCORER_WARNING,
    HashCosineScorer,
    build_score_prompts_parser,
    build_scorer,
    default_hypothesis,
    score_prompts,
)
from interp_lab.hf_records import load_prompt_records


class FakeScorer:
    """Deterministic stub for the scorer_factory seam: scores keyed on text markers."""

    def __init__(self, scorer_id="nli:fake-model", scores_by_marker=None, weak=False):
        self.id = scorer_id
        self.weak = weak
        self.scores_by_marker = scores_by_marker or {"EVAL": 0.9}
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score_batch(self, hypothesis, texts):
        self.calls.append((hypothesis, tuple(texts)))
        return [
            next(
                (score for marker, score in self.scores_by_marker.items() if marker in text),
                0.1,
            )
            for text in texts
        ]


def _fake_factory(fake):
    def factory(scorer, scorer_model):
        factory.requested = (scorer, scorer_model)
        return fake

    return factory


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _dataset(tmp_path: Path) -> Path:
    return _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {"prompt_id": "p1", "text": "an EVAL-looking prompt"},
            {"prompt_id": "p2", "text": "an ordinary request"},
        ],
    )


def test_score_prompts_with_fake_scorer_writes_loadable_scored_dataset(tmp_path: Path):
    fake = FakeScorer()
    out = tmp_path / "scored.jsonl"

    summary = score_prompts(
        _dataset(tmp_path),
        "evaluation awareness",
        out=out,
        scorer_factory=_fake_factory(fake),
    )

    assert summary["count"] == 2
    assert summary["scorer"] == "nli:fake-model"
    assert summary["out"] == str(out)
    assert summary["warnings"] == []
    assert "rows" not in summary  # with out set, the file is the payload
    # The output loads through the REAL scored-prompt loader (extra keys tolerated).
    records = load_prompt_records(out)
    assert [record.prompt_id for record in records] == ["p1", "p2"]
    assert [record.criterion_score for record in records] == [0.9, 0.1]
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert all(row["criterion_score_source"] == "nli:fake-model" for row in rows)
    assert summary["score_stats"] == {"min": 0.1, "max": 0.9, "mean": 0.5}


def test_score_prompts_records_default_hypothesis_and_passes_it_to_the_scorer(tmp_path: Path):
    fake = FakeScorer()

    summary = score_prompts(
        _dataset(tmp_path),
        "the model is aware it is being evaluated",
        scorer_factory=_fake_factory(fake),
    )

    expected = "This text clearly involves the model is aware it is being evaluated."
    assert default_hypothesis("the model is aware it is being evaluated") == expected
    assert summary["hypothesis"] == expected
    assert fake.calls[0][0] == expected


def test_score_prompts_hypothesis_override_is_used_and_recorded(tmp_path: Path):
    fake = FakeScorer()

    summary = score_prompts(
        _dataset(tmp_path),
        "evaluation awareness",
        hypothesis="This text describes a test of an AI system.",
        scorer_factory=_fake_factory(fake),
    )

    assert summary["hypothesis"] == "This text describes a test of an AI system."
    assert fake.calls[0][0] == "This text describes a test of an AI system."


def test_score_prompts_forwards_scorer_name_and_model_to_the_factory(tmp_path: Path):
    fake = FakeScorer()
    factory = _fake_factory(fake)

    score_prompts(
        _dataset(tmp_path),
        "evaluation awareness",
        scorer="nli",
        scorer_model="my-org/my-nli",
        scorer_factory=factory,
    )

    assert factory.requested == ("nli", "my-org/my-nli")


def test_score_prompts_without_out_returns_rows_inline(tmp_path: Path):
    summary = score_prompts(
        _dataset(tmp_path),
        "evaluation awareness",
        scorer_factory=_fake_factory(FakeScorer()),
    )

    assert summary["out"] is None
    assert [row["prompt_id"] for row in summary["rows"]] == ["p1", "p2"]
    assert all(row["criterion_score_source"] == "nli:fake-model" for row in summary["rows"])


def test_score_prompts_binarize_thresholds_scores_and_keeps_raw(tmp_path: Path):
    out = tmp_path / "scored.jsonl"

    summary = score_prompts(
        _dataset(tmp_path),
        "evaluation awareness",
        out=out,
        binarize=0.5,
        scorer_factory=_fake_factory(FakeScorer()),
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["criterion_score"] for row in rows] == [1.0, 0.0]
    assert [row["criterion_score_raw"] for row in rows] == [0.9, 0.1]
    assert summary["binarize"] == 0.5
    assert summary["binarized_positive_count"] == 1


def test_score_prompts_accepts_plain_text_lines_and_generates_prompt_ids(tmp_path: Path):
    dataset = tmp_path / "prompts.txt"
    dataset.write_text("an EVAL prompt\n\nan ordinary line\n", encoding="utf-8")
    out = tmp_path / "scored.jsonl"

    score_prompts(dataset, "evaluation awareness", out=out, scorer_factory=_fake_factory(FakeScorer()))

    records = load_prompt_records(out)
    assert [record.prompt_id for record in records] == ["prompt-001", "prompt-002"]
    assert records[0].criterion_score == 0.9


def test_score_prompts_accepts_existing_scored_dataset_rows(tmp_path: Path):
    # Re-scoring a dataset that already carries criterion_score must work: the
    # loader takes the text field and ignores the stale score.
    dataset = _write_jsonl(
        tmp_path / "scored-in.jsonl",
        [{"prompt_id": "a", "text": "EVAL text", "criterion_score": 0.0}],
    )

    summary = score_prompts(dataset, "evaluation awareness", scorer_factory=_fake_factory(FakeScorer()))

    assert summary["rows"][0]["criterion_score"] == 0.9


def test_score_prompts_dataset_diagnostics_carry_file_and_line(tmp_path: Path):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text('{"text": "fine"}\n{not json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"{dataset}:2") + ".*invalid prompt JSON"):
        score_prompts(dataset, "c", scorer_factory=_fake_factory(FakeScorer()))

    dataset.write_text('{"prompt_id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"{dataset}:1") + ".*text or prompt field"):
        score_prompts(dataset, "c", scorer_factory=_fake_factory(FakeScorer()))


def test_score_prompts_rejects_scorer_returning_wrong_count(tmp_path: Path):
    class BrokenScorer:
        id = "broken"

        def score_batch(self, hypothesis, texts):
            return [0.5]

    with pytest.raises(ValueError, match="returned 1 score"):
        score_prompts(
            _dataset(tmp_path),
            "evaluation awareness",
            scorer_factory=lambda scorer, model: BrokenScorer(),
        )


def test_nli_scorer_missing_extra_errors_cleanly(tmp_path: Path, monkeypatch):
    # transformers is absent in this environment; pin that down even if installed.
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(RuntimeError, match=re.escape("interp-lab[criteria]")) as excinfo:
        score_prompts(_dataset(tmp_path), "evaluation awareness", scorer="nli")
    assert str(excinfo.value) == CRITERIA_INSTALL_MESSAGE


def test_hash_scorer_is_always_labeled_weak(tmp_path: Path):
    dataset = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {"prompt_id": "p1", "text": "a passage about evaluation awareness"},
            {"prompt_id": "p2", "text": "a recipe for vegetable soup"},
        ],
    )
    out = tmp_path / "scored.jsonl"

    summary = score_prompts(dataset, "evaluation awareness", scorer="hash", out=out)

    assert summary["scorer"] == "hash_cosine"
    assert HASH_SCORER_WARNING in summary["warnings"]
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert all(row["criterion_score_source"] == "hash_cosine" for row in rows)
    # Lexical overlap orders the scores: the on-topic text beats the soup recipe.
    assert rows[0]["criterion_score"] > rows[1]["criterion_score"]
    # Cosine output is clamped to [0, 1].
    assert all(0.0 <= row["criterion_score"] <= 1.0 for row in rows)


def test_hash_scorer_clamps_to_unit_interval():
    scorer = HashCosineScorer()
    scores = scorer.score_batch("alpha beta", ["totally unrelated words here", "alpha beta"])
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores[1] == 1.0


def test_build_scorer_rejects_unknown_backend():
    with pytest.raises(ValueError, match="scorer must be one of"):
        build_scorer("bogus")


def test_score_prompts_requires_criterion_and_prompts(tmp_path: Path):
    with pytest.raises(ValueError, match="criterion is required"):
        score_prompts(_dataset(tmp_path), "  ", scorer_factory=_fake_factory(FakeScorer()))

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no prompts found"):
        score_prompts(empty, "c", scorer_factory=_fake_factory(FakeScorer()))


def test_score_prompts_parser_defaults():
    parser = build_score_prompts_parser()
    args = parser.parse_args(["--dataset", "d.jsonl", "--criterion", "c"])

    assert args.scorer == "nli"
    assert args.scorer_model is None
    assert args.hypothesis is None
    assert args.out is None
    assert args.binarize is None
    assert args.json is False
    # The default model id is a configurable default (any HF zero-shot/NLI id
    # works via --scorer-model), presented as such in the help text.
    assert DEFAULT_NLI_MODEL == "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1"
    assert "compact" in parser.format_help()
