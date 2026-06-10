"""Score prompt datasets against a natural-language criterion.

This is the SCORER stage of the criterion compiler: given candidate texts and a
criterion, it assigns each text a ``criterion_score`` in [0, 1] and writes the
scored-prompt JSONL every ``--dataset`` consumer in interp-lab already accepts
(``{"prompt_id", "text", "criterion_score"}`` -- the loaders tolerate the extra
provenance key this module adds).

Two scorer backends sit behind one factory seam:

- ``nli`` (default): zero-shot entailment with a compact NLI cross-encoder via
  transformers (optional extra ``criteria``). The score is P(entailment) of the
  scoring hypothesis against the text. The default model is a ~70M-parameter
  zero-shot checkpoint, but any Hugging Face zero-shot/NLI model id works via
  ``scorer_model``.
- ``hash``: cosine similarity between the lexical-hash embeddings of the
  hypothesis and the text, clamped to [0, 1]. Dependency-free, but it matches
  shared WORDS, not meaning -- every output produced with it is labeled weak.

Scoring-hypothesis discipline: criterion text is often model-internal language
("the model is aware it is being evaluated") while the scorer judges TEXT
properties. :func:`default_hypothesis` rewrites the criterion into a text-level
hypothesis, callers may override it, and every output records the hypothesis
and scorer id actually used.

``scorer_factory`` is the test/integration seam (mirroring ``llama_factory`` in
``gguf_records``): any callable ``(scorer_name, scorer_model) -> scorer`` where
the returned object exposes:

    id: str                                       # provenance stamp, e.g. "nli:<model>"
    score_batch(hypothesis, texts) -> list[float] # one score in [0, 1] per text

and optionally ``weak = True`` to mark lexical/advisory-only scorers.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from interp_lab.hf_records import load_prompt_records
from interp_lab.math_utils import clamp, cosine
from interp_lab.text_embedding import embed_text

CRITERIA_INSTALL_MESSAGE = "Install `interp-lab[criteria]` to score prompts with the NLI scorer (transformers + torch)."

DEFAULT_NLI_MODEL = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1"
SCORER_CHOICES = ("nli", "hash")

HASH_SCORER_WARNING = (
    "The hash scorer is weak/lexical: it matches shared words, not meaning. "
    'Treat these scores as advisory and install `interp-lab[criteria]` for the NLI scorer.'
)


def default_hypothesis(criterion: str) -> str:
    """Rewrite a (possibly model-internal) criterion as a text-level hypothesis."""
    return f"This text clearly involves {criterion.strip()}."


class HashCosineScorer:
    """Dependency-free lexical scorer: cosine(hash(hypothesis), hash(text)), clamped to [0, 1]."""

    id = "hash_cosine"
    weak = True

    def score_batch(self, hypothesis: str, texts: list[str]) -> list[float]:
        hypothesis_vector = embed_text(hypothesis)
        return [clamp(cosine(hypothesis_vector, embed_text(text))) for text in texts]


class NliEntailmentScorer:
    """Zero-shot NLI cross-encoder scorer (optional extra ``criteria``).

    score = P(entailment) of the hypothesis given the text, batched for CPU
    sanity. Any Hugging Face zero-shot/NLI sequence-classification model id
    works; the entailment logit index is read from the model config's labels.
    """

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, *, batch_size: int = 8) -> None:
        transformers = _optional_import("transformers", CRITERIA_INSTALL_MESSAGE)
        self._torch = _optional_import("torch", CRITERIA_INSTALL_MESSAGE)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        self._batch_size = max(1, batch_size)
        self.id = f"nli:{model_name}"
        self._entailment_index = _entailment_index(self._model.config)

    def score_batch(self, hypothesis: str, texts: list[str]) -> list[float]:
        scores: list[float] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                encoded = self._tokenizer(
                    batch,
                    [hypothesis] * len(batch),
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )
                logits = self._model(**encoded).logits
                probabilities = self._torch.softmax(logits, dim=-1)
                scores.extend(float(row[self._entailment_index]) for row in probabilities)
        return scores


def build_scorer(scorer: str = "nli", scorer_model: str | None = None) -> Any:
    """Default ``scorer_factory``: construct the named scorer backend."""
    if scorer == "nli":
        return NliEntailmentScorer(scorer_model or DEFAULT_NLI_MODEL)
    if scorer == "hash":
        return HashCosineScorer()
    raise ValueError(f"scorer must be one of: {', '.join(SCORER_CHOICES)} (got {scorer!r})")


def score_prompts(
    dataset: str | Path,
    criterion: str,
    *,
    hypothesis: str | None = None,
    scorer: str = "nli",
    scorer_model: str | None = None,
    out: str | Path | None = None,
    binarize: float | None = None,
    scorer_factory: Any | None = None,
) -> dict[str, Any]:
    """Score every prompt in ``dataset`` against ``criterion`` and summarize.

    ``dataset`` is a JSONL file of rows with ``text``/``prompt`` fields (extra
    keys, including an existing ``criterion_score``, are tolerated) or a plain
    text file with one prompt per line. With ``out`` set the scored rows are
    written in the standard scored-prompt JSONL format -- ``{"prompt_id",
    "text", "criterion_score"}`` plus a per-row ``criterion_score_source``
    provenance stamp -- and re-validated through the real prompt-record loader.
    Without ``out`` the scored rows ride along in the returned summary.

    ``binarize`` thresholds the continuous scores to 0/1 ``criterion_score``
    (the raw score is kept as ``criterion_score_raw``). Scores are continuous
    by default.
    """
    if not criterion or not criterion.strip():
        raise ValueError("criterion is required")
    rows = _load_dataset_rows(Path(dataset))
    if not rows:
        raise ValueError(f"{dataset}: no prompts found")
    resolved_hypothesis = hypothesis if hypothesis is not None else default_hypothesis(criterion)
    scorer_object = (scorer_factory or build_scorer)(scorer, scorer_model)
    scores = [float(value) for value in scorer_object.score_batch(resolved_hypothesis, [row["text"] for row in rows])]
    if len(scores) != len(rows):
        raise ValueError(
            f"scorer {scorer_object.id!r} returned {len(scores)} score(s) for {len(rows)} prompt(s)"
        )

    scored_rows: list[dict[str, Any]] = []
    for index, (row, score) in enumerate(zip(rows, scores), start=1):
        scored: dict[str, Any] = {
            "prompt_id": row["prompt_id"] or f"prompt-{index:03d}",
            "text": row["text"],
            "criterion_score": round(score, 6),
            "criterion_score_source": scorer_object.id,
        }
        if binarize is not None:
            scored["criterion_score_raw"] = round(score, 6)
            scored["criterion_score"] = 1.0 if score >= binarize else 0.0
        scored_rows.append(scored)

    out_path: Path | None = None
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for scored in scored_rows:
                handle.write(json.dumps(scored, sort_keys=True) + "\n")
        # Re-read through the real scored-prompt loader as a final format check.
        load_prompt_records(out_path)

    summary: dict[str, Any] = {
        "criterion": criterion,
        "hypothesis": resolved_hypothesis,
        "scorer": scorer_object.id,
        "count": len(scored_rows),
        "score_stats": _score_stats(scores),
        "binarize": binarize,
        "out": str(out_path) if out_path is not None else None,
        "warnings": [HASH_SCORER_WARNING] if getattr(scorer_object, "weak", False) else [],
    }
    if binarize is not None:
        summary["binarized_positive_count"] = sum(
            1 for scored in scored_rows if scored["criterion_score"] == 1.0
        )
    if out_path is None:
        summary["rows"] = scored_rows
    return summary


def build_score_prompts_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a prompt dataset against a natural-language criterion."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Prompt JSONL (text/prompt fields) or plain text file with one prompt per line.",
    )
    parser.add_argument("--criterion", required=True, help="Natural-language criterion.")
    parser.add_argument(
        "--hypothesis",
        help='Override the scoring hypothesis (default: "This text clearly involves <criterion>.").',
    )
    parser.add_argument(
        "--scorer",
        choices=list(SCORER_CHOICES),
        default="nli",
        help="Scorer backend: nli (needs the [criteria] extra) or hash (weak/lexical, no deps).",
    )
    parser.add_argument(
        "--scorer-model",
        help=(
            "Hugging Face zero-shot/NLI cross-encoder id for --scorer nli "
            f"(default: {DEFAULT_NLI_MODEL}, a compact ~70M-parameter checkpoint)."
        ),
    )
    parser.add_argument("--out", help="Output scored-prompt JSONL path. Omit to print the rows.")
    parser.add_argument(
        "--binarize",
        type=float,
        help="Threshold continuous scores to 0/1 criterion_score (raw kept as criterion_score_raw).",
    )
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    return parser


def run_score_prompts_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return score_prompts(
        args.dataset,
        args.criterion,
        hypothesis=args.hypothesis,
        scorer=args.scorer,
        scorer_model=args.scorer_model,
        out=args.out,
        binarize=args.binarize,
    )


def _load_dataset_rows(path: Path) -> list[dict[str, Any]]:
    """Load prompts: JSONL rows with text/prompt fields, or plain text lines."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            line_label = f"{path}:{line_number}"
            if stripped.startswith("{"):
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{line_label}: invalid prompt JSON: {exc.msg}") from exc
                text = data.get("text", data.get("prompt"))
                if text is None:
                    raise ValueError(f"{line_label}: prompt rows need a text or prompt field")
                rows.append(
                    {
                        "prompt_id": str(data.get("prompt_id", data.get("id", ""))),
                        "text": str(text),
                    }
                )
            else:
                rows.append({"prompt_id": "", "text": stripped})
    return rows


def _score_stats(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": round(min(scores), 6),
        "max": round(max(scores), 6),
        "mean": round(sum(scores) / len(scores), 6),
    }


def _entailment_index(config: Any) -> int:
    label2id = getattr(config, "label2id", None) or {}
    for label, index in label2id.items():
        if "entail" in str(label).lower():
            return int(index)
    return 0


def _optional_import(name: str, message: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(message) from exc
