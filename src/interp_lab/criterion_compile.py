"""Compile a natural-language criterion into a scored, gated prompt dataset.

The criterion compiler operationalizes a plain-language criterion through three
pluggable stages -- generate big, score tiny, verify everything:

- GENERATOR (pluggable, never bundled as a big model):
    ``heuristic``  -- zero-dependency template paraphrases (the floor; outputs
                      are clearly labeled heuristic).
    ``llamacpp``   -- a small local instruct GGUF via a ``llama_factory`` seam
                      (mirroring ``gguf_records``); the model is prompted for
                      JSON-lines candidates and its output is parsed
                      defensively with line-numbered diagnostics.
    ``agent``      -- NO model call. Writes a generation-request artifact
                      (``interp-lab.criterion_generation_request.v1``) plus
                      canonical ``agent_next_actions`` so the DRIVING agent
                      writes ``candidates.jsonl`` itself, then finishes with
                      ``compile-criterion --candidates``. Two-phase flow.
- SCORER: ``criterion_scoring`` (tiny NLI cross-encoder, or the weak lexical
  hash fallback), behind the same ``scorer_factory`` seam.
- GATE: NLI margin thresholds with per-prompt outlier exclusion (never silent),
  positive/negative balance trimming, and the REAL Criterion Lab assay
  validation (``build_criterion_assay_validation_report``) over the survivors.

Outputs (under ``out/``): ``prompts.jsonl`` (the scored dataset, standard
scored-prompt format plus per-row provenance), ``preset.json`` (loadable by the
real Criterion Lab preset loader, with the hypothesis and scorer cached), and
``compile-report.json`` + ``compile-report.md``. A gate failure still writes
the report and raises a ValueError pointing at it.

Candidates JSONL format (generator output / ``--candidates`` input): one JSON
object per line, ``{"label": "positive" | "negative", "text": "..."}``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path
from typing import Any

from interp_lab.agent_actions import next_action
from interp_lab.criterion_lab import build_criterion_assay_validation_report
from interp_lab.criterion_scoring import (
    HASH_SCORER_WARNING,
    SCORER_CHOICES,
    build_scorer,
    default_hypothesis,
)
from interp_lab.hf_records import load_prompt_records
from interp_lab.text_vectors import content_tokens

CRITERION_COMPILE_SCHEMA = "interp-lab.criterion_compile.v1"
GENERATION_REQUEST_SCHEMA = "interp-lab.criterion_generation_request.v1"

GENERATOR_CHOICES = ("heuristic", "llamacpp", "agent")

LLAMACPP_INSTALL_MESSAGE = "Install `interp-lab[gguf]` to generate candidates with a local GGUF model."

# Constraints stamped into every generation request: candidates that share
# template strings across sides, or that confound the criterion with length/
# topic/style, produce features that track the confound instead.
GENERATION_CONSTRAINTS = {
    "diversity": [
        "Vary length, topic, and style independently of the criterion.",
        "Cover several registers (notes, dialogue, reports, questions, first person).",
        "No two candidates may be near-duplicates of each other.",
    ],
    "confounds": [
        "No shared template strings between positive and negative candidates.",
        "Negative candidates must not mention the criterion or negate it verbatim.",
        "Keep mean length comparable across the positive and negative sides.",
    ],
}

_POSITIVE_TEMPLATES = (
    "The following passage clearly involves {topic}.",
    "Notes from a discussion centered on {topic}, with concrete details throughout.",
    "A first-person account in which {topic} shapes every decision described.",
    "Q: What is happening in this text? A: It is a clear case of {topic}.",
    "Report excerpt: observers documented {topic} from start to finish.",
    "She explained, step by step, how {topic} surfaced in the transcript.",
    "Summary line: this situation is best described as {topic}.",
    "In this example, {topic} is unmistakable from the very first sentence.",
)

_NEGATIVE_DISTRACTORS = (
    "a recipe for vegetable soup",
    "a commuter train timetable",
    "a review of a garden hose",
    "instructions for assembling a bookshelf",
    "a weather forecast for the weekend",
    "a child's birthday party plan",
    "a note about returning a library book",
    "a description of a quiet mountain lake",
)

_NEGATIVE_TEMPLATES = (
    "Here is {distractor}, written out in full.",
    "Jotted quickly before lunch: {distractor}, nothing more.",
    "Q: What does this text cover? A: Just {distractor}.",
    "Minutes of a short meeting devoted entirely to {distractor}.",
    "He read aloud {distractor} while the kettle boiled.",
    "Today's bulletin contains only {distractor}.",
    "For the archive: {distractor}, filed without comment.",
    "An everyday message about {distractor} and when it is needed.",
)


def compile_criterion(
    criterion: str,
    *,
    out: str | Path,
    generator: str = "heuristic",
    candidates: str | Path | None = None,
    n: int = 32,
    hypothesis: str | None = None,
    scorer: str = "nli",
    scorer_model: str | None = None,
    pos_threshold: float = 0.7,
    neg_threshold: float = 0.3,
    min_per_side: int = 8,
    scorer_factory: Any | None = None,
    llama_factory: Any | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compile ``criterion`` into a scored, gated prompt dataset under ``out``.

    Returns the compile report dict. With ``generator='agent'`` and no
    ``candidates``, no model is called: the generation-request payload is
    written to ``out/generation-request.json`` and returned instead (phase one
    of the two-phase agent flow; phase two passes ``candidates=``).

    Gate semantics: each positive candidate scoring below ``pos_threshold`` and
    each negative scoring above ``neg_threshold`` is EXCLUDED with a recorded
    reason; the sides are then balanced by trimming the over-represented side
    (lowest margin first, also recorded); the survivors run through the real
    Criterion Lab assay validation. With ``scorer='hash'`` the margin gate is
    advisory only (the lexical scorer cannot support exclusion decisions).
    Fewer than ``min_per_side`` survivors on either side raises ``ValueError``
    -- with the report already written and named in the error.
    """
    if not criterion or not criterion.strip():
        raise ValueError("criterion is required")
    if generator not in GENERATOR_CHOICES:
        raise ValueError(f"generator must be one of: {', '.join(GENERATOR_CHOICES)} (got {generator!r})")
    if n < 1:
        raise ValueError("n must be at least 1")
    if min_per_side < 1:
        raise ValueError("min_per_side must be at least 1")
    out_dir = Path(out)
    resolved_hypothesis = hypothesis if hypothesis is not None else default_hypothesis(criterion)

    if candidates is None and generator == "agent":
        return _write_generation_request(
            criterion,
            out_dir=out_dir,
            hypothesis=resolved_hypothesis,
            n=n,
            scorer=scorer,
            pos_threshold=pos_threshold,
            neg_threshold=neg_threshold,
            min_per_side=min_per_side,
        )

    if candidates is not None:
        candidate_rows = _load_candidates(Path(candidates))
        candidate_source = str(candidates)
    elif generator == "heuristic":
        candidate_rows = _heuristic_candidates(criterion, n=n)
        candidate_source = "heuristic"
    else:  # llamacpp
        if model_path is None:
            raise ValueError("model_path (--model) is required with generator='llamacpp'")
        candidate_rows = _llamacpp_candidates(
            criterion,
            hypothesis=resolved_hypothesis,
            n=n,
            model_path=model_path,
            llama_factory=llama_factory,
        )
        candidate_source = f"llamacpp:{Path(model_path).name}"
    if not candidate_rows:
        raise ValueError("no candidates to compile")

    scorer_object = (scorer_factory or build_scorer)(scorer, scorer_model)
    scorer_is_weak = bool(getattr(scorer_object, "weak", False))
    texts = [row["text"] for row in candidate_rows]
    scores = [float(value) for value in scorer_object.score_batch(resolved_hypothesis, texts)]
    if len(scores) != len(candidate_rows):
        raise ValueError(
            f"scorer {scorer_object.id!r} returned {len(scores)} score(s) for {len(candidate_rows)} candidate(s)"
        )
    scored = [
        {"label": row["label"], "text": row["text"], "score": round(score, 6)}
        for row, score in zip(candidate_rows, scores)
    ]
    positives = [item for item in scored if item["label"] == "positive"]
    negatives = [item for item in scored if item["label"] == "negative"]

    # --- Gate (a): NLI margins; per-prompt outliers excluded, never silently.
    exclusions: list[dict[str, Any]] = []
    warnings: list[str] = []
    candidate_mean_positive = _mean([item["score"] for item in positives])
    candidate_mean_negative = _mean([item["score"] for item in negatives])
    margin_mode = "advisory" if scorer_is_weak else "enforced"
    if scorer_is_weak:
        warnings.append(HASH_SCORER_WARNING)
        warnings.append(
            "MARGIN GATE IS ADVISORY ONLY: the hash scorer is lexical, so margin "
            "thresholds were NOT enforced and no candidates were excluded on score. "
            "Re-run with the NLI scorer before trusting these margins."
        )
        kept_positives, kept_negatives = list(positives), list(negatives)
    else:
        kept_positives = []
        for item in positives:
            if item["score"] < pos_threshold:
                exclusions.append(
                    {
                        "side": "positive",
                        "text": item["text"],
                        "score": item["score"],
                        "reason": f"positive score {item['score']} below pos_threshold {pos_threshold}",
                    }
                )
            else:
                kept_positives.append(item)
        kept_negatives = []
        for item in negatives:
            if item["score"] > neg_threshold:
                exclusions.append(
                    {
                        "side": "negative",
                        "text": item["text"],
                        "score": item["score"],
                        "reason": f"negative score {item['score']} above neg_threshold {neg_threshold}",
                    }
                )
            else:
                kept_negatives.append(item)

    # Margin verdict over the SURVIVING prompts (the dataset actually shipped).
    mean_positive = _mean([item["score"] for item in kept_positives])
    mean_negative = _mean([item["score"] for item in kept_negatives])
    margins_pass = (
        mean_positive is not None
        and mean_negative is not None
        and mean_positive >= pos_threshold
        and mean_negative <= neg_threshold
    )

    # --- Gate (b): balance -- trim the over-represented side, lowest margin first.
    balance_trimmed = 0
    if len(kept_positives) != len(kept_negatives):
        keep = min(len(kept_positives), len(kept_negatives))
        if len(kept_positives) > keep:
            kept_positives, trimmed = _trim_lowest_margin(kept_positives, keep, margin_key=lambda s: s)
        else:
            kept_negatives, trimmed = _trim_lowest_margin(kept_negatives, keep, margin_key=lambda s: -s)
        for item in trimmed:
            balance_trimmed += 1
            exclusions.append(
                {
                    "side": item["label"],
                    "text": item["text"],
                    "score": item["score"],
                    "reason": "balance trim: over-represented side, lowest margin first",
                }
            )

    # --- Outputs (written even on gate failure, so the error can point at them).
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = _write_prompts_jsonl(
        out_dir / "prompts.jsonl",
        positives=kept_positives,
        negatives=kept_negatives,
        score_source=scorer_object.id,
    )
    preset_path = _write_preset_json(
        out_dir / "preset.json",
        criterion=criterion,
        positives=kept_positives,
        negatives=kept_negatives,
        hypothesis=resolved_hypothesis,
        scorer_id=scorer_object.id,
        generator=generator if candidates is None else "candidates",
    )

    # --- Gate (c): duplicate/length-confound checks via the REAL assay validation.
    assay_validation = build_criterion_assay_validation_report(preset_file=preset_path)

    survivors_ok = len(kept_positives) >= min_per_side and len(kept_negatives) >= min_per_side
    status = "pass" if survivors_ok else "fail"

    report: dict[str, Any] = {
        "schema_version": CRITERION_COMPILE_SCHEMA,
        "status": status,
        "criterion": criterion,
        "hypothesis": resolved_hypothesis,
        "scorer": scorer_object.id,
        "scorer_weak": scorer_is_weak,
        "generator": generator,
        "candidate_source": candidate_source,
        "counts": {
            "candidates": len(scored),
            "positive_candidates": len(positives),
            "negative_candidates": len(negatives),
            "excluded": len(exclusions),
            "balance_trimmed": balance_trimmed,
            "positive_survivors": len(kept_positives),
            "negative_survivors": len(kept_negatives),
            "min_per_side": min_per_side,
        },
        "gates": {
            "margins": {
                "mode": margin_mode,
                "pass": bool(margins_pass),
                "pos_threshold": pos_threshold,
                "neg_threshold": neg_threshold,
                "mean_positive_score": mean_positive,
                "mean_negative_score": mean_negative,
                "candidate_mean_positive_score": candidate_mean_positive,
                "candidate_mean_negative_score": candidate_mean_negative,
            },
            "balance": {
                "pass": len(kept_positives) == len(kept_negatives),
                "trimmed": balance_trimmed,
            },
            "min_per_side": {
                "pass": survivors_ok,
                "required": min_per_side,
            },
            "assay_validation": assay_validation,
        },
        "exclusions": exclusions,
        "score_distributions": {
            "positive": _distribution([item["score"] for item in kept_positives]),
            "negative": _distribution([item["score"] for item in kept_negatives]),
        },
        "warnings": warnings,
        "outputs": {
            "prompts": str(prompts_path),
            "preset": str(preset_path),
            "report": str(out_dir / "compile-report.json"),
            "report_markdown": str(out_dir / "compile-report.md"),
        },
        "agent_next_actions": _compile_next_actions(
            criterion,
            prompts_path=prompts_path,
            hypothesis=resolved_hypothesis,
            scorer=scorer,
        ),
    }
    report_path = out_dir / "compile-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "compile-report.md").write_text(render_compile_report_markdown(report), encoding="utf-8")

    if not survivors_ok:
        raise ValueError(
            f"criterion compile gate failed: {len(kept_positives)} positive / "
            f"{len(kept_negatives)} negative survivor(s), need {min_per_side} per side; "
            f"see {report_path} for exclusions and reasons"
        )
    return report


def render_compile_report_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    margins = report["gates"]["margins"]
    assay = report["gates"]["assay_validation"]
    lines = [
        "# interp-lab Criterion Compile Report",
        "",
        f"Status: **{report['status']}**",
        f"Criterion: {report['criterion']}",
        f"Hypothesis: {report['hypothesis']}",
        f"Scorer: `{report['scorer']}`" + ("  (WEAK/lexical — margins advisory)" if report["scorer_weak"] else ""),
        f"Generator: `{report['generator']}` (source: {report['candidate_source']})",
        "",
        "## Gates",
        "",
        f"- Margins ({margins['mode']}): {'pass' if margins['pass'] else 'FAIL'} — "
        f"mean positive {margins['mean_positive_score']} (need >= {margins['pos_threshold']}), "
        f"mean negative {margins['mean_negative_score']} (need <= {margins['neg_threshold']})",
        f"- Balance: {'pass' if report['gates']['balance']['pass'] else 'FAIL'} "
        f"({report['gates']['balance']['trimmed']} trimmed)",
        f"- Min per side: {'pass' if report['gates']['min_per_side']['pass'] else 'FAIL'} "
        f"({counts['positive_survivors']} positive / {counts['negative_survivors']} negative, "
        f"need {counts['min_per_side']})",
        f"- Assay validation: {assay.get('status', 'unknown')} "
        f"({assay.get('summary', {}).get('issue_count', 0)} issue(s))",
        "",
        "## Counts",
        "",
        f"- Candidates: {counts['candidates']} "
        f"({counts['positive_candidates']} positive, {counts['negative_candidates']} negative)",
        f"- Excluded: {counts['excluded']} (of which {counts['balance_trimmed']} balance trims)",
        f"- Survivors: {counts['positive_survivors']} positive, {counts['negative_survivors']} negative",
    ]
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    if report["exclusions"]:
        lines.extend(["", "## Exclusions", ""])
        for exclusion in report["exclusions"]:
            lines.append(
                f"- [{exclusion['side']}] score={exclusion['score']}: {exclusion['reason']} — "
                f"{exclusion['text']}"
            )
    lines.extend(["", "## Outputs", ""])
    for name, path in report["outputs"].items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines).strip() + "\n"


def build_compile_criterion_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile a natural-language criterion into a scored, gated prompt dataset."
    )
    parser.add_argument("--criterion", required=True, help="Natural-language criterion.")
    parser.add_argument("--out", required=True, help="Output directory for the compiled artifacts.")
    parser.add_argument(
        "--generator",
        choices=list(GENERATOR_CHOICES),
        default="heuristic",
        help=(
            "Candidate generator: heuristic (zero-dep templates), llamacpp (local GGUF via "
            "--model), or agent (no model call: writes a generation request for the driving "
            "agent, two-phase flow finished with --candidates)."
        ),
    )
    parser.add_argument(
        "--candidates",
        help='Candidates JSONL ({"label": "positive"|"negative", "text": ...}); skips generation.',
    )
    parser.add_argument("--n", type=int, default=32, help="Candidates per side to generate (default 32).")
    parser.add_argument(
        "--hypothesis",
        help='Override the scoring hypothesis (default: "This text clearly involves <criterion>.").',
    )
    parser.add_argument(
        "--scorer",
        choices=list(SCORER_CHOICES),
        default="nli",
        help="Scorer backend: nli (needs the [criteria] extra) or hash (weak/lexical; margins advisory).",
    )
    parser.add_argument("--scorer-model", help="Hugging Face zero-shot/NLI model id for --scorer nli.")
    parser.add_argument("--model", help="GGUF model path for --generator llamacpp.")
    parser.add_argument(
        "--pos-threshold",
        type=float,
        default=0.7,
        help="Minimum score for positive candidates; lower-scoring positives are excluded (default 0.7).",
    )
    parser.add_argument(
        "--neg-threshold",
        type=float,
        default=0.3,
        help="Maximum score for negative candidates; higher-scoring negatives are excluded (default 0.3).",
    )
    parser.add_argument(
        "--min-per-side",
        type=int,
        default=8,
        help="Minimum surviving prompts per side; fewer fails the gate (default 8).",
    )
    parser.add_argument("--json", action="store_true", help="Print the report/request as JSON.")
    return parser


def run_compile_criterion_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return compile_criterion(
        args.criterion,
        out=args.out,
        generator=args.generator,
        candidates=args.candidates,
        n=args.n,
        hypothesis=args.hypothesis,
        scorer=args.scorer,
        scorer_model=args.scorer_model,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        min_per_side=args.min_per_side,
        model_path=args.model,
    )


# ------------------------------------------------------------------ generators


def _heuristic_candidates(criterion: str, *, n: int) -> list[dict[str, str]]:
    """Zero-dependency template paraphrases — the floor, clearly labeled heuristic."""
    topic = _short_topic(criterion)
    rows: list[dict[str, str]] = []
    for index in range(n):
        template = _POSITIVE_TEMPLATES[index % len(_POSITIVE_TEMPLATES)]
        text = template.format(topic=topic)
        if index >= len(_POSITIVE_TEMPLATES):
            text += f" (heuristic variant {index // len(_POSITIVE_TEMPLATES) + 1})"
        rows.append({"label": "positive", "text": text})
    for index in range(n):
        template = _NEGATIVE_TEMPLATES[index % len(_NEGATIVE_TEMPLATES)]
        distractor = _NEGATIVE_DISTRACTORS[index % len(_NEGATIVE_DISTRACTORS)]
        text = template.format(distractor=distractor)
        if index >= len(_NEGATIVE_TEMPLATES):
            text += f" (heuristic variant {index // len(_NEGATIVE_TEMPLATES) + 1})"
        rows.append({"label": "negative", "text": text})
    return rows


def _llamacpp_candidates(
    criterion: str,
    *,
    hypothesis: str,
    n: int,
    model_path: str | Path,
    llama_factory: Any | None,
) -> list[dict[str, str]]:
    """Generate candidates with a small instruct GGUF model via llama.cpp.

    ``llama_factory`` is the test/integration seam (mirroring ``gguf_records``):
    any callable ``(model_path) -> llama`` where the returned object exposes
    ``create_completion(prompt, max_tokens=..., temperature=...) ->
    {"choices": [{"text": str}]}``. The model output is parsed line by line as
    JSON candidates; malformed lines are reported with line-numbered
    diagnostics instead of silently dropped.
    """
    factory = llama_factory or _load_llama_for_generation
    llama = factory(str(model_path))
    prompt = _generation_prompt(criterion, hypothesis=hypothesis, n=n)
    completion = llama.create_completion(prompt, max_tokens=128 * n, temperature=0.8)
    raw = str(completion["choices"][0]["text"])
    rows, diagnostics = _parse_model_candidates(raw, label_source=f"{Path(model_path).name} output")
    positives = [row for row in rows if row["label"] == "positive"]
    negatives = [row for row in rows if row["label"] == "negative"]
    if not positives or not negatives:
        details = "; ".join(diagnostics) if diagnostics else "no parseable candidate lines"
        raise ValueError(
            f"{Path(model_path).name}: model output yielded {len(positives)} positive and "
            f"{len(negatives)} negative candidate(s); need both sides. Diagnostics: {details}"
        )
    return positives[:n] + negatives[:n]


def _generation_prompt(criterion: str, *, hypothesis: str, n: int) -> str:
    constraint_lines = "\n".join(
        f"- {constraint}"
        for constraint in (*GENERATION_CONSTRAINTS["diversity"], *GENERATION_CONSTRAINTS["confounds"])
    )
    return (
        f"You generate evaluation prompts for the criterion: {criterion}\n"
        f"A scorer will test each text against the hypothesis: {hypothesis}\n\n"
        f"Write exactly {n} POSITIVE texts (the hypothesis clearly holds) and "
        f"{n} NEGATIVE texts (the hypothesis clearly does not hold).\n"
        f"Constraints:\n{constraint_lines}\n\n"
        'Output ONLY JSON lines, one object per line, no prose:\n'
        '{"label": "positive", "text": "..."}\n'
        '{"label": "negative", "text": "..."}\n'
    )


def _parse_model_candidates(raw: str, *, label_source: str) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    diagnostics: list[str] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip().strip("`")
        if not stripped:
            continue
        if not stripped.startswith("{"):
            diagnostics.append(f"{label_source}:{line_number}: not a JSON object, skipped")
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            diagnostics.append(f"{label_source}:{line_number}: invalid JSON: {exc.msg}")
            continue
        label = str(data.get("label", "")).strip().lower()
        text = str(data.get("text", "")).strip()
        if label not in {"positive", "negative"}:
            diagnostics.append(f"{label_source}:{line_number}: label must be positive or negative")
            continue
        if not text:
            diagnostics.append(f"{label_source}:{line_number}: empty text")
            continue
        rows.append({"label": label, "text": text})
    return rows, diagnostics


def _load_llama_for_generation(model_path: str) -> Any:
    llama_cpp = _optional_import("llama_cpp", LLAMACPP_INSTALL_MESSAGE)
    return llama_cpp.Llama(model_path=model_path, n_ctx=4096, verbose=False)


# ------------------------------------------------------- agent two-phase flow


def _write_generation_request(
    criterion: str,
    *,
    out_dir: Path,
    hypothesis: str,
    n: int,
    scorer: str,
    pos_threshold: float,
    neg_threshold: float,
    min_per_side: int,
) -> dict[str, Any]:
    candidates_path = out_dir / "candidates.jsonl"
    request: dict[str, Any] = {
        "schema_version": GENERATION_REQUEST_SCHEMA,
        "criterion": criterion,
        "hypothesis": hypothesis,
        "counts": {"positive": n, "negative": n},
        "constraints": {key: list(value) for key, value in GENERATION_CONSTRAINTS.items()},
        "candidates_format": {
            "path": str(candidates_path),
            "encoding": "JSONL, one JSON object per line",
            "line": {"label": "positive | negative", "text": "the candidate prompt text"},
        },
        "scoring": {
            "scorer": scorer,
            "hypothesis": hypothesis,
            "pos_threshold": pos_threshold,
            "neg_threshold": neg_threshold,
            "min_per_side": min_per_side,
        },
        "out": str(out_dir),
        "agent_next_actions": [
            next_action(
                action_id="write_candidate_prompts",
                title=f"Write {n} positive + {n} negative candidate prompts as JSONL",
                instruction=(
                    f"Write {candidates_path}: one JSON object per line, "
                    '{"label": "positive"|"negative", "text": "..."}. '
                    f"Positives must clearly satisfy the hypothesis ({hypothesis!r}); negatives "
                    "clearly must not. Vary length, topic, and style independently of the "
                    "criterion and share no template strings between sides."
                ),
            ),
            next_action(
                action_id="finish_compile_criterion",
                title="Score, gate, and package the candidates",
                argv=[
                    "interp-lab",
                    "compile-criterion",
                    "--criterion",
                    criterion,
                    "--candidates",
                    str(candidates_path),
                    "--out",
                    str(out_dir),
                ],
                requires=[f"candidates JSONL at {candidates_path}"],
            ),
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    request_path = out_dir / "generation-request.json"
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
    request["request_path"] = str(request_path)
    return request


# ----------------------------------------------------------------- candidates


def _load_candidates(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            line_label = f"{path}:{line_number}"
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{line_label}: invalid JSON: {exc.msg}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"{line_label}: each candidate line must be a JSON object")
            label = str(data.get("label", "")).strip().lower()
            if label not in {"positive", "negative"}:
                raise ValueError(f"{line_label}: label must be 'positive' or 'negative'")
            text = str(data.get("text", "")).strip()
            if not text:
                raise ValueError(f"{line_label}: missing text")
            rows.append({"label": label, "text": text})
    return rows


# -------------------------------------------------------------------- outputs


def _write_prompts_jsonl(
    path: Path,
    *,
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    score_source: str,
) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for label, items in (("positive", positives), ("negative", negatives)):
            for index, item in enumerate(items, start=1):
                handle.write(
                    json.dumps(
                        {
                            "prompt_id": f"compiled-{label}-{index:03d}",
                            "text": item["text"],
                            "criterion_score": item["score"],
                            "criterion_score_source": score_source,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    # Re-read through the real scored-prompt loader as a final format check.
    load_prompt_records(path)
    return path


def _write_preset_json(
    path: Path,
    *,
    criterion: str,
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    hypothesis: str,
    scorer_id: str,
    generator: str,
) -> Path:
    name = _preset_name(criterion)
    preset = {
        "schema_version": "interp-lab.criterion_lab_preset.v1",
        "name": name,
        "label": f"Compiled: {criterion}",
        "criterion": criterion,
        "positive_prompts": [item["text"] for item in positives],
        "negative_prompts": [item["text"] for item in negatives],
        "defaults": {"workflow": "discovery", "layers": "all", "training_preset": "minimal"},
        # Compile provenance, cached for re-scoring. The Criterion Lab preset
        # loader tolerates (and preserves on disk) this extra block.
        "criterion_compile": {
            "schema_version": CRITERION_COMPILE_SCHEMA,
            "hypothesis": hypothesis,
            "scorer": scorer_id,
            "generator": generator,
        },
    }
    path.write_text(json.dumps(preset, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _compile_next_actions(
    criterion: str,
    *,
    prompts_path: Path,
    hypothesis: str,
    scorer: str,
) -> list[dict[str, Any]]:
    return [
        next_action(
            action_id="inspect_compiled_dataset",
            title="Export activations over the compiled dataset, then inspect against the criterion",
            argv=[
                "interp-lab",
                "inspect",
                "--model",
                "<model>",
                "--criterion",
                criterion,
                "--backend",
                "records",
                "--records",
                "<activation-records.jsonl>",
                "--out",
                "<report-dir>",
            ],
            requires=[
                f"activation records exported over {prompts_path} "
                "(e.g. export-hf-records / export-gguf-records --dataset)"
            ],
        ),
        next_action(
            action_id="rescore_compiled_prompts",
            title="Re-score the compiled dataset with the cached hypothesis (drift check)",
            argv=[
                "interp-lab",
                "score-prompts",
                "--dataset",
                str(prompts_path),
                "--criterion",
                criterion,
                "--hypothesis",
                hypothesis,
                "--scorer",
                scorer,
                "--out",
                "<rescored-prompts.jsonl>",
                "--json",
            ],
        ),
    ]


# -------------------------------------------------------------------- helpers


def _trim_lowest_margin(
    items: list[dict[str, Any]],
    keep: int,
    *,
    margin_key,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the ``keep`` highest-margin items (stable order), return (kept, trimmed)."""
    indexed = sorted(range(len(items)), key=lambda i: (margin_key(items[i]["score"]), -i))
    trim_indexes = set(indexed[: max(0, len(items) - keep)])
    kept = [item for index, item in enumerate(items) if index not in trim_indexes]
    trimmed = [
        {**item, "label": item.get("label", "")} for index, item in enumerate(items) if index in trim_indexes
    ]
    return kept, trimmed


def _short_topic(text: str) -> str:
    tokens = content_tokens(text)
    if not tokens:
        return "the criterion"
    return " ".join(tokens[:8])


def _preset_name(criterion: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", criterion.lower()).strip("-")
    return f"compiled-{slug[:48]}" if slug else "compiled-criterion"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _distribution(scores: list[float]) -> dict[str, float | None]:
    if not scores:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(scores),
        "min": round(min(scores), 6),
        "max": round(max(scores), 6),
        "mean": _mean(scores),
    }


def _optional_import(name: str, message: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(message) from exc
