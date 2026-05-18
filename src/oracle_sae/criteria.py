from __future__ import annotations

from oracle_sae.schema import Criterion
from oracle_sae.text_vectors import content_tokens


class CriterionCompiler:
    def compile(self, text: str) -> Criterion:
        raise NotImplementedError


class HeuristicCriterionCompiler(CriterionCompiler):
    """Small local compiler used until an LLM-backed compiler is configured."""

    def compile(self, text: str) -> Criterion:
        topic = _short_topic(text)
        return Criterion(
            text=text,
            positive_examples=[
                f"The context clearly expresses {topic}.",
                f"The model is internally tracking {topic}.",
                f"A response would change if {topic} were amplified.",
            ],
            negative_examples=[
                f"The context is unrelated to {topic}.",
                f"The model can answer without representing {topic}.",
                f"Amplifying unrelated details leaves {topic} unchanged.",
            ],
            scoring_notes=[
                "Prefer features with replicated behavior across paraphrases.",
                "Prioritize causal effects over text-label similarity.",
                "Treat natural-language explanations as hypotheses to validate.",
            ],
        )


def _short_topic(text: str) -> str:
    tokens = content_tokens(text)
    if not tokens:
        return "the criterion"
    return " ".join(tokens[:8])
