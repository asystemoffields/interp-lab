"""Pluggable text embedding for fingerprints and report similarity.

The default embedder is the dependency-free lexical hash (``hash_text_vector``):
deterministic, offline, and comparable across versions, but it matches shared
*words*, not meaning. Opt into a real semantic embedder to make cross-model
matching, text-pivot, and feature search compare *concepts* instead of vocabulary.

Selection (in order of precedence):

1. An embedder set programmatically via :func:`set_text_embedder`.
2. The ``name`` passed to :func:`configure_text_embedder` (e.g. the ``--text-embedder`` CLI flag).
3. The ``INTERP_LAB_TEXT_EMBEDDER`` environment variable.
4. The lexical hash default.

Accepted names: ``hash`` (default, lexical), ``minilm`` / ``semantic`` /
``sentence-transformers`` (local MiniLM via the ``[embeddings]`` extra), or
``st:<model-name>`` for any sentence-transformers model.

Every embedder exposes a stable ``id`` and ``dimensions``; the id is stamped into
each fingerprint so vectors produced by different embedders are never silently
compared (see ``matching.fingerprint_similarity``).
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from interp_lab.text_vectors import hash_text_vector

HASH_EMBEDDER_ID = "hash-v1"
_HASH_DIMENSIONS = 128

_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_SEMANTIC_ALIASES = {"minilm", "semantic", "st", "sentence-transformers", "sentence_transformers"}


@runtime_checkable
class TextEmbedder(Protocol):
    id: str
    dimensions: int

    def embed(self, text: str) -> list[float]:
        ...


class HashEmbedder:
    """Deterministic, dependency-free lexical embedder (the default)."""

    id = HASH_EMBEDDER_ID

    def __init__(self, dimensions: int = _HASH_DIMENSIONS) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        return hash_text_vector(text, dimensions=self.dimensions)


class SentenceTransformerEmbedder:
    """Local semantic embedder backed by sentence-transformers (the ``[embeddings]`` extra)."""

    def __init__(self, model_name: str = _MINILM_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra installed.
            raise RuntimeError(
                "Semantic text embedding needs the optional dependency. "
                'Install it with: pip install "interp-lab[embeddings]"'
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())
        self.id = f"st-{model_name.rsplit('/', 1)[-1]}"

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self.dimensions
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return [float(value) for value in vector]


_active: TextEmbedder | None = None


def _build_embedder(name: str | None) -> TextEmbedder:
    label = (name or "").strip().lower()
    if label in {"", "hash", "lexical", "hashing"}:
        return HashEmbedder()
    if label in _SEMANTIC_ALIASES:
        return SentenceTransformerEmbedder()
    if ":" in label:
        prefix, model_name = (name or "").split(":", 1)
        if prefix.strip().lower() in _SEMANTIC_ALIASES:
            return SentenceTransformerEmbedder(model_name.strip())
    raise ValueError(
        f"Unknown text embedder {name!r}. Use 'hash', 'minilm', or 'st:<model-name>'."
    )


def configure_text_embedder(name: str | None = None) -> TextEmbedder:
    """Select the active embedder by name, falling back to the env var, then hash.

    Pass ``None`` (the default) to honor ``INTERP_LAB_TEXT_EMBEDDER`` or keep the
    lexical default. Called once at CLI startup and usable from the Python API.
    """

    global _active
    chosen = name if name is not None else os.environ.get("INTERP_LAB_TEXT_EMBEDDER")
    _active = _build_embedder(chosen)
    return _active


def set_text_embedder(embedder: TextEmbedder) -> None:
    """Install a custom embedder instance (advanced/programmatic use)."""

    global _active
    _active = embedder


def reset_text_embedder() -> None:
    """Restore the lexical hash default (used by tests)."""

    global _active
    _active = None


def active_embedder() -> TextEmbedder:
    global _active
    if _active is None:
        _active = _build_embedder(os.environ.get("INTERP_LAB_TEXT_EMBEDDER"))
    return _active


def active_embedder_id() -> str:
    return active_embedder().id


def embed_text(text: str) -> list[float]:
    """Embed ``text`` with the active embedder (lexical hash unless configured)."""

    return active_embedder().embed(text)
