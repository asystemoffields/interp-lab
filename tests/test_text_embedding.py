from __future__ import annotations

import importlib.util

import pytest

from interp_lab.matching import fingerprint_similarity
from interp_lab.schema import FeatureFingerprint
from interp_lab.text_embedding import (
    HASH_EMBEDDER_ID,
    active_embedder_id,
    configure_text_embedder,
    embed_text,
    reset_text_embedder,
)

_HAS_ST = importlib.util.find_spec("sentence_transformers") is not None


@pytest.fixture(autouse=True)
def _restore_default_embedder():
    reset_text_embedder()
    yield
    reset_text_embedder()


def _fingerprint(text_vector, embedder):
    return FeatureFingerprint(
        feature_id="f",
        model="m",
        layer=0,
        text="t",
        text_vector=text_vector,
        activation_signature=[1.0, 0.0],
        decoder_signature=[1.0, 0.0],
        causal_vector=[0.5, 0.5],
        text_embedder=embedder,
    )


def test_default_embedder_is_lexical_hash():
    configure_text_embedder(None)
    assert active_embedder_id() == HASH_EMBEDDER_ID
    assert len(embed_text("a measurement in meters")) == 128


def test_env_var_selects_embedder(monkeypatch):
    monkeypatch.setenv("INTERP_LAB_TEXT_EMBEDDER", "hash")
    configure_text_embedder(None)
    assert active_embedder_id() == HASH_EMBEDDER_ID


def test_unknown_embedder_name_is_rejected():
    with pytest.raises(ValueError):
        configure_text_embedder("does-not-exist")


def test_matching_drops_text_for_embedder_mismatch():
    # Same content, but vectors come from different embedders / dimensions.
    left = _fingerprint([1.0, 0.0, 0.0], "st-all-MiniLM-L6-v2")
    right = _fingerprint([0.1] * 128, HASH_EMBEDDER_ID)
    score, components = fingerprint_similarity(left, right)
    assert "text" not in components
    assert components["text_embedder_mismatch"] == 1.0
    # Text is dropped and the remaining weights (activation/decoder/causal) are
    # renormalized, so the score reflects only the comparable components.
    assert score == pytest.approx(0.846154, abs=1e-5)


def test_matching_uses_text_when_embedders_match():
    left = _fingerprint([1.0, 0.0], HASH_EMBEDDER_ID)
    right = _fingerprint([1.0, 0.0], HASH_EMBEDDER_ID)
    _, components = fingerprint_similarity(left, right)
    assert "text" in components
    assert "text_embedder_mismatch" not in components


@pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers ([embeddings] extra) not installed")
def test_minilm_is_semantic_not_lexical():
    # Two phrasings of the same concept that share almost no words.
    a = "a unit of physical measurement such as meters or kilograms"
    b = "how long, heavy, or hot something is, expressed in standard units"
    unrelated = "a friendly greeting written to welcome a new coworker"

    from interp_lab.math_utils import cosine
    from interp_lab.text_vectors import hash_text_vector

    # Lexical hashing barely connects the paraphrases.
    lexical = (cosine(hash_text_vector(a), hash_text_vector(b)) + 1.0) / 2.0

    configure_text_embedder("minilm")
    assert active_embedder_id().startswith("st-")
    semantic_related = (cosine(embed_text(a), embed_text(b)) + 1.0) / 2.0
    semantic_unrelated = (cosine(embed_text(a), embed_text(unrelated)) + 1.0) / 2.0

    # Semantic embedding recognizes the paraphrase far better than lexical hashing,
    # and still separates the genuinely unrelated sentence.
    assert semantic_related > lexical
    assert semantic_related > semantic_unrelated
