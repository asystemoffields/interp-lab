from __future__ import annotations

import hashlib
import math
import re

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "being",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if len(token) > 2 and token not in STOPWORDS]


def hash_text_vector(text: str, *, dimensions: int = 128) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    length = math.sqrt(sum(value * value for value in vector))
    if length == 0:
        return vector
    return [value / length for value in vector]
