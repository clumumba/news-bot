from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_MULTISPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = _NON_WORD.sub(" ", normalized)
    normalized = _MULTISPACE.sub(" ", normalized)
    return normalized.strip()


def split_sentences(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    return [sentence.strip() for sentence in _SENTENCE_SPLIT.split(cleaned) if sentence.strip()]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


def extract_keywords(text: str, top_k: int = 8) -> list[str]:
    tokens = [token for token in tokenize(text) if token not in ENGLISH_STOP_WORDS and len(token) > 2]
    if not tokens:
        return []
    counts = Counter(tokens)
    return [token for token, _ in counts.most_common(top_k)]
