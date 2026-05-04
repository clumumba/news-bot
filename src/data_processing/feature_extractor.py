from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .text_preprocessor import normalize_text, split_sentences, tokenize


@dataclass
class FeatureExtractor:
    max_features: int = 20_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    vectorizer: TfidfVectorizer = field(init=False)

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            stop_words="english",
        )

    def _custom_features(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            normalized = normalize_text(text)
            tokens = tokenize(normalized)
            sentences = split_sentences(text)
            rows.append(
                [
                    len(text),
                    len(tokens),
                    len(sentences),
                    len(set(tokens)),
                    sum(1 for token in tokens if token.endswith("ing")),
                ]
            )
        return np.asarray(rows, dtype=float)

    def fit(self, texts: list[str]) -> "FeatureExtractor":
        self.vectorizer.fit([normalize_text(text) for text in texts])
        return self

    def transform(self, texts: list[str]) -> csr_matrix:
        tfidf = self.vectorizer.transform([normalize_text(text) for text in texts])
        custom = csr_matrix(self._custom_features(texts))
        return hstack([tfidf, custom])

    def fit_transform(self, texts: list[str]) -> csr_matrix:
        self.fit(texts)
        return self.transform(texts)

