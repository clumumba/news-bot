from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nlp_pipeline.modeling import load_model
from nlp_pipeline.newsbot import NewsBotSystem
from nlp_pipeline.preprocessing import normalize_text


@dataclass
class InferenceService:
    model_path: Path
    corpus_path: Path | None = None

    def __post_init__(self) -> None:
        self.pipeline = load_model(self.model_path)
        self.newsbot = NewsBotSystem(classifier_pipeline=self.pipeline)
        self.labels = list(getattr(self.pipeline.named_steps["classifier"], "classes_", []))
        if self.corpus_path and self.corpus_path.exists():
            with self.corpus_path.open("r", encoding="utf-8") as handle:
                corpus = json.load(handle)
            if isinstance(corpus, list):
                self.newsbot.ingest_articles(corpus)

    def predict_one(self, text: str) -> dict[str, Any]:
        clean_text = normalize_text(text)
        probabilities = self.pipeline.predict_proba([clean_text])[0]
        best_index = int(np.argmax(probabilities))
        label = self.labels[best_index]
        ranking = sorted(
            zip(self.labels, probabilities, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )
        return {
            "label": label,
            "confidence": float(probabilities[best_index]),
            "probabilities": {
                label_name: float(probability)
                for label_name, probability in zip(self.labels, probabilities, strict=False)
            },
            "alternatives": [
                {"label": label_name, "confidence": float(probability)}
                for label_name, probability in ranking[1:3]
            ],
        }

    def predict_many(self, texts: list[str]) -> list[dict[str, Any]]:
        return [self.predict_one(text) for text in texts]

    def analyze_article(self, text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.newsbot.analyze_article(text, metadata=metadata)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.newsbot.search(query, top_k=top_k)

    def chat(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.newsbot.process_query(query, conversation_context=context)
