from __future__ import annotations

from dataclasses import dataclass

from nlp_pipeline.preprocessing import extract_keywords


@dataclass
class ContentGenerator:
    def generate_headlines(self, article_text: str) -> list[str]:
        keywords = extract_keywords(article_text, top_k=5)
        if not keywords:
            return []
        return [
            " ".join(word.capitalize() for word in keywords[:3]),
            " ".join(word.capitalize() for word in keywords[:4]),
        ]

    def generate_brief(self, article_text: str) -> str:
        keywords = extract_keywords(article_text, top_k=6)
        return "Key themes: " + ", ".join(keywords) if keywords else "No strong themes detected."

