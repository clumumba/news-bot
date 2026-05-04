from __future__ import annotations

from dataclasses import dataclass, field

from nlp_pipeline.newsbot import IntelligentSummarizer


@dataclass
class Summarizer:
    engine: IntelligentSummarizer = field(default_factory=IntelligentSummarizer)

    def summarize(self, article_text: str, summary_type: str = "balanced") -> str:
        return self.engine.summarize_article(article_text, summary_type=summary_type)

    def summarize_many(self, articles, focus_topic: str | None = None) -> str:
        return self.engine.summarize_multiple_articles(articles, focus_topic=focus_topic)
