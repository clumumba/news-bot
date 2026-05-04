from __future__ import annotations

from dataclasses import dataclass, field

from nlp_pipeline.newsbot import MultilingualProcessor


@dataclass
class CrossLingualAnalyzer:
    processor: MultilingualProcessor = field(default_factory=MultilingualProcessor)

    def analyze(self, articles_by_language):
        return self.processor.analyze_cross_lingual(articles_by_language)
