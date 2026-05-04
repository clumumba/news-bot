from __future__ import annotations

from dataclasses import dataclass, field

from nlp_pipeline.newsbot import MultilingualProcessor


@dataclass
class LanguageDetector:
    processor: MultilingualProcessor = field(default_factory=MultilingualProcessor)

    def detect(self, text: str):
        return self.processor.detect_language(text)
