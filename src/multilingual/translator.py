from __future__ import annotations

from dataclasses import dataclass, field

from nlp_pipeline.newsbot import MultilingualProcessor


@dataclass
class Translator:
    processor: MultilingualProcessor = field(default_factory=MultilingualProcessor)

    def translate(self, text: str, target_language: str = "en"):
        return self.processor.translate_text(text, target_language=target_language)
