from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from nlp_pipeline.data import load_dataset


@dataclass
class DataValidator:
    required_columns: set[str] = field(default_factory=set)
    min_text_length: int = 25

    def validate_frame(self, frame: pd.DataFrame, text_column: str, label_column: str) -> pd.DataFrame:
        missing = {text_column, label_column} - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        cleaned = frame.dropna(subset=[text_column, label_column]).copy()
        cleaned[text_column] = cleaned[text_column].astype(str)
        cleaned = cleaned[cleaned[text_column].str.len() >= self.min_text_length]
        if cleaned.empty:
            raise ValueError("Dataset is empty after validation.")
        return cleaned.reset_index(drop=True)

    def validate_records(self, records: Iterable[dict], text_column: str = "text", label_column: str = "label") -> pd.DataFrame:
        return self.validate_frame(pd.DataFrame.from_records(list(records)), text_column, label_column)

    def load_and_validate(self, path: str | Path, config) -> pd.DataFrame:
        return load_dataset(Path(path), config)

