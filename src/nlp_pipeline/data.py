from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nlp_pipeline.config import DataConfig
from nlp_pipeline.preprocessing import normalize_text


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def _combine_text_columns(frame: pd.DataFrame, config: DataConfig) -> pd.Series:
    available_sources = [column for column in config.text_columns if column in frame.columns]
    if available_sources:
        combined = frame[available_sources].fillna("").astype(str).agg(" ".join, axis=1)
    elif config.text_column in frame.columns:
        combined = frame[config.text_column].fillna("").astype(str)
    else:
        raise ValueError(
            "Dataset is missing both the configured text columns "
            f"{config.text_columns} and text column {config.text_column!r}."
        )
    return combined


def load_dataset(path: Path, config: DataConfig) -> pd.DataFrame:
    frame = _read_table(path).copy()
    required_columns = {config.label_column}
    available_sources = [column for column in config.text_columns if column in frame.columns]
    if not available_sources and config.text_column not in frame.columns:
        required_columns.add(config.text_column)
    else:
        required_columns.update(available_sources)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset {path} is missing columns: {sorted(missing)}")

    frame[config.text_column] = _combine_text_columns(frame, config).map(normalize_text)
    frame = frame.dropna(subset=[config.label_column])
    frame = frame[frame[config.text_column].str.len() >= config.min_text_length]

    if config.allowed_labels:
        frame = frame[frame[config.label_column].isin(config.allowed_labels)]

    if config.drop_duplicates:
        frame = frame.drop_duplicates(subset=[config.text_column, config.label_column])

    if config.timestamp_column and config.timestamp_column in frame.columns:
        frame[config.timestamp_column] = pd.to_datetime(frame[config.timestamp_column], errors="coerce")

    if frame.empty:
        raise ValueError(f"Dataset {path} is empty after validation.")

    return frame.reset_index(drop=True)


def load_training_data(config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        load_dataset(config.train_path, config),
        load_dataset(config.validation_path, config),
    )


def dataframe_to_records(frame: pd.DataFrame) -> list[dict]:
    serializable = frame.copy()
    for column in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[column]):
            serializable[column] = serializable[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return json.loads(serializable.to_json(orient="records", date_format="iso"))
