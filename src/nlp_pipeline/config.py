from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    train_path: Path
    validation_path: Path
    text_column: str = "text"
    text_columns: list[str] = Field(default_factory=lambda: ["headline", "content"])
    label_column: str
    timestamp_column: str | None = None
    source_column: str | None = "source"
    language_column: str | None = "language"
    drop_duplicates: bool = True
    min_text_length: int = 25
    allowed_labels: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    max_features: int = 20_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    classifier: Literal["logistic_regression"] = "logistic_regression"
    random_state: int = 42


class MonitoringConfig(BaseModel):
    service_name: str = "nlp-inference-api"
    drift_sample_size: int = 1_000


class PipelineConfig(BaseModel):
    project_name: str = "newsbot-intelligence-system"
    artifacts_dir: Path = Path("artifacts")
    data: DataConfig
    model: ModelConfig
    monitoring: MonitoringConfig = MonitoringConfig()


def load_config(config_path: str | Path) -> PipelineConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return PipelineConfig.model_validate(payload)
