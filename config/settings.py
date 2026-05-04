from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from src.nlp_pipeline.config import PipelineConfig, load_config as load_pipeline_config
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

class AppSettings(BaseModel):
    project_name: str = "newsbot-intelligence-system"
    config_path: Path = Path("configs/pipeline.yaml")
    data_root: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    models_dir: Path = Path("data/models")
    results_dir: Path = Path("data/results")
    docs_dir: Path = Path("docs")
    reports_dir: Path = Path("reports")
    api_keys_template: Path = Path("ANTHROPIC_API_KEY", " ")
    notebook_dir: Path = Path("notebooks")

    def load_pipeline(self) -> PipelineConfig:
        return load_pipeline_config(self.config_path)


def load_settings(config_path: str | Path | None = None) -> AppSettings:
    settings = AppSettings()
    if config_path is not None:
        settings.config_path = Path(config_path)
    return settings

