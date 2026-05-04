from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from nlp_pipeline.config import PipelineConfig
from nlp_pipeline.data import dataframe_to_records, load_training_data
from nlp_pipeline.modeling import save_model, train_model
from nlp_pipeline.monitoring import build_drift_snapshot, save_json

LOGGER = logging.getLogger(__name__)


def train_pipeline(config: PipelineConfig) -> dict:
    artifacts_dir = config.artifacts_dir
    model_path = artifacts_dir / "model.joblib"
    metrics_path = artifacts_dir / "metrics.json"
    report_path = artifacts_dir / "classification_report.json"
    drift_path = artifacts_dir / "drift_snapshot.json"
    corpus_path = artifacts_dir / "news_corpus.json"
    manifest_path = artifacts_dir / "manifest.json"

    train_df, validation_df = load_training_data(config.data)
    training_artifacts = train_model(
        train_df=train_df,
        validation_df=validation_df,
        text_column=config.data.text_column,
        label_column=config.data.label_column,
        config=config.model,
    )
    save_model(training_artifacts.pipeline, model_path)
    save_json(training_artifacts.metrics, metrics_path)
    save_json(training_artifacts.report, report_path)

    corpus_df = pd.concat(
        [train_df.assign(dataset="train"), validation_df.assign(dataset="validation")],
        ignore_index=True,
    )
    corpus_records = dataframe_to_records(corpus_df)
    save_json(corpus_records, corpus_path)

    drift_snapshot = build_drift_snapshot(
        reference_df=train_df,
        current_df=validation_df.head(config.monitoring.drift_sample_size),
        text_column=config.data.text_column,
        label_column=config.data.label_column,
    )
    save_json(drift_snapshot, drift_path)

    manifest = {
        "project_name": config.project_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "drift_path": str(drift_path),
        "corpus_path": str(corpus_path),
        "metrics": training_artifacts.metrics,
        "dataset": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "labels": sorted({str(label) for label in train_df[config.data.label_column].unique()}),
        },
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    LOGGER.info("Training completed with metrics=%s", training_artifacts.metrics)
    return manifest
