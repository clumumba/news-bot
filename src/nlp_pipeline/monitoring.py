from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


REQUEST_COUNT = Counter(
    "nlp_requests_total",
    "Total inference requests.",
    labelnames=("endpoint", "status"),
)
REQUEST_LATENCY = Histogram(
    "nlp_request_latency_seconds",
    "Request latency for inference endpoints.",
    labelnames=("endpoint",),
)
PREDICTION_COUNT = Counter(
    "nlp_predictions_total",
    "Predictions emitted by label.",
    labelnames=("label",),
)


class RequestTimer:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.started = 0.0

    def __enter__(self) -> "RequestTimer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        REQUEST_LATENCY.labels(endpoint=self.endpoint).observe(time.perf_counter() - self.started)


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


def record_prediction(label: str) -> None:
    PREDICTION_COUNT.labels(label=label).inc()


def record_request(endpoint: str, status: str) -> None:
    REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()


def build_drift_snapshot(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> dict:
    reference_text_len = reference_df[text_column].str.len()
    current_text_len = current_df[text_column].str.len()

    return {
        "reference_rows": int(len(reference_df)),
        "current_rows": int(len(current_df)),
        "reference_avg_text_length": float(reference_text_len.mean()),
        "current_avg_text_length": float(current_text_len.mean()),
        "avg_text_length_delta": float(current_text_len.mean() - reference_text_len.mean()),
        "reference_label_distribution": reference_df[label_column].value_counts(normalize=True).to_dict(),
        "current_label_distribution": current_df[label_column].value_counts(normalize=True).to_dict(),
    }


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
