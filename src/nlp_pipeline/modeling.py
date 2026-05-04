from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from nlp_pipeline.config import ModelConfig


@dataclass
class TrainingArtifacts:
    pipeline: Pipeline
    metrics: dict[str, float]
    report: dict


def build_model(config: ModelConfig) -> Pipeline:
    if config.classifier != "logistic_regression":
        raise ValueError(f"Unsupported classifier: {config.classifier}")

    return Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    max_features=config.max_features,
                    ngram_range=config.ngram_range,
                    min_df=config.min_df,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def train_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    text_column: str,
    label_column: str,
    config: ModelConfig,
) -> TrainingArtifacts:
    model = build_model(config)
    model.fit(train_df[text_column], train_df[label_column])

    predictions = model.predict(validation_df[text_column])
    metrics = {
        "accuracy": float(accuracy_score(validation_df[label_column], predictions)),
        "macro_f1": float(f1_score(validation_df[label_column], predictions, average="macro")),
    }
    report = classification_report(
        validation_df[label_column],
        predictions,
        output_dict=True,
        zero_division=0,
    )
    return TrainingArtifacts(pipeline=model, metrics=metrics, report=report)


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)

###mpya
class NewsClassifier:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.pipeline = None

    def train(self, texts: pd.Series, labels: pd.Series) -> None:
        self.pipeline = build_model(self.config)
        self.pipeline.fit(texts, labels)

    def predict_with_confidence(self, text: str) -> dict:
        if self.pipeline is None:
            raise ValueError("Model not trained")
        probabilities = self.pipeline.predict_proba([text])[0]
        best_index = int(probabilities.argmax())
        labels = list(self.pipeline.named_steps["classifier"].classes_)
        return {
            "label": labels[best_index],
            "confidence": float(probabilities[best_index]),
        }
