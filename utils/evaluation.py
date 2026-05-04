from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score


@dataclass
class Evaluator:
    def evaluate_classification(self, y_true, y_pred) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        }

    def evaluate_summary(self, original_text: str, summary: str) -> dict[str, float]:
        return {
            "compression_ratio": float(len(summary) / max(len(original_text), 1)),
        }

