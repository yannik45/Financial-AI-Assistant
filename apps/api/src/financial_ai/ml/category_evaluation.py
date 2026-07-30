from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


class CategoryClassifier(Protocol):
    def predict(self, descriptions: pd.Series) -> np.ndarray: ...


@dataclass(frozen=True)
class CategoryEvaluation:
    accuracy: float
    macro_f1: float


def evaluate_category_classifier(
    model: CategoryClassifier,
    evaluation_data: pd.DataFrame,
) -> CategoryEvaluation:
    """Evaluate a fitted category classifier on labeled data."""

    predictions = model.predict(evaluation_data["description"])
    actual_labels = evaluation_data["target_category"]

    accuracy = float(accuracy_score(actual_labels, predictions))
    macro_f1 = float(
        f1_score(
            actual_labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    return CategoryEvaluation(
        accuracy=accuracy,
        macro_f1=macro_f1,
    )
