from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


class CategoryClassifier(Protocol):
    def predict(self, descriptions: pd.Series) -> np.ndarray: ...


@dataclass(frozen=True)
class CategoryMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class CategoryEvaluation:
    accuracy: float
    macro_f1: float
    per_category: tuple[CategoryMetrics, ...] = ()
    confusion_matrix: tuple[tuple[int, ...], ...] = ()


def evaluate_category_classifier(
    model: CategoryClassifier,
    evaluation_data: pd.DataFrame,
) -> CategoryEvaluation:
    """Evaluate a fitted category classifier on labeled data."""

    predictions = model.predict(evaluation_data["description"])
    actual_labels = evaluation_data["target_category"]
    labels = sorted(set(actual_labels) | set(predictions))

    accuracy = float(accuracy_score(actual_labels, predictions))
    macro_f1 = float(
        f1_score(
            actual_labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    precisions, recalls, f1_scores, supports = precision_recall_fscore_support(
        actual_labels,
        predictions,
        labels=labels,
        zero_division=0,
    )

    per_category = tuple(
        CategoryMetrics(
            label=label,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            support=int(support),
        )
        for label, precision, recall, f1, support in zip(
            labels,
            precisions,
            recalls,
            f1_scores,
            supports,
            strict=True,
        )
    )

    matrix = confusion_matrix(
        actual_labels,
        predictions,
        labels=labels,
    )

    converted_rows = []

    for row in matrix:
        converted_row = []

        for value in row:
            converted_row.append(int(value))

        converted_rows.append(tuple(converted_row))

    matrix = tuple(converted_rows)

    return CategoryEvaluation(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_category=per_category,
        confusion_matrix=matrix,
    )
