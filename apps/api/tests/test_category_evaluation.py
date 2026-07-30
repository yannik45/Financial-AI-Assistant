import numpy as np
import pandas as pd
import pytest
from financial_ai.ml.category_evaluation import evaluate_category_classifier


class ShoppingOnlyClassifier:
    def predict(self, descriptions: pd.Series) -> np.ndarray:
        return np.array(["shopping"] * len(descriptions))


def test_evaluate_category_classifier_calculates_accuracy_and_macro_f1():
    evaluation_data = pd.DataFrame(
        {
            "description": ["SHOP A", "SHOP B", "HOTEL A", "MARKET A"],
            "target_category": ["shopping", "shopping", "travel", "groceries"],
        }
    )

    result = evaluate_category_classifier(ShoppingOnlyClassifier(), evaluation_data)

    assert result.accuracy == pytest.approx(0.5)
    assert result.macro_f1 == pytest.approx(2 / 9)
    assert [metrics.label for metrics in result.per_category] == [
        "groceries",
        "shopping",
        "travel",
    ]
    assert result.per_category[0].support == 1
    assert result.per_category[0].f1 == pytest.approx(0.0)
    assert result.per_category[1].precision == pytest.approx(0.5)
    assert result.per_category[1].recall == pytest.approx(1.0)
    assert result.per_category[1].f1 == pytest.approx(2 / 3)
    assert result.per_category[1].support == 2
    assert result.per_category[2].support == 1
    assert result.per_category[2].f1 == pytest.approx(0.0)
    assert result.confusion_matrix == (
        (0, 1, 0),
        (0, 2, 0),
        (0, 1, 0),
    )
