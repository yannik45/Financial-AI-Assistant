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
