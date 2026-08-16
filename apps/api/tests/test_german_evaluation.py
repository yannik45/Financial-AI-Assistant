import numpy as np
import pandas as pd
from financial_ai.ml.transaction_classification.core.categories import ExpenseCategory
from financial_ai.ml.transaction_classification.evaluation.german_evaluation import (
    evaluate_german_challenge,
)


class CategoryEchoClassifier:
    def predict(self, descriptions: pd.Series) -> np.ndarray:
        return descriptions.str.rsplit(" ", n=1).str[-1].to_numpy()


def make_challenge_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": f"de_{category.value}_{number:03d}",
                "description": f"SYNTHETIC DESCRIPTION {category.value}",
                "target_category": category.value,
                "language": "de",
                "merchant_group": f"merchant_{category.value}_{number}",
                "merchant_scope": ("german_local" if number <= 5 else "international"),
            }
            for category in ExpenseCategory
            for number in range(1, 11)
        ]
    )


def test_evaluate_german_challenge_reports_overall_and_scope_slices():
    result = evaluate_german_challenge(
        CategoryEchoClassifier(),
        make_challenge_data(),
    )

    assert result.overall.accuracy == 1.0
    assert result.overall.macro_f1 == 1.0
    assert sum(metric.support for metric in result.overall.per_category) == 120
    assert result.german_local.accuracy == 1.0
    assert sum(metric.support for metric in result.german_local.per_category) == 60
    assert result.international.accuracy == 1.0
    assert sum(metric.support for metric in result.international.per_category) == 60
