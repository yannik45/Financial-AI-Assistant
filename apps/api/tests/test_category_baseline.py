import pandas as pd
from financial_ai.ml.transaction_classification.category_baseline import (
    train_majority_category_baseline,
)


def test_train_majority_category_baseline_predicts_most_frequent_category():
    training_data = pd.DataFrame(
        {
            "description": ["SHOP A", "SHOP B", "SHOP C", "HOTEL A"],
            "target_category": ["shopping", "shopping", "shopping", "travel"],
        }
    )

    model = train_majority_category_baseline(training_data)
    predictions = model.predict(pd.Series(["UNKNOWN ONE", "UNKNOWN TWO"]))

    assert predictions.tolist() == ["shopping", "shopping"]
    assert model.classes_.tolist() == ["shopping", "travel"]
