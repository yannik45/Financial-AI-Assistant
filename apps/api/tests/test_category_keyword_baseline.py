import pandas as pd
from financial_ai.ml.transaction_classification.category_keyword_baseline import (
    KeywordCategoryClassifier,
)


def test_keyword_classifier_matches_keywords_case_insensitively():
    model = KeywordCategoryClassifier()

    predictions = model.predict(
        pd.Series(
            [
                "MONTHLY RENT PAYMENT",
                "City Water Utility Bill",
                "airport TAXI service",
            ]
        )
    )

    assert predictions.tolist() == ["housing", "utilities", "transport"]


def test_keyword_classifier_uses_first_matching_rule():
    model = KeywordCategoryClassifier()

    predictions = model.predict(pd.Series(["HOTEL RESTAURANT PAYMENT"]))

    assert predictions.tolist() == ["travel"]


def test_keyword_classifier_falls_back_to_other():
    model = KeywordCategoryClassifier()

    predictions = model.predict(pd.Series(["UNKNOWN MERCHANT 12345"]))

    assert predictions.tolist() == ["other"]
