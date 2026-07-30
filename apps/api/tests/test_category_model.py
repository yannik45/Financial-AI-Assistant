import pandas as pd
from financial_ai.ml.category_model import train_tfidf_category_classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def make_training_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "description": [
                "SAFEWAY GROCERY MARKET",
                "ALDI GROCERY STORE",
                "WHOLE FOODS SUPERMARKET",
                "LOCAL SUPERMARKET PAYMENT",
                "CITY HOTEL BOOKING",
                "AIRLINE TICKET PAYMENT",
                "BEACH HOTEL RESERVATION",
                "INTERNATIONAL AIRWAYS TICKET",
                "MONTHLY APARTMENT RENT",
                "MORTGAGE PAYMENT",
                "PROPERTY RENT PAYMENT",
                "HOME MORTGAGE INSTALLMENT",
            ],
            "target_category": [
                "groceries",
                "groceries",
                "groceries",
                "groceries",
                "travel",
                "travel",
                "travel",
                "travel",
                "housing",
                "housing",
                "housing",
                "housing",
            ],
        }
    )


def test_train_tfidf_category_classifier_builds_expected_pipeline():
    model = train_tfidf_category_classifier(make_training_data())

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["tfidf"], TfidfVectorizer)
    assert model.named_steps["tfidf"].analyzer == "char_wb"
    assert model.named_steps["tfidf"].ngram_range == (3, 5)
    assert isinstance(model.named_steps["classifier"], LogisticRegression)
    assert model.named_steps["classifier"].class_weight == "balanced"
    assert model.named_steps["classifier"].random_state == 42


def test_train_tfidf_category_classifier_fits_and_predicts():
    model = train_tfidf_category_classifier(make_training_data(), random_state=123)

    predictions = model.predict(
        pd.Series(
            [
                "GROCERY SUPERMARKET",
                "AIRWAYS HOTEL BOOKING",
                "MONTHLY RENT PAYMENT",
            ]
        )
    )

    assert predictions.tolist() == ["groceries", "travel", "housing"]
    assert model.classes_.tolist() == ["groceries", "housing", "travel"]
    assert model.named_steps["classifier"].random_state == 123
