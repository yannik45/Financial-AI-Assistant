import pandas as pd
from financial_ai.ml.transaction_classification.category_model import (
    train_tfidf_category_classifier,
    train_word_char_tfidf_category_classifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


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


def test_train_word_char_tfidf_category_classifier_builds_expected_pipeline():
    model = train_word_char_tfidf_category_classifier(make_training_data())

    assert isinstance(model.named_steps["features"], FeatureUnion)
    feature_models = dict(model.named_steps["features"].transformer_list)
    assert feature_models["char_tfidf"].analyzer == "char_wb"
    assert feature_models["char_tfidf"].ngram_range == (3, 5)
    assert feature_models["word_tfidf"].analyzer == "word"
    assert feature_models["word_tfidf"].ngram_range == (1, 2)
    assert model.named_steps["classifier"].class_weight == "balanced"


def test_train_word_char_tfidf_category_classifier_fits_and_predicts():
    model = train_word_char_tfidf_category_classifier(
        make_training_data(),
        random_state=123,
    )

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
    assert model.named_steps["classifier"].random_state == 123
