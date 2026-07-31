import pandas as pd
from financial_ai.ml.category_split import CategoryDataSplits
from financial_ai.ml.multilingual_model_training import (
    build_language_balanced_training_data,
    train_and_evaluate_balanced_multilingual_validation,
    train_and_evaluate_multilingual_validation,
)
from sklearn.pipeline import Pipeline


def make_language_splits(language: str) -> CategoryDataSplits:
    train = pd.DataFrame(
        {
            "description": [
                f"{language} MARKET FOOD {number}" for number in range(6)
            ]
            + [f"{language} HOTEL TRAVEL {number}" for number in range(6)],
            "target_category": ["groceries"] * 6 + ["travel"] * 6,
            "provenance": [language] * 12,
        }
    )
    validation = pd.DataFrame(
        {
            "description": [
                f"{language} MARKET FOOD VALIDATION",
                f"{language} HOTEL TRAVEL VALIDATION",
            ],
            "target_category": ["groceries", "travel"],
            "provenance": [language] * 2,
        }
    )
    test = pd.DataFrame(
        {
            "description": [f"{language} FROZEN TEST"],
            "target_category": ["groceries"],
            "provenance": [language],
        }
    )
    return CategoryDataSplits(train=train, validation=validation, test=test)


def test_train_and_evaluate_multilingual_validation_combines_only_train_rows():
    english_splits = make_language_splits("ENGLISH")
    german_splits = make_language_splits("GERMAN")

    result = train_and_evaluate_multilingual_validation(
        english_splits,
        german_splits,
        random_state=17,
    )

    assert isinstance(result.model, Pipeline)
    assert result.english_training_rows == 12
    assert result.german_training_rows == 12
    assert result.combined_training_rows == 24
    assert sum(
        metric.support for metric in result.english_validation.per_category
    ) == len(english_splits.validation)
    assert sum(
        metric.support for metric in result.german_validation.per_category
    ) == len(german_splits.validation)


def test_build_language_balanced_training_data_matches_each_category():
    english_training_data = pd.DataFrame(
        {
            "description": [f"ENGLISH GROCERY {number}" for number in range(8)]
            + [f"ENGLISH TRAVEL {number}" for number in range(6)],
            "target_category": ["groceries"] * 8 + ["travel"] * 6,
            "source_category": ["English source"] * 14,
        }
    )
    german_training_data = pd.DataFrame(
        {
            "description": [f"GERMAN GROCERY {number}" for number in range(3)]
            + [f"GERMAN TRAVEL {number}" for number in range(4)],
            "target_category": ["groceries"] * 3 + ["travel"] * 4,
            "merchant_group": ["German merchant"] * 7,
        }
    )

    result = build_language_balanced_training_data(
        english_training_data,
        german_training_data,
        random_state=17,
    )

    assert list(result.english.columns) == ["description", "target_category"]
    assert list(result.german.columns) == ["description", "target_category"]
    assert result.english["target_category"].value_counts().to_dict() == {
        "travel": 4,
        "groceries": 3,
    }
    assert result.german["target_category"].value_counts().to_dict() == {
        "travel": 4,
        "groceries": 3,
    }
    assert len(result.combined) == 14


def test_build_language_balanced_training_data_is_deterministic():
    english_training_data = make_language_splits("ENGLISH").train
    german_training_data = make_language_splits("GERMAN").train

    first = build_language_balanced_training_data(
        english_training_data,
        german_training_data,
        random_state=17,
    )
    second = build_language_balanced_training_data(
        english_training_data.sample(frac=1, random_state=99),
        german_training_data,
        random_state=17,
    )

    pd.testing.assert_frame_equal(first.english, second.english)
    pd.testing.assert_frame_equal(first.german, second.german)
    pd.testing.assert_frame_equal(first.combined, second.combined)


def test_train_and_evaluate_balanced_multilingual_validation_uses_equal_sizes():
    english_splits = make_language_splits("ENGLISH")
    german_splits = make_language_splits("GERMAN")

    result = train_and_evaluate_balanced_multilingual_validation(
        english_splits,
        german_splits,
        random_state=17,
    )

    assert result.english_training_rows == result.german_training_rows == 12
    assert result.combined_training_rows == 24
    assert sum(
        metric.support for metric in result.english_validation.per_category
    ) == len(english_splits.validation)
    assert sum(
        metric.support for metric in result.german_validation.per_category
    ) == len(german_splits.validation)
