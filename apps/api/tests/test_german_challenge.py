from pathlib import Path

import pandas as pd
import pytest
from financial_ai.ml.categories import ExpenseCategory
from financial_ai.ml.german_challenge import validate_german_challenge_data

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GERMAN_CHALLENGE_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "evaluation"
    / "transaction_categories"
    / "german_challenge_v1.csv"
)


def make_german_challenge_data() -> pd.DataFrame:
    rows = [
        {
            "scenario_id": f"de_{category.value}_{example_number:03d}",
            "description": (
                f"SYNTHETISCHE BUCHUNG {category.value} {example_number}"
            ),
            "target_category": category.value,
            "language": "de",
            "merchant_group": f"merchant_{category.value}_{example_number}",
            "merchant_scope": "german_local",
        }
        for category in ExpenseCategory
        for example_number in range(1, 11)
    ]
    return pd.DataFrame(rows)


def test_validate_german_challenge_data_accepts_expected_schema():
    source_data = make_german_challenge_data()

    result = validate_german_challenge_data(source_data)

    pd.testing.assert_frame_equal(result, source_data)
    assert result is not source_data


@pytest.mark.parametrize(
    "columns",
    [
        ["scenario_id", "description"],
        [
            "scenario_id",
            "description",
            "target_category",
            "language",
            "merchant_group",
            "merchant_scope",
            "unexpected",
        ],
    ],
)
def test_validate_german_challenge_data_rejects_wrong_columns(columns):
    source_data = pd.DataFrame(columns=columns)

    with pytest.raises(ValueError, match="Expected German challenge columns"):
        validate_german_challenge_data(source_data)


@pytest.mark.parametrize("invalid_value", [None, "", "   "])
def test_validate_german_challenge_data_rejects_blank_descriptions(invalid_value):
    source_data = make_german_challenge_data()
    source_data.loc[0, "description"] = invalid_value

    with pytest.raises(ValueError, match="non-empty description"):
        validate_german_challenge_data(source_data)


@pytest.mark.parametrize("invalid_value", [None, "", "   "])
def test_validate_german_challenge_data_rejects_blank_scenario_ids(invalid_value):
    source_data = make_german_challenge_data()
    source_data.loc[0, "scenario_id"] = invalid_value

    with pytest.raises(ValueError, match="non-empty scenario_id"):
        validate_german_challenge_data(source_data)


def test_validate_german_challenge_data_rejects_duplicate_scenario_ids():
    source_data = make_german_challenge_data()
    source_data.loc[1, "scenario_id"] = source_data.loc[0, "scenario_id"]

    with pytest.raises(ValueError, match="unique scenario_id"):
        validate_german_challenge_data(source_data)


@pytest.mark.parametrize("invalid_language", [None, "", "en", "DE"])
def test_validate_german_challenge_data_rejects_invalid_language(invalid_language):
    source_data = make_german_challenge_data()
    source_data.loc[0, "language"] = invalid_language

    with pytest.raises(ValueError, match="language must be 'de'"):
        validate_german_challenge_data(source_data)


@pytest.mark.parametrize(
    "invalid_scope",
    [None, "", "local", "german-local", "INTERNATIONAL"],
)
def test_validate_german_challenge_data_rejects_invalid_merchant_scope(
    invalid_scope,
):
    source_data = make_german_challenge_data()
    source_data.loc[0, "merchant_scope"] = invalid_scope

    with pytest.raises(ValueError, match="valid merchant_scope"):
        validate_german_challenge_data(source_data)


@pytest.mark.parametrize(
    "invalid_category",
    [None, "", "Groceries", "income", "unknown"],
)
def test_validate_german_challenge_data_rejects_invalid_target_category(
    invalid_category,
):
    source_data = make_german_challenge_data()
    source_data.loc[0, "target_category"] = invalid_category

    with pytest.raises(ValueError, match="valid target_category"):
        validate_german_challenge_data(source_data)


def test_validate_german_challenge_data_rejects_missing_category_example():
    source_data = make_german_challenge_data().iloc[:-1].copy()

    with pytest.raises(ValueError, match="10 examples per target_category"):
        validate_german_challenge_data(source_data)


def test_validate_german_challenge_data_rejects_extra_category_example():
    source_data = make_german_challenge_data()
    extra_row = source_data.iloc[[0]].copy()
    extra_row.loc[:, "scenario_id"] = "de_groceries_011"
    extra_row.loc[:, "description"] = "SYNTHETISCHE BUCHUNG groceries 11"
    source_data = pd.concat([source_data, extra_row], ignore_index=True)

    with pytest.raises(ValueError, match="10 examples per target_category"):
        validate_german_challenge_data(source_data)


def test_german_challenge_v1_file_is_valid():
    source_data = pd.read_csv(GERMAN_CHALLENGE_PATH)

    result = validate_german_challenge_data(source_data)

    assert len(result) == 120
