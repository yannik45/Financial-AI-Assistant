import pandas as pd
import pytest
from financial_ai.ml.category_preprocessing import prepare_category_training_data


def test_prepare_category_training_data_maps_and_excludes_rows():
    source_data = pd.DataFrame(
        {
            "description": [
                "[debit] GREYSTAR RENT PAYMENT",
                "[debit] SAFEWAY #1197",
                "[credit] ACME CORP PAYROLL",
            ],
            "category": ["Rent", "Groceries", "Income"],
        }
    )

    result = prepare_category_training_data(source_data)

    assert result.to_dict(orient="records") == [
        {
            "description": "[debit] GREYSTAR RENT PAYMENT",
            "source_category": "Rent",
            "target_category": "housing",
        },
        {
            "description": "[debit] SAFEWAY #1197",
            "source_category": "Groceries",
            "target_category": "groceries",
        },
    ]


@pytest.mark.parametrize(
    "columns",
    [
        ["description"],
        ["category"],
        ["description", "category", "unexpected"],
    ],
)
def test_prepare_category_training_data_rejects_wrong_columns(columns):
    source_data = pd.DataFrame(columns=columns)

    with pytest.raises(ValueError, match="Expected dataset columns"):
        prepare_category_training_data(source_data)


@pytest.mark.parametrize("description", [None, "", "   "])
def test_prepare_category_training_data_rejects_missing_or_empty_descriptions(description):
    source_data = pd.DataFrame(
        {
            "description": [description],
            "category": ["Shopping"],
        }
    )

    with pytest.raises(ValueError, match="Descriptions must not be empty"):
        prepare_category_training_data(source_data)


def test_prepare_category_training_data_rejects_unknown_source_category():
    source_data = pd.DataFrame(
        {
            "description": ["[debit] EXAMPLE MERCHANT"],
            "category": ["Cryptocurrency"],
        }
    )

    with pytest.raises(ValueError, match="Unknown source category"):
        prepare_category_training_data(source_data)


def test_prepare_category_training_data_excludes_conflicting_descriptions():
    source_data = pd.DataFrame(
        {
            "description": ["AMBIGUOUS MERCHANT", "AMBIGUOUS MERCHANT", "SAFEWAY"],
            "category": ["Healthcare", "Shopping", "Groceries"],
        }
    )

    result = prepare_category_training_data(source_data)

    assert result.to_dict(orient="records") == [
        {
            "description": "SAFEWAY",
            "source_category": "Groceries",
            "target_category": "groceries",
        }
    ]


def test_prepare_category_training_data_removes_duplicate_descriptions():
    source_data = pd.DataFrame(
        {
            "description": ["AMBIGUOUS MERCHANT", "AMBIGUOUS MERCHANT", "BOOKINGCOM"],
            "category": ["Shopping", "Shopping", "Travel"],
        }
    )

    result = prepare_category_training_data(source_data)

    assert len(result) == 2
    assert result["description"].tolist() == [
        "AMBIGUOUS MERCHANT",
        "BOOKINGCOM",
    ]
    assert result.index.tolist() == [0, 1]
