import pytest
from financial_ai.ml.transaction_classification.core.categories import (
    ExpenseCategory,
    parse_expense_category,
)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("groceries", ExpenseCategory.GROCERIES),
        (" Groceries ", ExpenseCategory.GROCERIES),
        ("TRANSPORT", ExpenseCategory.TRANSPORT),
        ("healthCARE", ExpenseCategory.HEALTHCARE),
    ],
)
def test_parse_expense_category_normalizes_valid_values(raw_value, expected):
    assert parse_expense_category(raw_value) is expected


@pytest.mark.parametrize("raw_value", ["", "   ", "\t\n"])
def test_parse_expense_category_rejects_empty_values(raw_value):
    with pytest.raises(ValueError, match="Category must not be empty"):
        parse_expense_category(raw_value)


def test_parse_expense_category_rejects_unknown_values():
    with pytest.raises(ValueError, match="is not a valid ExpenseCategory"):
        parse_expense_category("cars")


def test_expense_category_contains_the_version_one_labels():
    assert {category.value for category in ExpenseCategory} == {
        "groceries",
        "dining",
        "transport",
        "housing",
        "utilities",
        "healthcare",
        "shopping",
        "entertainment",
        "travel",
        "insurance",
        "education",
        "other",
    }
