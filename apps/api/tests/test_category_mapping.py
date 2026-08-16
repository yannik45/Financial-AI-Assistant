import pytest
from financial_ai.ml.transaction_classification.core.categories import ExpenseCategory
from financial_ai.ml.transaction_classification.data.category_mapping import map_source_category


@pytest.mark.parametrize(
    ("source_category", "expected"),
    [
        ("Restaurants", ExpenseCategory.DINING),
        ("Groceries", ExpenseCategory.GROCERIES),
        ("Shopping", ExpenseCategory.SHOPPING),
        ("Transportation", ExpenseCategory.TRANSPORT),
        ("Entertainment", ExpenseCategory.ENTERTAINMENT),
        ("Utilities", ExpenseCategory.UTILITIES),
        ("Rent", ExpenseCategory.HOUSING),
        ("Mortgage", ExpenseCategory.HOUSING),
        ("Subscription", ExpenseCategory.OTHER),
        ("Healthcare", ExpenseCategory.HEALTHCARE),
        ("Insurance", ExpenseCategory.INSURANCE),
        ("Travel", ExpenseCategory.TRAVEL),
        ("Education", ExpenseCategory.EDUCATION),
    ],
)
def test_map_source_category_returns_target_category(source_category, expected):
    assert map_source_category(source_category) is expected


@pytest.mark.parametrize(
    "source_category",
    ["Income", "Transfer", "Fees", "Personal Care"],
)
def test_map_source_category_excludes_out_of_scope_categories(source_category):
    assert map_source_category(source_category) is None


def test_map_source_category_ignores_whitespace_and_case():
    assert map_source_category("  rEnT  ") is ExpenseCategory.HOUSING


def test_map_source_category_rejects_unknown_category():
    with pytest.raises(ValueError, match="Unknown source category"):
        map_source_category("Cryptocurrency")
