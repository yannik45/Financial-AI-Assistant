from enum import StrEnum


class ExpenseCategory(StrEnum):
    GROCERIES = "groceries"
    DINING = "dining"
    TRANSPORT = "transport"
    HOUSING = "housing"
    UTILITIES = "utilities"
    HEALTHCARE = "healthcare"
    SHOPPING = "shopping"
    ENTERTAINMENT = "entertainment"
    TRAVEL = "travel"
    INSURANCE = "insurance"
    EDUCATION = "education"
    OTHER = "other"


def parse_expense_category(value: str) -> ExpenseCategory:
    normalized = value.strip().casefold()
    if not normalized:
        raise ValueError("Category must not be empty")
    return ExpenseCategory(normalized)
