from dataclasses import dataclass
from enum import StrEnum

from financial_ai.ml.categories import ExpenseCategory

TAXONOMY_VERSION = "transaction-categories-v1"


class TransactionCategory(StrEnum):
    INCOME = "income"
    INVESTMENTS = "investments"
    FEES = "fees"
    TAXES = "taxes"
    SAVINGS = "savings"
    CASH = "cash"


class ClassificationRoute(StrEnum):
    DETERMINISTIC = "deterministic"
    EXPENSE_MODEL = "expense_model"
    NEEDS_REVIEW = "needs_review"


class ClassificationMethod(StrEnum):
    DETERMINISTIC = "deterministic"
    ML = "ml"
    NONE = "none"


@dataclass(frozen=True)
class ClassificationDecision:
    route: ClassificationRoute
    category: TransactionCategory | ExpenseCategory | None
    method: ClassificationMethod
    reason: str


DETERMINISTIC_CATEGORIES: dict[str, TransactionCategory] = {
    "salary": TransactionCategory.INCOME,
    "interest": TransactionCategory.INCOME,
    "dividend": TransactionCategory.INVESTMENTS,
    "security_buy": TransactionCategory.INVESTMENTS,
    "security_sell": TransactionCategory.INVESTMENTS,
    "fee": TransactionCategory.FEES,
    "tax": TransactionCategory.TAXES,
    "deposit": TransactionCategory.SAVINGS,
    "cash_withdrawal": TransactionCategory.CASH,
}

EXPENSE_MODEL_TYPES = {"card_payment", "direct_debit"}


def route_transaction_type(transaction_type: str) -> ClassificationDecision:
    normalized = transaction_type.strip().casefold()
    if not normalized:
        raise ValueError("Transaction type must not be empty")

    category = DETERMINISTIC_CATEGORIES.get(normalized)
    if category is not None:
        return ClassificationDecision(
            route=ClassificationRoute.DETERMINISTIC,
            category=category,
            method=ClassificationMethod.DETERMINISTIC,
            reason="Category is determined by the structured transaction type.",
        )

    if normalized in EXPENSE_MODEL_TYPES:
        return ClassificationDecision(
            route=ClassificationRoute.EXPENSE_MODEL,
            category=None,
            method=ClassificationMethod.ML,
            reason="Merchant-related expense requires the category model.",
        )

    return ClassificationDecision(
        route=ClassificationRoute.NEEDS_REVIEW,
        category=None,
        method=ClassificationMethod.NONE,
        reason="Transaction type does not provide enough context for automatic categorization.",
    )
