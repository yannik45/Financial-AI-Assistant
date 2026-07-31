from dataclasses import dataclass
from decimal import Decimal
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


def parse_product_category(value: str) -> str:
    normalized = value.strip().casefold()
    if not normalized:
        raise ValueError("Category must not be empty")
    valid_categories = {
        *(category.value for category in TransactionCategory),
        *(category.value for category in ExpenseCategory),
    }
    if normalized not in valid_categories:
        raise ValueError(f"Unsupported transaction category: {value}")
    return normalized


class ClassificationRoute(StrEnum):
    # Retained so records created by the pre-text-first service remain readable.
    DETERMINISTIC = "deterministic"
    TEXT_RULE = "text_rule"
    EXPENSE_MODEL = "expense_model"
    NEEDS_REVIEW = "needs_review"


class ClassificationMethod(StrEnum):
    # Retained so records created by the pre-text-first service remain readable.
    DETERMINISTIC = "deterministic"
    KEYWORD_RULE = "keyword_rule"
    ML = "ml"
    NONE = "none"


class FeedbackStatus(StrEnum):
    # Legacy value retained for records created before explicit confirmation tracking.
    ACCEPTED = "accepted"
    ACCEPTED_IMPLICIT = "accepted_implicit"
    ACCEPTED_EXPLICIT = "accepted_explicit"
    CORRECTED = "corrected"
    MANUAL = "manual"
    UNREVIEWED = "unreviewed"


OUTFLOW_ONLY_CATEGORIES = {
    TransactionCategory.FEES,
    TransactionCategory.TAXES,
    TransactionCategory.SAVINGS,
    TransactionCategory.CASH,
    *ExpenseCategory,
}


def determine_feedback_status(
    predicted_category: str | None,
    final_category: str | None,
    category_confirmed: bool = False,
) -> FeedbackStatus:
    normalized_prediction = predicted_category.strip().casefold() if predicted_category else None
    normalized_final = final_category.strip().casefold() if final_category else None
    if predicted_category is None:
        return FeedbackStatus.MANUAL
    if final_category is None:
        return FeedbackStatus.UNREVIEWED
    if normalized_prediction == normalized_final:
        return (
            FeedbackStatus.ACCEPTED_EXPLICIT
            if category_confirmed
            else FeedbackStatus.ACCEPTED_IMPLICIT
        )
    return FeedbackStatus.CORRECTED


@dataclass(frozen=True)
class ClassificationDecision:
    route: ClassificationRoute
    category: TransactionCategory | ExpenseCategory | None
    method: ClassificationMethod
    reason: str


def route_transaction_text(
    description: str,
    amount: Decimal,
    counterparty: str | None = None,
) -> ClassificationDecision:
    normalized_description = description.strip()
    if not normalized_description:
        raise ValueError("Description must not be empty")
    if amount == 0:
        raise ValueError("Transaction amount must not be zero")

    # Local import avoids a module cycle: the rule taxonomy uses the enums above.
    from financial_ai.ml.text_classification_rules import match_text_category

    text = " ".join(part for part in (normalized_description, (counterparty or "").strip()) if part)
    category = match_text_category(text)
    if category is not None:
        direction_conflict = (category is TransactionCategory.INCOME and amount < 0) or (
            category in OUTFLOW_ONLY_CATEGORIES and amount > 0
        )
        if direction_conflict:
            return ClassificationDecision(
                route=ClassificationRoute.NEEDS_REVIEW,
                category=None,
                method=ClassificationMethod.NONE,
                reason="Text category conflicts with the signed cash-flow direction.",
            )
        return ClassificationDecision(
            route=ClassificationRoute.TEXT_RULE,
            category=category,
            method=ClassificationMethod.KEYWORD_RULE,
            reason="Category matched a reviewable text rule in the experimental baseline.",
        )

    if amount < 0:
        return ClassificationDecision(
            route=ClassificationRoute.EXPENSE_MODEL,
            category=None,
            method=ClassificationMethod.ML,
            reason="Unmatched outgoing transaction requires the expense category model.",
        )

    return ClassificationDecision(
        route=ClassificationRoute.NEEDS_REVIEW,
        category=None,
        method=ClassificationMethod.NONE,
        reason="Unmatched incoming transaction requires manual review.",
    )
