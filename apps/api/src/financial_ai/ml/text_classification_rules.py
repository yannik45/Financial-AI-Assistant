import re
from dataclasses import dataclass

from financial_ai.ml.categories import ExpenseCategory
from financial_ai.ml.transaction_classification import TransactionCategory


@dataclass(frozen=True)
class TextCategoryRule:
    category: TransactionCategory | ExpenseCategory
    keywords: tuple[str, ...]


# Deliberately small, reviewable baseline. It is not intended to enumerate every
# merchant or replace a learned classifier.
TEXT_CATEGORY_RULES: tuple[TextCategoryRule, ...] = (
    TextCategoryRule(
        TransactionCategory.INCOME,
        (
            "salary",
            "payroll",
            "wages",
            "income",
            "interest",
            "gehalt",
            "lohn",
            "zinsen",
            "arbeitgeber",
        ),
    ),
    TextCategoryRule(
        TransactionCategory.INVESTMENTS,
        (
            "dividend",
            "dividende",
            "broker",
            "investment",
            "stock purchase",
            "security purchase",
            "etf purchase",
            "wertpapier",
            "aktienkauf",
        ),
    ),
    TextCategoryRule(
        TransactionCategory.FEES,
        ("fee", "bank fee", "account fee", "service fee", "gebühr", "gebuehr", "kontoführung"),
    ),
    TextCategoryRule(
        TransactionCategory.TAXES,
        ("tax", "tax payment", "income tax", "tax office", "steuer", "finanzamt"),
    ),
    TextCategoryRule(
        TransactionCategory.SAVINGS,
        ("savings", "savings account", "tagesgeld", "sparkonto", "spareinlage"),
    ),
    TextCategoryRule(
        TransactionCategory.CASH,
        ("cash withdrawal", "cash machine", "atm", "bargeld", "geldautomat"),
    ),
    TextCategoryRule(
        ExpenseCategory.INSURANCE,
        ("insurance", "premium", "versicherung"),
    ),
    TextCategoryRule(
        ExpenseCategory.UTILITIES,
        (
            "electricity",
            "energy bill",
            "gas bill",
            "water bill",
            "internet",
            "mobile phone",
            "strom",
            "wasser",
            "mobilfunk",
        ),
    ),
    TextCategoryRule(
        ExpenseCategory.HOUSING,
        (
            "rent",
            "mortgage",
            "house payment",
            "housing payment",
            "property management",
            "miete",
            "wohnung",
            "hauszahlung",
        ),
    ),
    TextCategoryRule(
        ExpenseCategory.HEALTHCARE,
        ("pharmacy", "hospital", "clinic", "dental", "doctor", "apotheke", "arzt"),
    ),
    TextCategoryRule(
        ExpenseCategory.EDUCATION,
        ("university", "tuition", "course", "school fee", "universität", "kurs", "schule"),
    ),
    TextCategoryRule(
        ExpenseCategory.TRAVEL,
        ("hotel", "airline", "airways", "flight", "motel", "flug", "reisebüro"),
    ),
    TextCategoryRule(
        ExpenseCategory.TRANSPORT,
        (
            "gas station",
            "fuel",
            "parking",
            "transit",
            "taxi",
            "train ticket",
            "tankstelle",
            "parken",
            "fahrkarte",
        ),
    ),
    TextCategoryRule(
        ExpenseCategory.DINING,
        ("restaurant", "cafe", "café", "pizza", "burger", "bakery", "bäckerei"),
    ),
    TextCategoryRule(
        ExpenseCategory.GROCERIES,
        ("grocery", "groceries", "supermarket", "lebensmittel", "supermarkt"),
    ),
    TextCategoryRule(
        ExpenseCategory.ENTERTAINMENT,
        ("cinema", "theater", "game", "streaming", "kino", "konzert"),
    ),
    TextCategoryRule(
        ExpenseCategory.SHOPPING,
        ("retail", "clothing", "electronics store", "fashion", "kleidung", "elektronik"),
    ),
)


def match_text_category(text: str) -> TransactionCategory | ExpenseCategory | None:
    normalized = " ".join(text.strip().casefold().split())
    if not normalized:
        return None
    for rule in TEXT_CATEGORY_RULES:
        if any(
            re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", normalized)
            for keyword in rule.keywords
        ):
            return rule.category
    return None
