from financial_ai.ml.categories import ExpenseCategory

SOURCE_CATEGORY_MAPPING: dict[str, ExpenseCategory] = {
    "Restaurants": ExpenseCategory.DINING,
    "Groceries": ExpenseCategory.GROCERIES,
    "Shopping": ExpenseCategory.SHOPPING,
    "Transportation": ExpenseCategory.TRANSPORT,
    "Entertainment": ExpenseCategory.ENTERTAINMENT,
    "Utilities": ExpenseCategory.UTILITIES,
    "Rent": ExpenseCategory.HOUSING,
    "Mortgage": ExpenseCategory.HOUSING,
    "Subscription": ExpenseCategory.OTHER,
    "Healthcare": ExpenseCategory.HEALTHCARE,
    "Insurance": ExpenseCategory.INSURANCE,
    "Travel": ExpenseCategory.TRAVEL,
    "Education": ExpenseCategory.EDUCATION,
}

EXCLUDED_SOURCE_CATEGORIES: set[str] = {
    "Income",
    "Transfer",
    "Fees",
    "Personal Care",
}


def map_source_category(value: str) -> ExpenseCategory | None:
    normalized = value.strip().casefold()
    normalized_mapping = {
        source.casefold(): target
        for source, target in SOURCE_CATEGORY_MAPPING.items()
    }
    normalized_excluded = {
        source.casefold() for source in EXCLUDED_SOURCE_CATEGORIES
    }

    if normalized in normalized_mapping:
        return normalized_mapping[normalized]

    if normalized in normalized_excluded:
        return None

    raise ValueError(f"Unknown source category: {value!r}")
