import pandas as pd

from financial_ai.ml.categories import ExpenseCategory

EXPECTED_GERMAN_CHALLENGE_COLUMNS = [
    "scenario_id",
    "description",
    "target_category",
    "language",
    "merchant_group",
    "merchant_scope",
]


def validate_german_challenge_data(source_data: pd.DataFrame) -> pd.DataFrame:
    """Validate the versioned synthetic German evaluation dataset."""
    actual_columns = list(source_data.columns)

    if actual_columns != EXPECTED_GERMAN_CHALLENGE_COLUMNS:
        raise ValueError(
            "Expected German challenge columns "
            f"{EXPECTED_GERMAN_CHALLENGE_COLUMNS}, got {actual_columns}"
        )

    descriptions = source_data["description"]

    invalid_descriptions = descriptions.isna() | descriptions.str.strip().eq("")
    if invalid_descriptions.any():
        raise ValueError("Expected non-empty description values")

    ids = source_data["scenario_id"]

    invalid_ids = ids.isna() | ids.str.strip().eq("")

    duplicate_ids = ids.duplicated()

    if invalid_ids.any():
        raise ValueError("Expected non-empty scenario_id values")

    if duplicate_ids.any():
        raise ValueError("Expected unique scenario_id values")

    languages = source_data["language"]
    invalid_languages = languages.ne("de")

    if invalid_languages.any():
        raise ValueError("German challenge language must be 'de'")

    merchant_scopes = source_data["merchant_scope"]
    allowed_merchant_scopes = {"german_local", "international"}
    invalid_merchant_scopes = ~merchant_scopes.isin(allowed_merchant_scopes)

    if invalid_merchant_scopes.any():
        raise ValueError("Expected a valid merchant_scope value")

    target_categories = source_data["target_category"]
    allowed_target_categories = {category.value for category in ExpenseCategory}
    invalid_target_categories = ~target_categories.isin(allowed_target_categories)

    if invalid_target_categories.any():
        raise ValueError("Expected a valid target_category value")

    category_counts = target_categories.value_counts().reindex(
        allowed_target_categories,
        fill_value=0,
    )
    invalid_category_counts = ~category_counts.eq(10)

    if invalid_category_counts.any():
        raise ValueError("Expected exactly 10 examples per target_category")

    return source_data.copy()
