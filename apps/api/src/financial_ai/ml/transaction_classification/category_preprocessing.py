import pandas as pd

from financial_ai.ml.transaction_classification.category_mapping import map_source_category

EXPECTED_SOURCE_COLUMNS = ["description", "category"]


def prepare_category_training_data(source_data: pd.DataFrame) -> pd.DataFrame:
    """Validate and map source rows into the versioned training taxonomy."""
    if list(source_data.columns) != EXPECTED_SOURCE_COLUMNS:
        raise ValueError(
            f"Expected dataset columns {EXPECTED_SOURCE_COLUMNS}, got {list(source_data.columns)}"
        )
    empty_descriptions = source_data["description"].fillna("").str.strip().eq("")

    if empty_descriptions.any():
        raise ValueError("Descriptions must not be empty")

    prepared_data = source_data.copy()
    prepared_data["target_category"] = prepared_data["category"].map(map_source_category)

    prepared_data = prepared_data[prepared_data["target_category"].notna()].copy()

    categories_per_description = prepared_data.groupby("description")["target_category"].nunique()
    conflicting_descriptions = categories_per_description[categories_per_description.gt(1)].index
    prepared_data = prepared_data[
        ~prepared_data["description"].isin(conflicting_descriptions)
    ].copy()

    prepared_data = prepared_data.drop_duplicates(subset=["description"], keep="first").reset_index(
        drop=True
    )

    prepared_data["target_category"] = prepared_data["target_category"].map(
        lambda category: category.value
    )

    prepared_data = prepared_data.rename(columns={"category": "source_category"})

    return prepared_data[["description", "source_category", "target_category"]].reset_index(
        drop=True
    )
