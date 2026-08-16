import pandas as pd

from financial_ai.ml.transaction_classification.data.category_split import CategoryDataSplits

REQUIRED_COLUMNS = {
    "description",
    "target_category",
    "merchant_group",
    "detail_group",
    "format_group",
    "split",
}


def split_declared_training_data(
    generated_data: pd.DataFrame,
) -> CategoryDataSplits:
    missing_columns = REQUIRED_COLUMNS.difference(generated_data.columns)
    if missing_columns:
        raise ValueError(f"Missing German v2 columns: {sorted(missing_columns)}")

    actual_splits = set(generated_data["split"])
    expected_splits = {"train", "validation", "test"}
    if actual_splits != expected_splits:
        raise ValueError(
            f"Expected German v2 splits {sorted(expected_splits)}, got {sorted(actual_splits)}"
        )

    return CategoryDataSplits(
        train=generated_data.loc[generated_data["split"].eq("train")].reset_index(drop=True),
        validation=generated_data.loc[generated_data["split"].eq("validation")].reset_index(
            drop=True
        ),
        test=generated_data.loc[generated_data["split"].eq("test")].reset_index(drop=True),
    )


def split_german_training_data_v2(
    generated_data: pd.DataFrame,
) -> CategoryDataSplits:
    return split_declared_training_data(generated_data)
