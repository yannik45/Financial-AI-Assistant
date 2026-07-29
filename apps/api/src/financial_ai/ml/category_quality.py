import pandas as pd


def build_category_quality_report(training_data: pd.DataFrame) -> dict[str, object]:
    row_count = len(training_data)
    category_counts = training_data["target_category"].value_counts().to_dict()
    duplicate_description_count = int(training_data["description"].duplicated().sum())

    categories_per_description = training_data.groupby("description")["target_category"].nunique()
    conflicting_description_count = int(categories_per_description.gt(1).sum())

    return {
        "row_count": row_count,
        "category_counts": category_counts,
        "duplicate_description_count": duplicate_description_count,
        "conflicting_description_count": conflicting_description_count,
    }
