from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class CategoryDataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_category_training_data(
    training_data: pd.DataFrame,
    random_state: int = 42,
) -> CategoryDataSplits:
    """Split prepared category data reproducibly while preserving class ratios."""

    train_data, remaining_data = train_test_split(
        training_data,
        test_size=0.30,
        random_state=random_state,
        stratify=training_data["target_category"],
    )

    validation_data, test_data = train_test_split(
        remaining_data,
        test_size=0.50,
        random_state=random_state,
        stratify=remaining_data["target_category"],
    )

    data_splits = CategoryDataSplits(
        train=train_data.reset_index(drop=True),
        validation=validation_data.reset_index(drop=True),
        test=test_data.reset_index(drop=True),
    )

    return data_splits
