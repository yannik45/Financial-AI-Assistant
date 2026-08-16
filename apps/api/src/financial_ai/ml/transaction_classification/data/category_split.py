from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from financial_ai.ml.transaction_classification.data.category_grouping import (
    normalize_description_group,
)


@dataclass(frozen=True)
class CategoryDataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_category_training_data(
    prepared_data: pd.DataFrame,
    random_state: int = 42,
) -> CategoryDataSplits:
    """Split prepared category data reproducibly while preserving class ratios."""

    train_data, remaining_data = train_test_split(
        prepared_data,
        test_size=0.30,
        random_state=random_state,
        stratify=prepared_data["target_category"],
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


def split_grouped_category_training_data(
    prepared_data: pd.DataFrame,
    random_state: int = 42,
) -> CategoryDataSplits:
    """Split prepared data while keeping normalized description groups together."""

    groups = prepared_data["description"].map(normalize_description_group)

    fold_ids = np.empty(len(prepared_data), dtype=int)

    splitter = StratifiedGroupKFold(
        n_splits=20,
        shuffle=True,
        random_state=random_state,
    )

    split = splitter.split(
        prepared_data,
        prepared_data["target_category"],
        groups=groups,
    )
    for fold_number, (_, held_out_indices) in enumerate(split):
        fold_ids[held_out_indices] = fold_number

    train_mask = fold_ids < 14
    validation_mask = (fold_ids >= 14) & (fold_ids < 17)
    test_mask = fold_ids >= 17

    train_data = prepared_data.loc[train_mask]
    validation_data = prepared_data.loc[validation_mask]
    test_data = prepared_data.loc[test_mask]

    return CategoryDataSplits(
        train=train_data.reset_index(drop=True),
        validation=validation_data.reset_index(drop=True),
        test=test_data.reset_index(drop=True),
    )
