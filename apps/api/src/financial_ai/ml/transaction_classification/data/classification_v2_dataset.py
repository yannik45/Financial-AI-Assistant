from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from financial_ai.ml.transaction_classification.data.manual_short_dataset import (
    DEFAULT_MANUAL_SHORT_PATH,
    load_manual_short_dataset,
)

DEFAULT_ENGLISH_PATH = Path("data/runtime/ml/transaction_categories/english_training_v1.csv")
DEFAULT_GERMAN_PATH = Path("data/runtime/ml/transaction_categories/german_training_v2.csv")
SPLIT_NAMES = ("train", "validation", "test")
GROUP_COLUMNS = ("merchant_group", "detail_group", "format_group")
REQUIRED_COLUMNS = {
    "example_id",
    "description",
    "target_category",
    "language",
    "template_id",
    *GROUP_COLUMNS,
    "split",
}


@dataclass(frozen=True)
class ClassificationV2Dataset:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def validate_declared_splits(data: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing classification dataset columns: {sorted(missing)}")

    actual_splits = set(data["split"])
    if actual_splits != set(SPLIT_NAMES):
        raise ValueError(f"Expected splits {sorted(SPLIT_NAMES)}, got {sorted(actual_splits)}")

    duplicate_ids = data.loc[data["example_id"].duplicated(), "example_id"].unique()
    if len(duplicate_ids):
        raise ValueError(f"Duplicate example IDs: {sorted(duplicate_ids)}")

    for column in GROUP_COLUMNS:
        split_counts = data.groupby(["language", column], dropna=False)["split"].nunique()
        leaking_groups = split_counts.loc[split_counts > 1]
        if not leaking_groups.empty:
            formatted = [f"{language}:{group}" for language, group in leaking_groups.index]
            raise ValueError(f"{column} values cross declared splits: {formatted}")


def load_classification_v2_dataset(
    english_path: Path = DEFAULT_ENGLISH_PATH,
    german_path: Path = DEFAULT_GERMAN_PATH,
    manual_short_path: Path | None = DEFAULT_MANUAL_SHORT_PATH,
) -> ClassificationV2Dataset:
    sources = []
    for path in (english_path, german_path):
        if not path.is_file():
            raise FileNotFoundError(f"Classification dataset not found: {path}")
        source = pd.read_csv(path)
        source["input_slice"] = "bank_feed"
        source["generalization_slice"] = "bank_feed"
        sources.append(source)

    if manual_short_path is not None:
        manual = load_manual_short_dataset(manual_short_path).rename(
            columns={
                "id": "example_id",
                "text": "description",
                "category": "target_category",
            }
        )
        manual["input_slice"] = "manual_short"
        sources.append(manual)

    validate_declared_splits(pd.concat(sources[:2], ignore_index=True))
    combined = pd.concat(sources, ignore_index=True)
    return ClassificationV2Dataset(
        train=combined.loc[combined["split"].eq("train")].reset_index(drop=True),
        validation=combined.loc[combined["split"].eq("validation")].reset_index(drop=True),
        test=combined.loc[combined["split"].eq("test")].reset_index(drop=True),
    )
