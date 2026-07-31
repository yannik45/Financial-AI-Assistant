import json

import pandas as pd
import pytest
from financial_ai.ml.categories import ExpenseCategory
from financial_ai.ml.english_training_generator_v1 import (
    GENERATOR_VERSION,
    OUTPUT_COLUMNS,
    generate_english_training_data_v1,
    write_english_training_dataset_v1,
)
from financial_ai.ml.german_training_split_v2 import split_declared_training_data


def test_generate_english_training_data_v1_is_deterministic_and_balanced():
    first = generate_english_training_data_v1(100, random_seed=7)
    second = generate_english_training_data_v1(100, random_seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == OUTPUT_COLUMNS
    assert len(first) == 100 * len(ExpenseCategory)
    assert first["description"].is_unique
    assert first["target_category"].value_counts().eq(100).all()
    assert set(first["language"]) == {"en"}


def test_english_controlled_splits_keep_provenance_groups_separate():
    generated = generate_english_training_data_v1(200, random_seed=7)
    splits = split_declared_training_data(generated)

    assert len(splits.train) == 150 * len(ExpenseCategory)
    assert len(splits.validation) == 25 * len(ExpenseCategory)
    assert len(splits.test) == 25 * len(ExpenseCategory)
    for column in ("merchant_group", "detail_group", "format_group"):
        train_groups = set(splits.train[column])
        validation_groups = set(splits.validation[column])
        test_groups = set(splits.test[column])
        assert train_groups.isdisjoint(validation_groups)
        assert train_groups.isdisjoint(test_groups)
        assert validation_groups.isdisjoint(test_groups)


def test_generate_english_training_data_v1_rejects_too_few_examples():
    with pytest.raises(ValueError, match="at least 8"):
        generate_english_training_data_v1(7)


def test_write_english_training_dataset_v1_writes_metadata(tmp_path):
    destination = tmp_path / "english_training_v1.csv"

    csv_path, metadata_path = write_english_training_dataset_v1(
        destination,
        examples_per_category=8,
        random_seed=11,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(pd.read_csv(csv_path)) == 8 * len(ExpenseCategory)
    assert metadata["generator_version"] == GENERATOR_VERSION
    assert metadata["legacy_train_rows_analyzed"] == 25_644
    assert metadata["sha256"]
