import json
from pathlib import Path

import pandas as pd
import pytest
from financial_ai.ml.transaction_classification.categories import ExpenseCategory
from financial_ai.ml.transaction_classification.german_training_generator_v2 import (
    GENERATOR_VERSION,
    OUTPUT_COLUMNS,
    generate_german_training_data_v2,
    write_german_training_dataset_v2,
)
from financial_ai.ml.transaction_classification.german_training_split_v2 import (
    split_german_training_data_v2,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GERMAN_CHALLENGE_PATH = (
    REPOSITORY_ROOT / "data" / "evaluation" / "transaction_categories" / "german_challenge_v1.csv"
)


def test_generate_german_training_data_v2_is_deterministic_and_balanced():
    first = generate_german_training_data_v2(
        examples_per_category=100,
        random_seed=7,
    )
    second = generate_german_training_data_v2(
        examples_per_category=100,
        random_seed=7,
    )

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == OUTPUT_COLUMNS
    assert len(first) == 100 * len(ExpenseCategory)
    assert first["description"].is_unique
    assert first["target_category"].value_counts().eq(100).all()
    assert set(first["language"]) == {"de"}


def test_german_v2_split_keeps_all_provenance_groups_separate():
    generated_data = generate_german_training_data_v2(
        examples_per_category=200,
        random_seed=7,
    )
    splits = split_german_training_data_v2(generated_data)

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


def test_german_v2_training_data_does_not_overlap_challenge_set():
    generated_data = generate_german_training_data_v2(
        examples_per_category=100,
        random_seed=7,
    )
    challenge_data = pd.read_csv(GERMAN_CHALLENGE_PATH)

    assert set(generated_data["description"]).isdisjoint(challenge_data["description"])
    assert set(generated_data["merchant_group"]).isdisjoint(challenge_data["merchant_group"])


def test_generate_german_training_data_v2_rejects_too_few_examples():
    with pytest.raises(ValueError, match="at least 8"):
        generate_german_training_data_v2(examples_per_category=7)


def test_write_german_training_dataset_v2_writes_versioned_metadata(tmp_path):
    destination = tmp_path / "german_training_v2.csv"

    csv_path, metadata_path = write_german_training_dataset_v2(
        destination=destination,
        examples_per_category=8,
        random_seed=11,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(pd.read_csv(csv_path)) == 8 * len(ExpenseCategory)
    assert metadata["generator_version"] == GENERATOR_VERSION
    assert metadata["split_strategy"] == "disjoint-merchant-detail-format-v1"
    assert metadata["sha256"]
