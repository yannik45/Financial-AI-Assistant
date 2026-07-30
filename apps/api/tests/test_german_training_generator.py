import json
from pathlib import Path

import pandas as pd
import pytest
from financial_ai.ml.categories import ExpenseCategory
from financial_ai.ml.german_training_generator import (
    OUTPUT_COLUMNS,
    generate_german_training_data,
    write_german_training_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GERMAN_CHALLENGE_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "evaluation"
    / "transaction_categories"
    / "german_challenge_v1.csv"
)


def test_generate_german_training_data_is_deterministic_and_balanced():
    first = generate_german_training_data(examples_per_category=20, random_seed=7)
    second = generate_german_training_data(examples_per_category=20, random_seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == OUTPUT_COLUMNS
    assert len(first) == 20 * len(ExpenseCategory)
    assert first["description"].is_unique
    assert set(first["language"]) == {"de"}
    assert first["target_category"].value_counts().eq(20).all()
    assert first["merchant_group"].str.startswith("generated_").all()


def test_generate_german_training_data_changes_with_seed():
    first = generate_german_training_data(examples_per_category=2, random_seed=7)
    second = generate_german_training_data(examples_per_category=2, random_seed=8)

    assert not first["description"].equals(second["description"])


def test_generated_training_data_does_not_overlap_challenge_set():
    generated_data = generate_german_training_data(
        examples_per_category=20,
        random_seed=7,
    )
    challenge_data = pd.read_csv(GERMAN_CHALLENGE_PATH)

    assert set(generated_data["description"]).isdisjoint(
        challenge_data["description"]
    )
    assert set(generated_data["merchant_group"]).isdisjoint(
        challenge_data["merchant_group"]
    )


def test_generate_german_training_data_rejects_non_positive_size():
    with pytest.raises(ValueError, match="must be positive"):
        generate_german_training_data(examples_per_category=0)


def test_write_german_training_dataset_writes_csv_and_metadata(tmp_path):
    destination = tmp_path / "german_training.csv"

    csv_path, metadata_path = write_german_training_dataset(
        destination=destination,
        examples_per_category=3,
        random_seed=11,
    )

    written_data = pd.read_csv(csv_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(written_data) == 3 * len(ExpenseCategory)
    assert metadata["generator_version"] == "german-training-generator-v1"
    assert metadata["row_count"] == len(written_data)
    assert metadata["random_seed"] == 11
    assert metadata["sha256"]
