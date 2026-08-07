import hashlib
from pathlib import Path
from unittest.mock import Mock, mock_open

import pandas as pd
import pytest
from financial_ai.ml.transaction_classification import category_dataset


def test_calculate_sha256():
    content = b"description,category\nExample,Shopping\n"
    dataset_path = Mock(spec=Path)
    dataset_path.open = mock_open(read_data=content)

    assert category_dataset.calculate_sha256(dataset_path) == hashlib.sha256(content).hexdigest()


def test_download_reuses_existing_verified_dataset(monkeypatch):
    dataset_path = Mock(spec=Path)
    dataset_path.exists.return_value = True
    monkeypatch.setattr(category_dataset, "calculate_sha256", lambda _: "expected")
    monkeypatch.setattr(category_dataset, "DATASET_SHA256", "expected")

    assert category_dataset.download_category_dataset(dataset_path) == dataset_path


def test_download_rejects_existing_file_with_wrong_checksum(monkeypatch):
    dataset_path = Mock(spec=Path)
    dataset_path.exists.return_value = True
    monkeypatch.setattr(category_dataset, "calculate_sha256", lambda _: "unexpected")
    monkeypatch.setattr(category_dataset, "DATASET_SHA256", "expected")

    with pytest.raises(ValueError, match="Existing dataset checksum mismatch"):
        category_dataset.download_category_dataset(dataset_path)


def test_load_category_training_data_reads_verified_dataset(monkeypatch):
    dataset_path = Mock(spec=Path)
    dataset_path.is_file.return_value = True
    source_data = pd.DataFrame({"description": ["example"], "category": ["Shopping"]})
    prepared_data = pd.DataFrame(
        {
            "description": ["example"],
            "source_category": ["Shopping"],
            "target_category": ["shopping"],
        }
    )
    monkeypatch.setattr(
        category_dataset,
        "calculate_sha256",
        lambda _: category_dataset.DATASET_SHA256,
    )
    monkeypatch.setattr(category_dataset.pd, "read_csv", lambda _: source_data)
    monkeypatch.setattr(
        category_dataset,
        "prepare_category_training_data",
        lambda _: prepared_data,
    )

    result = category_dataset.load_category_training_data(dataset_path)

    assert result is prepared_data


def test_load_category_training_data_rejects_missing_file():
    dataset_path = Mock(spec=Path)
    dataset_path.is_file.return_value = False

    with pytest.raises(FileNotFoundError, match="Category dataset not found"):
        category_dataset.load_category_training_data(dataset_path)


def test_load_category_training_data_rejects_wrong_checksum(monkeypatch):
    dataset_path = Mock(spec=Path)
    dataset_path.is_file.return_value = True
    monkeypatch.setattr(category_dataset, "calculate_sha256", lambda _: "unexpected")

    with pytest.raises(ValueError, match="Dataset checksum mismatch"):
        category_dataset.load_category_training_data(dataset_path)
