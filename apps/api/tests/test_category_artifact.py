import json

import pandas as pd
import pytest
from financial_ai.ml.transaction_classification.category_artifact import (
    ModelArtifactError,
    build_category_model_artifact,
    load_category_model_artifact,
)


def _controlled_data(language: str) -> pd.DataFrame:
    rows = []
    for split in ("train", "validation", "test"):
        for category, description in (
            ("groceries", "supermarket food"),
            ("dining", "restaurant dinner"),
        ):
            for copy_index in range(2):
                rows.append(
                    {
                        "description": f"{description} {language} {split} {copy_index}",
                        "target_category": category,
                        "merchant_group": f"{language}-{split}-{category}-merchant",
                        "detail_group": f"{language}-{split}-{category}-detail",
                        "format_group": f"{language}-{split}-format",
                        "split": split,
                    }
                )
    return pd.DataFrame(rows)


def test_model_artifact_is_built_from_train_and_validation_only(tmp_path):
    english_path = tmp_path / "english.csv"
    german_path = tmp_path / "german.csv"
    artifact_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "model.json"
    _controlled_data("en").to_csv(english_path, index=False)
    _controlled_data("de").to_csv(german_path, index=False)

    metadata = build_category_model_artifact(
        english_path,
        german_path,
        artifact_path,
        metadata_path,
    )
    loaded = load_category_model_artifact(artifact_path, metadata_path)

    assert metadata.training_rows == 16
    assert metadata.languages == ("en", "de")
    assert loaded.metadata.model_version == metadata.model_version
    assert set(loaded.model.classes_) == {"dining", "groceries"}
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["artifact_sha256"]
    assert metadata_payload["random_state"] == 42
    assert metadata_payload["feature_configuration"]["ngram_range"] == [3, 5]
    assert metadata_payload["model_parameters"]["class_weight"] == "balanced"
    assert metadata_payload["library_versions"]["scikit_learn"]


def test_model_artifact_rejects_missing_source_and_checksum_mismatch(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(ModelArtifactError, match="Training dataset not found"):
        build_category_model_artifact(missing, missing)

    english_path = tmp_path / "english.csv"
    german_path = tmp_path / "german.csv"
    artifact_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "model.json"
    _controlled_data("en").to_csv(english_path, index=False)
    _controlled_data("de").to_csv(german_path, index=False)
    build_category_model_artifact(english_path, german_path, artifact_path, metadata_path)
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")

    with pytest.raises(ModelArtifactError, match="checksum"):
        load_category_model_artifact(artifact_path, metadata_path)
