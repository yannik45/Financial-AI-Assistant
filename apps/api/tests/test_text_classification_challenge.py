import json

import pandas as pd
import pytest
from financial_ai.ml.text_classification_challenge import (
    build_text_classification_challenge,
    load_text_classification_challenge,
    validate_text_classification_challenge,
    write_text_classification_challenge,
)


def test_challenge_is_balanced_and_covers_declared_slices():
    challenge = build_text_classification_challenge()

    assert len(challenge) == 252
    assert set(challenge["language"]) == {"de", "en"}
    assert set(challenge["difficulty"]) == {"easy", "medium", "hard"}
    assert challenge.groupby(["expected_category", "language"]).size().eq(7).all()
    assert challenge["id"].is_unique


def test_challenge_excludes_known_development_examples():
    normalized_descriptions = {
        description.strip().casefold()
        for description in build_text_classification_challenge()["description"]
    }

    assert normalized_descriptions.isdisjoint(
        {"salary", "income", "house payment", "coffee shop", "amazon", "überweisung mama"}
    )


def test_challenge_validation_rejects_unbalanced_groups():
    invalid = build_text_classification_challenge().iloc[:-1].copy()

    with pytest.raises(ValueError, match="seven cases"):
        validate_text_classification_challenge(invalid)


def test_challenge_round_trip_writes_checksum_metadata(tmp_path):
    csv_path = tmp_path / "challenge.csv"
    metadata_path = tmp_path / "challenge.metadata.json"

    write_text_classification_challenge(csv_path, metadata_path)
    loaded = load_text_classification_challenge(csv_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    pd.testing.assert_frame_equal(loaded, build_text_classification_challenge())
    assert metadata["rows"] == 252
    assert len(metadata["sha256"]) == 64
    assert metadata["known_regression_cases_included"] is False
