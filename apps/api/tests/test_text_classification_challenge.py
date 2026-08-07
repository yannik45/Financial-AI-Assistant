import json

import pandas as pd
import pytest
from financial_ai.ml.transaction_classification.text_classification_challenge import (
    DEFAULT_CHALLENGE_PATH,
    DEFAULT_METADATA_PATH,
    V1_CHALLENGE_PATH,
    V1_METADATA_PATH,
    build_text_classification_challenge,
    calculate_canonical_csv_sha256,
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
    assert challenge["counterparty"].str.strip().ne("").any()


def test_challenge_v2_uses_cash_flow_consistent_investment_examples():
    challenge = build_text_classification_challenge()
    distributions = challenge[
        challenge["description"].str.contains(
            "dividend|distribution|Dividende|Ausschüttung", case=False, regex=True
        )
    ]

    assert not distributions.empty
    assert distributions["amount"].gt(0).all()


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


def test_committed_challenge_v2_matches_generator_and_checksum():
    committed = load_text_classification_challenge(DEFAULT_CHALLENGE_PATH, DEFAULT_METADATA_PATH)

    pd.testing.assert_frame_equal(committed, build_text_classification_challenge())


def test_frozen_v1_snapshot_checksum_remains_valid():
    metadata = json.loads(V1_METADATA_PATH.read_text(encoding="utf-8"))

    assert calculate_canonical_csv_sha256(V1_CHALLENGE_PATH) == metadata["sha256"]


def test_challenge_checksum_is_independent_of_platform_line_endings(tmp_path):
    lf_path = tmp_path / "lf.csv"
    crlf_path = tmp_path / "crlf.csv"
    lf_path.write_bytes(b"description,category\nCoffee,dining\n")
    crlf_path.write_bytes(b"description,category\r\nCoffee,dining\r\n")

    assert calculate_canonical_csv_sha256(lf_path) == calculate_canonical_csv_sha256(crlf_path)


def test_loader_rejects_tampered_challenge(tmp_path):
    csv_path = tmp_path / "challenge.csv"
    metadata_path = tmp_path / "challenge.metadata.json"
    write_text_classification_challenge(csv_path, metadata_path)
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        load_text_classification_challenge(csv_path, metadata_path)
