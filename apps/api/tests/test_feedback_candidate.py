import json
from pathlib import Path

import pandas as pd
import pytest
from financial_ai.ml.artifact_integrity import (
    calculate_canonical_text_sha256,
)
from financial_ai.ml.transaction_classification.category_artifact import (
    build_category_model_artifact,
    load_category_model_artifact,
)
from financial_ai.ml.transaction_classification.feedback_candidate import (
    FeedbackCandidateError,
    candidate_paths,
    train_feedback_candidate,
)
from financial_ai.ml.transaction_classification.feedback_export import (
    EXPORT_COLUMNS,
    EXPORT_SCHEMA_VERSION,
)
from financial_ai.ml.transaction_classification.feedback_promotion import (
    FeedbackPromotionError,
    promote_feedback_candidate,
)
from financial_ai.ml.transaction_classification.text_classification_challenge import (
    write_text_classification_challenge,
)


def _controlled_data(language: str) -> pd.DataFrame:
    rows = []
    for split in ("train", "validation", "test"):
        for category, description in (
            ("groceries", "supermarket provisions"),
            ("dining", "restaurant meal"),
        ):
            for copy_index in range(3):
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


def _write_feedback_snapshot(directory: Path, version: str, rows: int = 12) -> None:
    records = []
    for index in range(rows):
        category = "groceries" if index % 2 == 0 else "dining"
        records.append(
            {
                "text": f"Reviewed unique {category} merchant {index}",
                "target_category": category,
                "cash_flow": "outflow",
                "label_source": "corrected",
                "model_scope": "expense_model",
                "taxonomy_version": "transaction-categories-v1",
                "prediction_model_version": "active-test-v1",
            }
        )
    directory.mkdir(parents=True)
    csv_path = directory / f"transaction_category_feedback_{version}.csv"
    metadata_path = directory / f"transaction_category_feedback_{version}.metadata.json"
    pd.DataFrame(records, columns=EXPORT_COLUMNS).to_csv(csv_path, index=False, lineterminator="\n")
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "snapshot_version": version,
                "sha256": calculate_canonical_text_sha256(csv_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _candidate_fixture(tmp_path: Path):
    english_path = tmp_path / "english.csv"
    german_path = tmp_path / "german.csv"
    active_artifact = tmp_path / "active.pkl"
    active_metadata = tmp_path / "active.json"
    challenge_path = tmp_path / "challenge.csv"
    challenge_metadata = tmp_path / "challenge.metadata.json"
    feedback_directory = tmp_path / "feedback"
    candidate_directory = tmp_path / "candidates"
    _controlled_data("en").to_csv(english_path, index=False)
    _controlled_data("de").to_csv(german_path, index=False)
    build_category_model_artifact(
        english_path,
        german_path,
        active_artifact,
        active_metadata,
    )
    write_text_classification_challenge(challenge_path, challenge_metadata)
    _write_feedback_snapshot(feedback_directory, "reviewed-v1")
    paths, report = train_feedback_candidate(
        "reviewed-v1",
        "candidate-v1",
        feedback_directory=feedback_directory,
        candidate_directory=candidate_directory,
        english_path=english_path,
        german_path=german_path,
        challenge_path=challenge_path,
        active_artifact_path=active_artifact,
        active_metadata_path=active_metadata,
        minimum_rows=10,
        minimum_rows_per_category=5,
        minimum_distinct_categories=2,
    )
    return paths, report, active_artifact, active_metadata, candidate_directory


def test_feedback_candidate_is_versioned_and_evaluated(tmp_path):
    paths, report, _, _, _ = _candidate_fixture(tmp_path)

    assert paths == candidate_paths("candidate-v1", paths.artifact.parent)
    loaded = load_category_model_artifact(paths.artifact, paths.metadata)
    assert loaded.metadata.model_version == "transaction-category-feedback-candidate-v1"
    assert loaded.metadata.training_source_sha256["feedback"]
    assert report["feedback"]["training_rows"] == 10
    assert report["feedback"]["holdout_rows"] == 2
    assert set(report["gates"]) == {
        "challenge_macro_f1",
        "challenge_selective_accuracy",
        "feedback_holdout_macro_f1",
    }
    assert report["automatic_promotion"] is False
    assert json.loads(paths.evaluation.read_text(encoding="utf-8")) == report


def test_feedback_candidate_enforces_minimum_data_gate(tmp_path):
    feedback_directory = tmp_path / "feedback"
    _write_feedback_snapshot(feedback_directory, "too-small", rows=4)
    challenge_path = tmp_path / "challenge.csv"
    write_text_classification_challenge(
        challenge_path,
        tmp_path / "challenge.metadata.json",
    )
    english_path = tmp_path / "english.csv"
    german_path = tmp_path / "german.csv"
    _controlled_data("en").to_csv(english_path, index=False)
    _controlled_data("de").to_csv(german_path, index=False)

    with pytest.raises(FeedbackCandidateError, match="at least 10"):
        train_feedback_candidate(
            "too-small",
            "candidate-v1",
            feedback_directory=feedback_directory,
            english_path=english_path,
            german_path=german_path,
            challenge_path=challenge_path,
            minimum_rows=10,
            minimum_rows_per_category=2,
            minimum_distinct_categories=2,
        )


def test_promotion_requires_passing_gates_and_archives_active_model(tmp_path):
    paths, report, active_artifact, active_metadata, candidate_directory = _candidate_fixture(
        tmp_path
    )
    original_active = active_artifact.read_bytes()
    original_metadata = active_metadata.read_text(encoding="utf-8")

    report["eligible_for_promotion"] = False
    paths.evaluation.write_text(json.dumps(report) + "\n", encoding="utf-8")
    with pytest.raises(FeedbackPromotionError, match="promotion gate"):
        promote_feedback_candidate(
            "candidate-v1",
            candidate_directory=candidate_directory,
            active_artifact_path=active_artifact,
            active_metadata_path=active_metadata,
            archive_directory=tmp_path / "archive",
        )

    report["eligible_for_promotion"] = True
    report["gates"] = {gate: True for gate in report["gates"]}
    paths.evaluation.write_text(json.dumps(report) + "\n", encoding="utf-8")
    receipt_path = promote_feedback_candidate(
        "candidate-v1",
        candidate_directory=candidate_directory,
        active_artifact_path=active_artifact,
        active_metadata_path=active_metadata,
        archive_directory=tmp_path / "archive",
    )

    promoted = load_category_model_artifact(active_artifact, active_metadata)
    assert promoted.metadata.model_version == "transaction-category-feedback-candidate-v1"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["automatic_promotion"] is False
    archived_artifact = Path(receipt["archive_artifact"])
    archived_metadata = Path(receipt["archive_metadata"])
    assert archived_artifact.read_bytes() == original_active
    assert archived_metadata.read_text(encoding="utf-8") == original_metadata
