import json
from datetime import date, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest
from financial_ai.database import SessionLocal
from financial_ai.ml.artifact_integrity import (
    calculate_canonical_text_sha256,
)
from financial_ai.ml.transaction_classification.core.contracts import (
    TAXONOMY_VERSION,
    FeedbackStatus,
)
from financial_ai.ml.transaction_classification.feedback.feedback_export import (
    EXPORT_COLUMNS,
    load_feedback_snapshot,
    prepare_feedback_export,
    write_feedback_snapshot,
)
from financial_ai.models import Account, Transaction, TransactionClassificationRecord


def _add_feedback(
    *,
    name: str,
    final_category: str | None,
    status: FeedbackStatus,
    counterparty: str | None = None,
    amount: str = "-10.00",
    created_at: datetime | None = None,
) -> None:
    with SessionLocal() as session:
        account = session.query(Account).first()
        if account is None:
            account = Account(name="Feedback test account", account_type="cash", currency="EUR")
            session.add(account)
            session.flush()
        transaction = Transaction(
            account_id=account.id,
            booked_at=date(2026, 7, 31),
            name=name,
            amount=Decimal(amount),
            currency="EUR",
            transaction_type="cash",
            counterparty=counterparty,
            category=final_category,
            notes="must never be exported",
        )
        session.add(transaction)
        session.flush()
        session.add(
            TransactionClassificationRecord(
                transaction_id=transaction.id,
                predicted_category=final_category,
                final_category=final_category,
                route="text_rule",
                classification_method="keyword_rule",
                confidence=1.0,
                needs_review=False,
                feedback_status=status.value,
                reason="test",
                taxonomy_version=TAXONOMY_VERSION,
                model_version="test-model-v1",
                created_at=created_at or datetime(2026, 7, 31, 12, 0),
            )
        )
        session.commit()


def test_prepare_feedback_export_keeps_only_strong_valid_labels():
    _add_feedback(
        name="House Payment",
        counterparty="Example Landlord",
        final_category="housing",
        status=FeedbackStatus.ACCEPTED_EXPLICIT,
    )
    _add_feedback(
        name="Monthly salary",
        final_category="income",
        status=FeedbackStatus.ACCEPTED_IMPLICIT,
        amount="2500.00",
    )
    _add_feedback(
        name="Family transfer",
        final_category="income",
        status=FeedbackStatus.MANUAL,
        amount="50.00",
    )
    _add_feedback(
        name="Invalid historical label",
        final_category="not-in-taxonomy",
        status=FeedbackStatus.CORRECTED,
    )
    _add_feedback(
        name="Missing label",
        final_category=None,
        status=FeedbackStatus.UNREVIEWED,
    )

    with SessionLocal() as session:
        exported, report = prepare_feedback_export(session)

    assert list(exported.columns) == EXPORT_COLUMNS
    assert exported.to_dict("records") == [
        {
            "text": "House Payment Example Landlord",
            "target_category": "housing",
            "cash_flow": "outflow",
            "label_source": "accepted_explicit",
            "model_scope": "expense_model",
            "taxonomy_version": TAXONOMY_VERSION,
            "prediction_model_version": "test-model-v1",
        },
        {
            "text": "Family transfer",
            "target_category": "income",
            "cash_flow": "inflow",
            "label_source": "manual",
            "model_scope": "product_rule",
            "taxonomy_version": TAXONOMY_VERSION,
            "prediction_model_version": "test-model-v1",
        },
    ]
    assert report.source_records == 5
    assert report.eligible_records == 2
    assert report.excluded_missing_label == 1
    assert report.excluded_invalid_label == 1
    assert report.excluded_weak_or_unreviewed == 1


def test_prepare_feedback_export_removes_duplicates_and_label_conflicts():
    earlier = datetime(2026, 7, 30, 12, 0)
    _add_feedback(
        name=" Coffee   Shop ",
        final_category="dining",
        status=FeedbackStatus.MANUAL,
        created_at=earlier,
    )
    _add_feedback(
        name="coffee shop",
        final_category="dining",
        status=FeedbackStatus.ACCEPTED_EXPLICIT,
        created_at=earlier + timedelta(hours=1),
    )
    _add_feedback(
        name="Mystery merchant",
        final_category="shopping",
        status=FeedbackStatus.CORRECTED,
    )
    _add_feedback(
        name=" mystery  merchant ",
        final_category="groceries",
        status=FeedbackStatus.CORRECTED,
    )

    with SessionLocal() as session:
        exported, report = prepare_feedback_export(session)

    assert exported["text"].tolist() == ["coffee shop"]
    assert exported["label_source"].tolist() == ["accepted_explicit"]
    assert report.eligible_records == 4
    assert report.removed_duplicate_rows == 1
    assert report.excluded_conflicting_rows == 2


def test_write_feedback_snapshot_is_versioned_deterministic_and_private(tmp_path):
    _add_feedback(
        name="Pharmacy purchase",
        final_category="healthcare",
        status=FeedbackStatus.CORRECTED,
    )

    with SessionLocal() as session:
        csv_path, metadata_path, report = write_feedback_snapshot(
            session,
            "Reviewed-V1",
            tmp_path,
        )

    assert csv_path.name == "transaction_category_feedback_reviewed-v1.csv"
    exported = pd.read_csv(csv_path)
    assert list(exported.columns) == EXPORT_COLUMNS
    assert not {"amount", "notes", "transaction_id", "account_id"} & set(exported.columns)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["sha256"] == calculate_canonical_text_sha256(csv_path)
    assert metadata["report"] == {
        "source_records": 1,
        "eligible_records": 1,
        "exported_rows": 1,
        "excluded_missing_label": 0,
        "excluded_invalid_label": 0,
        "excluded_weak_or_unreviewed": 0,
        "excluded_conflicting_rows": 0,
        "removed_duplicate_rows": 0,
    }
    assert metadata["automatic_retraining"] is False
    assert report.exported_rows == 1

    with SessionLocal() as session, pytest.raises(FileExistsError):
        write_feedback_snapshot(session, "reviewed-v1", tmp_path)


@pytest.mark.parametrize("version", ["", "contains spaces", "../escape", "UPPER CASE"])
def test_write_feedback_snapshot_rejects_unsafe_versions(tmp_path, version):
    with SessionLocal() as session, pytest.raises(ValueError):
        write_feedback_snapshot(session, version, tmp_path)


def test_load_feedback_snapshot_rejects_tampered_csv(tmp_path):
    _add_feedback(
        name="Pharmacy purchase",
        final_category="healthcare",
        status=FeedbackStatus.CORRECTED,
    )
    with SessionLocal() as session:
        csv_path, _, _ = write_feedback_snapshot(session, "reviewed-v1", tmp_path)
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        load_feedback_snapshot("reviewed-v1", tmp_path)
