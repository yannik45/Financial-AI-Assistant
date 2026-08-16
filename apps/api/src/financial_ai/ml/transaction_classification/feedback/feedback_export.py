import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from financial_ai.database import SessionLocal
from financial_ai.ml.artifact_integrity import (
    calculate_canonical_text_sha256,
    normalize_artifact_version,
)
from financial_ai.ml.transaction_classification.core.categories import ExpenseCategory
from financial_ai.ml.transaction_classification.core.contracts import (
    FeedbackStatus,
    parse_product_category,
)
from financial_ai.models import Transaction, TransactionClassificationRecord

EXPORT_SCHEMA_VERSION = "transaction-feedback-export-v1"
DEFAULT_OUTPUT_DIRECTORY = Path("data/runtime/ml/feedback")
EXPORT_COLUMNS = [
    "text",
    "target_category",
    "cash_flow",
    "label_source",
    "model_scope",
    "taxonomy_version",
    "prediction_model_version",
]
ELIGIBLE_STATUSES = {
    FeedbackStatus.ACCEPTED_EXPLICIT.value,
    FeedbackStatus.CORRECTED.value,
    FeedbackStatus.MANUAL.value,
}
STATUS_PRIORITY = {
    FeedbackStatus.CORRECTED.value: 3,
    FeedbackStatus.ACCEPTED_EXPLICIT.value: 2,
    FeedbackStatus.MANUAL.value: 1,
}


@dataclass(frozen=True)
class FeedbackExportReport:
    source_records: int
    eligible_records: int
    exported_rows: int
    excluded_missing_label: int
    excluded_invalid_label: int
    excluded_weak_or_unreviewed: int
    excluded_conflicting_rows: int
    removed_duplicate_rows: int


@dataclass(frozen=True)
class _Candidate:
    text: str
    normalized_text: str
    target_category: str
    cash_flow: str
    label_source: str
    model_scope: str
    taxonomy_version: str
    prediction_model_version: str
    created_at_key: str
    stable_id: str


def _normalize_text(description: str, counterparty: str | None) -> tuple[str, str]:
    text = " ".join(part for part in (description.strip(), (counterparty or "").strip()) if part)
    text = " ".join(text.split())
    return text, text.casefold()


def prepare_feedback_export(session: Session) -> tuple[pd.DataFrame, FeedbackExportReport]:
    records = session.execute(
        select(TransactionClassificationRecord, Transaction)
        .join(Transaction, Transaction.id == TransactionClassificationRecord.transaction_id)
        .order_by(
            TransactionClassificationRecord.created_at,
            TransactionClassificationRecord.id,
        )
    ).all()
    candidates: list[_Candidate] = []
    excluded_missing_label = 0
    excluded_invalid_label = 0
    excluded_weak_or_unreviewed = 0

    expense_categories = {category.value for category in ExpenseCategory}
    for feedback, transaction in records:
        if feedback.final_category is None:
            excluded_missing_label += 1
            continue
        if feedback.feedback_status not in ELIGIBLE_STATUSES:
            excluded_weak_or_unreviewed += 1
            continue
        try:
            target_category = parse_product_category(feedback.final_category)
        except ValueError:
            excluded_invalid_label += 1
            continue
        text, normalized_text = _normalize_text(transaction.name, transaction.counterparty)
        candidates.append(
            _Candidate(
                text=text,
                normalized_text=normalized_text,
                target_category=target_category,
                cash_flow="inflow" if transaction.amount > 0 else "outflow",
                label_source=feedback.feedback_status,
                model_scope=(
                    "expense_model" if target_category in expense_categories else "product_rule"
                ),
                taxonomy_version=feedback.taxonomy_version,
                prediction_model_version=feedback.model_version or "",
                created_at_key=feedback.created_at.isoformat(),
                stable_id=feedback.id,
            )
        )

    categories_by_text: dict[str, set[str]] = {}
    for candidate in candidates:
        categories_by_text.setdefault(candidate.normalized_text, set()).add(
            candidate.target_category
        )
    conflicting_texts = {
        text for text, categories in categories_by_text.items() if len(categories) > 1
    }
    non_conflicting = [
        candidate for candidate in candidates if candidate.normalized_text not in conflicting_texts
    ]

    selected: dict[tuple[str, str], _Candidate] = {}
    for candidate in non_conflicting:
        key = (candidate.normalized_text, candidate.target_category)
        rank = (
            STATUS_PRIORITY[candidate.label_source],
            candidate.created_at_key,
            candidate.stable_id,
        )
        existing = selected.get(key)
        if existing is None or rank > (
            STATUS_PRIORITY[existing.label_source],
            existing.created_at_key,
            existing.stable_id,
        ):
            selected[key] = candidate

    selected_rows = sorted(
        selected.values(),
        key=lambda candidate: (
            candidate.target_category,
            candidate.normalized_text,
            candidate.label_source,
        ),
    )
    exported = pd.DataFrame(
        [
            {column: getattr(candidate, column) for column in EXPORT_COLUMNS}
            for candidate in selected_rows
        ],
        columns=EXPORT_COLUMNS,
    )
    report = FeedbackExportReport(
        source_records=len(records),
        eligible_records=len(candidates),
        exported_rows=len(exported),
        excluded_missing_label=excluded_missing_label,
        excluded_invalid_label=excluded_invalid_label,
        excluded_weak_or_unreviewed=excluded_weak_or_unreviewed,
        excluded_conflicting_rows=sum(
            candidate.normalized_text in conflicting_texts for candidate in candidates
        ),
        removed_duplicate_rows=len(non_conflicting) - len(selected),
    )
    return exported, report


def write_feedback_snapshot(
    session: Session,
    snapshot_version: str,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> tuple[Path, Path, FeedbackExportReport]:
    version = normalize_artifact_version(snapshot_version)
    exported, report = prepare_feedback_export(session)
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / f"transaction_category_feedback_{version}.csv"
    metadata_path = output_directory / f"transaction_category_feedback_{version}.metadata.json"
    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Feedback snapshot version already exists: {version}. "
            "Choose a new version instead of overwriting training evidence."
        )
    exported.to_csv(csv_path, index=False, lineterminator="\n")

    metadata = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "snapshot_version": version,
        "sha256": calculate_canonical_text_sha256(csv_path),
        "checksum_normalization": "utf-8-lf",
        "columns": EXPORT_COLUMNS,
        "eligibility": sorted(ELIGIBLE_STATUSES),
        "feedback_status_counts": dict(sorted(Counter(exported["label_source"]).items())),
        "target_category_counts": dict(sorted(Counter(exported["target_category"]).items())),
        "report": asdict(report),
        "privacy": {
            "excluded_fields": [
                "account_id",
                "transaction_id",
                "classification_id",
                "booked_at",
                "amount",
                "currency",
                "notes",
            ],
            "free_text_warning": (
                "The text field may still contain sensitive user-entered content."
            ),
        },
        "automatic_retraining": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return csv_path, metadata_path, report


def load_feedback_snapshot(
    snapshot_version: str,
    input_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> tuple[pd.DataFrame, dict]:
    version = normalize_artifact_version(snapshot_version)
    csv_path = input_directory / f"transaction_category_feedback_{version}.csv"
    metadata_path = input_directory / f"transaction_category_feedback_{version}.metadata.json"
    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Feedback snapshot not found: {version}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != EXPORT_SCHEMA_VERSION:
        raise ValueError("Feedback snapshot schema version is incompatible")
    if metadata.get("snapshot_version") != version:
        raise ValueError("Feedback snapshot version does not match its metadata")
    if metadata.get("sha256") != calculate_canonical_text_sha256(csv_path):
        raise ValueError("Feedback snapshot checksum does not match its metadata")
    feedback = pd.read_csv(csv_path, keep_default_na=False)
    if list(feedback.columns) != EXPORT_COLUMNS:
        raise ValueError("Feedback snapshot columns do not match the export schema")
    return feedback, metadata


def run() -> None:
    parser = argparse.ArgumentParser(description="Export reviewed transaction feedback")
    parser.add_argument("--version", required=True, help="Explicit snapshot version")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    args = parser.parse_args()
    with SessionLocal() as session:
        csv_path, metadata_path, report = write_feedback_snapshot(
            session,
            args.version,
            args.output_directory,
        )
    print(
        f"Feedback snapshot ready: {csv_path} ({report.exported_rows} rows; "
        f"metadata: {metadata_path})"
    )


if __name__ == "__main__":
    run()
