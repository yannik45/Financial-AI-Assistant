import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from financial_ai.ml.artifact_integrity import normalize_artifact_version
from financial_ai.ml.transaction_classification.feedback.feedback_candidate import (
    DEFAULT_CANDIDATE_DIRECTORY,
    candidate_paths,
)
from financial_ai.ml.transaction_classification.modeling.category_artifact import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_METADATA_PATH,
    calculate_sha256,
    load_category_model_artifact,
)

DEFAULT_ARCHIVE_DIRECTORY = Path("data/runtime/ml/models/archive")


class FeedbackPromotionError(RuntimeError):
    pass


def promote_feedback_candidate(
    candidate_version: str,
    *,
    candidate_directory: Path = DEFAULT_CANDIDATE_DIRECTORY,
    active_artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    active_metadata_path: Path = DEFAULT_METADATA_PATH,
    archive_directory: Path = DEFAULT_ARCHIVE_DIRECTORY,
) -> Path:
    version = normalize_artifact_version(candidate_version)
    paths = candidate_paths(version, candidate_directory)
    if not paths.evaluation.is_file():
        raise FeedbackPromotionError(f"Candidate evaluation not found: {paths.evaluation}")
    report = json.loads(paths.evaluation.read_text(encoding="utf-8"))
    if report.get("candidate_version") != version:
        raise FeedbackPromotionError("Candidate version does not match its evaluation report")
    if report.get("eligible_for_promotion") is not True:
        raise FeedbackPromotionError("Candidate did not pass every promotion gate")
    gates = report.get("gates")
    if not isinstance(gates, dict) or not gates or not all(gates.values()):
        raise FeedbackPromotionError("Candidate evaluation contains a failed promotion gate")

    candidate = load_category_model_artifact(paths.artifact, paths.metadata)
    if report.get("candidate_artifact_sha256") != candidate.metadata.artifact_sha256:
        raise FeedbackPromotionError("Candidate checksum does not match its evaluation report")
    active = load_category_model_artifact(active_artifact_path, active_metadata_path)
    active_sha256 = calculate_sha256(active_artifact_path)
    if report.get("baseline_artifact_sha256") != active_sha256:
        raise FeedbackPromotionError(
            "Active model changed after candidate evaluation; retrain and reevaluate the candidate"
        )

    archive_directory.mkdir(parents=True, exist_ok=True)
    archive_stem = f"{active.metadata.model_version}-{active_sha256[:12]}"
    archived_artifact = archive_directory / f"{archive_stem}.pkl"
    archived_metadata = archive_directory / f"{archive_stem}.json"
    if archived_artifact.exists() or archived_metadata.exists():
        raise FileExistsError(f"Active-model archive already exists: {archive_stem}")
    shutil.copy2(active_artifact_path, archived_artifact)
    shutil.copy2(active_metadata_path, archived_metadata)

    temporary_suffix = f".promotion-{uuid4().hex}.tmp"
    temporary_artifact = active_artifact_path.with_name(
        active_artifact_path.name + temporary_suffix
    )
    temporary_metadata = active_metadata_path.with_name(
        active_metadata_path.name + temporary_suffix
    )
    try:
        shutil.copy2(paths.artifact, temporary_artifact)
        shutil.copy2(paths.metadata, temporary_metadata)
        os.replace(temporary_artifact, active_artifact_path)
        os.replace(temporary_metadata, active_metadata_path)
        promoted = load_category_model_artifact(active_artifact_path, active_metadata_path)
        if promoted.metadata.artifact_sha256 != candidate.metadata.artifact_sha256:
            raise FeedbackPromotionError("Promoted artifact failed checksum verification")
    except Exception:
        shutil.copy2(archived_artifact, active_artifact_path)
        shutil.copy2(archived_metadata, active_metadata_path)
        raise
    finally:
        temporary_artifact.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)

    receipt_path = candidate_directory / f"transaction_category_{version}.promotion.json"
    receipt_path.write_text(
        json.dumps(
            {
                "candidate_version": version,
                "promoted_model_version": candidate.metadata.model_version,
                "promoted_artifact_sha256": candidate.metadata.artifact_sha256,
                "previous_model_version": active.metadata.model_version,
                "previous_artifact_sha256": active_sha256,
                "archive_artifact": str(archived_artifact),
                "archive_metadata": str(archived_metadata),
                "promoted_at": datetime.now(UTC).isoformat(),
                "automatic_promotion": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt_path


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly promote an eligible transaction-category candidate"
    )
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm replacement of the active local model after archiving it",
    )
    args = parser.parse_args()
    if not args.yes:
        parser.error("Promotion requires explicit confirmation with --yes")
    receipt = promote_feedback_candidate(args.candidate_version)
    print(f"Candidate promoted. Audit receipt: {receipt}. Restart the API to load it.")


if __name__ == "__main__":
    run()
