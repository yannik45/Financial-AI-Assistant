import argparse
import hashlib
import json
import pickle
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score

from financial_ai.ml.artifact_integrity import normalize_artifact_version
from financial_ai.ml.categories import ExpenseCategory
from financial_ai.ml.category_artifact import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_ENGLISH_PATH,
    DEFAULT_GERMAN_PATH,
    DEFAULT_METADATA_PATH,
    LoadedCategoryModel,
    ModelMetadata,
    calculate_sha256,
    load_category_model_artifact,
    training_partition,
)
from financial_ai.ml.category_model import train_tfidf_category_classifier
from financial_ai.ml.feedback_export import DEFAULT_OUTPUT_DIRECTORY, load_feedback_snapshot
from financial_ai.ml.text_classification_challenge import (
    DEFAULT_CHALLENGE_PATH,
    load_text_classification_challenge,
)
from financial_ai.ml.text_classification_evaluation import (
    MetricSummary,
    evaluate_text_classification_strategies,
)
from financial_ai.ml.transaction_classification import TAXONOMY_VERSION

CANDIDATE_PIPELINE_VERSION = "transaction-feedback-candidate-v1"
DEFAULT_CANDIDATE_DIRECTORY = Path("data/runtime/ml/candidates")
MIN_FEEDBACK_ROWS = 100
MIN_ROWS_PER_CATEGORY = 5
MIN_DISTINCT_CATEGORIES = 3
FEEDBACK_HOLDOUT_FRACTION = 0.2
MAX_CHALLENGE_MACRO_F1_REGRESSION = 0.01
MAX_SELECTIVE_ACCURACY_REGRESSION = 0.01


class FeedbackCandidateError(RuntimeError):
    pass


@dataclass(frozen=True)
class HoldoutMetrics:
    rows: int
    accuracy: float
    macro_f1: float


@dataclass(frozen=True)
class CandidatePaths:
    artifact: Path
    metadata: Path
    evaluation: Path


def candidate_paths(version: str, directory: Path = DEFAULT_CANDIDATE_DIRECTORY) -> CandidatePaths:
    normalized = normalize_artifact_version(version)
    stem = f"transaction_category_{normalized}"
    return CandidatePaths(
        artifact=directory / f"{stem}.pkl",
        metadata=directory / f"{stem}.json",
        evaluation=directory / f"{stem}.evaluation.json",
    )


def _normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _stable_order_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prepare_feedback(
    feedback: pd.DataFrame,
    challenge: pd.DataFrame,
    minimum_rows: int,
    minimum_rows_per_category: int,
    minimum_distinct_categories: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    expense_categories = {category.value for category in ExpenseCategory}
    expense_feedback = feedback.loc[
        feedback["model_scope"].eq("expense_model")
        & feedback["target_category"].isin(expense_categories)
    ].copy()
    expense_feedback["normalized_text"] = expense_feedback["text"].map(_normalized_text)

    challenge_text = challenge.apply(
        lambda row: " ".join(
            part for part in (str(row["description"]), str(row["counterparty"])) if part
        ),
        axis=1,
    ).map(_normalized_text)
    challenge_overlap = expense_feedback["normalized_text"].isin(set(challenge_text))
    excluded_challenge_overlap = int(challenge_overlap.sum())
    expense_feedback = expense_feedback.loc[~challenge_overlap].reset_index(drop=True)

    if len(expense_feedback) < minimum_rows:
        raise FeedbackCandidateError(
            f"Only {len(expense_feedback)} eligible feedback rows remain; "
            f"at least {minimum_rows} are required"
        )
    category_counts = Counter(expense_feedback["target_category"])
    if len(category_counts) < minimum_distinct_categories:
        raise FeedbackCandidateError(
            f"Feedback covers {len(category_counts)} expense categories; "
            f"at least {minimum_distinct_categories} are required"
        )
    sparse_categories = {
        category: count
        for category, count in category_counts.items()
        if count < minimum_rows_per_category
    }
    if sparse_categories:
        raise FeedbackCandidateError(
            "Feedback categories below the minimum row count: "
            f"{dict(sorted(sparse_categories.items()))}"
        )

    holdout_indexes: list[int] = []
    for _, group in expense_feedback.groupby("target_category", sort=True):
        ordered_indexes = sorted(
            group.index,
            key=lambda index: _stable_order_key(expense_feedback.at[index, "normalized_text"]),
        )
        holdout_count = max(1, round(len(ordered_indexes) * FEEDBACK_HOLDOUT_FRACTION))
        holdout_indexes.extend(ordered_indexes[:holdout_count])
    holdout_mask = expense_feedback.index.isin(holdout_indexes)
    feedback_train = expense_feedback.loc[~holdout_mask].reset_index(drop=True)
    feedback_holdout = expense_feedback.loc[holdout_mask].reset_index(drop=True)
    return (
        feedback_train,
        feedback_holdout,
        {
            "eligible_expense_rows": len(expense_feedback),
            "training_rows": len(feedback_train),
            "holdout_rows": len(feedback_holdout),
            "category_counts": dict(sorted(category_counts.items())),
            "excluded_challenge_overlap": excluded_challenge_overlap,
        },
    )


def _holdout_metrics(model: LoadedCategoryModel, holdout: pd.DataFrame) -> HoldoutMetrics:
    expected = holdout["target_category"]
    predicted = model.model.predict(holdout["text"])
    labels = sorted(set(expected))
    return HoldoutMetrics(
        rows=len(holdout),
        accuracy=float(accuracy_score(expected, predicted)),
        macro_f1=float(
            f1_score(expected, predicted, labels=labels, average="macro", zero_division=0)
        ),
    )


def _hybrid_expense_metrics(
    challenge: pd.DataFrame,
    model: LoadedCategoryModel,
) -> MetricSummary:
    evaluations = evaluate_text_classification_strategies(challenge, model)
    return next(
        evaluation.expense_only for evaluation in evaluations if evaluation.strategy == "hybrid"
    )


def _safe_selective_accuracy(metric: MetricSummary) -> float:
    return metric.selective_accuracy if metric.selective_accuracy is not None else 0.0


def train_feedback_candidate(
    feedback_version: str,
    candidate_version: str,
    *,
    feedback_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    candidate_directory: Path = DEFAULT_CANDIDATE_DIRECTORY,
    english_path: Path = DEFAULT_ENGLISH_PATH,
    german_path: Path = DEFAULT_GERMAN_PATH,
    challenge_path: Path = DEFAULT_CHALLENGE_PATH,
    active_artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    active_metadata_path: Path = DEFAULT_METADATA_PATH,
    minimum_rows: int = MIN_FEEDBACK_ROWS,
    minimum_rows_per_category: int = MIN_ROWS_PER_CATEGORY,
    minimum_distinct_categories: int = MIN_DISTINCT_CATEGORIES,
    random_state: int = 42,
) -> tuple[CandidatePaths, dict]:
    version = normalize_artifact_version(candidate_version)
    paths = candidate_paths(version, candidate_directory)
    if any(path.exists() for path in asdict(paths).values()):
        raise FileExistsError(f"Candidate version already exists: {version}")
    for source_path in (english_path, german_path, challenge_path):
        if not source_path.is_file():
            raise FeedbackCandidateError(f"Required dataset not found: {source_path}")

    feedback, feedback_metadata = load_feedback_snapshot(feedback_version, feedback_directory)
    challenge = load_text_classification_challenge(challenge_path)
    feedback_train, feedback_holdout, feedback_report = _prepare_feedback(
        feedback,
        challenge,
        minimum_rows,
        minimum_rows_per_category,
        minimum_distinct_categories,
    )
    active_model = load_category_model_artifact(active_artifact_path, active_metadata_path)

    english_training = training_partition(pd.read_csv(english_path))
    german_training = training_partition(pd.read_csv(german_path))
    controlled_training = pd.concat([english_training, german_training], ignore_index=True)
    candidate_feedback = feedback_train.rename(columns={"text": "description"})[
        ["description", "target_category"]
    ]
    combined_training = pd.concat(
        [controlled_training, candidate_feedback],
        ignore_index=True,
    ).drop_duplicates(subset=["description", "target_category"])
    conflicting_texts = (
        combined_training.assign(
            normalized_text=combined_training["description"].map(_normalized_text)
        )
        .groupby("normalized_text")["target_category"]
        .nunique()
    )
    if (conflicting_texts > 1).any():
        raise FeedbackCandidateError("Combined training data contains conflicting text labels")

    model = train_tfidf_category_classifier(combined_training, random_state=random_state)
    paths.artifact.parent.mkdir(parents=True, exist_ok=True)
    with paths.artifact.open("wb") as destination:
        pickle.dump(model, destination, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = ModelMetadata(
        model_version=f"transaction-category-feedback-{version}",
        taxonomy_version=TAXONOMY_VERSION,
        generator_version=CANDIDATE_PIPELINE_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        training_rows=len(combined_training),
        languages=("en", "de"),
        training_source_sha256={
            "english": calculate_sha256(english_path),
            "german": calculate_sha256(german_path),
            "feedback": str(feedback_metadata["sha256"]),
        },
        artifact_sha256=calculate_sha256(paths.artifact),
        random_state=random_state,
        feature_configuration={
            "vectorizer": "TfidfVectorizer",
            "analyzer": "char_wb",
            "ngram_range": [3, 5],
            "min_df": 2,
            "sublinear_tf": True,
        },
        model_parameters={
            "classifier": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
        },
        library_versions={
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    )
    paths.metadata.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")
    candidate_model = load_category_model_artifact(paths.artifact, paths.metadata)

    active_challenge = _hybrid_expense_metrics(challenge, active_model)
    candidate_challenge = _hybrid_expense_metrics(challenge, candidate_model)
    active_holdout = _holdout_metrics(active_model, feedback_holdout)
    candidate_holdout = _holdout_metrics(candidate_model, feedback_holdout)
    gates = {
        "challenge_macro_f1": (
            candidate_challenge.macro_f1
            >= active_challenge.macro_f1 - MAX_CHALLENGE_MACRO_F1_REGRESSION
        ),
        "challenge_selective_accuracy": (
            _safe_selective_accuracy(candidate_challenge)
            >= _safe_selective_accuracy(active_challenge) - MAX_SELECTIVE_ACCURACY_REGRESSION
        ),
        "feedback_holdout_macro_f1": (candidate_holdout.macro_f1 >= active_holdout.macro_f1),
    }
    report = {
        "pipeline_version": CANDIDATE_PIPELINE_VERSION,
        "candidate_version": version,
        "candidate_model_version": metadata.model_version,
        "candidate_artifact_sha256": metadata.artifact_sha256,
        "baseline_model_version": active_model.metadata.model_version,
        "baseline_artifact_sha256": calculate_sha256(active_artifact_path),
        "feedback_snapshot_version": normalize_artifact_version(feedback_version),
        "feedback_snapshot_sha256": str(feedback_metadata["sha256"]),
        "challenge_sha256": calculate_sha256(challenge_path),
        "feedback": feedback_report,
        "controlled_training_rows": len(controlled_training),
        "combined_training_rows": len(combined_training),
        "quality_configuration": {
            "minimum_rows": minimum_rows,
            "minimum_rows_per_category": minimum_rows_per_category,
            "minimum_distinct_categories": minimum_distinct_categories,
            "feedback_holdout_fraction": FEEDBACK_HOLDOUT_FRACTION,
            "maximum_challenge_macro_f1_regression": MAX_CHALLENGE_MACRO_F1_REGRESSION,
            "maximum_selective_accuracy_regression": MAX_SELECTIVE_ACCURACY_REGRESSION,
        },
        "challenge": {
            "baseline": asdict(active_challenge),
            "candidate": asdict(candidate_challenge),
        },
        "feedback_holdout": {
            "baseline": asdict(active_holdout),
            "candidate": asdict(candidate_holdout),
        },
        "gates": gates,
        "eligible_for_promotion": all(gates.values()),
        "automatic_promotion": False,
    }
    report = json.loads(json.dumps(report))
    paths.evaluation.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return paths, report


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a category-model candidate from reviewed feedback"
    )
    parser.add_argument("--feedback-version", required=True)
    parser.add_argument("--candidate-version", required=True)
    args = parser.parse_args()
    paths, report = train_feedback_candidate(args.feedback_version, args.candidate_version)
    print(
        f"Candidate ready: {paths.artifact} "
        f"(eligible_for_promotion={report['eligible_for_promotion']}; "
        f"evaluation={paths.evaluation})"
    )


if __name__ == "__main__":
    run()
