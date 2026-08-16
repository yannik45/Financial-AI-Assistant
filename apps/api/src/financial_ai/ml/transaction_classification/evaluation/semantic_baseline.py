import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from financial_ai.ml.transaction_classification.data.classification_v2_dataset import (
    ClassificationV2Dataset,
    load_classification_v2_dataset,
)
from financial_ai.ml.transaction_classification.modeling.category_model import (
    train_tfidf_category_classifier,
)
from financial_ai.ml.transaction_classification.modeling.semantic_embeddings import (
    ENCODER_ID,
    ENCODER_REVISION,
    SentenceEncoder,
    load_or_create_embedding_cache,
    load_sentence_encoder,
)

DEFAULT_REPORT_PATH = Path(
    "data/runtime/ml/transaction_categories/semantic_baseline_validation.json"
)
DEFAULT_TFIDF_REPORT_PATH = Path(
    "data/runtime/ml/transaction_categories/tfidf_v2_validation.json"
)
DEFAULT_MANUAL_PREDICTIONS_PATH = Path(
    "data/runtime/ml/transaction_categories/semantic_manual_validation_predictions.csv"
)
DEFAULT_MANUAL_CONFUSION_PATH = Path(
    "data/runtime/ml/transaction_categories/semantic_manual_validation_confusion.csv"
)
AUTO_ACCEPT_ACCURACY_TARGET = 0.90
RANDOM_STATE = 42


@dataclass(frozen=True)
class SliceEvaluation:
    rows: int
    accuracy: float
    macro_f1: float
    threshold: float
    automatic_coverage: float
    automatically_accepted_accuracy: float | None


@dataclass(frozen=True)
class RejectionEvaluation:
    rows: int
    threshold: float
    rejection_rate: float
    false_automatic_acceptance_rate: float


@dataclass(frozen=True)
class SemanticBaselineReport:
    candidate: str
    encoder_id: str
    encoder_revision: str
    classifier: str
    training_slice: str
    training_rows: int
    validation_rows: int
    embedding_dimensions: int
    embedding_cache_hit: bool
    auto_accept_accuracy_target: float
    validation: dict[str, SliceEvaluation]
    out_of_scope_validation: dict[str, RejectionEvaluation]
    test_partition_used: bool


@dataclass(frozen=True)
class SemanticValidationRun:
    report: SemanticBaselineReport
    validation: pd.DataFrame
    predictions: np.ndarray
    confidence: np.ndarray


def evaluate_prediction_slice(
    actual: pd.Series,
    predicted: np.ndarray,
    confidence: np.ndarray,
    *,
    accuracy_target: float = AUTO_ACCEPT_ACCURACY_TARGET,
) -> SliceEvaluation:
    if len(actual) == 0:
        raise ValueError("Evaluation slice must not be empty")
    correctness = predicted == actual.to_numpy()
    threshold = 1.0
    accepted = np.zeros(len(actual), dtype=bool)
    for candidate in sorted(np.unique(confidence)):
        candidate_accepted = confidence >= candidate
        candidate_accuracy = correctness[candidate_accepted].mean()
        if candidate_accuracy >= accuracy_target:
            threshold = float(candidate)
            accepted = candidate_accepted
            break

    accepted_accuracy = float(correctness[accepted].mean()) if accepted.any() else None
    return SliceEvaluation(
        rows=len(actual),
        accuracy=float(accuracy_score(actual, predicted)),
        macro_f1=float(f1_score(actual, predicted, average="macro", zero_division=0)),
        threshold=threshold,
        automatic_coverage=float(accepted.mean()),
        automatically_accepted_accuracy=accepted_accuracy,
    )


def evaluate_rejection_slice(
    confidence: np.ndarray,
    *,
    threshold: float,
) -> RejectionEvaluation:
    if len(confidence) == 0:
        raise ValueError("Rejection slice must not be empty")
    automatically_accepted = confidence >= threshold
    return RejectionEvaluation(
        rows=len(confidence),
        threshold=threshold,
        rejection_rate=float((~automatically_accepted).mean()),
        false_automatic_acceptance_rate=float(automatically_accepted.mean()),
    )


def evaluate_validation_predictions(
    validation: pd.DataFrame,
    predictions: np.ndarray,
    confidence: np.ndarray,
) -> tuple[dict[str, SliceEvaluation], dict[str, RejectionEvaluation]]:
    in_scope = validation["target_category"].ne("other")
    slices = {
        "all_in_scope": in_scope.to_numpy(),
        "bank_feed_in_scope": (
            validation["input_slice"].eq("bank_feed") & in_scope
        ).to_numpy(),
        "manual_short_in_scope": (
            validation["input_slice"].eq("manual_short") & in_scope
        ).to_numpy(),
        "manual_known_concept": validation["generalization_slice"]
        .eq("known_concept_new_phrase")
        .where(in_scope, False)
        .to_numpy(),
        "manual_novel_concept": validation["generalization_slice"]
        .eq("novel_concept")
        .where(in_scope, False)
        .to_numpy(),
    }
    evaluations = {
        name: evaluate_prediction_slice(
            validation.loc[mask, "target_category"],
            predictions[mask],
            confidence[mask],
        )
        for name, mask in slices.items()
    }
    out_of_scope_mask = (
        validation["input_slice"].eq("manual_short")
        & validation["target_category"].eq("other")
    ).to_numpy()
    manual_threshold = evaluations["manual_short_in_scope"].threshold
    return evaluations, {
        "manual_other": evaluate_rejection_slice(
            confidence[out_of_scope_mask],
            threshold=manual_threshold,
        )
    }


def run_semantic_validation_with_predictions(
    dataset: ClassificationV2Dataset,
    encoder: SentenceEncoder,
) -> SemanticValidationRun:
    training = dataset.train.loc[
        dataset.train["input_slice"].eq("bank_feed")
        & dataset.train["target_category"].ne("other")
    ].reset_index(drop=True)
    validation = dataset.validation.reset_index(drop=True)
    embedding_data = pd.concat(
        [
            training[["example_id", "description"]],
            validation[["example_id", "description"]],
        ],
        ignore_index=True,
    )
    cached = load_or_create_embedding_cache(embedding_data, encoder)
    training_embeddings = cached.values[: len(training)]
    validation_embeddings = cached.values[len(training) :]

    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1_000,
        random_state=RANDOM_STATE,
    )
    classifier.fit(training_embeddings, training["target_category"])
    predictions = classifier.predict(validation_embeddings)
    confidence = classifier.predict_proba(validation_embeddings).max(axis=1)

    evaluations, out_of_scope_evaluations = evaluate_validation_predictions(
        validation, predictions, confidence
    )
    report = SemanticBaselineReport(
        candidate="frozen-multilingual-e5-small-logistic-v1",
        encoder_id=ENCODER_ID,
        encoder_revision=ENCODER_REVISION,
        classifier="LogisticRegression(C=1.0,class_weight=balanced)",
        training_slice="bank_feed_only",
        training_rows=len(training),
        validation_rows=len(validation),
        embedding_dimensions=cached.metadata.dimensions,
        embedding_cache_hit=cached.cache_hit,
        auto_accept_accuracy_target=AUTO_ACCEPT_ACCURACY_TARGET,
        validation=evaluations,
        out_of_scope_validation=out_of_scope_evaluations,
        test_partition_used=False,
    )
    return SemanticValidationRun(
        report=report,
        validation=validation,
        predictions=predictions,
        confidence=confidence,
    )


def write_manual_validation_diagnostics(
    validation_run: SemanticValidationRun,
    *,
    predictions_destination: Path = DEFAULT_MANUAL_PREDICTIONS_PATH,
    confusion_destination: Path = DEFAULT_MANUAL_CONFUSION_PATH,
) -> tuple[Path, Path]:
    manual_mask = validation_run.validation["input_slice"].eq("manual_short")
    manual = validation_run.validation.loc[manual_mask].copy()
    manual["predicted_category"] = validation_run.predictions[manual_mask]
    manual["confidence"] = validation_run.confidence[manual_mask]
    manual["correct"] = manual["target_category"].eq(manual["predicted_category"])
    output_columns = [
        "example_id",
        "description",
        "language",
        "generalization_slice",
        "target_category",
        "predicted_category",
        "confidence",
        "correct",
    ]
    predictions_destination.parent.mkdir(parents=True, exist_ok=True)
    manual.loc[:, output_columns].to_csv(predictions_destination, index=False)

    labels = sorted(
        set(manual["target_category"]).union(manual["predicted_category"])
    )
    confusion = pd.crosstab(
        manual["target_category"],
        manual["predicted_category"],
        rownames=["actual"],
        colnames=["predicted"],
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)
    confusion.to_csv(confusion_destination)
    return predictions_destination, confusion_destination


def run_tfidf_validation_with_predictions(
    dataset: ClassificationV2Dataset,
) -> SemanticValidationRun:
    training = dataset.train.loc[
        dataset.train["input_slice"].eq("bank_feed")
        & dataset.train["target_category"].ne("other")
    ].reset_index(drop=True)
    validation = dataset.validation.reset_index(drop=True)
    classifier = train_tfidf_category_classifier(
        training[["description", "target_category"]],
        random_state=RANDOM_STATE,
    )
    predictions = classifier.predict(validation["description"])
    confidence = classifier.predict_proba(validation["description"]).max(axis=1)
    evaluations, out_of_scope_evaluations = evaluate_validation_predictions(
        validation, predictions, confidence
    )
    report = SemanticBaselineReport(
        candidate="character-tfidf-logistic-v2-comparison",
        encoder_id="TfidfVectorizer(char_wb,3-5)",
        encoder_revision="scikit-learn-pipeline",
        classifier="LogisticRegression(class_weight=balanced)",
        training_slice="bank_feed_only",
        training_rows=len(training),
        validation_rows=len(validation),
        embedding_dimensions=len(classifier.named_steps["tfidf"].vocabulary_),
        embedding_cache_hit=False,
        auto_accept_accuracy_target=AUTO_ACCEPT_ACCURACY_TARGET,
        validation=evaluations,
        out_of_scope_validation=out_of_scope_evaluations,
        test_partition_used=False,
    )
    return SemanticValidationRun(
        report=report,
        validation=validation,
        predictions=predictions,
        confidence=confidence,
    )


def write_semantic_validation_report(
    report: SemanticBaselineReport,
    destination: Path = DEFAULT_REPORT_PATH,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    return destination


def run() -> None:
    dataset = load_classification_v2_dataset()
    tfidf_report = run_tfidf_validation_with_predictions(dataset).report
    write_semantic_validation_report(tfidf_report, DEFAULT_TFIDF_REPORT_PATH)
    encoder = load_sentence_encoder()
    validation_run = run_semantic_validation_with_predictions(dataset, encoder)
    destination = write_semantic_validation_report(validation_run.report)
    predictions_path, confusion_path = write_manual_validation_diagnostics(
        validation_run
    )
    print(f"Semantic validation report: {destination}")
    print(f"Manual validation predictions: {predictions_path}")
    print(f"Manual validation confusion matrix: {confusion_path}")


if __name__ == "__main__":
    run()
