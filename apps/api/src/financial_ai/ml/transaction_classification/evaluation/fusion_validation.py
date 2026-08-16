import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from financial_ai.demo_bank_feed import generate_demo_bank_feed
from financial_ai.ml.transaction_classification.data.classification_v2_dataset import (
    ClassificationV2Dataset,
    load_classification_v2_dataset,
)
from financial_ai.ml.transaction_classification.evaluation.semantic_baseline import (
    AUTO_ACCEPT_ACCURACY_TARGET,
    RANDOM_STATE,
    SemanticBaselineReport,
    SemanticValidationRun,
    evaluate_validation_predictions,
)
from financial_ai.ml.transaction_classification.evaluation.validation_diagnostics import (
    HybridEvaluation,
    evaluate_hybrid_bank_predictions,
)
from financial_ai.ml.transaction_classification.modeling.fusion import (
    FusionCategoryClassifier,
    train_fusion_category_classifier,
)
from financial_ai.ml.transaction_classification.modeling.semantic_embeddings import (
    ENCODER_ID,
    ENCODER_REVISION,
    SentenceEncoder,
    load_or_create_embedding_cache,
    load_sentence_encoder,
)

DEFAULT_REPORT_PATH = Path("data/runtime/ml/transaction_categories/fusion_validation.json")
DEFAULT_DEMO_REPORT_PATH = Path(
    "data/runtime/ml/transaction_categories/fusion_demo_validation.json"
)


@dataclass(frozen=True)
class FusionValidationRun(SemanticValidationRun):
    model: FusionCategoryClassifier


def run_fusion_validation(
    dataset: ClassificationV2Dataset,
    encoder: SentenceEncoder,
) -> FusionValidationRun:
    training = dataset.train.loc[
        dataset.train["input_slice"].eq("bank_feed") & dataset.train["target_category"].ne("other")
    ].reset_index(drop=True)
    validation = dataset.validation.reset_index(drop=True)
    embedding_rows = pd.concat(
        [
            training[["example_id", "description"]],
            validation[["example_id", "description"]],
        ],
        ignore_index=True,
    )
    cached = load_or_create_embedding_cache(embedding_rows, encoder)
    training_embeddings = cached.values[: len(training)]
    validation_embeddings = cached.values[len(training) :]
    model = train_fusion_category_classifier(
        training[["description", "target_category"]],
        training_embeddings,
        random_state=RANDOM_STATE,
    )
    predictions = model.predict(validation["description"], validation_embeddings)
    confidence = model.predict_proba(validation["description"], validation_embeddings).max(axis=1)
    evaluations, out_of_scope = evaluate_validation_predictions(validation, predictions, confidence)
    report = SemanticBaselineReport(
        candidate="character-tfidf-multilingual-e5-fusion-logistic-v1",
        encoder_id=ENCODER_ID,
        encoder_revision=ENCODER_REVISION,
        classifier="LogisticRegression(C=1.0,class_weight=balanced)",
        training_slice="bank_feed_only",
        training_rows=len(training),
        validation_rows=len(validation),
        embedding_dimensions=(len(model.vectorizer.vocabulary_) + model.embedding_dimensions),
        embedding_cache_hit=cached.cache_hit,
        auto_accept_accuracy_target=AUTO_ACCEPT_ACCURACY_TARGET,
        validation=evaluations,
        out_of_scope_validation=out_of_scope,
        test_partition_used=False,
    )
    return FusionValidationRun(
        report=report,
        validation=validation,
        predictions=predictions,
        confidence=confidence,
        model=model,
    )


def evaluate_demo_feed(
    validation_run: FusionValidationRun,
    encoder: SentenceEncoder,
) -> dict[str, HybridEvaluation]:
    items = [
        item
        for seed in range(100)
        for item in generate_demo_bank_feed(
            seed=seed,
            year=2026,
            month=8,
            variable_count=12,
        )
    ]
    text = pd.Series(
        [
            " ".join(part for part in (item.description, item.counterparty or "") if part.strip())
            for item in items
        ]
    )
    embedding_rows = pd.DataFrame(
        {
            "example_id": [f"demo-{index}" for index in range(len(items))],
            "description": text,
        }
    )
    embeddings = load_or_create_embedding_cache(embedding_rows, encoder).values
    predictions = validation_run.model.predict(text, embeddings)
    confidence = validation_run.model.predict_proba(text, embeddings).max(axis=1)
    base_rows = pd.DataFrame(
        {
            "description": text,
            "target_category": [item.expected_category for item in items],
            "predicted_category": predictions,
            "confidence": confidence,
        }
    )
    base_rows["correct"] = base_rows["target_category"].eq(base_rows["predicted_category"])
    reports = {}
    for name, threshold in {
        "controlled_validation_threshold": validation_run.report.validation[
            "bank_feed_in_scope"
        ].threshold,
        "conservative_product_threshold": 0.65,
    }.items():
        rows = base_rows.copy()
        rows["model_accepted"] = rows["confidence"].ge(threshold)
        report, _ = evaluate_hybrid_bank_predictions(rows)
        reports[name] = report
    return reports


def run() -> None:
    dataset = load_classification_v2_dataset()
    encoder = load_sentence_encoder()
    result = run_fusion_validation(dataset, encoder)
    DEFAULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT_PATH.write_text(
        json.dumps(asdict(result.report), indent=2) + "\n",
        encoding="utf-8",
    )
    demo_reports = evaluate_demo_feed(result, encoder)
    DEFAULT_DEMO_REPORT_PATH.write_text(
        json.dumps(
            {name: asdict(report) for name, report in demo_reports.items()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Fusion validation report: {DEFAULT_REPORT_PATH}")
    print(f"Fusion demo validation report: {DEFAULT_DEMO_REPORT_PATH}")


if __name__ == "__main__":
    run()
