import numpy as np
import pandas as pd
from financial_ai.ml.transaction_classification.evaluation.semantic_baseline import (
    SemanticBaselineReport,
    SemanticValidationRun,
    evaluate_prediction_slice,
    evaluate_rejection_slice,
    write_manual_validation_diagnostics,
)


def test_selective_evaluation_maximizes_coverage_at_accuracy_target() -> None:
    actual = pd.Series(["shopping", "shopping", "dining", "dining"])
    predicted = np.asarray(["shopping", "shopping", "dining", "shopping"])
    confidence = np.asarray([0.95, 0.90, 0.80, 0.40])

    result = evaluate_prediction_slice(
        actual,
        predicted,
        confidence,
        accuracy_target=0.90,
    )

    assert result.accuracy == 0.75
    assert result.threshold == 0.80
    assert result.automatic_coverage == 0.75
    assert result.automatically_accepted_accuracy == 1.0


def test_rejection_evaluation_reuses_in_scope_threshold() -> None:
    result = evaluate_rejection_slice(
        np.asarray([0.95, 0.79, 0.40]),
        threshold=0.80,
    )

    assert result.threshold == 0.80
    assert result.rejection_rate == 2 / 3
    assert result.false_automatic_acceptance_rate == 1 / 3


def test_write_manual_validation_diagnostics(tmp_path) -> None:
    validation = pd.DataFrame(
        {
            "example_id": ["bank-1", "manual-1", "manual-2"],
            "description": ["CARD SHOP", "Klamotten", "Kino"],
            "language": ["de", "de", "de"],
            "input_slice": ["bank_feed", "manual_short", "manual_short"],
            "generalization_slice": ["bank_feed", "novel_concept", "novel_concept"],
            "target_category": ["shopping", "shopping", "entertainment"],
        }
    )
    validation_run = SemanticValidationRun(
        report=SemanticBaselineReport(
            candidate="candidate",
            encoder_id="encoder",
            encoder_revision="revision",
            classifier="classifier",
            training_slice="bank_feed_only",
            training_rows=1,
            validation_rows=3,
            embedding_dimensions=2,
            embedding_cache_hit=True,
            auto_accept_accuracy_target=0.9,
            validation={},
            out_of_scope_validation={},
            test_partition_used=False,
        ),
        validation=validation,
        predictions=np.asarray(["shopping", "shopping", "shopping"]),
        confidence=np.asarray([0.9, 0.8, 0.6]),
    )

    predictions_path, confusion_path = write_manual_validation_diagnostics(
        validation_run,
        predictions_destination=tmp_path / "predictions.csv",
        confusion_destination=tmp_path / "confusion.csv",
    )

    predictions = pd.read_csv(predictions_path)
    confusion = pd.read_csv(confusion_path, index_col="actual")
    assert predictions["example_id"].tolist() == ["manual-1", "manual-2"]
    assert predictions["correct"].tolist() == [True, False]
    assert confusion.loc["entertainment", "shopping"] == 1
    assert confusion.loc["shopping", "shopping"] == 1
