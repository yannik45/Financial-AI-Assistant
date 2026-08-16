import numpy as np
import pandas as pd
from financial_ai.ml.transaction_classification.evaluation.semantic_baseline import (
    SemanticBaselineReport,
    SemanticValidationRun,
    SliceEvaluation,
)
from financial_ai.ml.transaction_classification.evaluation.validation_diagnostics import (
    build_bank_prediction_rows,
    evaluate_group_generalization,
    evaluate_hybrid_bank_predictions,
)


def test_hybrid_evaluation_applies_rules_before_model_threshold() -> None:
    validation = pd.DataFrame(
        {
            "description": ["Monthly rent", "Corner shop", "Other purpose"],
            "target_category": ["housing", "shopping", "shopping"],
            "input_slice": ["bank_feed"] * 3,
        }
    )
    slice_evaluation = SliceEvaluation(
        rows=3,
        accuracy=2 / 3,
        macro_f1=0.5,
        threshold=0.8,
        automatic_coverage=1 / 3,
        automatically_accepted_accuracy=1.0,
    )
    run = SemanticValidationRun(
        report=SemanticBaselineReport(
            candidate="candidate",
            encoder_id="encoder",
            encoder_revision="revision",
            classifier="classifier",
            training_slice="bank_feed_only",
            training_rows=10,
            validation_rows=3,
            embedding_dimensions=2,
            embedding_cache_hit=True,
            auto_accept_accuracy_target=0.9,
            validation={"bank_feed_in_scope": slice_evaluation},
            out_of_scope_validation={},
            test_partition_used=False,
        ),
        validation=validation,
        predictions=np.asarray(["shopping", "shopping", "housing"]),
        confidence=np.asarray([0.4, 0.9, 0.5]),
    )

    rows = build_bank_prediction_rows(run)
    report, evaluated = evaluate_hybrid_bank_predictions(rows)

    assert evaluated["hybrid_route"].tolist() == [
        "text_rule",
        "expense_model",
        "needs_review",
    ]
    assert report.rules_accepted == 1
    assert report.model_accepted == 1
    assert report.needs_review == 1
    assert report.automatically_accepted_accuracy == 1.0


def test_group_generalization_weights_each_group_once() -> None:
    rows = pd.DataFrame(
        {
            "merchant_group": ["large"] * 4 + ["small"],
            "detail_group": ["detail-large"] * 4 + ["detail-small"],
            "format_group": ["format-large"] * 4 + ["format-small"],
            "correct": [True, True, True, True, False],
        }
    )

    result = evaluate_group_generalization(rows)["merchant_group"]

    assert rows["correct"].mean() == 0.8
    assert result.mean_group_accuracy == 0.5
    assert result.fully_correct_group_rate == 0.5
    assert result.fully_failed_group_rate == 0.5
