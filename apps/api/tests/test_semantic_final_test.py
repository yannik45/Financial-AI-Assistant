import numpy as np
import pandas as pd
from financial_ai.ml.transaction_classification.evaluation.semantic_final_test import (
    evaluate_manual_test_predictions,
)


def test_manual_final_test_separates_in_scope_and_other(monkeypatch) -> None:
    test = pd.DataFrame(
        {
            "target_category": ["shopping", "healthcare", "other"],
            "generalization_slice": [
                "known_concept_new_phrase",
                "novel_concept",
                "novel_concept",
            ],
        }
    )
    monkeypatch.setattr(
        "financial_ai.ml.transaction_classification.evaluation.semantic_final_test.calculate_manual_short_sha256",
        lambda: "dataset-sha",
    )
    monkeypatch.setattr(
        "financial_ai.ml.transaction_classification.evaluation.semantic_final_test._sha256",
        lambda path: "selection-sha",
    )

    report = evaluate_manual_test_predictions(
        test,
        np.asarray(["shopping", "shopping", "shopping"]),
        np.asarray([0.9, 0.4, 0.3]),
        threshold=0.5,
        training_rows=10,
    )

    assert report.in_scope_rows == 2
    assert report.accuracy == 0.5
    assert report.automatic_coverage == 0.5
    assert report.automatically_accepted_accuracy == 1.0
    assert report.other_rejection_rate == 1.0
    assert report.test_partition_used is True
