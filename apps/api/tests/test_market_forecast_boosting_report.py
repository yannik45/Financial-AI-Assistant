import json

import pytest
from financial_ai.ml.market_forecast.boosting import (
    BoostingCandidateEvaluation,
    BoostingFoldEvaluation,
    XGBoostCandidate,
    XGBoostConfig,
)
from financial_ai.ml.market_forecast.boosting_report import (
    _serialize_candidate_evaluation,
    build_boosting_selection_report,
    write_boosting_selection_report,
)
from financial_ai.ml.market_forecast.evaluation import VolatilityForecastMetrics


def candidate_evaluation() -> BoostingCandidateEvaluation:
    return BoostingCandidateEvaluation(
        candidate=XGBoostCandidate(
            name="balanced",
            config=XGBoostConfig(max_depth=3, min_child_weight=10.0),
        ),
        folds=(
            BoostingFoldEvaluation(
                validation_year=2019,
                boosting_rounds=60,
                metrics=VolatilityForecastMetrics(mae=0.05, rmse=0.07, qlike=0.2),
            ),
            BoostingFoldEvaluation(
                validation_year=2020,
                boosting_rounds=80,
                metrics=VolatilityForecastMetrics(mae=0.07, rmse=0.1, qlike=0.3),
            ),
        ),
        mean_metrics=VolatilityForecastMetrics(mae=0.06, rmse=0.085, qlike=0.25),
        median_boosting_rounds=70,
    )


def test_candidate_evaluation_serialization_preserves_config_metrics_and_rank():
    result = _serialize_candidate_evaluation(1, candidate_evaluation())

    assert result["rank"] == 1
    assert result["candidate"] == "balanced"
    assert result["configuration"]["max_depth"] == 3
    assert result["mean_metrics"] == {"mae": 0.06, "rmse": 0.085, "qlike": 0.25}
    assert result["median_boosting_rounds"] == 70
    assert result["folds"][0] == {
        "validation_year": 2019,
        "boosting_rounds": 60,
        "metrics": {"mae": 0.05, "rmse": 0.07, "qlike": 0.2},
    }


def test_selection_report_records_inner_cv_ranking_without_outer_evaluation(monkeypatch):
    dataset = object()
    folds = (object(), object(), object())
    evaluation = candidate_evaluation()

    def fake_build_folds(received_dataset):
        assert received_dataset is dataset
        return folds

    def fake_compare(received_folds):
        assert received_folds is folds
        return (evaluation,)

    monkeypatch.setattr(
        "financial_ai.ml.market_forecast.boosting_report.build_expanding_training_folds",
        fake_build_folds,
    )
    monkeypatch.setattr(
        "financial_ai.ml.market_forecast.boosting_report.compare_xgboost_candidates",
        fake_compare,
    )

    report = build_boosting_selection_report(
        dataset,
        dataset_metadata={
            "dataset_version": "dataset-v1",
            "sha256": "dataset-checksum",
            "splits": {"train": {"row_count": 100}},
        },
    )

    assert report["evaluation_scope"] == "inner_cross_validation"
    assert report["outer_validation_evaluated"] is False
    assert report["test_split_evaluated"] is False
    assert report["dataset"] == {
        "version": "dataset-v1",
        "sha256": "dataset-checksum",
        "outer_train": {"row_count": 100},
    }
    assert report["fold_strategy"]["validation_years"] == [2019, 2020, 2021]
    assert report["ranking"][0]["rank"] == 1
    assert report["ranking"][0]["candidate"] == "balanced"


def test_boosting_selection_report_is_immutable(tmp_path):
    report = {"schema_version": "test"}

    destination = write_boosting_selection_report(
        report,
        "selection-v1",
        output_directory=tmp_path,
    )

    assert json.loads(destination.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="already exists"):
        write_boosting_selection_report(
            report,
            "selection-v1",
            output_directory=tmp_path,
        )
