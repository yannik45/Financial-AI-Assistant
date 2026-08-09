import numpy as np
import pandas as pd
import pytest
from financial_ai.ml.market_forecast.boosting import (
    BoostingCandidateEvaluation,
    BoostingFoldEvaluation,
    XGBoostCandidate,
    XGBoostConfig,
    build_xgboost_validation_predictions,
    compare_xgboost_candidates,
    evaluate_xgboost_candidate,
    evaluate_xgboost_fold,
)
from financial_ai.ml.market_forecast.evaluation import VolatilityForecastMetrics
from financial_ai.ml.market_forecast.features import FEATURE_COLUMNS
from financial_ai.ml.market_forecast.model_selection import TemporalValidationFold
from financial_ai.ml.market_forecast.targets import TARGET_COLUMN


def temporal_fold() -> TemporalValidationFold:
    rows = []
    for index in range(80):
        feature_level = 0.01 * (index + 1)
        row = {
            "symbol": "AAPL" if index % 2 == 0 else "MSFT",
            "observed_on": pd.Timestamp("2018-01-02") + pd.offsets.BDay(index),
            TARGET_COLUMN: float(np.exp(-2.0 + feature_level + 0.2 * np.sin(index / 5))),
        }
        row.update(
            {
                feature: feature_level * (feature_index + 1)
                for feature_index, feature in enumerate(FEATURE_COLUMNS)
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    return TemporalValidationFold(
        validation_year=2019,
        train=frame.iloc[:60].reset_index(drop=True),
        validation=frame.iloc[60:].reset_index(drop=True),
    )


def test_xgboost_fold_returns_original_scale_metrics_and_early_stopped_rounds():
    config = XGBoostConfig(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=2,
        min_child_weight=1.0,
        early_stopping_rounds=5,
    )

    result = evaluate_xgboost_fold(temporal_fold(), config)

    assert result.validation_year == 2019
    assert 1 <= result.boosting_rounds <= config.n_estimators
    assert result.metrics.mae >= 0
    assert result.metrics.rmse >= result.metrics.mae
    assert result.metrics.qlike >= 0


def test_xgboost_fold_does_not_modify_input_frames():
    fold = temporal_fold()
    expected_train = fold.train.copy(deep=True)
    expected_validation = fold.validation.copy(deep=True)
    config = XGBoostConfig(n_estimators=10, early_stopping_rounds=2)

    evaluate_xgboost_fold(fold, config)

    pd.testing.assert_frame_equal(fold.train, expected_train)
    pd.testing.assert_frame_equal(fold.validation, expected_validation)


def test_xgboost_candidate_aggregates_fold_metrics_and_median_rounds(monkeypatch):
    folds = tuple(
        TemporalValidationFold(year, pd.DataFrame(), pd.DataFrame()) for year in (2019, 2020, 2021)
    )
    fold_results = {
        2019: BoostingFoldEvaluation(2019, 60, VolatilityForecastMetrics(0.06, 0.09, 0.30)),
        2020: BoostingFoldEvaluation(2020, 100, VolatilityForecastMetrics(0.09, 0.14, 0.45)),
        2021: BoostingFoldEvaluation(2021, 80, VolatilityForecastMetrics(0.03, 0.07, 0.15)),
    }

    monkeypatch.setattr(
        "financial_ai.ml.market_forecast.boosting.evaluate_xgboost_fold",
        lambda fold, config: fold_results[fold.validation_year],
    )
    candidate = XGBoostCandidate("balanced", XGBoostConfig())

    result = evaluate_xgboost_candidate(candidate, folds)

    assert result.candidate == candidate
    assert result.folds == tuple(fold_results.values())
    assert result.mean_metrics.mae == pytest.approx(0.06)
    assert result.mean_metrics.rmse == pytest.approx(0.10)
    assert result.mean_metrics.qlike == pytest.approx(0.30)
    assert result.median_boosting_rounds == 80


def test_xgboost_comparison_evaluates_and_ranks_candidates_by_mean_mae(monkeypatch):
    candidates = tuple(
        XGBoostCandidate(name, XGBoostConfig()) for name in ("shallow", "balanced", "flexible")
    )
    mean_mae = {"shallow": 0.07, "balanced": 0.05, "flexible": 0.06}
    evaluated_names = []

    def fake_evaluate(candidate, folds):
        evaluated_names.append(candidate.name)
        return BoostingCandidateEvaluation(
            candidate=candidate,
            folds=(),
            mean_metrics=VolatilityForecastMetrics(mean_mae[candidate.name], 0.1, 0.2),
            median_boosting_rounds=80,
        )

    monkeypatch.setattr(
        "financial_ai.ml.market_forecast.boosting.evaluate_xgboost_candidate",
        fake_evaluate,
    )

    result = compare_xgboost_candidates((), candidates)

    assert evaluated_names == ["shallow", "balanced", "flexible"]
    assert [evaluation.candidate.name for evaluation in result] == [
        "balanced",
        "flexible",
        "shallow",
    ]


def test_xgboost_comparison_rejects_duplicate_candidate_names():
    candidates = (
        XGBoostCandidate("same", XGBoostConfig(max_depth=2)),
        XGBoostCandidate("same", XGBoostConfig(max_depth=4)),
    )

    with pytest.raises(ValueError, match="unique"):
        compare_xgboost_candidates((), candidates)


def test_xgboost_validation_fit_uses_fixed_rounds_and_returns_validation_rows(monkeypatch):
    fold = temporal_fold()
    train = fold.train.copy()
    train["split"] = "train"
    validation = fold.validation.copy()
    validation["split"] = "validation"
    test = fold.validation.copy()
    test["split"] = "test"
    dataset = pd.concat([train, validation, test], ignore_index=True)
    captured = {}

    class FakeRegressor:
        def __init__(self, **kwargs):
            captured["configuration"] = kwargs

        def fit(self, features, targets):
            captured["fit_rows"] = len(features)
            captured["fit_targets"] = targets.copy()
            return self

        def predict(self, features):
            captured["prediction_rows"] = len(features)
            return np.full(len(features), np.log(0.2))

    monkeypatch.setattr(
        "financial_ai.ml.market_forecast.boosting.XGBRegressor",
        FakeRegressor,
    )

    result = build_xgboost_validation_predictions(
        dataset,
        config=XGBoostConfig(max_depth=4),
        boosting_rounds=144,
    )

    assert captured["configuration"]["n_estimators"] == 144
    assert "early_stopping_rounds" not in captured["configuration"]
    assert captured["fit_rows"] == len(train)
    assert captured["prediction_rows"] == len(validation)
    assert len(result) == len(validation)
    assert result["predicted_volatility"].eq(0.2).all()
    assert list(result.columns) == [
        "symbol",
        "observed_on",
        TARGET_COLUMN,
        "predicted_volatility",
    ]


@pytest.mark.parametrize("boosting_rounds", [0, -1])
def test_xgboost_validation_fit_rejects_non_positive_rounds(boosting_rounds):
    with pytest.raises(ValueError, match="boosting rounds"):
        build_xgboost_validation_predictions(
            pd.DataFrame(),
            config=XGBoostConfig(),
            boosting_rounds=boosting_rounds,
        )
