import json
from dataclasses import asdict

import pandas as pd
import pytest
from financial_ai.ml.market_forecast.boosting import XGBoostConfig
from financial_ai.ml.market_forecast.boosting_validation_report import (
    build_boosting_validation_report,
    write_boosting_validation_report,
)
from financial_ai.ml.market_forecast.evaluation import VolatilityForecastMetrics


def provenance():
    dataset_metadata = {
        "dataset_version": "dataset-v1",
        "sha256": "dataset-checksum",
        "source_snapshot_version": "snapshot-v1",
        "source_snapshot_sha256": "snapshot-checksum",
        "splits": {"validation": {"row_count": 20}},
    }
    snapshot_metadata = {
        "snapshot_version": "snapshot-v1",
        "sha256": "snapshot-checksum",
        "provider": "test",
        "feed": "test",
    }
    selection_report = {
        "dataset": {"version": "dataset-v1", "sha256": "dataset-checksum"},
        "ranking": [
            {
                "rank": 1,
                "candidate": "flexible",
                "configuration": asdict(XGBoostConfig(max_depth=4, min_child_weight=5.0)),
                "median_boosting_rounds": 144,
            }
        ],
    }
    return dataset_metadata, snapshot_metadata, selection_report


def test_boosting_validation_report_uses_frozen_selection_and_excludes_test(monkeypatch):
    dataset_metadata, snapshot_metadata, selection_report = provenance()
    captured = {}
    dataset = pd.DataFrame({"split": ["validation"], "observed_on": ["2023-12-01"]})
    daily_bars = pd.DataFrame(
        {"observed_on": ["2023-12-01", "2024-01-02"], "marker": ["included", "excluded"]}
    )

    def fake_ewma(received_dataset, bars):
        captured["ewma_markers"] = bars["marker"].tolist()
        return VolatilityForecastMetrics(0.07, 0.1, 0.25)

    monkeypatch.setattr(
        "financial_ai.ml.market_forecast.boosting_validation_report.evaluate_ewma_validation",
        fake_ewma,
    )
    monkeypatch.setattr(
        "financial_ai.ml.market_forecast.boosting_validation_report.evaluate_ridge_validation",
        lambda dataset: VolatilityForecastMetrics(0.06, 0.09, 0.24),
    )

    def fake_evaluate(dataset, *, config, boosting_rounds):
        captured["config"] = config
        captured["boosting_rounds"] = boosting_rounds
        return VolatilityForecastMetrics(0.05, 0.08, 0.2)

    monkeypatch.setattr(
        "financial_ai.ml.market_forecast.boosting_validation_report.evaluate_xgboost_validation",
        fake_evaluate,
    )

    report = build_boosting_validation_report(
        dataset,
        daily_bars,
        dataset_metadata=dataset_metadata,
        snapshot_metadata=snapshot_metadata,
        selection_report=selection_report,
        selection_version="inner-cv-v1",
        selection_sha256="selection-checksum",
    )

    assert report["evaluation_scope"] == "outer_validation"
    assert report["test_split_evaluated"] is False
    assert report["model_selection"]["sha256"] == "selection-checksum"
    assert captured["config"].max_depth == 4
    assert captured["boosting_rounds"] == 144
    assert captured["ewma_markers"] == ["included"]
    assert report["evaluations"]["xgboost"]["mae"] == 0.05


def test_boosting_validation_report_rejects_selection_dataset_mismatch():
    dataset_metadata, snapshot_metadata, selection_report = provenance()
    selection_report["dataset"]["sha256"] = "different-checksum"

    with pytest.raises(ValueError, match="checksums"):
        build_boosting_validation_report(
            pd.DataFrame(),
            pd.DataFrame(),
            dataset_metadata=dataset_metadata,
            snapshot_metadata=snapshot_metadata,
            selection_report=selection_report,
            selection_version="inner-cv-v1",
            selection_sha256="selection-checksum",
        )


def test_boosting_validation_report_is_immutable(tmp_path):
    report = {"schema_version": "test"}

    destination = write_boosting_validation_report(
        report,
        "validation-v1",
        output_directory=tmp_path,
    )

    assert json.loads(destination.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="already exists"):
        write_boosting_validation_report(
            report,
            "validation-v1",
            output_directory=tmp_path,
        )
