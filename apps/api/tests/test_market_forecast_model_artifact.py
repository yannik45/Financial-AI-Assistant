import financial_ai.ml.market_forecast.model_artifact as model_artifact_module
import numpy as np
import pandas as pd
import pytest
from financial_ai.ml.market_forecast.features import FEATURE_COLUMNS
from financial_ai.ml.market_forecast.model_artifact import (
    MODEL_ARTIFACT_SCHEMA_VERSION,
    MarketForecastArtifactError,
    build_market_forecast_model_artifact,
    load_market_forecast_model_artifact,
    predict_volatility,
    prepare_deployment_training_data,
)
from financial_ai.ml.market_forecast.targets import TARGET_COLUMN


def model_dataset() -> pd.DataFrame:
    rows = []
    for index, split in enumerate(("train", "validation", "test"), start=1):
        row = {
            "symbol": "AAPL" if index < 3 else "MSFT",
            "observed_on": pd.Timestamp("2021-01-04") + pd.DateOffset(years=index - 1),
            "split": split,
            TARGET_COLUMN: 0.1 * index,
        }
        row.update(
            {
                feature: float(index + feature_index)
                for feature_index, feature in enumerate(FEATURE_COLUMNS)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def model_dataset_metadata() -> dict[str, object]:
    return {
        "dataset_version": "test-v1",
        "sha256": "a" * 64,
        "source_provider": "alpaca",
        "source_feed": "sip",
    }


def test_deployment_training_data_uses_all_splits_and_preserves_feature_contract():
    dataset = model_dataset()
    expected = dataset.copy(deep=True)

    result = prepare_deployment_training_data(dataset)

    assert result.row_count == 3
    assert list(result.features.columns) == list(FEATURE_COLUMNS)
    assert result.log_targets == pytest.approx(np.log([0.1, 0.2, 0.3]))
    assert result.date_from == "2021-01-04"
    assert result.date_to == "2023-01-04"
    assert result.symbols == ("AAPL", "MSFT")
    pd.testing.assert_frame_equal(dataset, expected)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (FEATURE_COLUMNS[0], np.inf, "finite"),
        (TARGET_COLUMN, 0.0, "positive"),
    ],
)
def test_deployment_training_data_rejects_invalid_numeric_values(column, value, message):
    dataset = model_dataset()
    dataset.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        prepare_deployment_training_data(dataset)


def test_deployment_training_data_requires_all_evaluation_splits():
    dataset = model_dataset().loc[lambda rows: rows["split"] != "test"]

    with pytest.raises(ValueError, match="train, validation, and test"):
        prepare_deployment_training_data(dataset)


def test_market_forecast_artifact_round_trip_preserves_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(model_artifact_module, "FINAL_BOOSTING_ROUNDS", 2)
    dataset = model_dataset()
    artifact_path = tmp_path / "model.ubj"
    metadata_path = tmp_path / "model.metadata.json"

    metadata = build_market_forecast_model_artifact(
        dataset,
        dataset_metadata=model_dataset_metadata(),
        artifact_path=artifact_path,
        metadata_path=metadata_path,
    )
    first_load = load_market_forecast_model_artifact(artifact_path, metadata_path)
    predictions_before_reload = first_load.model.predict(dataset.loc[:, FEATURE_COLUMNS])
    second_load = load_market_forecast_model_artifact(artifact_path, metadata_path)
    predictions_after_reload = second_load.model.predict(dataset.loc[:, FEATURE_COLUMNS])
    prediction = predict_volatility(
        second_load,
        dataset.loc[[0], list(reversed(FEATURE_COLUMNS))],
    )

    assert artifact_path.is_file()
    assert metadata_path.is_file()
    assert metadata.schema_version == MODEL_ARTIFACT_SCHEMA_VERSION
    assert metadata.training_purpose == "post_evaluation_deployment_refit"
    assert metadata.training_dataset_version == "test-v1"
    assert metadata.training_dataset_sha256 == "a" * 64
    assert metadata.training_source_provider == "alpaca"
    assert metadata.training_source_feed == "sip"
    assert metadata.training_rows == len(dataset)
    assert metadata.feature_columns == tuple(FEATURE_COLUMNS)
    assert predictions_after_reload == pytest.approx(predictions_before_reload)
    assert np.isfinite(predictions_after_reload).all()
    assert prediction == pytest.approx(float(np.exp(predictions_after_reload[0])))


def test_market_forecast_artifact_rejects_checksum_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(model_artifact_module, "FINAL_BOOSTING_ROUNDS", 2)
    artifact_path = tmp_path / "model.ubj"
    metadata_path = tmp_path / "model.metadata.json"
    build_market_forecast_model_artifact(
        model_dataset(),
        dataset_metadata=model_dataset_metadata(),
        artifact_path=artifact_path,
        metadata_path=metadata_path,
    )
    artifact_path.write_bytes(artifact_path.read_bytes() + b"corrupt")

    with pytest.raises(MarketForecastArtifactError, match="checksum"):
        load_market_forecast_model_artifact(artifact_path, metadata_path)


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (model_dataset().loc[[], FEATURE_COLUMNS], "exactly one row"),
        (model_dataset().loc[[0, 1], FEATURE_COLUMNS], "exactly one row"),
        (model_dataset().loc[[0], FEATURE_COLUMNS[1:]], "missing"),
        (
            model_dataset().loc[[0], FEATURE_COLUMNS].assign(unexpected=1.0),
            "unexpected",
        ),
        (
            model_dataset().loc[[0], FEATURE_COLUMNS].assign(**{FEATURE_COLUMNS[0]: np.inf}),
            "finite",
        ),
    ],
)
def test_market_forecast_prediction_rejects_invalid_features(
    features,
    message,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(model_artifact_module, "FINAL_BOOSTING_ROUNDS", 2)
    artifact_path = tmp_path / "model.ubj"
    metadata_path = tmp_path / "model.metadata.json"
    build_market_forecast_model_artifact(
        model_dataset(),
        dataset_metadata=model_dataset_metadata(),
        artifact_path=artifact_path,
        metadata_path=metadata_path,
    )
    loaded_model = load_market_forecast_model_artifact(artifact_path, metadata_path)

    with pytest.raises(ValueError, match=message):
        predict_volatility(loaded_model, features)
