"""Versioned deployment artifacts for the selected market volatility model."""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

from financial_ai.ml.market_forecast.boosting import DEFAULT_XGBOOST_CANDIDATES
from financial_ai.ml.market_forecast.dataset import (
    DEFAULT_DATASET_DIRECTORY,
    load_model_dataset,
)
from financial_ai.ml.market_forecast.features import FEATURE_COLUMNS
from financial_ai.ml.market_forecast.splits import SPLIT_COLUMN
from financial_ai.ml.market_forecast.targets import TARGET_COLUMN

MODEL_ARTIFACT_SCHEMA_VERSION = "market-volatility-model-artifact-v1"
MODEL_VERSION = "market-volatility-xgboost-v1"
FINAL_CANDIDATE_NAME = "flexible"
FINAL_BOOSTING_ROUNDS = 144
DEFAULT_ARTIFACT_PATH = Path(
    "data/runtime/ml/market_forecast/models/market_volatility_xgboost_v1.ubj"
)
DEFAULT_METADATA_PATH = Path(
    "data/runtime/ml/market_forecast/models/market_volatility_xgboost_v1.metadata.json"
)


class MarketForecastArtifactError(RuntimeError):
    """Raised when a market forecast artifact violates its contract."""


@dataclass(frozen=True)
class DeploymentTrainingData:
    """Validated feature matrix, transformed target, and training coverage."""

    features: pd.DataFrame
    log_targets: np.ndarray
    row_count: int
    date_from: str
    date_to: str
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class MarketForecastModelMetadata:
    """Provenance and compatibility contract for a deployable forecast model."""

    schema_version: str
    model_version: str
    created_at: str
    training_purpose: str
    candidate_name: str
    boosting_rounds: int
    feature_columns: tuple[str, ...]
    target_column: str
    target_transformation: str
    training_dataset_version: str
    training_dataset_sha256: str
    training_source_provider: str
    training_source_feed: str
    training_rows: int
    training_date_from: str
    training_date_to: str
    training_symbols: tuple[str, ...]
    artifact_sha256: str
    model_parameters: dict[str, Any]
    library_versions: dict[str, str]


@dataclass(frozen=True)
class LoadedMarketForecastModel:
    """Checksum-verified model and its deployment metadata."""

    model: XGBRegressor
    metadata: MarketForecastModelMetadata


def prepare_deployment_training_data(dataset: pd.DataFrame) -> DeploymentTrainingData:
    """Prepare all labeled rows for the post-evaluation deployment refit."""
    required_columns = {
        "symbol",
        "observed_on",
        SPLIT_COLUMN,
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Deployment dataset has missing columns: {names}")

    deployment_data = dataset.copy(deep=True)
    available_splits = set(deployment_data[SPLIT_COLUMN].dropna().astype(str).unique())
    required_splits = {"train", "validation", "test"}
    if not required_splits.issubset(available_splits):
        raise ValueError("Deployment dataset must contain train, validation, and test rows")

    observed_on = pd.to_datetime(deployment_data["observed_on"], errors="coerce")
    if observed_on.isna().any():
        raise ValueError("Observed dates must be valid")

    features = deployment_data.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Feature values must be finite")

    targets = pd.to_numeric(deployment_data[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(targets).all():
        raise ValueError("Target values must be finite")
    if not (targets > 0.0).all():
        raise ValueError("Target values must be positive")

    symbols = deployment_data["symbol"].dropna().astype(str).str.strip()
    if len(symbols) != len(deployment_data) or symbols.eq("").any():
        raise ValueError("Symbols must be non-empty")

    return DeploymentTrainingData(
        features=features,
        log_targets=np.log(targets),
        row_count=len(deployment_data),
        date_from=observed_on.min().date().isoformat(),
        date_to=observed_on.max().date().isoformat(),
        symbols=tuple(sorted(symbols.unique())),
    )


def calculate_file_sha256(path: Path) -> str:
    """Calculate a binary SHA-256 checksum without loading the full file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_xgboost_parameters() -> dict[str, Any]:
    """Return the frozen candidate parameters used for the deployment refit."""
    candidate = next(
        (item for item in DEFAULT_XGBOOST_CANDIDATES if item.name == FINAL_CANDIDATE_NAME),
        None,
    )
    if candidate is None:
        raise MarketForecastArtifactError(
            f"Selected XGBoost candidate is unavailable: {FINAL_CANDIDATE_NAME}"
        )
    configuration = replace(
        candidate.config,
        n_estimators=FINAL_BOOSTING_ROUNDS,
        early_stopping_rounds=None,
    )
    return configuration.__dict__.copy()


def build_market_forecast_model_artifact(
    dataset: pd.DataFrame,
    *,
    dataset_metadata: dict[str, object],
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> MarketForecastModelMetadata:
    """Refit the frozen model on all labeled data and write a versioned artifact."""
    dataset_version = dataset_metadata.get("dataset_version")
    dataset_sha256 = dataset_metadata.get("sha256")
    source_provider = dataset_metadata.get("source_provider")
    source_feed = dataset_metadata.get("source_feed")
    if not isinstance(dataset_version, str) or not dataset_version.strip():
        raise ValueError("Training dataset metadata has no dataset version")
    if not isinstance(dataset_sha256, str) or not dataset_sha256.strip():
        raise ValueError("Training dataset metadata has no SHA-256 checksum")
    if not isinstance(source_provider, str) or not source_provider.strip():
        raise ValueError("Training dataset metadata has no source provider")
    if not isinstance(source_feed, str) or not source_feed.strip():
        raise ValueError("Training dataset metadata has no source feed")

    training_data = prepare_deployment_training_data(dataset)
    parameters = selected_xgboost_parameters()
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        **parameters,
    )
    model.fit(training_data.features, training_data.log_targets)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(artifact_path)
    metadata = MarketForecastModelMetadata(
        schema_version=MODEL_ARTIFACT_SCHEMA_VERSION,
        model_version=MODEL_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        training_purpose="post_evaluation_deployment_refit",
        candidate_name=FINAL_CANDIDATE_NAME,
        boosting_rounds=FINAL_BOOSTING_ROUNDS,
        feature_columns=tuple(FEATURE_COLUMNS),
        target_column=TARGET_COLUMN,
        target_transformation="natural_log",
        training_dataset_version=dataset_version,
        training_dataset_sha256=dataset_sha256,
        training_source_provider=source_provider,
        training_source_feed=source_feed,
        training_rows=training_data.row_count,
        training_date_from=training_data.date_from,
        training_date_to=training_data.date_to,
        training_symbols=training_data.symbols,
        artifact_sha256=calculate_file_sha256(artifact_path),
        model_parameters={
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 1,
            **parameters,
        },
        library_versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgboost.__version__,
        },
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(asdict(metadata), indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_market_forecast_model_artifact(
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> LoadedMarketForecastModel:
    """Load a compatible native XGBoost artifact after integrity checks."""
    if not artifact_path.is_file() or not metadata_path.is_file():
        raise MarketForecastArtifactError(
            "Market forecast model artifact is unavailable. Run the build command first."
        )
    try:
        payload: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["feature_columns"] = tuple(payload["feature_columns"])
        payload["training_symbols"] = tuple(payload["training_symbols"])
        metadata = MarketForecastModelMetadata(**payload)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise MarketForecastArtifactError("Market forecast metadata is invalid") from error

    if metadata.schema_version != MODEL_ARTIFACT_SCHEMA_VERSION:
        raise MarketForecastArtifactError("Market forecast artifact schema is incompatible")
    if metadata.model_version != MODEL_VERSION:
        raise MarketForecastArtifactError("Market forecast model version is incompatible")
    if metadata.feature_columns != tuple(FEATURE_COLUMNS):
        raise MarketForecastArtifactError("Market forecast feature contract is incompatible")
    if calculate_file_sha256(artifact_path) != metadata.artifact_sha256:
        raise MarketForecastArtifactError(
            "Market forecast artifact checksum does not match its metadata"
        )

    model = XGBRegressor()
    try:
        model.load_model(artifact_path)
    except xgboost.core.XGBoostError as error:
        raise MarketForecastArtifactError("Market forecast model cannot be loaded") from error
    return LoadedMarketForecastModel(model=model, metadata=metadata)


def predict_volatility(
    loaded_model: LoadedMarketForecastModel,
    features: pd.DataFrame,
) -> float:
    """Predict original-scale annualized volatility for one feature row."""
    if len(features) != 1:
        raise ValueError("Market volatility prediction requires exactly one row")

    expected_columns = loaded_model.metadata.feature_columns
    missing_columns = set(expected_columns).difference(features.columns)
    unexpected_columns = set(features.columns).difference(expected_columns)
    if missing_columns or unexpected_columns:
        details = []
        if missing_columns:
            details.append(f"missing: {', '.join(sorted(missing_columns))}")
        if unexpected_columns:
            details.append(f"unexpected: {', '.join(sorted(unexpected_columns))}")
        raise ValueError(f"Market forecast feature contract is invalid ({'; '.join(details)})")

    model_features = features.loc[:, expected_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(model_features.to_numpy(dtype=float)).all():
        raise ValueError("Market forecast feature values must be finite")

    log_prediction = np.asarray(loaded_model.model.predict(model_features), dtype=float)
    if log_prediction.shape != (1,) or not np.isfinite(log_prediction).all():
        raise MarketForecastArtifactError("Market forecast model returned an invalid prediction")
    prediction = float(np.exp(log_prediction[0]))
    if not np.isfinite(prediction) or prediction <= 0.0:
        raise MarketForecastArtifactError("Market forecast prediction must be finite and positive")
    return prediction


def run() -> None:
    """Build the deployable market forecast model from a verified dataset."""
    parser = argparse.ArgumentParser(description="Build the market volatility model artifact")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--dataset-directory", type=Path, default=DEFAULT_DATASET_DIRECTORY)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    args = parser.parse_args()

    dataset, dataset_metadata = load_model_dataset(
        args.dataset_version,
        args.dataset_directory,
    )
    metadata = build_market_forecast_model_artifact(
        dataset,
        dataset_metadata=dataset_metadata,
        artifact_path=args.artifact_path,
        metadata_path=args.metadata_path,
    )
    print(f"Market forecast model: {args.artifact_path}")
    print(f"Metadata: {args.metadata_path}")
    print(
        f"Training coverage: {metadata.training_rows} rows, "
        f"{metadata.training_date_from} to {metadata.training_date_to}"
    )


if __name__ == "__main__":
    run()
