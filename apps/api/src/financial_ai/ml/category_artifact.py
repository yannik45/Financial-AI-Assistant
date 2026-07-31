import hashlib
import json
import pickle
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from financial_ai.ml.category_model import train_tfidf_category_classifier
from financial_ai.ml.german_training_split_v2 import split_declared_training_data
from financial_ai.ml.transaction_classification import TAXONOMY_VERSION

MODEL_VERSION = "transaction-category-char-tfidf-bilingual-v1"
GENERATOR_VERSION = "category-artifact-builder-v1"
DEFAULT_ENGLISH_PATH = Path("data/runtime/ml/transaction_categories/english_training_v1.csv")
DEFAULT_GERMAN_PATH = Path("data/runtime/ml/transaction_categories/german_training_v2.csv")
DEFAULT_ARTIFACT_PATH = Path("data/runtime/ml/models/transaction_category_bilingual_v1.pkl")
DEFAULT_METADATA_PATH = Path("data/runtime/ml/models/transaction_category_bilingual_v1.json")
MODEL_COLUMNS = ["description", "target_category"]


class ModelArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelMetadata:
    model_version: str
    taxonomy_version: str
    generator_version: str
    created_at: str
    training_rows: int
    languages: tuple[str, ...]
    training_source_sha256: dict[str, str]
    artifact_sha256: str
    random_state: int | None = None
    feature_configuration: dict[str, Any] | None = None
    model_parameters: dict[str, Any] | None = None
    library_versions: dict[str, str] | None = None


@dataclass(frozen=True)
class LoadedCategoryModel:
    model: Pipeline
    metadata: ModelMetadata


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_partition(data: pd.DataFrame) -> pd.DataFrame:
    splits = split_declared_training_data(data)
    return pd.concat(
        [splits.train[MODEL_COLUMNS], splits.validation[MODEL_COLUMNS]],
        ignore_index=True,
    )


def build_category_model_artifact(
    english_path: Path = DEFAULT_ENGLISH_PATH,
    german_path: Path = DEFAULT_GERMAN_PATH,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    random_state: int = 42,
) -> ModelMetadata:
    for source_path in (english_path, german_path):
        if not source_path.is_file():
            raise ModelArtifactError(f"Training dataset not found: {source_path}")

    english_training = training_partition(pd.read_csv(english_path))
    german_training = training_partition(pd.read_csv(german_path))
    training_data = pd.concat([english_training, german_training], ignore_index=True)
    model = train_tfidf_category_classifier(training_data, random_state=random_state)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as destination:
        pickle.dump(model, destination, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = ModelMetadata(
        model_version=MODEL_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        generator_version=GENERATOR_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        training_rows=len(training_data),
        languages=("en", "de"),
        training_source_sha256={
            "english": calculate_sha256(english_path),
            "german": calculate_sha256(german_path),
        },
        artifact_sha256=calculate_sha256(artifact_path),
        random_state=random_state,
        feature_configuration={
            "vectorizer": "TfidfVectorizer",
            "analyzer": "char_wb",
            "ngram_range": [3, 5],
            "min_df": 2,
            "sublinear_tf": True,
        },
        model_parameters={
            "classifier": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
        },
        library_versions={
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")
    return metadata


def load_category_model_artifact(
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> LoadedCategoryModel:
    if not artifact_path.is_file() or not metadata_path.is_file():
        raise ModelArtifactError(
            "Category model artifact is unavailable. Run the model build command first."
        )

    metadata_payload: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_payload["languages"] = tuple(metadata_payload["languages"])
    metadata = ModelMetadata(**metadata_payload)
    if metadata.taxonomy_version != TAXONOMY_VERSION:
        raise ModelArtifactError("Model artifact taxonomy version is incompatible")
    if calculate_sha256(artifact_path) != metadata.artifact_sha256:
        raise ModelArtifactError("Model artifact checksum does not match its metadata")

    # Pickle must only be loaded from this locally generated, checksum-verified path.
    with artifact_path.open("rb") as source:
        model = pickle.load(source)  # noqa: S301
    if not isinstance(model, Pipeline):
        raise ModelArtifactError("Model artifact does not contain a scikit-learn Pipeline")
    return LoadedCategoryModel(model=model, metadata=metadata)


def run() -> None:
    built_metadata = build_category_model_artifact()
    print(
        f"Category model ready: {DEFAULT_ARTIFACT_PATH} "
        f"({built_metadata.training_rows} training rows)"
    )


if __name__ == "__main__":
    run()
