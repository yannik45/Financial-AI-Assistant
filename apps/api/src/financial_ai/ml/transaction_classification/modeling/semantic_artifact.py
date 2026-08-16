import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from financial_ai.ml.transaction_classification.core.contracts import TAXONOMY_VERSION
from financial_ai.ml.transaction_classification.data.classification_v2_dataset import (
    load_classification_v2_dataset,
)
from financial_ai.ml.transaction_classification.modeling.semantic_embeddings import (
    ENCODER_ID,
    ENCODER_REVISION,
    SentenceEncoder,
    load_or_create_embedding_cache,
    prepare_semantic_text,
)

MODEL_VERSION = "transaction-category-e5-head-v1"
DEFAULT_ARTIFACT_PATH = Path("data/runtime/ml/models/transaction_category_e5_head_v1.npz")
DEFAULT_METADATA_PATH = Path("data/runtime/ml/models/transaction_category_e5_head_v1.json")


class SemanticArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class SemanticHeadMetadata:
    model_version: str
    taxonomy_version: str
    encoder_id: str
    encoder_revision: str
    created_at: str
    training_rows: int
    embedding_dimensions: int
    artifact_sha256: str


@dataclass(frozen=True)
class LoadedSemanticHead:
    coefficients: np.ndarray
    intercepts: np.ndarray
    classes: np.ndarray
    metadata: SemanticHeadMetadata

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.metadata.embedding_dimensions:
            raise ValueError("Embedding shape does not match semantic head")
        logits = embeddings @ self.coefficients.T + self.intercepts
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_semantic_head_artifact(
    encoder: SentenceEncoder,
    *,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> SemanticHeadMetadata:
    dataset = load_classification_v2_dataset()
    training = dataset.train.loc[
        dataset.train["input_slice"].eq("bank_feed")
        & dataset.train["target_category"].ne("other")
    ].reset_index(drop=True)
    embedding_rows = training[["example_id", "description"]]
    embeddings = load_or_create_embedding_cache(embedding_rows, encoder).values
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1_000,
        random_state=42,
    ).fit(embeddings, training["target_category"])

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = artifact_path.with_suffix(".npz.part")
    with temporary_path.open("wb") as destination:
        np.savez_compressed(
            destination,
            coefficients=classifier.coef_.astype(np.float32),
            intercepts=classifier.intercept_.astype(np.float32),
            classes=np.asarray(classifier.classes_, dtype=str),
        )
    temporary_path.replace(artifact_path)
    metadata = SemanticHeadMetadata(
        model_version=MODEL_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        encoder_id=ENCODER_ID,
        encoder_revision=ENCODER_REVISION,
        created_at=datetime.now(UTC).isoformat(),
        training_rows=len(training),
        embedding_dimensions=embeddings.shape[1],
        artifact_sha256=_sha256(artifact_path),
    )
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")
    return metadata


def load_semantic_head_artifact(
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> LoadedSemanticHead:
    if not artifact_path.is_file() or not metadata_path.is_file():
        raise SemanticArtifactError("Semantic classification artifact is unavailable")
    metadata = SemanticHeadMetadata(**json.loads(metadata_path.read_text(encoding="utf-8")))
    if (
        metadata.model_version != MODEL_VERSION
        or metadata.taxonomy_version != TAXONOMY_VERSION
        or metadata.encoder_id != ENCODER_ID
        or metadata.encoder_revision != ENCODER_REVISION
    ):
        raise SemanticArtifactError("Semantic classification metadata is incompatible")
    if _sha256(artifact_path) != metadata.artifact_sha256:
        raise SemanticArtifactError("Semantic classification artifact checksum mismatch")
    with np.load(artifact_path, allow_pickle=False) as values:
        coefficients = values["coefficients"]
        intercepts = values["intercepts"]
        classes = values["classes"]
    if coefficients.shape != (len(classes), metadata.embedding_dimensions):
        raise SemanticArtifactError("Semantic classification artifact shape is invalid")
    if intercepts.shape != (len(classes),):
        raise SemanticArtifactError("Semantic classification intercept shape is invalid")
    return LoadedSemanticHead(coefficients, intercepts, classes, metadata)


def semantic_probabilities(
    texts: list[str],
    encoder: SentenceEncoder,
    head: LoadedSemanticHead,
) -> np.ndarray:
    prepared = [prepare_semantic_text(text) for text in texts]
    embeddings = np.asarray(
        encoder.encode(
            prepared,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    return head.predict_proba(embeddings)
