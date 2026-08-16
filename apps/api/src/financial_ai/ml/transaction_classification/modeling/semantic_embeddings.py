import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

ENCODER_ID = "intfloat/multilingual-e5-small"
ENCODER_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
PREPROCESSING_VERSION = "transaction-semantic-text-v1"
DEFAULT_CACHE_DIRECTORY = Path("data/runtime/ml/transaction_categories/embedding_cache")
DEFAULT_ENCODER_DIRECTORY = Path("data/runtime/ml/transaction_categories/encoders")


class SentenceEncoder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class EmbeddingCacheMetadata:
    dataset_sha256: str
    encoder_id: str
    encoder_revision: str
    preprocessing_version: str
    rows: int
    dimensions: int
    normalized: bool
    artifact_sha256: str


@dataclass(frozen=True)
class CachedEmbeddings:
    values: np.ndarray
    metadata: EmbeddingCacheMetadata
    cache_hit: bool


def prepare_semantic_text(value: str) -> str:
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError("Embedding text must not be empty")
    return f"query: {normalized}"


def calculate_embedding_dataset_sha256(data: pd.DataFrame) -> str:
    required = {"example_id", "description"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing embedding dataset columns: {sorted(missing)}")
    canonical_rows = data.loc[:, ["example_id", "description"]].astype(str)
    payload = canonical_rows.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(dataset_sha256: str, encoder_id: str, encoder_revision: str) -> str:
    payload = json.dumps(
        {
            "dataset_sha256": dataset_sha256,
            "encoder_id": encoder_id,
            "encoder_revision": encoder_revision,
            "preprocessing_version": PREPROCESSING_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


@lru_cache(maxsize=4)
def load_sentence_encoder(
    *,
    encoder_id: str = ENCODER_ID,
    encoder_revision: str = ENCODER_REVISION,
    cache_directory: Path = DEFAULT_ENCODER_DIRECTORY,
    allow_download: bool = True,
) -> SentenceEncoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Semantic dependencies are unavailable. Run `uv sync --extra semantic`."
        ) from exc
    return SentenceTransformer(
        encoder_id,
        revision=encoder_revision,
        cache_folder=str(cache_directory),
        local_files_only=not allow_download,
    )


def load_or_create_embedding_cache(
    data: pd.DataFrame,
    encoder: SentenceEncoder,
    *,
    cache_directory: Path = DEFAULT_CACHE_DIRECTORY,
    encoder_id: str = ENCODER_ID,
    encoder_revision: str = ENCODER_REVISION,
    batch_size: int = 64,
) -> CachedEmbeddings:
    dataset_sha256 = calculate_embedding_dataset_sha256(data)
    key = _cache_key(dataset_sha256, encoder_id, encoder_revision)
    artifact_path = cache_directory / f"{key}.npz"
    metadata_path = cache_directory / f"{key}.metadata.json"

    if artifact_path.is_file() and metadata_path.is_file():
        metadata = EmbeddingCacheMetadata(**json.loads(metadata_path.read_text(encoding="utf-8")))
        if (
            metadata.dataset_sha256 == dataset_sha256
            and metadata.encoder_id == encoder_id
            and metadata.encoder_revision == encoder_revision
            and metadata.preprocessing_version == PREPROCESSING_VERSION
            and metadata.rows == len(data)
            and metadata.artifact_sha256 == _artifact_sha256(artifact_path)
        ):
            with np.load(artifact_path, allow_pickle=False) as cached:
                values = cached["embeddings"]
            if values.shape == (metadata.rows, metadata.dimensions):
                return CachedEmbeddings(values=values, metadata=metadata, cache_hit=True)

    texts = [prepare_semantic_text(value) for value in data["description"].astype(str)]
    values = np.asarray(
        encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    if values.ndim != 2 or values.shape[0] != len(data):
        raise ValueError("Encoder returned an unexpected embedding shape")

    cache_directory.mkdir(parents=True, exist_ok=True)
    temporary_path = artifact_path.with_suffix(".npz.part")
    with temporary_path.open("wb") as destination:
        np.savez_compressed(destination, embeddings=values)
    temporary_path.replace(artifact_path)
    metadata = EmbeddingCacheMetadata(
        dataset_sha256=dataset_sha256,
        encoder_id=encoder_id,
        encoder_revision=encoder_revision,
        preprocessing_version=PREPROCESSING_VERSION,
        rows=len(data),
        dimensions=values.shape[1],
        normalized=True,
        artifact_sha256=_artifact_sha256(artifact_path),
    )
    metadata_path.write_text(
        json.dumps(asdict(metadata), indent=2) + "\n",
        encoding="utf-8",
    )
    return CachedEmbeddings(values=values, metadata=metadata, cache_hit=False)
