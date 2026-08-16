import json

import numpy as np
import pandas as pd
from financial_ai.ml.transaction_classification.modeling.semantic_embeddings import (
    PREPROCESSING_VERSION,
    calculate_embedding_dataset_sha256,
    load_or_create_embedding_cache,
    prepare_semantic_text,
)


class StubEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, sentences, **_):
        self.calls += 1
        return np.asarray(
            [[float(len(sentence)), float(index)] for index, sentence in enumerate(sentences)],
            dtype=np.float32,
        )


def test_semantic_text_is_normalized_for_e5() -> None:
    assert prepare_semantic_text("  Neue   Kleidung ") == "query: Neue Kleidung"


def test_embedding_cache_is_reused_and_invalidated_by_dataset(tmp_path) -> None:
    data = pd.DataFrame(
        {
            "example_id": ["1", "2"],
            "description": ["Klamotten", "Restaurant"],
        }
    )
    encoder = StubEncoder()

    created = load_or_create_embedding_cache(
        data,
        encoder,
        cache_directory=tmp_path,
        encoder_id="test-encoder",
        encoder_revision="revision-1",
    )
    reused = load_or_create_embedding_cache(
        data,
        encoder,
        cache_directory=tmp_path,
        encoder_id="test-encoder",
        encoder_revision="revision-1",
    )
    changed = load_or_create_embedding_cache(
        data.assign(description=["Kleidung", "Restaurant"]),
        encoder,
        cache_directory=tmp_path,
        encoder_id="test-encoder",
        encoder_revision="revision-1",
    )

    assert created.cache_hit is False
    assert reused.cache_hit is True
    assert changed.cache_hit is False
    assert encoder.calls == 2
    assert np.array_equal(created.values, reused.values)
    assert created.metadata.preprocessing_version == PREPROCESSING_VERSION
    assert created.metadata.dataset_sha256 == calculate_embedding_dataset_sha256(data)

    metadata_path = next(tmp_path.glob("*.metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["encoder_revision"] == "revision-1"
