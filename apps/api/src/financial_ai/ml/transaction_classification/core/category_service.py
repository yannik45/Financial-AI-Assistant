from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import pandas as pd

from financial_ai.config import get_settings
from financial_ai.ml.transaction_classification.core.contracts import (
    TAXONOMY_VERSION,
    ClassificationInputSource,
    ClassificationMethod,
    ClassificationRoute,
    route_transaction_text,
)
from financial_ai.ml.transaction_classification.modeling.category_artifact import (
    LoadedCategoryModel,
    ModelArtifactError,
    load_category_model_artifact,
)
from financial_ai.ml.transaction_classification.modeling.semantic_artifact import (
    LoadedSemanticHead,
    SemanticArtifactError,
    load_semantic_head_artifact,
    semantic_probabilities,
)
from financial_ai.ml.transaction_classification.modeling.semantic_embeddings import (
    SentenceEncoder,
    load_sentence_encoder,
)

DEFAULT_REVIEW_THRESHOLD = 0.65
MANUAL_REVIEW_THRESHOLD = 0.44150763750076294


@dataclass(frozen=True)
class TransactionClassification:
    category: str | None
    route: ClassificationRoute
    method: ClassificationMethod
    confidence: float | None
    needs_review: bool
    reason: str
    taxonomy_version: str
    model_version: str | None
    input_source: ClassificationInputSource = ClassificationInputSource.MANUAL_ENTRY
    alternative_category: str | None = None
    alternative_model_version: str | None = None
    model_agreement: bool | None = None


@dataclass(frozen=True)
class ClassificationServiceStatus:
    status: str
    mode: str
    tfidf_model_version: str | None
    semantic_model_version: str | None
    reason: str | None = None


class TransactionClassifier:
    def __init__(
        self,
        loaded_model: LoadedCategoryModel | None = None,
        review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
        artifact_path: Path | None = None,
        metadata_path: Path | None = None,
        semantic_head: LoadedSemanticHead | None = None,
        semantic_encoder: SentenceEncoder | None = None,
        semantic_enabled: bool = True,
    ) -> None:
        if not 0 < review_threshold <= 1:
            raise ValueError("Review threshold must be greater than 0 and at most 1")
        self._loaded_model = loaded_model
        self._review_threshold = review_threshold
        self._artifact_path = artifact_path
        self._metadata_path = metadata_path
        self._semantic_head = semantic_head
        self._semantic_encoder = semantic_encoder
        self._semantic_enabled = semantic_enabled

    def classify(
        self,
        description: str,
        amount: Decimal,
        counterparty: str | None = None,
        input_source: ClassificationInputSource = ClassificationInputSource.MANUAL_ENTRY,
    ) -> TransactionClassification:
        return self.classify_many(
            [(description, amount, counterparty)], input_source=input_source
        )[0]

    def classify_many(
        self,
        inputs: list[tuple[str, Decimal, str | None]],
        *,
        input_source: ClassificationInputSource,
    ) -> list[TransactionClassification]:
        results: list[TransactionClassification | None] = [None] * len(inputs)
        model_rows: list[tuple[int, str]] = []
        for index, (description, amount, counterparty) in enumerate(inputs):
            decision = route_transaction_text(description, amount, counterparty)
            if decision.route in {
                ClassificationRoute.DETERMINISTIC,
                ClassificationRoute.TEXT_RULE,
            }:
                assert decision.category is not None
                results[index] = TransactionClassification(
                    category=decision.category.value,
                    route=decision.route,
                    method=decision.method,
                    confidence=None,
                    needs_review=False,
                    reason=decision.reason,
                    taxonomy_version=TAXONOMY_VERSION,
                    model_version=None,
                    input_source=input_source,
                )
            elif decision.route is ClassificationRoute.NEEDS_REVIEW:
                results[index] = TransactionClassification(
                    category=None,
                    route=decision.route,
                    method=decision.method,
                    confidence=None,
                    needs_review=True,
                    reason=decision.reason,
                    taxonomy_version=TAXONOMY_VERSION,
                    model_version=None,
                    input_source=input_source,
                )
            else:
                text = " ".join(
                    part
                    for part in (description.strip(), (counterparty or "").strip())
                    if part
                )
                model_rows.append((index, text))

        if model_rows:
            texts = [text for _, text in model_rows]
            tfidf = self._get_model()
            tfidf_probabilities = tfidf.model.predict_proba(pd.Series(texts))
            semantic_ready = True
            try:
                if not self._semantic_enabled:
                    raise SemanticArtifactError("Semantic classification is disabled")
                semantic_head, semantic_encoder = self._get_semantic_model()
                semantic_values = semantic_probabilities(
                    texts, semantic_encoder, semantic_head
                )
            except (SemanticArtifactError, RuntimeError, OSError):
                semantic_ready = False
                semantic_head = None
                semantic_values = None

            for row, (result_index, _) in enumerate(model_rows):
                tfidf_index = int(tfidf_probabilities[row].argmax())
                tfidf_category = str(tfidf.model.classes_[tfidf_index])
                tfidf_confidence = float(tfidf_probabilities[row, tfidf_index])
                if not semantic_ready or semantic_head is None or semantic_values is None:
                    needs_review = tfidf_confidence < self._review_threshold
                    results[result_index] = TransactionClassification(
                        category=tfidf_category,
                        route=ClassificationRoute.EXPENSE_MODEL,
                        method=ClassificationMethod.ML,
                        confidence=tfidf_confidence,
                        needs_review=needs_review,
                        reason=(
                            "Semantic model unavailable; conservative TF-IDF fallback "
                            "requires review."
                            if needs_review
                            else "Semantic model unavailable; conservative TF-IDF fallback "
                            "accepted."
                        ),
                        taxonomy_version=tfidf.metadata.taxonomy_version,
                        model_version=tfidf.metadata.model_version,
                        input_source=input_source,
                    )
                    continue

                semantic_index = int(semantic_values[row].argmax())
                semantic_category = str(semantic_head.classes[semantic_index])
                semantic_confidence = float(semantic_values[row, semantic_index])
                agreement = semantic_category == tfidf_category
                if input_source is ClassificationInputSource.BANK_FEED:
                    needs_review = not agreement
                    reason = (
                        "TF-IDF and semantic models agree."
                        if agreement
                        else "TF-IDF and semantic models disagree; user review is required."
                    )
                else:
                    needs_review = semantic_confidence < MANUAL_REVIEW_THRESHOLD
                    reason = (
                        "Semantic suggestion is below the manual review threshold."
                        if needs_review
                        else "Semantic category suggestion is ready for confirmation."
                    )
                results[result_index] = TransactionClassification(
                    category=semantic_category,
                    route=ClassificationRoute.EXPENSE_MODEL,
                    method=ClassificationMethod.ML,
                    confidence=semantic_confidence,
                    needs_review=needs_review,
                    reason=reason,
                    taxonomy_version=TAXONOMY_VERSION,
                    model_version=semantic_head.metadata.model_version,
                    input_source=input_source,
                    alternative_category=tfidf_category,
                    alternative_model_version=tfidf.metadata.model_version,
                    model_agreement=agreement,
                )

        if any(result is None for result in results):
            raise RuntimeError("Classification result assembly failed")
        return [result for result in results if result is not None]

    def status(self) -> ClassificationServiceStatus:
        try:
            tfidf = self._get_model()
        except ModelArtifactError:
            return ClassificationServiceStatus(
                "unavailable", "none", None, None, "tfidf_unavailable"
            )
        try:
            if not self._semantic_enabled:
                raise SemanticArtifactError("Semantic classification is disabled")
            semantic_head, _ = self._get_semantic_model()
        except (SemanticArtifactError, RuntimeError, OSError):
            return ClassificationServiceStatus(
                "degraded",
                "tfidf_v1_fallback",
                tfidf.metadata.model_version,
                None,
                "semantic_artifact_unavailable",
            )
        return ClassificationServiceStatus(
            "ready",
            "agreement_v2",
            tfidf.metadata.model_version,
            semantic_head.metadata.model_version,
        )

    def _get_semantic_model(self) -> tuple[LoadedSemanticHead, SentenceEncoder]:
        if self._semantic_head is None:
            self._semantic_head = load_semantic_head_artifact()
        if self._semantic_encoder is None:
            self._semantic_encoder = load_sentence_encoder(allow_download=False)
        return self._semantic_head, self._semantic_encoder

    def _get_model(self) -> LoadedCategoryModel:
        if self._loaded_model is None:
            if self._artifact_path is None or self._metadata_path is None:
                self._loaded_model = load_category_model_artifact()
            else:
                self._loaded_model = load_category_model_artifact(
                    self._artifact_path,
                    self._metadata_path,
                )
        return self._loaded_model


@lru_cache
def get_transaction_classifier() -> TransactionClassifier:
    settings = get_settings()
    return TransactionClassifier(
        review_threshold=settings.category_review_threshold,
        artifact_path=settings.category_model_artifact_path,
        metadata_path=settings.category_model_metadata_path,
    )
