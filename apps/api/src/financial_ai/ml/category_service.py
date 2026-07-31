from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from financial_ai.config import get_settings
from financial_ai.ml.categories import ExpenseCategory
from financial_ai.ml.category_artifact import LoadedCategoryModel, load_category_model_artifact
from financial_ai.ml.transaction_classification import (
    TAXONOMY_VERSION,
    ClassificationMethod,
    ClassificationRoute,
    route_transaction_type,
)

DEFAULT_REVIEW_THRESHOLD = 0.65


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


class TransactionClassifier:
    def __init__(
        self,
        loaded_model: LoadedCategoryModel | None = None,
        review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
        artifact_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        if not 0 < review_threshold <= 1:
            raise ValueError("Review threshold must be greater than 0 and at most 1")
        self._loaded_model = loaded_model
        self._review_threshold = review_threshold
        self._artifact_path = artifact_path
        self._metadata_path = metadata_path

    def classify(
        self,
        transaction_type: str,
        description: str,
        counterparty: str | None = None,
    ) -> TransactionClassification:
        decision = route_transaction_type(transaction_type)
        if decision.route is ClassificationRoute.DETERMINISTIC:
            assert decision.category is not None
            return TransactionClassification(
                category=decision.category.value,
                route=decision.route,
                method=decision.method,
                confidence=1.0,
                needs_review=False,
                reason=decision.reason,
                taxonomy_version=TAXONOMY_VERSION,
                model_version=None,
            )
        if decision.route is ClassificationRoute.NEEDS_REVIEW:
            return TransactionClassification(
                category=None,
                route=decision.route,
                method=decision.method,
                confidence=None,
                needs_review=True,
                reason=decision.reason,
                taxonomy_version=TAXONOMY_VERSION,
                model_version=None,
            )

        normalized_description = description.strip()
        if not normalized_description:
            raise ValueError("Description must not be empty for expense classification")
        model = self._get_model()
        model_input = " ".join(
            part for part in (normalized_description, (counterparty or "").strip()) if part
        )
        probabilities = model.model.predict_proba(pd.Series([model_input]))[0]
        best_index = int(probabilities.argmax())
        predicted_value = str(model.model.classes_[best_index])
        predicted_category = ExpenseCategory(predicted_value)
        confidence = float(probabilities[best_index])
        needs_review = confidence < self._review_threshold
        return TransactionClassification(
            category=predicted_category.value,
            route=decision.route,
            method=decision.method,
            confidence=confidence,
            needs_review=needs_review,
            reason=(
                "Model confidence is below the review threshold."
                if needs_review
                else "Expense category predicted by the versioned model artifact."
            ),
            taxonomy_version=model.metadata.taxonomy_version,
            model_version=model.metadata.model_version,
        )

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
