from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline

from financial_ai.ml.category_evaluation import (
    CategoryEvaluation,
    evaluate_category_classifier,
)
from financial_ai.ml.category_model import train_tfidf_category_classifier
from financial_ai.ml.category_split import CategoryDataSplits

MODEL_COLUMNS = ["description", "target_category"]


@dataclass(frozen=True)
class MultilingualValidationRun:
    model: Pipeline
    english_validation: CategoryEvaluation
    german_validation: CategoryEvaluation
    english_training_rows: int
    german_training_rows: int
    combined_training_rows: int


@dataclass(frozen=True)
class LanguageBalancedTrainingData:
    english: pd.DataFrame
    german: pd.DataFrame
    combined: pd.DataFrame


def build_language_balanced_training_data(
    english_training_data: pd.DataFrame,
    german_training_data: pd.DataFrame,
    random_state: int = 42,
) -> LanguageBalancedTrainingData:
    """Match English row counts to German row counts within each category."""
    german_model_data = (
        german_training_data[MODEL_COLUMNS].sort_values(MODEL_COLUMNS).reset_index(drop=True)
    )
    german_category_counts = german_model_data["target_category"].value_counts()
    english_samples = []

    for category, required_count in german_category_counts.items():
        category_mask = english_training_data["target_category"].eq(category)
        category_candidates = (
            english_training_data.loc[category_mask, MODEL_COLUMNS]
            .sort_values(MODEL_COLUMNS)
            .reset_index(drop=True)
        )
        sampled_candidates = category_candidates.sample(
            n=required_count,
            random_state=random_state,
        )
        english_samples.append(sampled_candidates)

    balanced_english_data = (
        pd.concat(english_samples, ignore_index=True)
        .sort_values(MODEL_COLUMNS)
        .reset_index(drop=True)
    )
    combined_data = pd.concat(
        [balanced_english_data, german_model_data],
        ignore_index=True,
    )

    return LanguageBalancedTrainingData(
        english=balanced_english_data,
        german=german_model_data,
        combined=combined_data,
    )


def train_and_evaluate_multilingual_validation(
    english_splits: CategoryDataSplits,
    german_splits: CategoryDataSplits,
    random_state: int = 42,
    training_function: Callable[[pd.DataFrame, int], Pipeline] = (train_tfidf_category_classifier),
) -> MultilingualValidationRun:
    """Fit one model on both train splits and evaluate each language separately."""
    combined_training_data = pd.concat(
        [
            english_splits.train[MODEL_COLUMNS],
            german_splits.train[MODEL_COLUMNS],
        ],
        ignore_index=True,
    )

    model = training_function(
        combined_training_data,
        random_state,
    )
    english_validation = evaluate_category_classifier(
        model,
        english_splits.validation,
    )
    german_validation = evaluate_category_classifier(
        model,
        german_splits.validation,
    )

    return MultilingualValidationRun(
        model=model,
        english_validation=english_validation,
        german_validation=german_validation,
        english_training_rows=len(english_splits.train),
        german_training_rows=len(german_splits.train),
        combined_training_rows=len(combined_training_data),
    )


def train_and_evaluate_balanced_multilingual_validation(
    english_splits: CategoryDataSplits,
    german_splits: CategoryDataSplits,
    random_state: int = 42,
    training_function: Callable[[pd.DataFrame, int], Pipeline] = (train_tfidf_category_classifier),
) -> MultilingualValidationRun:
    """Fit on category-balanced language samples and evaluate both languages."""
    balanced_data = build_language_balanced_training_data(
        english_splits.train,
        german_splits.train,
        random_state=random_state,
    )
    model = training_function(
        balanced_data.combined,
        random_state,
    )
    english_validation = evaluate_category_classifier(
        model,
        english_splits.validation,
    )
    german_validation = evaluate_category_classifier(
        model,
        german_splits.validation,
    )
    return MultilingualValidationRun(
        model=model,
        english_validation=english_validation,
        german_validation=german_validation,
        english_training_rows=len(balanced_data.english),
        german_training_rows=len(balanced_data.german),
        combined_training_rows=len(balanced_data.combined),
    )
