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
from financial_ai.ml.german_training_split import (
    split_german_training_data_by_merchant,
)
from financial_ai.ml.german_training_split_v2 import split_declared_training_data


@dataclass(frozen=True)
class GermanValidationRun:
    model: Pipeline
    splits: CategoryDataSplits
    validation: CategoryEvaluation


def train_and_evaluate_german_validation(
    generated_data: pd.DataFrame,
    random_state: int = 42,
) -> GermanValidationRun:
    """Fit on grouped train data and evaluate once on grouped validation data."""
    splits = split_german_training_data_by_merchant(generated_data, random_state=random_state)

    model = train_tfidf_category_classifier(splits.train, random_state=random_state)

    validation = evaluate_category_classifier(model=model, evaluation_data=splits.validation)

    return GermanValidationRun(
        model=model,
        splits=splits,
        validation=validation,
    )


def train_and_evaluate_german_v2_validation(
    generated_data: pd.DataFrame,
    random_state: int = 42,
) -> GermanValidationRun:
    """Fit on German v2 train groups and evaluate on its validation groups."""
    splits = split_declared_training_data(generated_data)
    model = train_tfidf_category_classifier(
        splits.train,
        random_state=random_state,
    )
    validation = evaluate_category_classifier(model, splits.validation)
    return GermanValidationRun(
        model=model,
        splits=splits,
        validation=validation,
    )


def train_and_evaluate_controlled_validation(
    generated_data: pd.DataFrame,
    random_state: int = 42,
    training_function: Callable[[pd.DataFrame, int], Pipeline] = (train_tfidf_category_classifier),
) -> GermanValidationRun:
    """Fit and validate any dataset with declared provenance-group splits."""
    splits = split_declared_training_data(generated_data)
    model = training_function(
        splits.train,
        random_state,
    )
    validation = evaluate_category_classifier(model, splits.validation)
    return GermanValidationRun(
        model=model,
        splits=splits,
        validation=validation,
    )
