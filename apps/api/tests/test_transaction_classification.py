import numpy as np
import pandas as pd
import pytest
from financial_ai.ml.category_artifact import LoadedCategoryModel, ModelMetadata
from financial_ai.ml.category_model import train_tfidf_category_classifier
from financial_ai.ml.category_service import TransactionClassifier
from financial_ai.ml.transaction_classification import (
    ClassificationMethod,
    ClassificationRoute,
    FeedbackStatus,
    TransactionCategory,
    determine_feedback_status,
    parse_product_category,
    route_transaction_type,
)


@pytest.mark.parametrize(
    ("transaction_type", "expected_category"),
    [
        ("salary", TransactionCategory.INCOME),
        ("interest", TransactionCategory.INCOME),
        ("dividend", TransactionCategory.INVESTMENTS),
        ("security_buy", TransactionCategory.INVESTMENTS),
        ("security_sell", TransactionCategory.INVESTMENTS),
        ("fee", TransactionCategory.FEES),
        ("tax", TransactionCategory.TAXES),
        ("deposit", TransactionCategory.SAVINGS),
        ("cash_withdrawal", TransactionCategory.CASH),
    ],
)
def test_deterministic_transaction_types_are_routed(transaction_type, expected_category):
    decision = route_transaction_type(transaction_type)
    assert decision.route is ClassificationRoute.DETERMINISTIC
    assert decision.category is expected_category
    assert decision.method is ClassificationMethod.DETERMINISTIC


@pytest.mark.parametrize("transaction_type", ["card_payment", "direct_debit"])
def test_merchant_expenses_are_routed_to_model(transaction_type):
    assert route_transaction_type(transaction_type).route is ClassificationRoute.EXPENSE_MODEL


@pytest.mark.parametrize("transaction_type", ["transfer", "withdrawal"])
def test_ambiguous_transaction_types_require_review(transaction_type):
    assert route_transaction_type(transaction_type).route is ClassificationRoute.NEEDS_REVIEW


def test_transaction_type_routing_normalizes_and_rejects_empty_values():
    assert route_transaction_type(" SALARY ").category is TransactionCategory.INCOME
    with pytest.raises(ValueError, match="must not be empty"):
        route_transaction_type("  ")


def _loaded_model() -> LoadedCategoryModel:
    training_data = pd.DataFrame(
        {
            "description": [
                "supermarket food",
                "supermarket groceries",
                "restaurant dinner",
                "restaurant lunch",
            ],
            "target_category": ["groceries", "groceries", "dining", "dining"],
        }
    )
    model = train_tfidf_category_classifier(training_data)
    metadata = ModelMetadata(
        model_version="test-model-v1",
        taxonomy_version="transaction-categories-v1",
        generator_version="test",
        created_at="2026-07-31T00:00:00+00:00",
        training_rows=4,
        languages=("en",),
        training_source_sha256={"test": "abc"},
        artifact_sha256="def",
    )
    return LoadedCategoryModel(model=model, metadata=metadata)


def test_classifier_returns_deterministic_result_without_loading_model():
    result = TransactionClassifier().classify("salary", "Monthly salary")
    assert result.category == "income"
    assert result.confidence == 1.0
    assert result.model_version is None
    assert result.needs_review is False


def test_classifier_predicts_expense_and_reports_model_provenance():
    classifier = TransactionClassifier(_loaded_model(), review_threshold=0.01)
    result = classifier.classify("card_payment", "supermarket food")
    assert result.category == "groceries"
    assert result.method is ClassificationMethod.ML
    assert result.confidence is not None
    assert result.needs_review is False
    assert result.model_version == "test-model-v1"


def test_classifier_flags_low_confidence_and_validates_expense_description():
    loaded_model = _loaded_model()
    classifier = TransactionClassifier(loaded_model, review_threshold=1.0)
    result = classifier.classify("direct_debit", "unknown merchant")
    assert result.needs_review is True
    assert result.reason == "Model confidence is below the review threshold."
    with pytest.raises(ValueError, match="Description must not be empty"):
        classifier.classify("card_payment", "  ")


def test_classifier_rejects_invalid_review_threshold():
    with pytest.raises(ValueError, match="Review threshold"):
        TransactionClassifier(review_threshold=0)


def test_model_probability_order_matches_classifier_classes():
    loaded_model = _loaded_model()
    probabilities = loaded_model.model.predict_proba(pd.Series(["restaurant dinner"]))[0]
    predicted_index = int(np.argmax(probabilities))
    assert loaded_model.model.classes_[predicted_index] == "dining"


@pytest.mark.parametrize(
    ("prediction", "final", "expected"),
    [
        ("groceries", "Groceries", FeedbackStatus.ACCEPTED),
        ("groceries", "dining", FeedbackStatus.CORRECTED),
        (None, "dining", FeedbackStatus.MANUAL),
        ("groceries", None, FeedbackStatus.UNREVIEWED),
    ],
)
def test_feedback_status_is_derived_from_prediction_and_final_category(
    prediction, final, expected
):
    assert determine_feedback_status(prediction, final) is expected


def test_product_category_parser_normalizes_and_rejects_unknown_values():
    assert parse_product_category(" Groceries ") == "groceries"
    assert parse_product_category("INCOME") == "income"
    with pytest.raises(ValueError, match="Unsupported transaction category"):
        parse_product_category("unknown")
