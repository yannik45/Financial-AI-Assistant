from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from financial_ai.ml.transaction_classification.core.categories import ExpenseCategory
from financial_ai.ml.transaction_classification.core.category_service import TransactionClassifier
from financial_ai.ml.transaction_classification.core.contracts import (
    ClassificationInputSource,
    ClassificationMethod,
    ClassificationRoute,
    FeedbackStatus,
    TransactionCategory,
    determine_feedback_status,
    parse_product_category,
    route_transaction_text,
)
from financial_ai.ml.transaction_classification.modeling.category_artifact import (
    LoadedCategoryModel,
    ModelMetadata,
)
from financial_ai.ml.transaction_classification.modeling.category_model import (
    train_tfidf_category_classifier,
)
from financial_ai.ml.transaction_classification.modeling.semantic_artifact import (
    LoadedSemanticHead,
    SemanticHeadMetadata,
)


@pytest.mark.parametrize(
    ("description", "amount", "expected_category"),
    [
        ("Monthly salary", "2500", TransactionCategory.INCOME),
        ("Zinsen Sparkonto", "15", TransactionCategory.INCOME),
        ("Dividend payment", "40", TransactionCategory.INVESTMENTS),
        ("Bank fee", "-5", TransactionCategory.FEES),
        ("Tax office payment", "-200", TransactionCategory.TAXES),
        ("Transfer to savings account", "-300", TransactionCategory.SAVINGS),
        ("ATM cash withdrawal", "-50", TransactionCategory.CASH),
        ("House Payment", "-950", ExpenseCategory.HOUSING),
        ("Miete Wohnung", "-950", ExpenseCategory.HOUSING),
        ("Health insurance premium", "-200", ExpenseCategory.INSURANCE),
    ],
)
def test_reviewable_text_rules_route_product_categories(description, amount, expected_category):
    decision = route_transaction_text(description, Decimal(amount))
    assert decision.route is ClassificationRoute.TEXT_RULE
    assert decision.category is expected_category
    assert decision.method is ClassificationMethod.KEYWORD_RULE


def test_unmatched_outgoing_text_is_routed_to_expense_model():
    decision = route_transaction_text("Unknown merchant reference", Decimal("-20"))
    assert decision.route is ClassificationRoute.EXPENSE_MODEL


def test_text_rules_use_boundaries_so_coffee_does_not_match_fee():
    decision = route_transaction_text("Coffee shop", Decimal("-4"))
    assert decision.route is ClassificationRoute.EXPENSE_MODEL


def test_unmatched_incoming_text_requires_review():
    decision = route_transaction_text("Unknown incoming transfer", Decimal("20"))
    assert decision.route is ClassificationRoute.NEEDS_REVIEW


@pytest.mark.parametrize(
    ("description", "amount"),
    [
        ("Monthly salary", "-2500"),
        ("ATM cash withdrawal", "50"),
        ("Tax office payment", "200"),
        ("Apartment rent", "950"),
    ],
)
def test_text_rule_conflicting_with_cash_flow_requires_review(description, amount):
    decision = route_transaction_text(description, Decimal(amount))

    assert decision.route is ClassificationRoute.NEEDS_REVIEW
    assert decision.category is None
    assert decision.method is ClassificationMethod.NONE
    assert "conflicts" in decision.reason


def test_investment_text_supports_incoming_distributions_and_outgoing_purchases():
    incoming = route_transaction_text("Quarterly dividend", Decimal("25"))
    outgoing = route_transaction_text("Broker securities purchase", Decimal("-250"))

    assert incoming.category is TransactionCategory.INVESTMENTS
    assert outgoing.category is TransactionCategory.INVESTMENTS


def test_text_routing_rejects_empty_description_and_zero_amount():
    with pytest.raises(ValueError, match="Description must not be empty"):
        route_transaction_text("  ", Decimal("10"))
    with pytest.raises(ValueError, match="amount must not be zero"):
        route_transaction_text("Salary", Decimal("0"))


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


class _SemanticEncoder:
    def encode(self, texts, **_):
        return np.asarray(
            [
                [5.0, 0.0]
                if "disagree" in text
                else ([0.0, 5.0] if "market" in text else [5.0, 0.0])
                for text in texts
            ],
            dtype=np.float32,
        )


def _semantic_head() -> LoadedSemanticHead:
    metadata = SemanticHeadMetadata(
        model_version="semantic-test-v1",
        taxonomy_version="transaction-categories-v1",
        encoder_id="test",
        encoder_revision="test",
        created_at="2026-08-16T00:00:00+00:00",
        training_rows=4,
        embedding_dimensions=2,
        artifact_sha256="test",
    )
    return LoadedSemanticHead(
        coefficients=np.eye(2, dtype=np.float32),
        intercepts=np.zeros(2, dtype=np.float32),
        classes=np.asarray(["dining", "groceries"]),
        metadata=metadata,
    )


def test_classifier_returns_text_rule_result_without_loading_model():
    result = TransactionClassifier().classify("Monthly salary", Decimal("2500"))
    assert result.category == "income"
    assert result.method is ClassificationMethod.KEYWORD_RULE
    assert result.confidence is None
    assert result.model_version is None
    assert result.needs_review is False


def test_classifier_predicts_expense_and_reports_model_provenance():
    classifier = TransactionClassifier(
        _loaded_model(), review_threshold=0.01, semantic_enabled=False
    )
    result = classifier.classify("market food purchase", Decimal("-20"))
    assert result.category == "groceries"
    assert result.method is ClassificationMethod.ML
    assert result.confidence is not None
    assert result.needs_review is False
    assert result.model_version == "test-model-v1"


def test_classifier_flags_low_confidence_and_validates_expense_description():
    loaded_model = _loaded_model()
    classifier = TransactionClassifier(loaded_model, review_threshold=1.0, semantic_enabled=False)
    result = classifier.classify("unknown merchant", Decimal("-20"))
    assert result.needs_review is True
    assert result.reason == (
        "Semantic model unavailable; conservative TF-IDF fallback requires review."
    )
    with pytest.raises(ValueError, match="Description must not be empty"):
        classifier.classify("  ", Decimal("-20"))


def test_bank_feed_requires_model_agreement_and_batches_semantic_inference():
    classifier = TransactionClassifier(
        _loaded_model(),
        semantic_head=_semantic_head(),
        semantic_encoder=_SemanticEncoder(),
    )

    agreed, disagreed = classifier.classify_many(
        [
            ("market food purchase", Decimal("-20"), None),
            ("market food purchase disagree", Decimal("-30"), None),
        ],
        input_source=ClassificationInputSource.BANK_FEED,
    )

    assert agreed.category == "groceries"
    assert agreed.alternative_category == "groceries"
    assert agreed.model_agreement is True
    assert agreed.needs_review is False
    assert disagreed.category == "dining"
    assert disagreed.alternative_category == "groceries"
    assert disagreed.model_agreement is False
    assert disagreed.needs_review is True


def test_manual_semantic_suggestion_preserves_tfidf_provenance():
    classifier = TransactionClassifier(
        _loaded_model(),
        semantic_head=_semantic_head(),
        semantic_encoder=_SemanticEncoder(),
    )
    result = classifier.classify("market purchase", Decimal("-20"))

    assert result.category == "groceries"
    assert result.model_version == "semantic-test-v1"
    assert result.alternative_model_version == "test-model-v1"
    assert result.input_source is ClassificationInputSource.MANUAL_ENTRY
    assert classifier.status().mode == "agreement_v2"


def test_classifier_rejects_invalid_review_threshold():
    with pytest.raises(ValueError, match="Review threshold"):
        TransactionClassifier(review_threshold=0)


def test_model_probability_order_matches_classifier_classes():
    loaded_model = _loaded_model()
    probabilities = loaded_model.model.predict_proba(pd.Series(["restaurant dinner"]))[0]
    predicted_index = int(np.argmax(probabilities))
    assert loaded_model.model.classes_[predicted_index] == "dining"


@pytest.mark.parametrize(
    ("prediction", "final", "category_confirmed", "expected"),
    [
        ("groceries", "Groceries", False, FeedbackStatus.ACCEPTED_IMPLICIT),
        ("groceries", "Groceries", True, FeedbackStatus.ACCEPTED_EXPLICIT),
        ("groceries", "dining", True, FeedbackStatus.CORRECTED),
        (None, "dining", True, FeedbackStatus.MANUAL),
        ("groceries", None, False, FeedbackStatus.UNREVIEWED),
    ],
)
def test_feedback_status_is_derived_from_prediction_and_final_category(
    prediction, final, category_confirmed, expected
):
    assert determine_feedback_status(prediction, final, category_confirmed) is expected


def test_product_category_parser_normalizes_and_rejects_unknown_values():
    assert parse_product_category(" Groceries ") == "groceries"
    assert parse_product_category("INCOME") == "income"
    with pytest.raises(ValueError, match="Unsupported transaction category"):
        parse_product_category("unknown")
