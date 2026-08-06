from financial_ai.ml.transaction_classification.german_model_training import (
    train_and_evaluate_controlled_validation,
    train_and_evaluate_german_v2_validation,
    train_and_evaluate_german_validation,
)
from financial_ai.ml.transaction_classification.german_training_generator import (
    generate_german_training_data,
)
from financial_ai.ml.transaction_classification.german_training_generator_v2 import (
    generate_german_training_data_v2,
)


def test_train_and_evaluate_german_validation_uses_grouped_splits():
    generated_data = generate_german_training_data(
        examples_per_category=100,
        random_seed=7,
    )

    result = train_and_evaluate_german_validation(
        generated_data,
        random_state=17,
    )

    train_groups = set(result.splits.train["merchant_group"])
    validation_groups = set(result.splits.validation["merchant_group"])
    test_groups = set(result.splits.test["merchant_group"])

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)
    assert sum(metric.support for metric in result.validation.per_category) == len(
        result.splits.validation
    )
    assert 0.0 <= result.validation.accuracy <= 1.0
    assert 0.0 <= result.validation.macro_f1 <= 1.0


def test_train_and_evaluate_german_v2_validation_uses_declared_splits():
    generated_data = generate_german_training_data_v2(
        examples_per_category=100,
        random_seed=7,
    )

    result = train_and_evaluate_german_v2_validation(
        generated_data,
        random_state=17,
    )

    assert len(result.splits.train) == 76 * 12
    assert len(result.splits.validation) == 12 * 12
    assert len(result.splits.test) == 12 * 12
    assert sum(metric.support for metric in result.validation.per_category) == len(
        result.splits.validation
    )


def test_train_and_evaluate_controlled_validation_uses_declared_splits():
    generated_data = generate_german_training_data_v2(
        examples_per_category=100,
        random_seed=7,
    )

    result = train_and_evaluate_controlled_validation(generated_data)

    assert len(result.splits.train) == 76 * 12
    assert len(result.splits.validation) == 12 * 12
    assert len(result.splits.test) == 12 * 12
