import pandas as pd
import pytest
from financial_ai.ml.categories import ExpenseCategory
from financial_ai.ml.german_training_split import (
    split_german_training_data_by_merchant,
)


def make_grouped_training_data(groups_per_category: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "description": (
                    f"SYNTHETIC {category.value} MERCHANT {merchant_number} ROW {row_number}"
                ),
                "target_category": category.value,
                "merchant_group": (f"generated_{category.value}_merchant_{merchant_number}"),
            }
            for category in ExpenseCategory
            for merchant_number in range(1, groups_per_category + 1)
            for row_number in range(1, 3)
        ]
    )


def test_split_german_training_data_keeps_merchant_groups_separate():
    splits = split_german_training_data_by_merchant(make_grouped_training_data())

    train_groups = set(splits.train["merchant_group"])
    validation_groups = set(splits.validation["merchant_group"])
    test_groups = set(splits.test["merchant_group"])

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)


def test_split_german_training_data_assigns_six_one_one_groups_per_category():
    splits = split_german_training_data_by_merchant(make_grouped_training_data())

    for category in ExpenseCategory:
        category_value = category.value
        assert (
            splits.train.loc[
                splits.train["target_category"].eq(category_value),
                "merchant_group",
            ].nunique()
            == 6
        )
        assert (
            splits.validation.loc[
                splits.validation["target_category"].eq(category_value),
                "merchant_group",
            ].nunique()
            == 1
        )
        assert (
            splits.test.loc[
                splits.test["target_category"].eq(category_value),
                "merchant_group",
            ].nunique()
            == 1
        )


def test_split_german_training_data_is_deterministic():
    training_data = make_grouped_training_data()

    first = split_german_training_data_by_merchant(training_data, random_state=17)
    second = split_german_training_data_by_merchant(training_data, random_state=17)

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.validation, second.validation)
    pd.testing.assert_frame_equal(first.test, second.test)


def test_split_german_training_data_is_independent_of_row_order():
    training_data = make_grouped_training_data()
    reordered_data = training_data.sample(frac=1, random_state=99).reset_index(drop=True)

    original = split_german_training_data_by_merchant(
        training_data,
        random_state=17,
    )
    reordered = split_german_training_data_by_merchant(
        reordered_data,
        random_state=17,
    )

    assert set(original.train["merchant_group"]) == set(reordered.train["merchant_group"])
    assert set(original.validation["merchant_group"]) == set(reordered.validation["merchant_group"])
    assert set(original.test["merchant_group"]) == set(reordered.test["merchant_group"])


@pytest.mark.parametrize(
    ("groups_per_category", "expected_split_counts"),
    [
        (7, (5, 1, 1)),
        (9, (7, 1, 1)),
        (16, (12, 2, 2)),
    ],
)
def test_split_german_training_data_supports_variable_group_counts(
    groups_per_category,
    expected_split_counts,
):
    training_data = make_grouped_training_data(groups_per_category=groups_per_category)
    expected_train, expected_validation, expected_test = expected_split_counts

    splits = split_german_training_data_by_merchant(training_data)

    assert len(splits.train) + len(splits.validation) + len(splits.test) == len(training_data)
    for category in ExpenseCategory:
        category_value = category.value
        assert (
            splits.train.loc[
                splits.train["target_category"].eq(category_value),
                "merchant_group",
            ].nunique()
            == expected_train
        )
        assert (
            splits.validation.loc[
                splits.validation["target_category"].eq(category_value),
                "merchant_group",
            ].nunique()
            == expected_validation
        )
        assert (
            splits.test.loc[
                splits.test["target_category"].eq(category_value),
                "merchant_group",
            ].nunique()
            == expected_test
        )


def test_split_german_training_data_requires_three_groups_per_category():
    training_data = make_grouped_training_data(groups_per_category=2)

    with pytest.raises(ValueError, match="at least 3 merchant groups"):
        split_german_training_data_by_merchant(training_data)
