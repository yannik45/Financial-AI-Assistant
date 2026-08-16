import pandas as pd
from financial_ai.ml.transaction_classification.data.category_grouping import (
    normalize_description_group,
)
from financial_ai.ml.transaction_classification.data.category_split import (
    split_category_training_data,
    split_grouped_category_training_data,
)


def make_prepared_data() -> pd.DataFrame:
    categories = ["groceries", "housing", "shopping", "travel"]
    rows = [
        {
            "description": f"{category} merchant {index}",
            "source_category": category.title(),
            "target_category": category,
        }
        for category in categories
        for index in range(20)
    ]
    return pd.DataFrame(rows)


def test_split_category_training_data_uses_expected_sizes_and_all_rows():
    prepared_data = make_prepared_data()

    splits = split_category_training_data(prepared_data)

    assert len(splits.train) == 56
    assert len(splits.validation) == 12
    assert len(splits.test) == 12

    all_descriptions = {
        *splits.train["description"],
        *splits.validation["description"],
        *splits.test["description"],
    }
    assert all_descriptions == set(prepared_data["description"])


def test_split_category_training_data_has_no_overlap():
    splits = split_category_training_data(make_prepared_data())

    train_descriptions = set(splits.train["description"])
    validation_descriptions = set(splits.validation["description"])
    test_descriptions = set(splits.test["description"])

    assert train_descriptions.isdisjoint(validation_descriptions)
    assert train_descriptions.isdisjoint(test_descriptions)
    assert validation_descriptions.isdisjoint(test_descriptions)


def test_split_category_training_data_preserves_category_distribution():
    splits = split_category_training_data(make_prepared_data())

    assert splits.train["target_category"].value_counts().to_dict() == {
        "groceries": 14,
        "housing": 14,
        "shopping": 14,
        "travel": 14,
    }
    assert splits.validation["target_category"].value_counts().to_dict() == {
        "groceries": 3,
        "housing": 3,
        "shopping": 3,
        "travel": 3,
    }
    assert splits.test["target_category"].value_counts().to_dict() == {
        "groceries": 3,
        "housing": 3,
        "shopping": 3,
        "travel": 3,
    }


def test_split_category_training_data_is_reproducible():
    prepared_data = make_prepared_data()

    first = split_category_training_data(prepared_data, random_state=123)
    second = split_category_training_data(prepared_data, random_state=123)

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.validation, second.validation)
    pd.testing.assert_frame_equal(first.test, second.test)


def make_grouped_prepared_data() -> pd.DataFrame:
    categories = ["groceries", "housing", "shopping", "travel"]
    rows = [
        {
            "description": f"[debit] {category} merchant {group_index} #{reference}",
            "source_category": category.title(),
            "target_category": category,
        }
        for category in categories
        for group_index in range(20)
        for reference in (1001, 2002)
    ]
    return pd.DataFrame(rows)


def test_grouped_split_uses_expected_sizes_and_all_rows():
    prepared_data = make_grouped_prepared_data()

    splits = split_grouped_category_training_data(prepared_data)

    assert len(splits.train) == 112
    assert len(splits.validation) == 24
    assert len(splits.test) == 24
    assert len(splits.train) + len(splits.validation) + len(splits.test) == len(prepared_data)


def test_grouped_split_has_no_normalized_group_overlap():
    splits = split_grouped_category_training_data(make_grouped_prepared_data())

    split_groups = [
        set(split["description"].map(normalize_description_group))
        for split in (splits.train, splits.validation, splits.test)
    ]

    assert split_groups[0].isdisjoint(split_groups[1])
    assert split_groups[0].isdisjoint(split_groups[2])
    assert split_groups[1].isdisjoint(split_groups[2])


def test_grouped_split_preserves_category_distribution():
    splits = split_grouped_category_training_data(make_grouped_prepared_data())

    expected_counts = {
        "train": 28,
        "validation": 6,
        "test": 6,
    }
    for split_name, split in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        assert split["target_category"].value_counts().to_dict() == {
            category: expected_counts[split_name]
            for category in ("groceries", "housing", "shopping", "travel")
        }


def test_grouped_split_is_reproducible():
    prepared_data = make_grouped_prepared_data()

    first = split_grouped_category_training_data(prepared_data, random_state=123)
    second = split_grouped_category_training_data(prepared_data, random_state=123)

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.validation, second.validation)
    pd.testing.assert_frame_equal(first.test, second.test)
