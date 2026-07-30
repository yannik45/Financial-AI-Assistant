import pandas as pd
from financial_ai.ml.category_split import split_category_training_data


def make_training_data() -> pd.DataFrame:
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
    training_data = make_training_data()

    splits = split_category_training_data(training_data)

    assert len(splits.train) == 56
    assert len(splits.validation) == 12
    assert len(splits.test) == 12

    all_descriptions = {
        *splits.train["description"],
        *splits.validation["description"],
        *splits.test["description"],
    }
    assert all_descriptions == set(training_data["description"])


def test_split_category_training_data_has_no_overlap():
    splits = split_category_training_data(make_training_data())

    train_descriptions = set(splits.train["description"])
    validation_descriptions = set(splits.validation["description"])
    test_descriptions = set(splits.test["description"])

    assert train_descriptions.isdisjoint(validation_descriptions)
    assert train_descriptions.isdisjoint(test_descriptions)
    assert validation_descriptions.isdisjoint(test_descriptions)


def test_split_category_training_data_preserves_category_distribution():
    splits = split_category_training_data(make_training_data())

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
    training_data = make_training_data()

    first = split_category_training_data(training_data, random_state=123)
    second = split_category_training_data(training_data, random_state=123)

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.validation, second.validation)
    pd.testing.assert_frame_equal(first.test, second.test)
