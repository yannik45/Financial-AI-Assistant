import pandas as pd
from financial_ai.ml.transaction_classification.data.category_quality import (
    build_category_quality_report,
)


def test_build_category_quality_report_counts_categories_and_quality_issues():
    training_data = pd.DataFrame(
        {
            "description": ["A", "A", "B", "C", "C"],
            "source_category": ["Shopping", "Shopping", "Travel", "Rent", "Utilities"],
            "target_category": [
                "shopping",
                "shopping",
                "travel",
                "housing",
                "utilities",
            ],
        }
    )

    assert build_category_quality_report(training_data) == {
        "row_count": 5,
        "category_counts": {
            "shopping": 2,
            "travel": 1,
            "housing": 1,
            "utilities": 1,
        },
        "duplicate_description_count": 2,
        "conflicting_description_count": 1,
    }


def test_build_category_quality_report_handles_empty_data():
    training_data = pd.DataFrame(columns=["description", "source_category", "target_category"])

    assert build_category_quality_report(training_data) == {
        "row_count": 0,
        "category_counts": {},
        "duplicate_description_count": 0,
        "conflicting_description_count": 0,
    }
