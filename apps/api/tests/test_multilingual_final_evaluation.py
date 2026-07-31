import pandas as pd
from financial_ai.ml.categories import ExpenseCategory
from financial_ai.ml.english_training_generator_v1 import (
    generate_english_training_data_v1,
)
from financial_ai.ml.german_training_generator_v2 import (
    generate_german_training_data_v2,
)
from financial_ai.ml.multilingual_final_evaluation import (
    build_final_report,
    run_final_model_comparison,
)


def make_challenge_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": f"test_{category.value}_{number:03d}",
                "description": f"TEST CHALLENGE {category.value} {number}",
                "target_category": category.value,
                "language": "de",
                "merchant_group": f"challenge_{category.value}_{number}",
                "merchant_scope": ("german_local" if number <= 5 else "international"),
            }
            for category in ExpenseCategory
            for number in range(1, 11)
        ]
    )


def test_run_final_model_comparison_uses_train_and_validation_only_for_fitting():
    english_data = generate_english_training_data_v1(100, random_seed=7)
    german_data = generate_german_training_data_v2(100, random_seed=7)

    result = run_final_model_comparison(
        english_data,
        german_data,
        make_challenge_data(),
        random_state=17,
    )

    rows_per_language = (76 + 12) * len(ExpenseCategory)
    assert result.english_only.fitting_rows == rows_per_language
    assert result.german_only.fitting_rows == rows_per_language
    assert result.multilingual.fitting_rows == rows_per_language * 2
    assert sum(
        metric.support for metric in result.english_only.controlled_test.per_category
    ) == 12 * len(ExpenseCategory)
    assert sum(
        metric.support for metric in result.german_only.controlled_test.per_category
    ) == 12 * len(ExpenseCategory)
    assert sum(
        metric.support for metric in result.multilingual.english_controlled_test.per_category
    ) == 12 * len(ExpenseCategory)
    assert sum(
        metric.support for metric in result.multilingual.german_controlled_test.per_category
    ) == 12 * len(ExpenseCategory)
    assert (
        sum(metric.support for metric in result.german_only.challenge.overall.per_category) == 120
    )
    assert (
        sum(metric.support for metric in result.multilingual.german_challenge.overall.per_category)
        == 120
    )

    report = build_final_report(result)
    assert report["english_only"]["fitting_rows"] == rows_per_language
    assert "model" not in report["english_only"]
