import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from financial_ai.ml.transaction_classification.category_evaluation import (
    CategoryEvaluation,
    evaluate_category_classifier,
)
from financial_ai.ml.transaction_classification.category_model import (
    train_tfidf_category_classifier,
)
from financial_ai.ml.transaction_classification.category_split import CategoryDataSplits
from financial_ai.ml.transaction_classification.german_evaluation import (
    GermanChallengeEvaluation,
    evaluate_german_challenge,
)
from financial_ai.ml.transaction_classification.german_training_split_v2 import (
    split_declared_training_data,
)

MODEL_COLUMNS = ["description", "target_category"]
DEFAULT_ENGLISH_PATH = Path("data/runtime/ml/transaction_categories/english_training_v1.csv")
DEFAULT_GERMAN_PATH = Path("data/runtime/ml/transaction_categories/german_training_v2.csv")
DEFAULT_GERMAN_CHALLENGE_PATH = Path(
    "data/evaluation/transaction_categories/german_challenge_v1.csv"
)
DEFAULT_REPORT_PATH = Path("data/runtime/ml/transaction_categories/final_model_evaluation_v1.json")


@dataclass(frozen=True)
class EnglishFinalEvaluation:
    model: Pipeline
    fitting_rows: int
    controlled_test: CategoryEvaluation


@dataclass(frozen=True)
class GermanFinalEvaluation:
    model: Pipeline
    fitting_rows: int
    controlled_test: CategoryEvaluation
    challenge: GermanChallengeEvaluation


@dataclass(frozen=True)
class MultilingualFinalEvaluation:
    model: Pipeline
    fitting_rows: int
    english_controlled_test: CategoryEvaluation
    german_controlled_test: CategoryEvaluation
    german_challenge: GermanChallengeEvaluation


@dataclass(frozen=True)
class FinalModelComparison:
    english_only: EnglishFinalEvaluation
    german_only: GermanFinalEvaluation
    multilingual: MultilingualFinalEvaluation


def _combine_train_and_validation(splits: CategoryDataSplits) -> pd.DataFrame:
    return pd.concat(
        [
            splits.train[MODEL_COLUMNS],
            splits.validation[MODEL_COLUMNS],
        ],
        ignore_index=True,
    )


def run_final_model_comparison(
    english_data: pd.DataFrame,
    german_data: pd.DataFrame,
    german_challenge_data: pd.DataFrame,
    random_state: int = 42,
) -> FinalModelComparison:
    """Fit frozen character models and evaluate predeclared final datasets."""
    english_splits = split_declared_training_data(english_data)
    german_splits = split_declared_training_data(german_data)
    english_fitting_data = _combine_train_and_validation(english_splits)
    german_fitting_data = _combine_train_and_validation(german_splits)
    multilingual_fitting_data = pd.concat(
        [english_fitting_data, german_fitting_data],
        ignore_index=True,
    )

    english_model = train_tfidf_category_classifier(
        english_fitting_data,
        random_state=random_state,
    )
    german_model = train_tfidf_category_classifier(
        german_fitting_data,
        random_state=random_state,
    )
    multilingual_model = train_tfidf_category_classifier(
        multilingual_fitting_data,
        random_state=random_state,
    )

    return FinalModelComparison(
        english_only=EnglishFinalEvaluation(
            model=english_model,
            fitting_rows=len(english_fitting_data),
            controlled_test=evaluate_category_classifier(
                english_model,
                english_splits.test,
            ),
        ),
        german_only=GermanFinalEvaluation(
            model=german_model,
            fitting_rows=len(german_fitting_data),
            controlled_test=evaluate_category_classifier(
                german_model,
                german_splits.test,
            ),
            challenge=evaluate_german_challenge(
                german_model,
                german_challenge_data,
            ),
        ),
        multilingual=MultilingualFinalEvaluation(
            model=multilingual_model,
            fitting_rows=len(multilingual_fitting_data),
            english_controlled_test=evaluate_category_classifier(
                multilingual_model,
                english_splits.test,
            ),
            german_controlled_test=evaluate_category_classifier(
                multilingual_model,
                german_splits.test,
            ),
            german_challenge=evaluate_german_challenge(
                multilingual_model,
                german_challenge_data,
            ),
        ),
    )


def build_final_report(comparison: FinalModelComparison) -> dict[str, object]:
    return {
        "english_only": {
            "fitting_rows": comparison.english_only.fitting_rows,
            "controlled_test": asdict(comparison.english_only.controlled_test),
        },
        "german_only": {
            "fitting_rows": comparison.german_only.fitting_rows,
            "controlled_test": asdict(comparison.german_only.controlled_test),
            "german_challenge": asdict(comparison.german_only.challenge),
        },
        "multilingual": {
            "fitting_rows": comparison.multilingual.fitting_rows,
            "english_controlled_test": asdict(comparison.multilingual.english_controlled_test),
            "german_controlled_test": asdict(comparison.multilingual.german_controlled_test),
            "german_challenge": asdict(comparison.multilingual.german_challenge),
        },
    }


def write_final_report(
    report: dict[str, object],
    destination: Path = DEFAULT_REPORT_PATH,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


if __name__ == "__main__":
    final_comparison = run_final_model_comparison(
        pd.read_csv(DEFAULT_ENGLISH_PATH),
        pd.read_csv(DEFAULT_GERMAN_PATH),
        pd.read_csv(DEFAULT_GERMAN_CHALLENGE_PATH),
    )
    report_path = write_final_report(build_final_report(final_comparison))
    print(f"Final evaluation report ready: {report_path}")
