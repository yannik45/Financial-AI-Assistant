import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from financial_ai.ml.transaction_classification.category_dataset import load_category_training_data
from financial_ai.ml.transaction_classification.category_evaluation import (
    CategoryClassifier,
    CategoryEvaluation,
    evaluate_category_classifier,
)
from financial_ai.ml.transaction_classification.category_model import (
    train_tfidf_category_classifier,
)
from financial_ai.ml.transaction_classification.category_split import (
    split_grouped_category_training_data,
)
from financial_ai.ml.transaction_classification.german_challenge import (
    validate_german_challenge_data,
)

DEFAULT_GERMAN_CHALLENGE_PATH = (
    Path("data/evaluation/transaction_categories") / "german_challenge_v1.csv"
)


@dataclass(frozen=True)
class GermanChallengeEvaluation:
    overall: CategoryEvaluation
    german_local: CategoryEvaluation
    international: CategoryEvaluation


def evaluate_german_challenge(
    model: CategoryClassifier,
    challenge_data: pd.DataFrame,
) -> GermanChallengeEvaluation:
    validated_data = validate_german_challenge_data(challenge_data)

    return GermanChallengeEvaluation(
        overall=evaluate_category_classifier(model, validated_data),
        german_local=evaluate_category_classifier(
            model,
            validated_data.loc[validated_data["merchant_scope"].eq("german_local")],
        ),
        international=evaluate_category_classifier(
            model,
            validated_data.loc[validated_data["merchant_scope"].eq("international")],
        ),
    )


def run_german_zero_shot_evaluation(
    challenge_path: Path = DEFAULT_GERMAN_CHALLENGE_PATH,
) -> GermanChallengeEvaluation:
    prepared_english_data = load_category_training_data()
    grouped_splits = split_grouped_category_training_data(prepared_english_data)
    fitting_data = pd.concat(
        [grouped_splits.train, grouped_splits.validation],
        ignore_index=True,
    )
    model = train_tfidf_category_classifier(fitting_data)

    challenge_data = pd.read_csv(challenge_path)
    return evaluate_german_challenge(model, challenge_data)


if __name__ == "__main__":
    result = run_german_zero_shot_evaluation()
    print(json.dumps(asdict(result), indent=2))
