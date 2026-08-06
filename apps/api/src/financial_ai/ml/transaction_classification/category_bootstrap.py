from financial_ai.ml.transaction_classification.category_artifact import (
    DEFAULT_ARTIFACT_PATH,
    build_category_model_artifact,
)
from financial_ai.ml.transaction_classification.english_training_generator_v1 import (
    write_english_training_dataset_v1,
)
from financial_ai.ml.transaction_classification.german_training_generator_v2 import (
    write_german_training_dataset_v2,
)


def bootstrap_category_model() -> None:
    english_path, _ = write_english_training_dataset_v1()
    german_path, _ = write_german_training_dataset_v2()
    metadata = build_category_model_artifact(
        english_path=english_path,
        german_path=german_path,
    )
    print(f"Category model ready: {DEFAULT_ARTIFACT_PATH} ({metadata.training_rows} training rows)")


def run() -> None:
    bootstrap_category_model()


if __name__ == "__main__":
    run()
