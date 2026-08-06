import pandas as pd
from sklearn.dummy import DummyClassifier


def train_majority_category_baseline(training_data: pd.DataFrame) -> DummyClassifier:
    """Train a reference classifier that always predicts the majority category."""
    model = DummyClassifier(strategy="most_frequent")

    model.fit(training_data["description"], training_data["target_category"])

    return model
