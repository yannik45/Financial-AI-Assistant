import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_tfidf_category_classifier(
    training_data: pd.DataFrame,
    random_state: int = 42,
) -> Pipeline:
    """Train the first learned transaction category baseline."""
    tfidf_model = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
    )

    logreg_model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=random_state
    )

    model = Pipeline(
        [
            ("tfidf", tfidf_model),
            ("classifier", logreg_model),
        ]
    )

    model.fit(training_data["description"], training_data["target_category"])

    return model
