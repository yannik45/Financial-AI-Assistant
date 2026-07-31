import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


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


def train_word_char_tfidf_category_classifier(
    training_data: pd.DataFrame,
    random_state: int = 42,
) -> Pipeline:
    """Train a linear classifier on combined word and character TF-IDF."""
    features = FeatureUnion(
        [
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )
    model = Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )
    model.fit(training_data["description"], training_data["target_category"])
    return model
