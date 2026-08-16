from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class FusionCategoryClassifier:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    embedding_dimensions: int

    def predict(self, text: pd.Series, embeddings: np.ndarray) -> np.ndarray:
        return self.classifier.predict(self._features(text, embeddings))

    def predict_proba(self, text: pd.Series, embeddings: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self._features(text, embeddings))

    def _features(self, text: pd.Series, embeddings: np.ndarray) -> csr_matrix:
        tfidf = self.vectorizer.transform(text)
        if embeddings.shape != (len(text), self.embedding_dimensions):
            raise ValueError("Embedding shape does not match fusion model input")
        return hstack((tfidf, csr_matrix(embeddings)), format="csr")


def train_fusion_category_classifier(
    training_data: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    random_state: int = 42,
) -> FusionCategoryClassifier:
    if len(training_data) != len(embeddings) or embeddings.ndim != 2:
        raise ValueError("Training rows and embeddings must align")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(training_data["description"])
    features = hstack((tfidf, csr_matrix(embeddings)), format="csr")
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1_000,
        random_state=random_state,
    )
    classifier.fit(features, training_data["target_category"])
    return FusionCategoryClassifier(
        vectorizer=vectorizer,
        classifier=classifier,
        embedding_dimensions=embeddings.shape[1],
    )
