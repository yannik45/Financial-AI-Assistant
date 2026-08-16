import numpy as np
import pandas as pd
import pytest
from financial_ai.ml.transaction_classification.modeling.fusion import (
    train_fusion_category_classifier,
)


def test_fusion_classifier_combines_text_and_aligned_embeddings() -> None:
    training = pd.DataFrame(
        {
            "description": ["clothes shop", "new clothes", "city bus", "bus ticket"],
            "target_category": ["shopping", "shopping", "transport", "transport"],
        }
    )
    embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
    )
    model = train_fusion_category_classifier(training, embeddings)

    probabilities = model.predict_proba(
        pd.Series(["clothes", "bus"]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    assert probabilities.shape == (2, 2)
    assert model.predict(
        pd.Series(["clothes", "bus"]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    ).tolist() == ["shopping", "transport"]


def test_fusion_classifier_rejects_misaligned_embeddings() -> None:
    training = pd.DataFrame(
        {
            "description": ["clothes", "clothes shop", "bus", "bus ticket"],
            "target_category": ["shopping", "shopping", "transport", "transport"],
        }
    )
    model = train_fusion_category_classifier(
        training,
        np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
    )

    with pytest.raises(ValueError, match="Embedding shape"):
        model.predict(pd.Series(["clothes"]), np.asarray([[1.0, 0.0, 0.0]]))
