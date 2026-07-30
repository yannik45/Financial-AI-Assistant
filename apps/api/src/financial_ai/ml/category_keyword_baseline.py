import numpy as np
import pandas as pd

KEYWORD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("housing", ("rent", "mortgage", "lease payment")),
    ("utilities", ("electric", "water utility", "internet provider")),
    ("insurance", ("insurance", "premium")),
    ("education", ("university", "tuition", "course")),
    ("healthcare", ("pharmacy", "dental", "hospital", "clinic")),
    ("travel", ("hotel", "airline", "airways", "motel")),
    ("transport", ("gas station", "parking", "transit", "taxi")),
    ("dining", ("restaurant", "cafe", "pizza", "burger")),
    ("groceries", ("grocery", "supermarket")),
    ("entertainment", ("cinema", "theater", "sportsbook", "game")),
    ("shopping", ("retail", "clothing", "electronics store")),
    ("other", ("subscription", "monthly plan")),
)


class KeywordCategoryClassifier:
    """Classify descriptions with a small ordered set of deterministic rules."""

    def predict(self, descriptions: pd.Series) -> np.ndarray:
        predictions = []
        for description in descriptions:
            normalized_description = description.casefold()
            predicted_category = "other"

            for category, keywords in KEYWORD_RULES:
                if any(keyword in normalized_description for keyword in keywords):
                    predicted_category = category
                    break

            predictions.append(predicted_category)

        return np.array(predictions)
