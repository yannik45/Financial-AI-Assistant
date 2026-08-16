from pathlib import Path

import pandas as pd
import pytest
from financial_ai.ml.transaction_classification.data.classification_v2_dataset import (
    load_classification_v2_dataset,
    validate_declared_splits,
)


def _row(example_id: str, split: str, merchant: str, detail: str, format_group: str) -> dict:
    return {
        "example_id": example_id,
        "description": f"Transaction {example_id}",
        "target_category": "shopping",
        "language": "de",
        "template_id": f"template-{format_group}",
        "merchant_group": merchant,
        "detail_group": detail,
        "format_group": format_group,
        "split": split,
    }


def test_declared_splits_reject_merchant_or_text_family_leakage() -> None:
    merchant_leakage = pd.DataFrame(
        [
            _row("1", "train", "merchant-a", "detail-a", "format-a"),
            _row("2", "validation", "merchant-a", "detail-b", "format-b"),
            _row("3", "test", "merchant-c", "detail-c", "format-c"),
        ]
    )
    with pytest.raises(ValueError, match="merchant_group values cross"):
        validate_declared_splits(merchant_leakage)

    template_leakage = pd.DataFrame(
        [
            _row("1", "train", "merchant-a", "detail-a", "shared-format"),
            _row("2", "validation", "merchant-b", "detail-b", "shared-format"),
            _row("3", "test", "merchant-c", "detail-c", "format-c"),
        ]
    )
    with pytest.raises(ValueError, match="format_group values cross"):
        validate_declared_splits(template_leakage)


def test_v2_loader_combines_languages_without_changing_declared_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = pd.DataFrame(
        [
            _row("1", "train", "merchant-a", "detail-a", "format-a"),
            _row("2", "validation", "merchant-b", "detail-b", "format-b"),
            _row("3", "test", "merchant-c", "detail-c", "format-c"),
        ]
    )
    english = rows.assign(
        example_id=lambda frame: "en-" + frame["example_id"],
        language="en",
    )
    german = rows.assign(
        example_id=lambda frame: "de-" + frame["example_id"],
        language="de",
    )
    english_path = Path("english.csv")
    german_path = Path("german.csv")
    monkeypatch.setattr(Path, "is_file", lambda self: self in {english_path, german_path})
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda path: english.copy() if path == english_path else german.copy(),
    )

    dataset = load_classification_v2_dataset(
        english_path,
        german_path,
        manual_short_path=None,
    )

    assert len(dataset.train) == len(dataset.validation) == len(dataset.test) == 2
    assert set(dataset.train["input_slice"]) == {"bank_feed"}
    assert set(dataset.train["language"]) == {"en", "de"}
