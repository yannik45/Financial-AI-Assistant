import hashlib
from pathlib import Path

import httpx
import pandas as pd

from financial_ai.ml.transaction_classification.data.category_preprocessing import (
    prepare_category_training_data,
)

DATASET_REVISION = "c60ce79c60e31532d1f018275cc5c2e06e88af3f"
DATASET_FILENAME = "transactions-synthetic.csv"
DATASET_URL = (
    "https://huggingface.co/datasets/DoDataThings/"
    f"us-bank-transaction-categories-v2/resolve/{DATASET_REVISION}/{DATASET_FILENAME}"
)
DATASET_SHA256 = "0424aed6e76f74a5b3b1ff61ccec43bc321622e6806da353b910b3b2c8108f6e"
DEFAULT_DATASET_PATH = Path("data/runtime/ml/transaction_categories") / DATASET_FILENAME


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_category_dataset(destination: Path = DEFAULT_DATASET_PATH) -> Path:
    if destination.exists():
        actual_hash = calculate_sha256(destination)
        if actual_hash != DATASET_SHA256:
            raise ValueError(
                f"Existing dataset checksum mismatch: expected {DATASET_SHA256}, got {actual_hash}"
            )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(f"{destination.suffix}.part")

    with httpx.stream("GET", DATASET_URL, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as file:
            for chunk in response.iter_bytes():
                file.write(chunk)

    actual_hash = calculate_sha256(temporary_path)
    if actual_hash != DATASET_SHA256:
        raise ValueError(
            f"Downloaded dataset checksum mismatch: expected {DATASET_SHA256}, got {actual_hash}"
        )

    temporary_path.replace(destination)
    return destination


def load_category_training_data(path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Category dataset not found: {path}")

    actual_hash = calculate_sha256(path)
    if actual_hash != DATASET_SHA256:
        raise ValueError(f"Dataset checksum mismatch: expected {DATASET_SHA256}, got {actual_hash}")

    source_data = pd.read_csv(path)
    return prepare_category_training_data(source_data)


if __name__ == "__main__":
    dataset_path = download_category_dataset()
    print(f"Dataset ready: {dataset_path}")
