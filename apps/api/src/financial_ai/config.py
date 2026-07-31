from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FINANCIAL_AI_", env_file=".env", extra="ignore")

    app_name: str = "Financial AI Assistant"
    database_url: str = "sqlite:///./data/runtime/financial_ai.db"
    cors_origins: list[str] = ["http://localhost:5173"]
    ecb_fx_path: Path = Path("data/market/ecb_fx.csv")
    category_model_artifact_path: Path = Path(
        "data/runtime/ml/models/transaction_category_bilingual_v1.pkl"
    )
    category_model_metadata_path: Path = Path(
        "data/runtime/ml/models/transaction_category_bilingual_v1.json"
    )
    category_review_threshold: float = 0.65


@lru_cache
def get_settings() -> Settings:
    return Settings()
