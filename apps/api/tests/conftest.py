import os
from pathlib import Path

TEST_DB = Path(__file__).parent / "test.db"
os.environ["FINANCIAL_AI_DATABASE_URL"] = f"sqlite:///{TEST_DB.as_posix()}"

import pytest
from fastapi.testclient import TestClient

from financial_ai.database import Base, engine
from financial_ai.main import app


@pytest.fixture(autouse=True)
def clean_database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    TEST_DB.unlink(missing_ok=True)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
