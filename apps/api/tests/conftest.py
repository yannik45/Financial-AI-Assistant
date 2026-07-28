import os
from pathlib import Path

TEST_DB = Path(__file__).parent / "test.db"
os.environ["FINANCIAL_AI_DATABASE_URL"] = f"sqlite:///{TEST_DB.as_posix()}"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from financial_ai.database import Base, engine  # noqa: E402
from financial_ai.main import app  # noqa: E402


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
