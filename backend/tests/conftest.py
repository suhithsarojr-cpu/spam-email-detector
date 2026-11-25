"""Test setup: isolated temp DB + temp model trained from the synthetic sample data."""
import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="spam_tests_"))
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp / 'test.db'}"
os.environ["MODEL_DIR"] = str(_tmp / "models")
os.environ["STORE_EMAIL_TEXT"] = "true"

import pytest  # noqa: E402

from scripts.generate_sample_data import main as _gen  # noqa: E402,F401
from scripts.generate_sample_data import build_rows  # noqa: E402
from scripts.train_model import train  # noqa: E402
import csv  # noqa: E402

DATA = _tmp / "sample.csv"
with open(DATA, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["subject", "body", "label"])
    w.writeheader()
    w.writerows(build_rows(150, 42))
METADATA = train(DATA, _tmp / "models")


@pytest.fixture(scope="session")
def metadata():
    return METADATA


@pytest.fixture(scope="session")
def api_key():
    from app.auth import create_user
    from app.database import SessionLocal, init_db
    init_db()
    with SessionLocal() as db:
        return create_user(db, "tester")[1]


@pytest.fixture()
def client(api_key):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app, headers={"X-API-Key": api_key}) as c:  # lifespan: tables + model load
        yield c


@pytest.fixture()
def anon_client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c
