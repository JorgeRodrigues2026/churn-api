"""Testes de integração da API. Executar: pytest -v"""
from unittest.mock import patch

import joblib
import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.7, "TotalCharges": 191.4,
}
LOYAL = {**VALID, "tenure": 60, "Contract": "Two year", "PaymentMethod": "Credit card (automatic)",
         "InternetService": "DSL", "OnlineSecurity": "Yes", "TechSupport": "Yes",
         "MonthlyCharges": 60.0, "TotalCharges": 3600.0}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # o "with" dispara o lifespan
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
    assert r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/v1/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in {"baixo", "medio", "alto"}


def test_predict_ranks_risk_coherently(client):
    risky = client.post("/v1/predict", json=VALID).json()["churn_probability"]
    loyal = client.post("/v1/predict", json=LOYAL).json()["churn_probability"]
    assert risky > loyal


def test_batch_ok(client):
    r = client.post("/v1/predict/batch", json={"customers": [VALID, LOYAL]})
    assert r.status_code == 200
    assert r.json()["total"] == 2


@pytest.mark.parametrize("field,value", [
    ("tenure", "dez"),           # texto em campo numérico
    ("tenure", -1),              # abaixo do ge=0
    ("MonthlyCharges", 0),       # gt=0
    ("Contract", "Weekly"),      # fora do Literal
])
def test_invalid_payload_returns_422(client, field, value):
    r = client.post("/v1/predict", json={**VALID, field: value})
    assert r.status_code == 422


def test_missing_and_extra_fields_return_422(client):
    missing = {k: v for k, v in VALID.items() if k != "tenure"}
    assert client.post("/v1/predict", json=missing).status_code == 422
    assert client.post("/v1/predict", json={**VALID, "hacker": 1}).status_code == 422


def test_business_rule_returns_400(client):
    r = client.post("/v1/predict", json={**VALID, "InternetService": "No"})
    assert r.status_code == 400
    assert "internet" in r.json()["detail"]


def test_empty_batch_returns_422(client):
    assert client.post("/v1/predict/batch", json={"customers": []}).status_code == 422


def test_model_loaded_once_not_per_request():
    """Fase 3: o disco é lido 1x no startup e nunca nas requisições."""
    with patch("app.services.model_service.joblib.load", wraps=joblib.load) as spy:
        with TestClient(app) as c:
            for _ in range(5):
                assert c.post("/v1/predict", json=VALID).status_code == 200
        assert spy.call_count == 1


def test_model_missing_returns_503(tmp_path):
    with patch("app.main.MODEL_PATH", tmp_path / "nao_existe.joblib"):
        with TestClient(app) as c:
            assert c.get("/health").json()["status"] == "degraded"
            assert c.post("/v1/predict", json=VALID).status_code == 503
