import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(model_file):
    from api.main import app
    with TestClient(app) as c:   # context manager runs the startup (model load)
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client, sample_record):
    r = client.post("/predict", json=sample_record)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["model_name"] == "test_lr"
    assert r.headers["x-request-id"] == body["request_id"]


def test_predict_without_optional_fields(client, sample_record):
    record = {k: v for k, v in sample_record.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=record).status_code == 200


@pytest.mark.parametrize("field,value", [("age", -5), ("cp", 9), ("thal", 5),
                                         ("chol", "high")])
def test_invalid_input_rejected(client, sample_record, field, value):
    assert client.post("/predict", json={**sample_record, field: value}).status_code == 422


def test_missing_required_field_rejected(client, sample_record):
    record = dict(sample_record)
    del record["age"]
    assert client.post("/predict", json=record).status_code == 422


def test_metrics_exposed(client, sample_record):
    client.post("/predict", json=sample_record)
    text = client.get("/metrics/").text
    assert "api_requests_total" in text and "model_predictions_total" in text


def test_health_503_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    from api.main import app
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
