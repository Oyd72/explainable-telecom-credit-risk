import numpy as np
from fastapi.testclient import TestClient

import src.api.app as api_module


client = TestClient(api_module.app)

FEATURES = [
    "cnt_ma_rech90",
    "daily_decr30",
    "last_rech_date_ma",
    "sumamnt_ma_rech90",
    "aon",
    "last_rech_amt_ma",
    "daily_decr90",
    "sumamnt_ma_rech30",
    "medianamnt_ma_rech30",
    "medianmarechprebal90",
    "rental30",
    "cnt_ma_rech30",
]


class FakeImputer:
    statistics_ = np.arange(1, len(FEATURES) + 1, dtype=float)


class FakeBaseModel:
    named_steps = {"imputer": FakeImputer()}

    def predict_proba(self, frame):
        positive = np.full(len(frame), 0.30, dtype=float)
        return np.column_stack([1.0 - positive, positive])


class FakeCalibrator:
    def predict(self, raw):
        return np.asarray(raw, dtype=float) * 0.5


def fake_package():
    return {
        "base_model": FakeBaseModel(),
        "calibrator": FakeCalibrator(),
        "features": FEATURES,
        "operating_threshold": 0.1751,
    }


def valid_payload():
    return {feature: float(index + 1) for index, feature in enumerate(FEATURES)}


def test_health_endpoint_is_liveness_only():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "explainable-telecom-credit-risk-api",
    }


def test_ready_endpoint_returns_503_when_model_unavailable(monkeypatch):
    def unavailable():
        raise FileNotFoundError("model missing")

    monkeypatch.setattr(api_module, "get_model_package", unavailable)

    response = client.get("/ready")

    assert response.status_code == 503
    assert "Model is not ready" in response.json()["detail"]


def test_ready_endpoint_returns_200_when_model_available(monkeypatch):
    monkeypatch.setattr(api_module, "get_model_package", fake_package)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model": "available"}


def test_predict_endpoint_scores_valid_payload(monkeypatch):
    monkeypatch.setattr(api_module, "get_model_package", fake_package)

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["raw_delinquency_probability"] == 0.30
    assert payload["calibrated_delinquency_probability"] == 0.15
    assert payload["higher_risk_flag"] is False


def test_predict_endpoint_rejects_extra_fields(monkeypatch):
    monkeypatch.setattr(api_module, "get_model_package", fake_package)
    payload = valid_payload()
    payload["unexpected"] = 1

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_model_info_exposes_aggregate_imputation_medians(monkeypatch):
    monkeypatch.setattr(api_module, "get_model_package", fake_package)

    response = client.get("/model-info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["features"] == FEATURES
    assert payload["operating_threshold"] == 0.1751
    assert payload["imputation_medians"]["cnt_ma_rech90"] == 1.0


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "explainable-telecom-credit-risk-api"
