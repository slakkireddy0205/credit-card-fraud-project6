from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200


columns = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
payload = {c: 0.0 for c in columns}


def test_predict_valid():
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in (0,1)


def test_predict_missing_field():
    bad = dict(payload)
    del bad["Amount"]
    response = client.post("/predict", json=bad)
    assert response.status_code == 422