"""Serving tests: hermetic via a stand-in keras model (no artifact needed)."""
import io

import pytest
from fastapi.testclient import TestClient

import src.serve as serve
from src.data import CLASS_NAMES
from src.serve import app


@pytest.fixture()
def client(monkeypatch):
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.models import Sequential

    standin = Sequential([Flatten(input_shape=(64, 64, 3)), Dense(10, activation="softmax")])
    standin.compile(optimizer="adam", loss="categorical_crossentropy")
    monkeypatch.setattr(serve, "_model", standin)
    return TestClient(app)


def _png_bytes(color=(128, 128, 128)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "healthy"}


def test_predict_valid_image(client):
    r = client.post("/predict", files={"file": ("patch.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in CLASS_NAMES
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 10


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert r.status_code == 422
