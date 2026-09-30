"""FastAPI serving: single EuroSAT patch -> class + confidence.

Model loads lazily behind get_model() — a missing artifact fails as a
request-time 503, never an import-time crash (hermetic tests).
"""
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.data import CLASS_NAMES
from src.preprocessing import IMG_SIZE, preprocess_input

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "eurosat_cnn_best.keras"

app = FastAPI(title="EuroSAT Land-Use Classifier")

_model = None


def get_model():
    """Load once on first request (never at import: hermetic tests)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail=f"model artifact missing: {MODEL_PATH}")
        from tensorflow.keras.models import load_model

        _model = load_model(MODEL_PATH)
    return _model


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=422, detail="upload must be an image file")
    try:
        img = Image.open(BytesIO(file.file.read())).convert("RGB").resize(IMG_SIZE)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"unreadable image: {e}")
    import numpy as np

    probs = get_model().predict(preprocess_input(np.array(img)), verbose=0)[0]
    idx = int(probs.argmax())
    return {
        "predicted_class": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
