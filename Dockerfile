FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code only (no data/, notebooks/, tests/).
COPY src/ ./src/
COPY api/ ./api/

# Production model: fetched by exact release version, SHA256-verified.
# Never a moving tag, never baked from a developer laptop. Provenance:
# https://github.com/Bakr1m/eurosat-landuse-cnn/releases/tag/v1.0.0
ARG MODEL_TAG=v1.0.0
ARG MODEL_SHA256=734015a28d04d90f8d9c32eee486d237cc7012cf011dd5d1a0b3f9f620285a8c
RUN mkdir -p models && \
    curl -fsSL -o models/eurosat_cnn_best.keras \
      "https://github.com/Bakr1m/eurosat-landuse-cnn/releases/download/${MODEL_TAG}/eurosat_cnn_best.keras" && \
    echo "${MODEL_SHA256}  models/eurosat_cnn_best.keras" | sha256sum -c - && \
    python -c "from tensorflow.keras.models import load_model; m=load_model('models/eurosat_cnn_best.keras'); print('artifact OK:', m.count_params(), 'params')"

EXPOSE 8000

CMD ["python", "api/main.py"]
