# EuroSAT Land-Use CNN

**Thesis project, productionized | Custom CNN: data → model → evaluation → API → Docker**

## Business Context

Earth Observation missions (Copernicus Sentinel-2) generate far more
imagery than analysts can label by hand. This project automates Land Use
/ Land Cover (LULC) mapping: a custom convolutional network classifies
satellite patches into 10 land-use classes, with the hard cases being
classes whose RGB signatures overlap (crops, pasture, vegetation).

## Dataset

- **Source**: EuroSAT RGB (Helber et al. 2019; Sentinel-2, CC BY-SA 4.0),
  27,000 64×64 patches — reproduced via `scripts/download_data.sh`
  (data/ is gitignored)
- **Classes (10)**: AnnualCrop, Forest, HerbaceousVegetation, Highway,
  Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- **Split**: stratified 70/15/15 (train 18,900 / val 4,050 / test 4,050),
  `random_state=42`
- **Note**: RGB only — the 10 Sentinel-2 spectral bands are collapsed to
  visual channels, which is exactly what makes similar classes confusable

## Approach

1. **Pipeline** (`src/data.py`, `src/preprocessing.py`): stratified split;
   train-only augmentation (rotation 20°, shifts/shear/zoom 0.1, flip),
   rescale-only eval — augmentation can never leak (tested).
2. **Architecture** (`src/model.py`): custom 3-block CNN —
   3× [Conv2D → BatchNorm → MaxPool → Dropout] → Dense(512) → softmax,
   4,296,138 params. Adam(1e-3), categorical crossentropy.
3. **Training** (`src/train.py`): up to 50 epochs, ModelCheckpoint on
   `val_accuracy`, EarlyStopping on `val_loss` (patience 10, restore best).
4. **Evaluation**: consolidated test report + confusion matrix.
5. **Serving** (`src/serve.py`): FastAPI `/predict` (image upload → class
   + confidence + full probability vector), lazy model load.
6. **Container**: release-pinned `.keras` artifact, SHA256-verified,
   parity-checked against local serving.

## Results (held-out test, 4,050 patches — independent rerun)

- **Accuracy 0.8763**, loss 0.3675 (macro F1 0.87).
- Strongest: Forest (F1 0.94), SeaLake (0.95), AnnualCrop (0.89).
- Weakest: HerbaceousVegetation (F1 0.79 — confused with Pasture and
  AnnualCrop), Industrial recall 0.78.

*Provenance note: the thesis notebook reported 0.9123 on the same
protocol. This repo's `make train` rerun (CPU, TF 2.19, same seed,
same architecture) reproduces 0.8763 with the identical error pattern
— same weakest classes, same confusions. The gap is consistent with
GPU/CPU nondeterminism plus best-of-several reporting in the original;
the rerun number above is the one this repo's artifact actually scores,
and the shipped model is the rerun.*

## Limitations

1. **RGB only** — multispectral bands unused; crop/pasture/vegetation
   confusion is the error mass.
2. **Single dataset, single sensor mix** — no cross-sensor or
   multi-region validation.
3. **No calibration** — softmax scores rank, they don't quantify uncertainty.
4. **64×64 tiles** — no landscape context beyond the patch.
5. **Small test set** (4,050) for a 10-class problem; rare confusions are noisy.

## Ethical Considerations

EO mapping supports land administration and environmental monitoring, but
misclassification at scale could misinform land-use decisions affecting
livelihoods. Demo only — operational use needs multispectral inputs,
multi-region validation, and human review.

## Run Instructions

```bash
make install-dev              # serving deps + pytest/ruff/httpx
make test lint                # 10 tests, ruff clean
bash scripts/download_data.sh # fetch + unpack EuroSAT RGB (~90 MB)
make install-train && make train  # retrain (CPU, ~1-2h); saves models/eurosat_cnn_best.keras
make serve                    # API on :8000
curl -X POST http://localhost:8000/predict -F "file=@patch.png"
```

Docker (prebuilt, release-pinned artifact inside):

```bash
docker pull bakr1m/eurosat-api:latest
docker run -p 8000:8000 bakr1m/eurosat-api:latest
```

## Provenance

Refactored from the author's final-thesis notebook (preserved under
`notebooks/01_eurosat_cnn_thesis.ipynb`): same architecture, same split
seed, same training protocol. Colab/Drive-specific cells were removed;
everything reproducible runs from `src/` + `make`.
