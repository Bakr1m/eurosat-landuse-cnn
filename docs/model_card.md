# Model Card — EuroSAT Land-Use CNN

## Intended use
Land-use / land-cover classification of Sentinel-2 RGB patches for Earth
Observation mapping support. Research/thesis artifact — not an operational
mapping product.

## Data
- **Source**: EuroSAT RGB (Helber et al. 2019; Sentinel-2, CC BY-SA 4.0),
  27,000 64×64 patches, 10 classes, via Zenodo mirror (record 7711810).
- **Split**: stratified 70/15/15 (train 18,900 / val 4,050 / test 4,050),
  `random_state=42`.
- **Inputs**: RGB only (the 10 Sentinel-2 spectral bands collapsed to
  visual channels) — overlapping spectral signatures are the core difficulty.

## Model
- Custom 3-block CNN: 3× [Conv2D → BatchNorm → MaxPool → Dropout] →
  Dense(512) → softmax; 4,296,138 params.
- Adam(1e-3), categorical crossentropy; train augmentation (rotation 20°,
  shifts/shear/zoom 0.1, horizontal flip); eval rescale-only.
- ModelCheckpoint on `val_accuracy` + EarlyStopping on `val_loss`
  (patience 10, restore best).

## Metrics (held-out test, 4,050 patches — independent rerun)
- **Accuracy 0.8763**, loss 0.3675, macro F1 0.87.
- Strongest: SeaLake (F1 0.95), Forest (0.94), AnnualCrop (0.89).
  Weakest: HerbaceousVegetation (F1 0.79 — confused with
  Pasture/AnnualCrop), Industrial recall 0.78.
- The thesis notebook reported 0.9123 on the same protocol; the rerun
  scores 0.8763 with the identical error pattern (see README provenance
  note). Shipped artifact = the rerun.

## Limitations
- RGB only — multispectral bands unused; spectrally similar classes
  (crop/pasture/vegetation) are the error mass.
- Single dataset, single geography mix; no cross-sensor validation.
- No calibration; softmax scores are ranking scores.
- 64×64 patches lack landscape context beyond the tile.

## Ethics
EO mapping can support land administration and environmental monitoring;
misclassification at scale could misinform land-use decisions. Demo only —
any operational use needs multispectral inputs, multi-region validation,
and human review.
