# Notebooks — exploratory trail (docs, not the deploy path)

- `01_eurosat_cnn_thesis.ipynb` — original final-thesis notebook: EuroSAT
  download, 70/15/15 stratified split, augmentation, custom 3-block CNN,
  ModelCheckpoint + EarlyStopping, test accuracy 0.9123, per-class report +
  confusion matrix, single-image inference demo. Colab/Drive-specific cells
  are kept as history; the reproducible path is `src/` + `make train`.
