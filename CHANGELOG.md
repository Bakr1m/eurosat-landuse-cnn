# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-09-30

### Added
- Custom 3-block CNN for 10-class EuroSAT LULC (test accuracy 0.8763 on
  the independent `make train` rerun; thesis reported 0.9123 — see README),
  refactored from the final-thesis notebook into `src/`:
  `data` (download + stratified 70/15/15 split), `preprocessing`
  (train augmentation isolated from eval rescale), `model` (4,296,138
  params), `train` (checkpoint + early stopping), `serve` (FastAPI).
- FastAPI `/predict` (image upload → class + confidence + probabilities),
  lazy model load, 422 on non-image uploads.
- Hermetic pytest suite (split stratification, exact tensor contract,
  architecture param-count fidelity, SmallCNN stand-in, serving contract);
  ruff lint; GitHub Actions CI with test gate.
- Dockerized serving image (`bakr1m/eurosat-api`) built from the
  release-pinned `.keras` artifact (SHA256-verified), parity-checked.
- Professional repo hygiene: LICENSE, CONTRIBUTING, CHANGELOG, CI workflow,
  Makefile, model card, thesis notebook preserved under `notebooks/`.
