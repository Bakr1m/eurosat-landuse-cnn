#!/usr/bin/env bash
# Fetch + unpack the EuroSAT RGB archive into data/ (gitignored).
# Source: Zenodo record 7711810 (community mirror of EuroSAT RGB,
# Sentinel-2; CC BY-SA 4.0, Helber et al. 2019).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d data/EuroSAT/EuroSAT_RGB ] || [ -d data/EuroSAT/2750 ]; then
  echo "EuroSAT patches already present — nothing to do."
  exit 0
fi

mkdir -p data
if [ ! -f data/EuroSAT_RGB.zip ]; then
  curl -fsSL -o data/EuroSAT_RGB.zip \
    https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip
fi
.venv/bin/python -c "from src.data import extract_eurosat; print(extract_eurosat())"
