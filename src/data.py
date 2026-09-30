"""EuroSAT dataset: download, verify, stratified split.

Source: EuroSAT RGB (Sentinel-2, 27,000 64x64 patches, 10 LULC classes),
via the Zenodo community mirror (record 7711810) — the same archive the
thesis notebook used after the primary host blocked datacenter IPs.
"""
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

EUROSAT_URL = "https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip"
EUROSAT_ZIP = DATA_DIR / "EuroSAT_RGB.zip"

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def download_eurosat(url: str = EUROSAT_URL, dest: Path = EUROSAT_ZIP) -> Path:
    """Fetch the EuroSAT RGB archive (skip if already present)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"archive already present: {dest}")
        return dest
    print(f"downloading EuroSAT RGB (~90 MB) from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"saved to {dest}")
    return dest


def extract_eurosat(archive: Path = EUROSAT_ZIP) -> Path:
    """Unpack the archive; return the directory holding the 10 class folders."""
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA_DIR / "EuroSAT")
    for candidate in ("EuroSAT_RGB", "2750"):
        root = DATA_DIR / "EuroSAT" / candidate
        if root.is_dir():
            return root
    raise FileNotFoundError("extracted EuroSAT class directory not found")


def build_dataframe(image_root: Path) -> pd.DataFrame:
    """One row per patch: filepath + label (sorted classes, stable order)."""
    filepaths, labels = [], []
    for class_name in sorted(os.listdir(image_root)):
        class_path = image_root / class_name
        if not class_path.is_dir():
            continue
        for filename in sorted(os.listdir(class_path)):
            filepaths.append(str(class_path / filename))
            labels.append(class_name)
    return pd.DataFrame({"filepath": filepaths, "label": labels})


def stratified_split(df: pd.DataFrame, seed: int = SEED):
    """70/15/15 stratified split, reproducible (thesis: random_state=42)."""
    train_val, test = train_test_split(
        df, test_size=TEST_RATIO, stratify=df["label"], random_state=seed
    )
    val_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(
        train_val, test_size=val_relative, stratify=train_val["label"], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
