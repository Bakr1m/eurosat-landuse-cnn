"""Input pipeline: augmentation for train, rescale-only for eval.

The thesis used keras ImageDataGenerators; this module exposes the two
configs as data so tests can assert augmentation never leaks into eval.
"""
from dataclasses import dataclass
from typing import Any

IMG_SIZE = (64, 64)
BATCH_SIZE = 32


@dataclass(frozen=True)
class TrainAugmentation:
    """Train-only augmentation (thesis cell 4). Never used for val/test."""

    rescale: float = 1.0 / 255
    rotation_range: int = 20
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    shear_range: float = 0.1
    zoom_range: float = 0.1
    horizontal_flip: bool = True
    fill_mode: str = "nearest"


@dataclass(frozen=True)
class EvalPreprocessing:
    """Deterministic eval preprocessing: rescale only."""

    rescale: float = 1.0 / 255


def make_train_datagen():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    cfg = TrainAugmentation()
    return ImageDataGenerator(
        rescale=cfg.rescale,
        rotation_range=cfg.rotation_range,
        width_shift_range=cfg.width_shift_range,
        height_shift_range=cfg.height_shift_range,
        shear_range=cfg.shear_range,
        zoom_range=cfg.zoom_range,
        horizontal_flip=cfg.horizontal_flip,
        fill_mode=cfg.fill_mode,
    )


def make_eval_datagen():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator(rescale=EvalPreprocessing().rescale)


def flow_from_df(datagen, df, *, shuffle: bool):
    return datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=shuffle,
        seed=42,
    )


def preprocess_input(img_array: Any) -> Any:
    """Single-image eval path: float32, [0, 1], batched (1, 64, 64, 3)."""
    import numpy as np

    arr = np.asarray(img_array, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)
