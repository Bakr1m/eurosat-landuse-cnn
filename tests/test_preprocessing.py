"""Preprocessing tests: exact tensor contract + augmentation isolation."""
import numpy as np

from src.preprocessing import (
    EvalPreprocessing,
    TrainAugmentation,
    make_eval_datagen,
    make_train_datagen,
    preprocess_input,
)


def test_preprocess_input_exact_contract():
    img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype("uint8")
    out = preprocess_input(img)
    assert out.shape == (1, 64, 64, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_train_datagen_has_augmentation_eval_does_not():
    train_gen, eval_gen = make_train_datagen(), make_eval_datagen()
    assert train_gen.rotation_range == TrainAugmentation.rotation_range == 20
    assert train_gen.horizontal_flip is True
    assert eval_gen.rotation_range == 0
    assert eval_gen.horizontal_flip is False
    assert eval_gen.rescale == EvalPreprocessing.rescale == 1.0 / 255
