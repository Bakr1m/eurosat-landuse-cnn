"""Model tests: architecture fidelity + SmallCNN stand-in contract."""
import numpy as np
import tensorflow as tf

from src.model import INPUT_SHAPE, NUM_CLASSES, create_cnn_model


def test_create_cnn_model_matches_thesis():
    model = create_cnn_model()
    assert model.input_shape[1:] == INPUT_SHAPE
    assert model.output_shape[-1] == NUM_CLASSES
    total = model.count_params()
    assert total == 4296138, total  # thesis summary: 4,296,138 params


def test_smallcnn_standin_trains_on_synthetic_tiles():
    """Contract: the training loop works end-to-end on fake data."""
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential

    rng = np.random.default_rng(0)
    X = rng.random((32, 64, 64, 3), dtype=np.float32)
    y = tf.keras.utils.to_categorical(rng.integers(0, 10, 32), 10)
    standin = Sequential(
        [Conv2D(8, 3, activation="relu", input_shape=INPUT_SHAPE),
         MaxPooling2D(8), Flatten(), Dense(10, activation="softmax")]
    )
    standin.compile(optimizer="adam", loss="categorical_crossentropy")
    history = standin.fit(X, y, epochs=1, verbose=0)
    assert "loss" in history.history
    assert standin.predict(X[:2], verbose=0).shape == (2, 10)
