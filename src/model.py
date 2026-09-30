"""Custom 3-block CNN for EuroSAT (thesis cell 5, verbatim architecture).

3x [Conv2D -> BatchNorm -> MaxPool -> Dropout] -> Dense(512) -> softmax.
4,296,138 params (16.39 MB). Adam(1e-3), categorical crossentropy.
"""
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

INPUT_SHAPE = (64, 64, 3)
NUM_CLASSES = 10
LEARNING_RATE = 0.001


def create_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(512, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
